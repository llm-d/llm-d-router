/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package sglanggrpc provides a request and response parser for SGLang's native
// gRPC API (sglang.api.v1.SglangService).
package sglanggrpc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

const (
	// SGLangGRPCParserType is the canonical type name used to register the plugin.
	SGLangGRPCParserType = "sglanggrpc-parser"
)

var sglangGeneratePath = strings.TrimPrefix(sglangapiv1.SglangService_Generate_FullMethodName, "/")

var (
	_ fwkrh.Parser           = &SGLangGRPCParser{}
	_ fwkrh.PriorityRewriter = &SGLangGRPCParser{}
)

// SGLangGRPCParser implements fwkrh.Parser for SGLang's native gRPC Generate RPC.
type SGLangGRPCParser struct {
	typedName fwkplugin.TypedName
}

// NewSGLangGRPCParser creates a new SGLangGRPCParser.
func NewSGLangGRPCParser() *SGLangGRPCParser {
	return &SGLangGRPCParser{
		typedName: fwkplugin.TypedName{
			Type: SGLangGRPCParserType,
			Name: SGLangGRPCParserType,
		},
	}
}

// SGLangGRPCParserPluginFactory is the factory function used to register the plugin.
func SGLangGRPCParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewSGLangGRPCParser().WithName(name), nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *SGLangGRPCParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// WithName sets the plugin instance name.
func (p *SGLangGRPCParser) WithName(name string) *SGLangGRPCParser {
	p.typedName.Name = name
	return p
}

// Claims returns the gRPC method path and protocol claimed by this parser.
func (p *SGLangGRPCParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths:     []string{sglangGeneratePath},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C},
	}
}

// ParseRequest reads a framed sglang.api.v1.GenerateRequest into an
// InferenceRequestBody. It scans the wire format for the fields the EPP needs
// and never decodes the full message, so media and token IDs are not copied
// into Go memory. The original bytes are forwarded unless a rewriter changes
// them.
func (p *SGLangGRPCParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := strings.TrimSuffix(strings.TrimSpace(fwkrequest.GetRequestPath(headers)), "/")
	if path != sglangapiv1.SglangService_Generate_FullMethodName && path != sglangGeneratePath {
		return nil, fmt.Errorf("unsupported gRPC path: %s", path)
	}

	payload, err := grpcframe.ParsePayload(body)
	if err != nil {
		return nil, fmt.Errorf("invalid gRPC generate request: %w", err)
	}

	scan, err := scanGenerateRequest(payload)
	if err != nil {
		return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
	}

	extracted, err := convertGenerateRequest(&scan)
	if err != nil {
		return nil, err
	}
	extracted.RawBody = body
	extracted.Payload = fwkrequest.PayloadBytes(body)
	log.FromContext(ctx).V(logutil.TRACE).Info("parsed SGLang gRPC GenerateRequest")
	return &fwkrh.ParseResult{Body: extracted, SkipResponseProcessing: false}, nil
}

func convertGenerateRequest(scan *generateRequestScan) (*fwkrequest.InferenceRequestBody, error) {
	if err := rejectBatchFields(scan); err != nil {
		return nil, err
	}

	hasInputIDs := scan.inputIDs.present
	hasText := scan.text.present
	if !hasInputIDs && !hasText {
		return nil, errors.New("invalid generate request: input_ids or text must be provided")
	}

	var tokenIDs []uint32
	var text string
	if hasInputIDs {
		ids, err := scanTokenIDs(scan.inputIDs.one)
		if err != nil {
			return nil, fmt.Errorf("invalid generate request: %w", err)
		}
		if len(ids) == 0 {
			return nil, errors.New("invalid generate request: input_ids cannot be empty")
		}
		tokenIDs = ids
	} else {
		text = string(scan.text.one)
		if text == "" {
			return nil, errors.New("invalid generate request: text cannot be empty")
		}
	}

	cacheSalt, err := parsers.ComposeSGLangCacheSalt(string(scan.extraKey.one), string(scan.cacheSalt.one))
	if err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}

	var maxTokens *int64
	if scan.samplingParams.present {
		v, ok, err := scanMaxNewTokens(scan.samplingParams.one)
		if err != nil {
			return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
		}
		if ok && v >= 0 {
			maxTokens = &v
		}
	}

	features, err := extractMMFeatures(scan)
	if err != nil {
		return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
	}

	return &fwkrequest.InferenceRequestBody{
		Generate: &fwkrequest.GenerateRequest{
			TokenIDs:  tokenIDs,
			Text:      text,
			CacheSalt: cacheSalt,
			Features:  features,
		},
		MaxOutputTokens: maxTokens,
		Stream:          scan.stream,
	}, nil
}

func rejectBatchFields(scan *generateRequestScan) error {
	switch {
	case scan.rid.many:
		return errors.New("unsupported generate request: batched rid is not supported")
	case scan.inputIDs.many:
		return errors.New("unsupported generate request: batched input_ids is not supported")
	case scan.text.many:
		return errors.New("unsupported generate request: batched text is not supported")
	case scan.cacheSalt.many:
		return errors.New("unsupported generate request: cache_salt must be a single string")
	case scan.extraKey.many:
		return errors.New("unsupported generate request: extra_key must be a single string")
	case scan.samplingParams.many:
		return errors.New("unsupported generate request: batched sampling_params is not supported")
	case scan.mmHashes.many:
		return errors.New("unsupported generate request: batched mm_hashes is not supported")
	}
	return nil
}

func extractMMFeatures(scan *generateRequestScan) (*tokenization.MultiModalFeatures, error) {
	if !scan.mmHashes.present {
		return nil, nil //nolint:nilnil
	}
	hashes, err := scanStringList(scan.mmHashes.one)
	if err != nil {
		return nil, err
	}
	if len(hashes) == 0 {
		return nil, nil //nolint:nilnil
	}
	modality := reqcommon.ModalityImage
	switch {
	case scan.hasVideo && !scan.hasImage && !scan.hasAudio:
		modality = reqcommon.ModalityVideo
	case scan.hasAudio && !scan.hasImage && !scan.hasVideo:
		modality = reqcommon.ModalityAudio
	}
	return &tokenization.MultiModalFeatures{
		MMHashes: map[string][]string{string(modality): hashes},
	}, nil
}

// RewritePriority sets the EPP-resolved priority on the raw GenerateRequest
// frame: it copies the payload without any client priority field and appends
// the new one, in one allocation.
func (p *SGLangGRPCParser) RewritePriority(_ fwkrh.PriorityRewriteContext, payload fwkrequest.MarshalablePayload, priority int) (fwkrequest.MarshalablePayload, bool, error) {
	raw, ok := payload.(fwkrequest.PayloadBytes)
	if !ok {
		return payload, false, nil
	}
	body, err := grpcframe.ParsePayload(raw)
	if err != nil {
		return nil, false, fmt.Errorf("invalid gRPC generate request: %w", err)
	}
	framed, err := framePriority(body, int64(priority))
	if err != nil {
		return nil, false, fmt.Errorf("rewriting priority: %w", err)
	}
	return fwkrequest.PayloadBytes(framed), true, nil
}

// ParseResponse counts streamed frames and reads token usage from terminal
// GenerateStreamItem frames. It scans each frame at the wire level and never
// decodes a message, so a token frame costs no allocation beyond the result.
//
// A request with sampling_params.n > 1 ends with one terminal frame per
// index. The usage reported here is that of the last terminal frame in the
// chunk; the handler merges chunks with last-wins semantics, so output tokens
// of the other indexes are not counted.
func (p *SGLangGRPCParser) ParseResponse(_ context.Context, body []byte, _ map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}

	var result *fwkrh.ParsedResponse
	var errs []error
	for len(body) > 0 {
		payload, err := grpcframe.ParsePayload(body)
		if err != nil {
			errs = append(errs, fmt.Errorf("parsing gRPC response payload: %w", err))
			break
		}
		body = body[grpcframe.HeaderLen+len(payload):]

		item, err := scanStreamItem(payload)
		if err != nil {
			errs = append(errs, fmt.Errorf("unmarshaling GenerateStreamItem: %w", err))
			continue
		}
		if result == nil {
			result = &fwkrh.ParsedResponse{}
		}
		if !item.hasFrame {
			continue
		}
		result.StreamedEvents++
		if !item.finished {
			continue
		}
		promptTokens := clampUint64ToInt(uint64(item.promptTokens))
		completionTokens := clampUint64ToInt(item.completionTokens)
		if promptTokens > 0 || completionTokens > 0 {
			result.Usage = &fwkrequest.Usage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
				TotalTokens:      promptTokens + completionTokens,
				PromptTokenDetails: &fwkrequest.PromptTokenDetails{
					CachedTokens: clampUint64ToInt(uint64(item.cachedTokens)),
				},
			}
		}
	}
	return result, errors.Join(errs...)
}

func clampUint64ToInt(v uint64) int {
	if v > uint64(math.MaxInt) {
		return math.MaxInt
	}
	return int(v) //#nosec G115 -- guarded by MaxInt check above
}
