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
	"slices"
	"strings"

	"google.golang.org/protobuf/proto"
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

// ParseRequest decodes a framed sglang.api.v1.GenerateRequest into an InferenceRequestBody.
func (p *SGLangGRPCParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := strings.TrimSuffix(strings.TrimSpace(fwkrequest.GetRequestPath(headers)), "/")
	if path != sglangapiv1.SglangService_Generate_FullMethodName && path != sglangGeneratePath {
		return nil, fmt.Errorf("unsupported gRPC path: %s", path)
	}

	parsedPayload, err := grpcframe.ParsePayload(body)
	if err != nil {
		return nil, fmt.Errorf("invalid gRPC generate request: %w", err)
	}

	var req sglangapiv1.GenerateRequest
	if err := proto.Unmarshal(parsedPayload, &req); err != nil {
		return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
	}

	extracted, err := convertGenerateRequest(&req)
	if err != nil {
		return nil, err
	}
	log.FromContext(ctx).V(logutil.TRACE).Info("parsed SGLang gRPC GenerateRequest")
	return &fwkrh.ParseResult{Body: extracted, SkipResponseProcessing: false}, nil
}

func convertGenerateRequest(req *sglangapiv1.GenerateRequest) (*fwkrequest.InferenceRequestBody, error) {
	if err := rejectBatchFields(req); err != nil {
		return nil, err
	}

	hasInputIDs := req.GetInputIds() != nil && req.GetInputIds().GetValue() != nil
	hasText := req.GetText() != nil && req.GetText().GetValue() != nil
	if !hasInputIDs && !hasText {
		return nil, errors.New("invalid generate request: input_ids or text must be provided")
	}

	var tokenIDs []uint32
	var text string
	if hasInputIDs {
		ids := req.GetInputIds().GetOne().GetIds()
		if len(ids) == 0 {
			return nil, errors.New("invalid generate request: input_ids cannot be empty")
		}
		tokenIDs = make([]uint32, len(ids))
		for i, id := range ids {
			if id < 0 || id > math.MaxUint32 {
				return nil, fmt.Errorf("invalid generate request: input_ids[%d] out of uint32 range: %d", i, id)
			}
			tokenIDs[i] = uint32(id) //#nosec G115 -- guarded by [0, math.MaxUint32] check above
		}
	} else {
		text = req.GetText().GetOne()
		if text == "" {
			return nil, errors.New("invalid generate request: text cannot be empty")
		}
	}

	cacheSalt, err := parsers.ComposeSGLangCacheSalt(
		req.GetExtraKey().GetOne(),
		req.GetCacheSalt().GetOne(),
	)
	if err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}

	var maxTokens *int64
	if sp := req.GetSamplingParams().GetOne(); sp != nil && sp.MaxNewTokens != nil && *sp.MaxNewTokens >= 0 {
		v := *sp.MaxNewTokens
		maxTokens = &v
	}

	return &fwkrequest.InferenceRequestBody{
		Generate: &fwkrequest.GenerateRequest{
			TokenIDs:  tokenIDs,
			Text:      text,
			CacheSalt: cacheSalt,
			Features:  extractMMFeatures(req),
		},
		Payload:         fwkrequest.PayloadProto{Message: req},
		MaxOutputTokens: maxTokens,
		Stream:          req.GetStream(),
	}, nil
}

func rejectBatchFields(req *sglangapiv1.GenerateRequest) error {
	if _, ok := req.GetRid().GetValue().(*sglangapiv1.StringOrList_Many); ok {
		return errors.New("unsupported generate request: batched rid is not supported")
	}
	if _, ok := req.GetInputIds().GetValue().(*sglangapiv1.TokenIdsOrList_Many); ok {
		return errors.New("unsupported generate request: batched input_ids is not supported")
	}
	if _, ok := req.GetText().GetValue().(*sglangapiv1.StringOrList_Many); ok {
		return errors.New("unsupported generate request: batched text is not supported")
	}
	if _, ok := req.GetCacheSalt().GetValue().(*sglangapiv1.StringOrList_Many); ok {
		return errors.New("unsupported generate request: cache_salt must be a single string")
	}
	if _, ok := req.GetExtraKey().GetValue().(*sglangapiv1.StringOrList_Many); ok {
		return errors.New("unsupported generate request: extra_key must be a single string")
	}
	if _, ok := req.GetSamplingParams().GetValue().(*sglangapiv1.SamplingParamsOrList_Many); ok {
		return errors.New("unsupported generate request: batched sampling_params is not supported")
	}
	if _, ok := req.GetMmHashes().GetValue().(*sglangapiv1.StringListOrList_Many); ok {
		return errors.New("unsupported generate request: batched mm_hashes is not supported")
	}
	return nil
}

func extractMMFeatures(req *sglangapiv1.GenerateRequest) *tokenization.MultiModalFeatures {
	hashes := req.GetMmHashes().GetOne().GetItems()
	if len(hashes) == 0 {
		return nil
	}
	modality := reqcommon.ModalityImage
	switch {
	case req.GetVideoData() != nil && req.GetImageData() == nil && req.GetAudioData() == nil:
		modality = reqcommon.ModalityVideo
	case req.GetAudioData() != nil && req.GetImageData() == nil && req.GetVideoData() == nil:
		modality = reqcommon.ModalityAudio
	}
	return &tokenization.MultiModalFeatures{
		MMHashes: map[string][]string{
			string(modality): slices.Clone(hashes),
		},
	}
}

// RewritePriority sets the EPP-resolved priority on a cloned GenerateRequest message.
func (p *SGLangGRPCParser) RewritePriority(_ fwkrh.PriorityRewriteContext, payload fwkrequest.MarshalablePayload, priority int) (fwkrequest.MarshalablePayload, bool, error) {
	pp, ok := payload.(fwkrequest.PayloadProto)
	if !ok {
		return payload, false, nil
	}
	req, ok := pp.Message.(*sglangapiv1.GenerateRequest)
	if !ok || req == nil {
		return payload, false, nil
	}
	cloned, ok := proto.Clone(req).(*sglangapiv1.GenerateRequest)
	if !ok {
		return payload, false, nil
	}
	cloned.Priority = proto.Int64(int64(priority))
	return fwkrequest.PayloadProto{Message: cloned}, true, nil
}

// ParseResponse extracts streamed frame counts and terminal token usage from
// framed sglang.api.v1.GenerateStreamItem messages.
func (p *SGLangGRPCParser) ParseResponse(_ context.Context, body []byte, _ map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}

	var errs []error
	payloads, err := grpcframe.ParsePayloads(body)
	if err != nil {
		errs = append(errs, fmt.Errorf("parsing gRPC response payload: %w", err))
	}

	var result *fwkrh.ParsedResponse
	for _, payload := range payloads {
		var item sglangapiv1.GenerateStreamItem
		if err := proto.Unmarshal(payload, &item); err != nil {
			errs = append(errs, fmt.Errorf("unmarshaling GenerateStreamItem: %w", err))
			continue
		}
		if result == nil {
			result = &fwkrh.ParsedResponse{}
		}
		frame := item.GetFrame()
		if frame == nil {
			continue
		}
		result.StreamedEvents++
		meta := frame.GetMetaInfo()
		if meta == nil || meta.GetFinishReason() == nil {
			continue
		}
		promptTokens := clampUint64ToInt(uint64(meta.GetPromptTokens()))
		completionTokens := clampUint64ToInt(meta.GetCompletionTokens())
		if promptTokens > 0 || completionTokens > 0 {
			result.Usage = &fwkrequest.Usage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
				TotalTokens:      promptTokens + completionTokens,
				PromptTokenDetails: &fwkrequest.PromptTokenDetails{
					CachedTokens: clampUint64ToInt(uint64(meta.GetCachedTokens())),
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
