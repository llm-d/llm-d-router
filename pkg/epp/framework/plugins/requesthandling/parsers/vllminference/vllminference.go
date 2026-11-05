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

// Package vllminference provides a request and response parser for the
// upstream vLLM gRPC service (vllm.Inference).
package vllminference

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
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

const (
	// VllmInferenceParserType is the canonical type name used to register the plugin.
	VllmInferenceParserType = "vllminference-parser"
)

var (
	vllmGeneratePath       = strings.TrimPrefix(vllmpb.Inference_Generate_FullMethodName, "/")
	vllmGenerateStreamPath = strings.TrimPrefix(vllmpb.Inference_GenerateStream_FullMethodName, "/")
)

var (
	_ fwkrh.Parser            = &VllmInferenceParser{}
	_ fwkrh.ModelNameRewriter = &VllmInferenceParser{}
	_ fwkrh.PriorityRewriter  = &VllmInferenceParser{}
)

// VllmInferenceParser implements fwkrh.Parser for vllm.Inference Generate and
// GenerateStream RPCs.
type VllmInferenceParser struct {
	typedName fwkplugin.TypedName
}

// NewVllmInferenceParser creates a new VllmInferenceParser.
func NewVllmInferenceParser() *VllmInferenceParser {
	return &VllmInferenceParser{
		typedName: fwkplugin.TypedName{
			Type: VllmInferenceParserType,
			Name: VllmInferenceParserType,
		},
	}
}

// VllmInferenceParserPluginFactory is the factory function used to register the plugin.
func VllmInferenceParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewVllmInferenceParser().WithName(name), nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *VllmInferenceParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// WithName sets the plugin instance name.
func (p *VllmInferenceParser) WithName(name string) *VllmInferenceParser {
	p.typedName.Name = name
	return p
}

// Claims returns the gRPC method paths and protocol claimed by this parser.
func (p *VllmInferenceParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths: []string{
			vllmGeneratePath,
			vllmGenerateStreamPath,
		},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C},
	}
}

// ParseRequest reads a framed vllm.GenerateRequest into an InferenceRequestBody.
// Through vllmwire, it scans the protobuf wire format without copying media
// payloads or MessagePack tensors.
func (p *VllmInferenceParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := strings.TrimSuffix(strings.TrimSpace(fwkrequest.GetRequestPath(headers)), "/")
	var stream bool
	switch path {
	case vllmpb.Inference_Generate_FullMethodName, vllmGeneratePath:
		stream = false
	case vllmpb.Inference_GenerateStream_FullMethodName, vllmGenerateStreamPath:
		stream = true
	default:
		return nil, fmt.Errorf("unsupported gRPC path: %s", path)
	}

	payload, err := parsers.ParseUnaryFrame(body)
	if err != nil {
		return nil, fmt.Errorf("invalid gRPC generate request: %w", err)
	}

	req, err := vllmwire.ScanGenerateRequest(payload)
	if err != nil {
		return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
	}

	extracted, err := convertGenerateRequest(&req, stream)
	if err != nil {
		return nil, err
	}
	extracted.RawBody = body
	extracted.Payload = fwkrequest.PayloadBytes(body)
	if logger := log.FromContext(ctx).V(logutil.TRACE); logger.Enabled() {
		logger.Info("parsed vllm.Inference GenerateRequest", "stream", stream)
	}
	return &fwkrh.ParseResult{Body: extracted, SkipResponseProcessing: false}, nil
}

func convertGenerateRequest(req *vllmwire.Request, stream bool) (*fwkrequest.InferenceRequestBody, error) {
	if n, ok := req.NumSequences(); ok && n > 1 {
		return nil, errors.New("unsupported generate request: num_sequences > 1 is not supported")
	}

	var tokenIDs []uint32
	var text string
	switch req.Prompt {
	case vllmwire.PromptTokenIDs:
		ids, err := req.TokenIDs()
		if err != nil {
			return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
		}
		if len(ids) == 0 {
			return nil, errors.New("invalid generate request: token_ids cannot be empty")
		}
		tokenIDs = ids
	case vllmwire.PromptText:
		text = string(req.Text())
		if text == "" {
			return nil, errors.New("invalid generate request: text cannot be empty")
		}
	default:
		return nil, errors.New("invalid generate request: prompt (text or token_ids) must be provided")
	}

	features, err := extractMMFeatures(req)
	if err != nil {
		return nil, err
	}

	// Upstream vLLM validates GenerateRequest.model against base model names and
	// resolves LoRA adapters from GenerateRequest.lora_name. EPP scorers and
	// KV-cache block keys identify LoRA requests by TargetModel.
	model := req.LoraName()
	if model == "" {
		model = req.Model()
	}

	var maxTokens *int64
	if maxNew, ok := req.MaxNewTokens(); ok && maxNew > 0 {
		v := int64(maxNew)
		maxTokens = &v
	}

	salt, _ := req.CacheSalt()

	return &fwkrequest.InferenceRequestBody{
		Model: model,
		Generate: &fwkrequest.GenerateRequest{
			TokenIDs:  tokenIDs,
			Text:      text,
			CacheSalt: string(salt),
			Features:  features,
		},
		MaxOutputTokens: maxTokens,
		Stream:          stream,
	}, nil
}

func extractMMFeatures(req *vllmwire.Request) (*tokenization.MultiModalFeatures, error) {
	if !req.HasMedia() {
		return nil, nil //nolint:nilnil
	}
	items, err := req.MediaItems()
	if err != nil {
		return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
	}
	var hashes map[string][]string
	var placeholders map[string][]kvblock.PlaceholderRange
	for i := range items {
		item := &items[i]
		if !item.HasFeatures {
			if item.UUID == "" {
				continue
			}
			mod, err := modalityName(item.Modality)
			if err != nil {
				return nil, err
			}
			if hashes == nil {
				hashes = make(map[string][]string, 1)
			}
			key := string(mod)
			hashes[key] = append(hashes[key], item.UUID)
			continue
		}
		mod, err := modalityName(item.Modality)
		if err != nil {
			return nil, err
		}
		if item.Offset > math.MaxInt || item.Length > math.MaxInt {
			return nil, errors.New("invalid generate request: media placeholder range exceeds int limits")
		}
		hash := string(item.Identifier)
		if hash == "" && item.HasMMHash {
			hash = string(item.MMHash)
		}
		if hash == "" {
			hash = item.UUID
		}
		if hash == "" {
			continue
		}
		if hashes == nil {
			hashes = make(map[string][]string, 1)
		}
		if placeholders == nil {
			placeholders = make(map[string][]kvblock.PlaceholderRange, 1)
		}
		key := string(mod)
		hashes[key] = append(hashes[key], hash)
		placeholders[key] = append(placeholders[key], kvblock.PlaceholderRange{
			Offset: int(item.Offset), //#nosec G115 -- guarded by MaxInt check above
			Length: int(item.Length), //#nosec G115 -- guarded by MaxInt check above
		})
	}
	if len(hashes) == 0 {
		return nil, nil //nolint:nilnil
	}
	return &tokenization.MultiModalFeatures{
		MMHashes:       hashes,
		MMPlaceholders: placeholders,
	}, nil
}

func modalityName(m vllmpb.Modality) (reqcommon.Modality, error) {
	switch m {
	case vllmpb.Modality_MODALITY_IMAGE:
		return reqcommon.ModalityImage, nil
	case vllmpb.Modality_MODALITY_VIDEO:
		return reqcommon.ModalityVideo, nil
	case vllmpb.Modality_MODALITY_AUDIO:
		return reqcommon.ModalityAudio, nil
	default:
		return "", errors.New("invalid generate request: media modality is required")
	}
}

// RewriteModelName writes the resolved model into the framed GenerateRequest in
// one allocation. When lora_name is non-empty, it rewrites lora_name and keeps
// GenerateRequest.model as the base model. A target equal to the base model
// clears lora_name, because an empty lora_name selects the base model.
func (p *VllmInferenceParser) RewriteModelName(payload fwkrequest.MarshalablePayload, model string) (fwkrequest.MarshalablePayload, error) {
	raw, ok := payload.(fwkrequest.PayloadBytes)
	if !ok {
		return payload, nil
	}
	body, err := grpcframe.ParsePayload(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid gRPC generate request: %w", err)
	}
	req, err := vllmwire.ScanGenerateRequest(body)
	if err != nil {
		return nil, fmt.Errorf("rewriting model name: %w", err)
	}
	rewrite := vllmwire.RewriteModel
	if req.LoraName() != "" {
		rewrite = vllmwire.RewriteLoraName
		if model == req.Model() {
			model = ""
		}
	}
	framed, err := rewrite(raw, model)
	if err != nil {
		return nil, fmt.Errorf("rewriting model name: %w", err)
	}
	return fwkrequest.PayloadBytes(framed), nil
}

// RewritePriority negates the EPP priority (higher is more urgent) to match
// the vLLM scheduler (lower is more urgent) and splices it into the framed
// GenerateRequest in one allocation. When the client omits priority and the EPP
// priority is 0, RewritePriority returns payload without copying.
func (p *VllmInferenceParser) RewritePriority(_ fwkrh.PriorityRewriteContext, payload fwkrequest.MarshalablePayload, priority int) (fwkrequest.MarshalablePayload, bool, error) {
	raw, ok := payload.(fwkrequest.PayloadBytes)
	if !ok {
		return payload, false, nil
	}
	framed, err := vllmwire.RewritePriority(raw, toVLLMPriority(priority))
	if err != nil {
		return nil, false, fmt.Errorf("rewriting priority: %w", err)
	}
	if &framed[0] == &raw[0] {
		return payload, false, nil
	}
	return fwkrequest.PayloadBytes(framed), true, nil
}

// toVLLMPriority negates EPP priority (sourced from InferenceObjective's int32
// Spec.Priority) into vLLM scheduler priority.
func toVLLMPriority(priority int) int32 {
	if priority <= math.MinInt32 {
		return math.MaxInt32
	}
	return -int32(priority) //#nosec G115 -- InferenceObjective.Spec.Priority is int32
}

// ParseResponse scans framed vllm.GenerateResponse messages for token usage and
// streamed output events. GenerateStream emits prompt_info on the start frame
// and finish_info on the final frame. The response handler merges usage across
// chunks.
func (p *VllmInferenceParser) ParseResponse(_ context.Context, body []byte, _ map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}

	var (
		result           *fwkrh.ParsedResponse
		errs             []error
		promptTokens     int
		completionTokens int
		hasCachedTokens  bool
		cachedTokens     int
	)
	if err := grpcframe.ForEachPayload(body, func(payload []byte) {
		item, err := vllmwire.ScanGenerateResponse(payload)
		if err != nil {
			errs = append(errs, fmt.Errorf("unmarshaling GenerateResponse: %w", err))
			return
		}
		if result == nil {
			result = &fwkrh.ParsedResponse{}
		}
		if item.HasOutputs {
			result.StreamedEvents++
		}
		if item.HasPromptInfo && item.NumPromptTokens > 0 {
			promptTokens = parsers.ClampToInt(item.NumPromptTokens)
		}
		if item.HasFinishInfo {
			if item.NumOutputTokens > 0 {
				completionTokens = parsers.ClampToInt(item.NumOutputTokens)
			}
			if item.HasCachedTokens {
				hasCachedTokens = true
				cachedTokens = parsers.ClampToInt(item.NumCachedTokens)
			}
		}
	}); err != nil {
		errs = append(errs, fmt.Errorf("parsing gRPC response payload: %w", err))
	}

	if result != nil && (promptTokens > 0 || completionTokens > 0 || hasCachedTokens) {
		var cached *int
		if hasCachedTokens {
			cached = &cachedTokens
		}
		result.Usage = parsers.NewUsageOptionalCached(promptTokens, completionTokens, cached)
	}
	return result, errors.Join(errs...)
}
