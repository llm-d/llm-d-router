/*
Copyright 2026 The Kubernetes Authors.
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

package parsers

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwkplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins"
)

const (
	// MethodPathKey is the header key for the request path.
	MethodPathKey = ":path"
)

// NewUsage builds a Usage struct from prompt, completion, and cached prompt
// token counts.
func NewUsage(promptTokens, completionTokens, cachedTokens int) *fwkrequest.Usage {
	return NewUsageOptionalCached(promptTokens, completionTokens, &cachedTokens)
}

// NewUsageOptionalCached is NewUsage for an engine that can omit the cached
// prompt token count. A nil cachedTokens leaves PromptTokenDetails nil.
func NewUsageOptionalCached(promptTokens, completionTokens int, cachedTokens *int) *fwkrequest.Usage {
	usage := &fwkrequest.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	}
	if cachedTokens != nil {
		usage.PromptTokenDetails = &fwkrequest.PromptTokenDetails{
			CachedTokens: *cachedTokens,
		}
	}
	return usage
}

// ParseUnaryFrame returns the payload of body, which must hold exactly one
// gRPC frame. A unary request carries one message, so bytes after the frame
// are a malformed request that the parser rejects before scheduling runs.
func ParseUnaryFrame(body []byte) ([]byte, error) {
	payload, err := grpcframe.ParsePayload(body)
	if err != nil {
		return nil, err
	}
	if grpcframe.HeaderLen+len(payload) != len(body) {
		return nil, fmt.Errorf("%w: %d bytes after the frame", grpcframe.ErrMalformed, len(body)-grpcframe.HeaderLen-len(payload))
	}
	return payload, nil
}

// ClampToInt converts an unsigned token count from a protobuf message to int.
// It saturates at math.MaxInt.
func ClampToInt[T ~uint32 | ~uint64](v T) int {
	if uint64(v) > uint64(math.MaxInt) {
		return math.MaxInt
	}
	return int(v) //#nosec G115 -- guarded by MaxInt check above
}

// ComposeSGLangCacheSalt combines SGLang's extra_key and cache_salt into the
// canonical cache_salt string used for prefix-cache indexing. SGLang partitions
// its cache on (extra_key, cache_salt) while publishing only cache_salt in KV
// events; encoding the pair keeps extra_key requests from matching blocks
// indexed under a bare cache_salt.
func ComposeSGLangCacheSalt(extraKey, cacheSalt string) (string, error) {
	if extraKey == "" {
		return cacheSalt, nil
	}
	pair, err := json.Marshal([]string{extraKey, cacheSalt})
	if err != nil {
		return "", err
	}
	return string(pair), nil
}

// RewritePriority strips any client-supplied priority from a JSON-map payload and
// writes the EPP-resolved priority; it reports whether the payload changed.
// Non-map payloads are returned unchanged. Callers decide whether priority
// propagation is enabled (see requestHandler.propagatePriority); this helper
// always injects when invoked. EPP priority follows SGLang semantics (higher =
// more urgent), matching the InferenceObjective convention; vLLM's native
// scheduler is the opposite (lower = more urgent), so the value is negated for
// vLLM targets to keep the same relative ordering across backends.
func RewritePriority(ctx fwkrh.PriorityRewriteContext, payload fwkrequest.MarshalablePayload, priority int) (fwkrequest.MarshalablePayload, bool, error) {
	m, ok := payload.(fwkrequest.PayloadMap)
	if !ok {
		return payload, false, nil
	}
	delete(m, "priority")
	if isVLLMTarget(ctx) {
		priority = -priority
	}
	m["priority"] = priority
	return m, true, nil
}

// isVLLMTarget reports whether the scheduled endpoint is a vLLM backend.
func isVLLMTarget(ctx fwkrh.PriorityRewriteContext) bool {
	meta := ctx.TargetEndpoint
	if meta == nil || meta.Labels == nil {
		return false
	}
	return strings.EqualFold(meta.Labels[fwkplugins.EngineTypeLabelKey], "vllm")
}
