/*
Copyright 2026 The Kubernetes Authors.

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
	"strings"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwkplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins"
)

const (
	// MethodPathKey is the header key for the request path.
	MethodPathKey = ":path"
)

// RewritePriority strips any client-supplied priority from a JSON-map payload and
// writes the EPP-resolved priority; it reports whether the payload changed.
// Non-map payloads are returned unchanged. Callers decide whether priority
// propagation is enabled (see requestHandler.propagatePriority); this helper
// always injects when invoked. EPP priority follows SGLang semantics (higher =
// more urgent), matching the InferenceObjective convention; vLLM's native
// scheduler is the opposite (lower = more urgent), so the value is negated for
// vLLM targets to keep the same relative ordering across backends.
func RewritePriority(ctx fwkrh.PriorityRewriteContext, payload fwkrh.MarshalablePayload, priority int) (fwkrh.MarshalablePayload, bool, error) {
	m, ok := payload.(fwkrh.PayloadMap)
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
	engineType := meta.Labels[fwkplugins.EngineTypeLabelKey]
	if engineType == "" {
		// Fall back to the pre-migration GAIE label for backward compatibility.
		engineType = meta.Labels["inference.networking.k8s.io/engine-type"]
	}
	return strings.EqualFold(engineType, "vllm")
}
