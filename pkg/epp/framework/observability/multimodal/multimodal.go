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

// Package multimodal provides span-attribute helpers derived from a request's
// multimodal features. It lives under pkg/epp so the sidecar build (which only
// pulls pkg/common) does not pick up EPP-internal imports.
package multimodal

import (
	"sort"
	"strings"

	"go.opentelemetry.io/otel/attribute"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ModalityNone is the mm.modality value when no multimodal content is present.
const ModalityNone = "none"

// SpanAttributes derives mm.modality and mm.hash_count from req.
func SpanAttributes(req *scheduling.InferenceRequest) []attribute.KeyValue {
	modality, hashCount := Summary(req)
	return []attribute.KeyValue{
		attribute.String("mm.modality", modality),
		attribute.Int("mm.hash_count", hashCount),
	}
}

// Summary returns the modality string (sorted comma-joined, or "none") and
// hash count derived from req.
func Summary(req *scheduling.InferenceRequest) (modality string, hashCount int) {
	features := requestMMFeatures(req)
	if len(features) == 0 {
		return ModalityNone, 0
	}
	seen := map[string]struct{}{}
	for _, f := range features {
		if f.Modality == "" {
			continue
		}
		seen[string(f.Modality)] = struct{}{}
	}
	if len(seen) == 0 {
		return ModalityNone, len(features)
	}
	modalities := make([]string, 0, len(seen))
	for m := range seen {
		modalities = append(modalities, m)
	}
	sort.Strings(modalities)
	return strings.Join(modalities, ","), len(features)
}

func requestMMFeatures(req *scheduling.InferenceRequest) []fwkrh.MultiModalFeature {
	if req == nil || req.Body == nil || req.Body.TokenizedPrompt == nil {
		return nil
	}
	return req.Body.TokenizedPrompt.MultiModalFeatures
}
