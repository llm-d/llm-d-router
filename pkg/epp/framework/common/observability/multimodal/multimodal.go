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
	"slices"
	"sort"
	"strings"

	"go.opentelemetry.io/otel/attribute"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ModalityNone is the mm.modality value when no multimodal content is present.
const ModalityNone = "none"

// ModalityOther is the metric label value for a modality outside image, audio
// and video.
const ModalityOther = metricsutil.OverflowValue

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

func requestMMFeatures(req *scheduling.InferenceRequest) []fwkrequest.MultiModalFeature {
	if req == nil || req.Body == nil || req.Body.TokenizedRequest == nil {
		return nil
	}
	var features []fwkrequest.MultiModalFeature
	for _, p := range req.Body.TokenizedRequest.Prompts {
		features = append(features, p.MultiModalFeatures...)
	}
	return features
}

// MetricLabel bounds a modality value for use as a Prometheus label. Token-in
// requests (vLLM /inference/v1/generate, SGLang /generate) key mm_hashes by
// whatever the client sends, and every distinct key would otherwise add series
// that are never removed. image, audio and video keep their names, ModalityNone
// and the empty string pass through, and any other modality becomes
// ModalityOther. A comma-joined value from Summary is folded one modality at a
// time and stays sorted and deduplicated, so it takes at most 16 values.
func MetricLabel(modality string) string {
	if modality == "" || modality == ModalityNone || knownModality(modality) {
		return modality
	}
	var labels []string
	for _, m := range strings.Split(modality, ",") {
		if !knownModality(m) {
			m = ModalityOther
		}
		if !slices.Contains(labels, m) {
			labels = append(labels, m)
		}
	}
	slices.Sort(labels)
	return strings.Join(labels, ",")
}

func knownModality(m string) bool {
	switch reqcommon.Modality(m) {
	case reqcommon.ModalityImage, reqcommon.ModalityAudio, reqcommon.ModalityVideo:
		return true
	}
	return false
}
