/*
Copyright 2025 The Kubernetes Authors.

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

package metrics

import (
	dto "github.com/prometheus/client_model/go"

	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// LoRASpec extends the standard Spec to allow special case
// handling for retrieving the latest metrics value for LoRAs.
type LoRASpec struct {
	*Spec
}

// parseStringToLoRASpec parses the metric specification but
// wraps the return in a LoRASpec.
func parseStringToLoRASpec(spec string) (*LoRASpec, error) {
	baseSpec, err := parseStringToSpec(spec)
	if err != nil {
		return nil, err
	}
	if baseSpec == nil {
		// empty string → disabled; preserve nil-means-disabled contract
		return nil, nil //nolint:nilnil
	}
	return &LoRASpec{Spec: baseSpec}, nil
}

// getLatestMetric retrieves the latest LoRA metric based on Spec.
// We can't use the standard Spec method since, in the case of
// LoRA (i.e., `vllm:lora_requests_info`), each label key-value pair permutation
// generates new series and only most recent should be used. The value of each
// series is its creation timestamp so we can retrieve the latest by sorting on
// that the value first.
//
// vLLM only emits vllm:lora_requests_info once an adapter has been loaded, so a
// vanilla deployment scrape legitimately has no family present. Both "family
// missing" and "family present but no matching labels" are reported as a nil
// metric so the extractor can skip the LoRA section silently rather than
// incrementing DataLayerExtractErrorsTotal on every poll (#926).
func (spec *LoRASpec) getLatestMetric(families sourcemetrics.PrometheusMetricMap) *dto.Metric {
	family, exists := families[spec.Name]
	if !exists || len(family.GetMetric()) == 0 {
		return nil
	}

	var latest *dto.Metric
	var recent float64 = -1

	for _, metric := range family.GetMetric() {
		if spec.labelsMatch(metric.GetLabel()) {
			value := extractValue(metric) // metric value is its creation timestamp
			if value > recent {
				recent = value
				latest = metric
			}
		}
	}

	return latest
}
