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

package metrics

import (
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// The model_name label is populated from the request body, which no
// coordinator config validates against a closed set. Prometheus *Vec types
// never evict label combinations, so an unbounded number of distinct model
// names would grow the time series set without limit and exhaust memory. The
// cap matches EPP's cap for the same reason and for consistency between the
// two components.
const maxModelLabelValues = 1000

var modelLabelLimiter = metricsutil.NewBoundedLabel(maxModelLabelValues)

// boundModel maps a request-derived model name to the label value emitted on
// coordinator metrics. Empty resolves to ModelUnknown before the cap is
// consulted, so a client that never sends "model" cannot exhaust the cap on
// its own.
func boundModel(modelName string) string {
	if modelName == "" {
		return ModelUnknown
	}
	return modelLabelLimiter.Bound(modelName)
}
