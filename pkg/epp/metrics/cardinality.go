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
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// Model name labels are populated from the request body, which is not validated
// against a closed set of served models. Prometheus *Vec types never evict label
// combinations, so an unbounded number of distinct model names would grow the time
// series set without limit and exhaust memory.
//
// Names configured through InferenceModelRewrite rules (exact-match sources and
// rewrite targets) are pinned: they always emit their real label and do not count
// against the cap, so a flood of unconfigured names cannot displace a configured
// model into the overflow bucket.
const maxModelLabelValues = 1000

var modelLabelLimiter = metricsutil.NewBoundedLabel(maxModelLabelValues)

// boundFairnessID caps the request-derived fairness_id label.
func boundFairnessID(fairnessID string) string {
	return metricsutil.BoundFairnessID(fairnessID)
}

// PreAdmitModelLabels pins the given model names so they always emit their real
// label value on model-labeled metrics, regardless of how many unconfigured
// names have been admitted. The datastore calls this when InferenceModelRewrite
// rules are loaded or updated.
func PreAdmitModelLabels(names ...string) {
	for _, n := range names {
		modelLabelLimiter.Pin(n)
	}
}

// boundModel caps a single request-derived model-name label.
func boundModel(modelName string) string {
	return modelLabelLimiter.Bound(modelName)
}

// boundModels caps the model-name labels shared by request metrics.
func boundModels(modelName, targetModelName string) (string, string) {
	return modelLabelLimiter.Bound(modelName), modelLabelLimiter.Bound(targetModelName)
}

// maxModelServerEndpointLabelValues caps the model_server_endpoint label of metrics
// that attribute an observation to a single inference endpoint. Endpoint names
// normally come from the datastore and are therefore bounded by the pool's
// endpoint count; the cap is a backstop for discovery sources that can emit an
// unbounded number of identities (per-rank endpoints, restarted pods with fresh
// names), because Prometheus *Vec types never evict label combinations.
const maxModelServerEndpointLabelValues = 1000

var modelServerEndpointLabelLimiter = metricsutil.NewBoundedLabel(maxModelServerEndpointLabelValues)

// boundModelServerEndpoint caps a model_server_endpoint label value.
func boundModelServerEndpoint(endpointName string) string {
	return modelServerEndpointLabelLimiter.Bound(endpointName)
}
