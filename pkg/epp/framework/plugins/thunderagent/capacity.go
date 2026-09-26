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

package thunderagent

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// capacitySource labels where a pod's capacity value came from.
type capacitySource string

const (
	capacityReal     capacitySource = "real"
	capacityFallback capacitySource = "fallback"
)

// meteredEndpoint is the common surface of scheduling and datalayer
// endpoints, so capacity resolution serves both the scorer and the
// saturation detector.
type meteredEndpoint interface {
	GetMetrics() *fwkdl.Metrics
}

// endpointCapacity returns a pod's KV cache capacity in tokens. The real
// value is block_size * num_gpu_blocks scraped from the engine's
// cache_config_info by the datalayer metrics extractor; it handles
// heterogeneous pools where a single configured value cannot. When the
// scraped values are absent the configured capacityTokens is used, unless
// requireRealCapacity is set, in which case 0 is returned and callers treat
// the pod as having no usable capacity.
func (a *ThunderAgent) endpointCapacity(endpoint meteredEndpoint) (float64, capacitySource) {
	if endpoint != nil {
		if m := endpoint.GetMetrics(); m != nil && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
			return float64(m.CacheBlockSize) * float64(m.CacheNumBlocks), capacityReal
		}
	}
	if a.requireRealCapacity {
		return 0, capacityFallback
	}
	return a.capacityTokens, capacityFallback
}
