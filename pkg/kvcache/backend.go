/*
Copyright 2025 The llm-d Authors.

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

package kvcache

import "strings"

// GPUTier is the tier name of blocks resident in GPU memory. KV events that
// carry no medium are indexed under it.
const GPUTier = "gpu"

// KVCacheBackendConfig assigns a scoring weight to a device tier, identified
// by the medium field in KV-cache events.
type KVCacheBackendConfig struct {
	// Name is matched against the lowercased medium string in KV-cache
	// events, for example "gpu", "cpu", "storage".
	Name string `json:"name"`
	// Weight is the scoring weight for blocks stored on this medium
	Weight float64 `json:"weight"`
}

// DefaultKVCacheBackendConfig returns the default tier weights, tunable via
// IndexerConfig.BackendConfigs. "storage" covers the vLLM tiering offload;
// "shared_storage" and "object_store" cover the llm-d filesystem backend.
// Offloaded tiers score conservatively because promotion speed varies widely
// across media (NVMe vs CephFS vs S3); deployments on fast storage should
// raise these values.
func DefaultKVCacheBackendConfig() []*KVCacheBackendConfig {
	return []*KVCacheBackendConfig{
		{Name: GPUTier, Weight: 1.0},
		{Name: "cpu", Weight: 0.8},
		{Name: "shared_storage", Weight: 0.4},
		{Name: "storage", Weight: 0.3},
		{Name: "object_store", Weight: 0.2},
	}
}

// tierWeightsFromBackends maps each configured backend's device tier to its
// scoring weight.
func tierWeightsFromBackends(backends []*KVCacheBackendConfig) map[string]float64 {
	weights := make(map[string]float64, len(backends))
	for _, medium := range backends {
		weights[strings.ToLower(medium.Name)] = medium.Weight
	}
	return weights
}
