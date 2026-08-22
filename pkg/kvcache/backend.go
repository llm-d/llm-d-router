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

// KVCacheBackendConfig assigns a scoring weight to a device tier, identified
// by the medium field in KV-cache events.
type KVCacheBackendConfig struct {
	// Name must match the medium string in KV-cache events (lowercased): "gpu", "cpu", "storage".
	Name string `json:"name"`
	// Weight is the scoring weight for blocks stored on this medium
	Weight float64 `json:"weight"`
}

// DefaultKVCacheBackendConfig returns the default tier weights.
// "storage" is set conservatively at 0.3 because promotion speed varies
// widely across media (NVMe vs CephFS vs S3). Deployments with fast
// storage should raise this value; slow storage may lower it further.
func DefaultKVCacheBackendConfig() []*KVCacheBackendConfig {
	return []*KVCacheBackendConfig{
		{Name: "gpu", Weight: 1.0},
		{Name: "cpu", Weight: 0.8},
		{Name: "storage", Weight: 0.3},
	}
}
