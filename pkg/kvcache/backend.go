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

type KVCacheBackendConfig struct {
	// Name is the identifier for this medium (e.g., "gpu", "cpu", "disk")
	Name string `json:"name"`
	// Weight is the scoring weight for blocks stored on this medium
	Weight float64 `json:"weight"`
	// TailOnly scores this medium's blocks only at positions past the
	// longest chain, from the first block, that any candidate pod holds in
	// a configured medium without TailOnly or speculatively. Set it on media
	// that are slower to load from than memory, such as storage, so they
	// steer requests no pod holds in memory without drawing requests away
	// from pods that do.
	TailOnly bool `json:"tailOnly,omitempty"`
}

// DefaultKVCacheBackendConfig returns the default backend weights, tunable
// via IndexerConfig.BackendConfigs.
func DefaultKVCacheBackendConfig() []*KVCacheBackendConfig {
	return []*KVCacheBackendConfig{
		{Name: GPUTier, Weight: 1.0},
		{Name: "cpu", Weight: 0.8},
		// SGLang reports blocks in its host-memory (HiCache) tier as CPU_PINNED.
		{Name: "cpu_pinned", Weight: 0.8},
		{Name: "storage", Weight: 0.6},
		{Name: "shared_storage", Weight: 0.4},
		{Name: "object_store", Weight: 0.2},
	}
}

// tierScoring is the matcher's per-tier configuration.
type tierScoring struct {
	// weights maps each configured device tier to its scoring weight.
	weights map[string]float64
	// tailOnly holds the configured tiers with TailOnly set.
	tailOnly map[string]bool
}

// tierScoringFromBackends resolves each configured backend's device tier.
func tierScoringFromBackends(backends []*KVCacheBackendConfig) tierScoring {
	ts := tierScoring{weights: make(map[string]float64, len(backends))}
	for _, medium := range backends {
		tier := strings.ToLower(medium.Name)
		ts.weights[tier] = medium.Weight
		if medium.TailOnly {
			if ts.tailOnly == nil {
				ts.tailOnly = map[string]bool{}
			}
			ts.tailOnly[tier] = true
		}
	}
	return ts
}
