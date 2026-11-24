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

package kvblock

// SampledKey reports whether key passes the index's power-of-two sampling
// test. A modulus of 0 or 1 samples every key; a larger power-of-two modulus
// samples the keys whose low bits are zero, so roughly one key in modulus is
// kept. The request key is the sampling coordinate because every consumer
// holds it: the pool computes it at Add time, the index keys on it, and the
// matcher holds the query's request-key chain. Selection is a pure function
// of that value, so all sites agree without coordination state.
func SampledKey(key BlockHash, modulus int) bool {
	return modulus <= 1 || key&BlockHash(modulus-1) == 0
}

// filterSampledKeys returns the keys passing the sampling test as a fresh
// slice. Callers with modulus <= 1 skip the call and keep the input.
func filterSampledKeys(keys []BlockHash, modulus int) []BlockHash {
	var out []BlockHash
	for _, k := range keys {
		if SampledKey(k, modulus) {
			out = append(out, k)
		}
	}
	return out
}

// validSampleModulus reports whether modulus is 0, 1, or a power of two.
func validSampleModulus(modulus int) bool {
	return modulus <= 1 || modulus&(modulus-1) == 0
}

// ConfigSampleModulus returns the sampling modulus of the backend cfg
// selects, zero when the configuration selects no sampling. The kvcache
// matcher filters its query keys by the same test, so it reads the modulus
// from the configuration rather than through the index.
func ConfigSampleModulus(cfg *IndexConfig) int {
	if cfg == nil {
		return 0
	}
	switch {
	case cfg.DenseMemoryConfig != nil:
		return cfg.DenseMemoryConfig.SampleModulus
	case cfg.InMemoryConfig != nil:
		return cfg.InMemoryConfig.SampleModulus
	}
	return 0
}
