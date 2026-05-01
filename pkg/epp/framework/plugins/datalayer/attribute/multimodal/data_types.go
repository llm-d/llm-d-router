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

package multimodal

import (
	"maps"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
)

const (
	// EncoderCacheMatchInfoKey is attached to endpoints by the multimodal data
	// producer and consumed by scorer/latency plugins that need encoder-cache locality.
	EncoderCacheMatchInfoKey = "MultiModalEncoderCacheMatchInfoKey"
)

// EncoderCacheMatchInfo summarizes how much of a request's multimodal encoder
// work is likely already cached on one endpoint.
type EncoderCacheMatchInfo struct {
	matchedWeight int
	totalWeight   int
	matchedHashes map[string]int
}

// NewEncoderCacheMatchInfo creates endpoint-local multimodal cache match data.
func NewEncoderCacheMatchInfo(matchedWeight int, totalWeight int, matchedHashes map[string]int) *EncoderCacheMatchInfo {
	return &EncoderCacheMatchInfo{
		matchedWeight: matchedWeight,
		totalWeight:   totalWeight,
		matchedHashes: maps.Clone(matchedHashes),
	}
}

// MatchedWeight returns the weighted multimodal content already cached.
func (m *EncoderCacheMatchInfo) MatchedWeight() int {
	return m.matchedWeight
}

// TotalWeight returns the total weighted multimodal content in the request.
func (m *EncoderCacheMatchInfo) TotalWeight() int {
	return m.totalWeight
}

// MatchedHashes returns a copy of the matched hash weights.
func (m *EncoderCacheMatchInfo) MatchedHashes() map[string]int {
	return maps.Clone(m.matchedHashes)
}

// Clone implements datalayer.Cloneable.
func (m *EncoderCacheMatchInfo) Clone() fwkdl.Cloneable {
	if m == nil {
		return nil
	}
	return &EncoderCacheMatchInfo{
		matchedWeight: m.matchedWeight,
		totalWeight:   m.totalWeight,
		matchedHashes: maps.Clone(m.matchedHashes),
	}
}
