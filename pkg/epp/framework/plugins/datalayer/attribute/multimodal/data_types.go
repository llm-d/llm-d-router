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
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
)

const (
	// EncoderCacheMatchInfoKey is attached to endpoints by the multimodal data
	// producer and consumed by scorer/latency plugins that need encoder-cache locality.
	EncoderCacheMatchInfoKey = "MultiModalEncoderCacheMatchInfoKey"
)

// MatchItem describes one unique multimodal item involved in encoder-cache
// affinity matching.
type MatchItem struct {
	Hash string
	Size int
}

// EncoderCacheMatchInfo carries endpoint-local multimodal cache match data.
// Consumers choose how to interpret item sizes and compute scores.
type EncoderCacheMatchInfo struct {
	matchedItems []MatchItem
	requestItems []MatchItem
}

// NewEncoderCacheMatchInfo creates endpoint-local multimodal cache match data.
func NewEncoderCacheMatchInfo(matchedItems []MatchItem, requestItems []MatchItem) *EncoderCacheMatchInfo {
	return &EncoderCacheMatchInfo{
		matchedItems: CloneMatchItems(matchedItems),
		requestItems: CloneMatchItems(requestItems),
	}
}

// MatchedItems returns endpoint-local request items that are likely already cached.
func (m *EncoderCacheMatchInfo) MatchedItems() []MatchItem {
	if m == nil {
		return nil
	}
	return CloneMatchItems(m.matchedItems)
}

// RequestItems returns all unique multimodal request items.
func (m *EncoderCacheMatchInfo) RequestItems() []MatchItem {
	if m == nil {
		return nil
	}
	return CloneMatchItems(m.requestItems)
}

// Clone implements datalayer.Cloneable.
func (m *EncoderCacheMatchInfo) Clone() fwkdl.Cloneable {
	if m == nil {
		return nil
	}
	return &EncoderCacheMatchInfo{
		matchedItems: CloneMatchItems(m.matchedItems),
		requestItems: CloneMatchItems(m.requestItems),
	}
}

// CloneMatchItems creates a deep copy of a MatchItem slice.
func CloneMatchItems(items []MatchItem) []MatchItem {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]MatchItem, len(items))
	copy(cloned, items)
	return cloned
}
