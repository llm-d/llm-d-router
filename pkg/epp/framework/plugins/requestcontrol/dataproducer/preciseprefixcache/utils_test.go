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

package preciseprefixcache

import (
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"

	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestMatchedBlocks(t *testing.T) {
	const (
		podA = "10.0.0.1:8000"
		podB = "10.0.0.2:8000"
	)
	keys := []kvblock.BlockHash{1, 2, 3, 4}

	// gpu/cpu tiers must count identically in the any-tier count.
	gpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"} }
	cpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "cpu"} }
	speculative := func(pod string) kvblock.PodEntry {
		return kvblock.PodEntry{PodIdentifier: pod, Speculative: true}
	}

	tests := []struct {
		name       string
		keys       []kvblock.BlockHash
		keyToPods  map[kvblock.BlockHash][]kvblock.PodEntry
		podID      string
		wantCount  int
		wantByTier map[string]int
	}{
		{
			name: "all blocks held on one tier count fully",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {cpu(podA)}, 2: {cpu(podA)}, 3: {cpu(podA)}, 4: {cpu(podA)},
			},
			podID:      podA,
			wantCount:  4,
			wantByTier: map[string]int{"cpu": 4},
		},
		{
			name: "single block counts as one, not zero",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {cpu(podA)},
			},
			podID:      podA,
			wantCount:  1,
			wantByTier: map[string]int{"cpu": 1},
		},
		{
			name: "stops at first missing block",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA)}, 2: {gpu(podA)}, 4: {gpu(podA)}, // block 3 missing
			},
			podID:      podA,
			wantCount:  2,
			wantByTier: map[string]int{"gpu": 2},
		},
		{
			name: "pod absent from first block yields zero and empty map",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podB)}, 2: {gpu(podA)},
			},
			podID:      podA,
			wantCount:  0,
			wantByTier: map[string]int{},
		},
		{
			name: "counts are per-pod independent",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podB)}, 2: {gpu(podA)}, 3: {cpu(podB)},
			},
			podID:      podA,
			wantCount:  2,
			wantByTier: map[string]int{"gpu": 2},
		},
		{
			name: "dual-tier block counts once per tier",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podA)}, 2: {gpu(podA)},
			},
			podID:      podA,
			wantCount:  2,
			wantByTier: map[string]int{"gpu": 2, "cpu": 1},
		},
		{
			name: "tier-specific gap stops that tier only",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podA)}, 2: {gpu(podA), cpu(podA)}, 3: {gpu(podA)}, 4: {gpu(podA), cpu(podA)},
			},
			podID:      podA,
			wantCount:  4,
			wantByTier: map[string]int{"gpu": 4, "cpu": 2},
		},
		{
			// gpu chain ends at block 2, but the pod holds every block, so the
			// any-tier count keeps growing past the tier break.
			name: "tier chain ends while any-tier count continues",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA)}, 2: {cpu(podA)}, 3: {gpu(podA)}, 4: {gpu(podA)},
			},
			podID:      podA,
			wantCount:  4,
			wantByTier: map[string]int{"gpu": 1},
		},
		{
			name: "speculative entries count under the speculative key",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {speculative(podA), gpu(podA)}, 2: {speculative(podA)},
			},
			podID:      podA,
			wantCount:  2,
			wantByTier: map[string]int{"gpu": 1, attrprefix.SpeculativeTierKey: 2},
		},
		{
			name:       "empty index yields zero",
			keyToPods:  map[kvblock.BlockHash][]kvblock.PodEntry{},
			podID:      podA,
			wantCount:  0,
			wantByTier: map[string]int{},
		},
		{
			name:       "no keys yields zero",
			keys:       []kvblock.BlockHash{},
			keyToPods:  map[kvblock.BlockHash][]kvblock.PodEntry{},
			podID:      podA,
			wantCount:  0,
			wantByTier: map[string]int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blockKeys := keys // per-case override; nil means the shared keys
			if tt.keys != nil {
				blockKeys = tt.keys
			}
			gotCount, gotByTier := matchedBlocks(blockKeys, tt.keyToPods, tt.podID)
			assert.Equal(t, tt.wantCount, gotCount)
			assert.NotNil(t, gotByTier)
			assert.Equal(t, tt.wantByTier, gotByTier)
			// Each tier's contiguous count never exceeds the any-tier count.
			for tier, count := range gotByTier {
				assert.LessOrEqual(t, count, gotCount, "tier %q", tier)
			}
		})
	}
}
