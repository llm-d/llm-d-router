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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
)

func TestMatchedBlockCount(t *testing.T) {
	const (
		podA = "10.0.0.1:8000"
		podB = "10.0.0.2:8000"
	)
	keys := []kvblock.BlockHash{1, 2, 3, 4}

	// gpu/cpu tiers must count identically — the unweighted count ignores tier.
	gpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"} }
	cpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "cpu"} }

	tests := []struct {
		name      string
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry
		podID     string
		want      int
	}{
		{
			name: "all blocks held on RAM/cpu tier count fully (unweighted)",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {cpu(podA)}, 2: {cpu(podA)}, 3: {cpu(podA)}, 4: {cpu(podA)},
			},
			podID: podA,
			want:  4,
		},
		{
			name: "single RAM block counts as one, not zero",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {cpu(podA)},
			},
			podID: podA,
			want:  1,
		},
		{
			name: "stops at first missing block",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA)}, 2: {gpu(podA)}, 4: {gpu(podA)}, // block 3 missing
			},
			podID: podA,
			want:  2,
		},
		{
			name: "pod absent from first block yields zero",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podB)}, 2: {gpu(podA)},
			},
			podID: podA,
			want:  0,
		},
		{
			name: "counts are per-pod independent",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podB)}, 2: {gpu(podA)}, 3: {cpu(podB)},
			},
			podID: podA,
			want:  2,
		},
		{
			name:      "empty index yields zero",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{},
			podID:     podA,
			want:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, matchedBlockCount(keys, tt.keyToPods, tt.podID))
		})
	}
}

func TestMatchedBlockCountByTier(t *testing.T) {
	const (
		podA = "10.0.0.1:8000"
		podB = "10.0.0.2:8000"
	)
	keys := []kvblock.BlockHash{1, 2, 3, 4}

	gpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"} }
	cpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "cpu"} }
	speculative := func(pod string) kvblock.PodEntry {
		return kvblock.PodEntry{PodIdentifier: pod, Speculative: true}
	}

	tests := []struct {
		name      string
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry
		podID     string
		want      map[string]int
	}{
		{
			name: "all blocks on one tier",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA)}, 2: {gpu(podA)}, 3: {gpu(podA)}, 4: {gpu(podA)},
			},
			podID: podA,
			want:  map[string]int{"gpu": 4},
		},
		{
			name: "dual-tier block counts once per tier",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podA)}, 2: {gpu(podA)},
			},
			podID: podA,
			want:  map[string]int{"gpu": 2, "cpu": 1},
		},
		{
			name: "tier-specific gap stops that tier only",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podA)}, 2: {gpu(podA), cpu(podA)}, 3: {gpu(podA)}, 4: {gpu(podA), cpu(podA)},
			},
			podID: podA,
			want:  map[string]int{"gpu": 4, "cpu": 2},
		},
		{
			name: "pod absent from first block yields empty map",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podB)}, 2: {gpu(podA)},
			},
			podID: podA,
			want:  map[string]int{},
		},
		{
			name: "counts are per-pod independent",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podB)}, 2: {gpu(podA), cpu(podB)}, 3: {cpu(podB)},
			},
			podID: podA,
			want:  map[string]int{"gpu": 2},
		},
		{
			name: "speculative entries count under the empty tier",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {speculative(podA), gpu(podA)}, 2: {speculative(podA)},
			},
			podID: podA,
			want:  map[string]int{"gpu": 1, "": 2},
		},
		{
			name:      "empty index yields empty map",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{},
			podID:     podA,
			want:      map[string]int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := matchedBlockCountByTier(keys, tt.keyToPods, tt.podID)
			assert.NotNil(t, got)
			assert.Equal(t, tt.want, got)
			// Each tier's contiguous count never exceeds the any-tier count.
			anyTier := matchedBlockCount(keys, tt.keyToPods, tt.podID)
			for tier, count := range got {
				assert.LessOrEqual(t, count, anyTier, "tier %q", tier)
			}
		})
	}
}
