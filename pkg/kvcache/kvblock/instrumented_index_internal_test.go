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

package kvblock

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMaxContiguousPodHits(t *testing.T) {
	gpu := func(pod string) PodEntry { return PodEntry{PodIdentifier: pod, DeviceTier: "gpu"} }
	cpu := func(pod string) PodEntry { return PodEntry{PodIdentifier: pod, DeviceTier: "cpu"} }
	keys := []BlockHash{1, 2, 3}

	tests := []struct {
		name       string
		keys       []BlockHash
		keyToPods  map[BlockHash][]PodEntry
		wantResult int
	}{
		{
			name:       "no request keys",
			keys:       nil,
			keyToPods:  map[BlockHash][]PodEntry{},
			wantResult: 0,
		},
		{
			name:       "first key held by nobody breaks every chain",
			keys:       keys,
			keyToPods:  map[BlockHash][]PodEntry{2: {gpu("pod-a")}},
			wantResult: 0,
		},
		{
			name: "one pod holds the whole prefix",
			keys: keys,
			keyToPods: map[BlockHash][]PodEntry{
				1: {gpu("pod-a")}, 2: {gpu("pod-a")}, 3: {gpu("pod-a")},
			},
			wantResult: 3,
		},
		{
			name: "longest chain wins",
			keys: keys,
			keyToPods: map[BlockHash][]PodEntry{
				1: {gpu("pod-a"), gpu("pod-b")}, 2: {gpu("pod-a")}, 3: {gpu("pod-a")},
			},
			wantResult: 3,
		},
		{
			name: "a gap ends the chain even if the pod reappears",
			keys: keys,
			keyToPods: map[BlockHash][]PodEntry{
				1: {gpu("pod-a")}, 3: {gpu("pod-a")},
			},
			wantResult: 1,
		},
		{
			name: "a pod that joins after the first key never starts a chain",
			keys: keys,
			keyToPods: map[BlockHash][]PodEntry{
				1: {gpu("pod-a")}, 2: {gpu("pod-a"), gpu("pod-b")}, 3: {gpu("pod-b")},
			},
			wantResult: 2,
		},
		{
			name: "duplicate device tiers at one key count once",
			keys: keys,
			keyToPods: map[BlockHash][]PodEntry{
				1: {gpu("pod-a"), cpu("pod-a")},
				2: {gpu("pod-a"), cpu("pod-a")},
				3: {gpu("pod-a"), cpu("pod-a")},
			},
			wantResult: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantResult, maxContiguousPodHits(tt.keys, tt.keyToPods))
		})
	}
}
