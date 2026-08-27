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

package kvblock_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

func TestPseudoPodIdentifiers(t *testing.T) {
	tests := []struct {
		id       string
		isPseudo bool
		isPool   bool
		inFilter bool
	}{
		{id: kvblock.NodePseudoPod("n1"), isPseudo: true, isPool: false, inFilter: false},
		{id: kvblock.PoolPseudoPod("fs"), isPseudo: true, isPool: true, inFilter: true},
		{id: "10.0.0.1:8000", isPseudo: false, isPool: false, inFilter: true},
		{id: "10.0.0.9:8000", isPseudo: false, isPool: false, inFilter: false},
		{id: "nodeless", isPseudo: false, isPool: false, inFilter: false},
		{id: "", isPseudo: false, isPool: false, inFilter: false},
	}
	filter := sets.New("10.0.0.1:8000")
	for _, tt := range tests {
		t.Run(tt.id, func(t *testing.T) {
			assert.Equal(t, tt.isPseudo, kvblock.IsPseudoPod(tt.id))
			assert.Equal(t, tt.isPool, kvblock.IsPoolPseudoPod(tt.id))
			assert.Equal(t, tt.inFilter, kvblock.InPodFilter(filter, tt.id))
		})
	}
	assert.Equal(t, "node:n1", kvblock.NodePseudoPod("n1"))
	assert.Equal(t, "pool:fs", kvblock.PoolPseudoPod("fs"))
}

func TestResolvePseudoPods(t *testing.T) {
	byNode := map[string][]string{"n1": {"a", "b"}, "n2": {"c"}}
	all := []string{"a", "b", "c"}
	in := map[kvblock.BlockHash][]kvblock.PodEntry{
		1: {
			{PodIdentifier: "a", DeviceTier: "gpu"},
			{PodIdentifier: "node:n1", DeviceTier: "lmcache-l1"},
		},
		2: {{PodIdentifier: "pool:fs", DeviceTier: "lmcache-l2-fs"}},
		3: {{PodIdentifier: "node:unknown", DeviceTier: "lmcache-l1"}},
		4: {{PodIdentifier: "c", DeviceTier: "gpu"}},
	}
	got := kvblock.ResolvePseudoPods(in, byNode, all)
	assert.ElementsMatch(t, []kvblock.PodEntry{
		{PodIdentifier: "a", DeviceTier: "gpu"},
		{PodIdentifier: "a", DeviceTier: "lmcache-l1"},
		{PodIdentifier: "b", DeviceTier: "lmcache-l1"},
	}, got[1])
	assert.ElementsMatch(t, []kvblock.PodEntry{
		{PodIdentifier: "a", DeviceTier: "lmcache-l2-fs"},
		{PodIdentifier: "b", DeviceTier: "lmcache-l2-fs"},
		{PodIdentifier: "c", DeviceTier: "lmcache-l2-fs"},
	}, got[2])
	assert.Empty(t, got[3])
	assert.Equal(t, []kvblock.PodEntry{{PodIdentifier: "c", DeviceTier: "gpu"}}, got[4])
}
