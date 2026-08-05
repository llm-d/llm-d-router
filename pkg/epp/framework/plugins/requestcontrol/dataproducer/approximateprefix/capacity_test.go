/*
Copyright 2026 The Kubernetes Authors.

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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestMakeServerCapacity(t *testing.T) {
	tests := []struct {
		name         string
		config       config
		metrics      *fwkdl.Metrics
		wantCapacity int
		wantSource   capacitySource
	}{
		{
			name:   "total token capacity metric wins and converts at server block size",
			config: config{AutoTune: true, LRUCapacityPerServer: 5000},
			metrics: &fwkdl.Metrics{
				KvCacheMaxTokenCapacity: 800000,
				CacheBlockSize:          64,
				CacheNumBlocks:          1570,
			},
			wantCapacity: 12500,
			wantSource:   capacitySourceTotalTokens,
		},
		{
			name:   "token capacity converts using configured block size when metric absent",
			config: config{AutoTune: true, BlockSizeTokens: 128},
			metrics: &fwkdl.Metrics{
				KvCacheMaxTokenCapacity: 800000,
			},
			wantCapacity: 6250,
			wantSource:   capacitySourceTotalTokens,
		},
		{
			name:   "GPU blocks only, no offload",
			config: config{AutoTune: true},
			metrics: &fwkdl.Metrics{
				CacheNumBlocks: 1570,
			},
			wantCapacity: 1570,
			wantSource:   capacitySourceGPUBlocks,
		},
		{
			name:   "GPU blocks with offload detected flags undercount",
			config: config{AutoTune: true},
			metrics: &fwkdl.Metrics{
				CacheNumBlocks:         1570,
				KvCacheOffloadDetected: true,
			},
			wantCapacity: 1570,
			wantSource:   capacitySourceGPUBlocksUndercount,
		},
		{
			name:         "autotune without usable metrics falls back to configured capacity",
			config:       config{AutoTune: true, LRUCapacityPerServer: 5000},
			metrics:      &fwkdl.Metrics{},
			wantCapacity: 5000,
			wantSource:   capacitySourceConfigured,
		},
		{
			name:         "autotune disabled uses configured capacity",
			config:       config{AutoTune: false, LRUCapacityPerServer: 5000},
			metrics:      &fwkdl.Metrics{CacheNumBlocks: 1570},
			wantCapacity: 5000,
			wantSource:   capacitySourceConfigured,
		},
		{
			name:         "no metrics and no configured capacity uses default",
			config:       config{AutoTune: true},
			metrics:      &fwkdl.Metrics{},
			wantCapacity: defaultLRUCapacityPerServer,
			wantSource:   capacitySourceDefault,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &dataProducer{config: tt.config}
			endpoint := fwksched.NewEndpoint(
				&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: "pod1"}},
				tt.metrics,
				fwkdl.NewAttributes(),
			)
			s := p.makeserver(endpoint)
			assert.Equal(t, tt.wantCapacity, s.LRUCapacityBlocks)
			assert.Equal(t, tt.wantSource, s.CapacitySource)
		})
	}
}

// TestIndexerResizeOnCapacityChange verifies that a pod's LRU grows when a
// later Add carries a larger capacity, e.g. after tier capacity gauges appear
// in a subsequent metrics scrape.
func TestIndexerResizeOnCapacityChange(t *testing.T) {
	i := newIndexer(context.Background(), 10, "test-name", "test-type").(*indexer)
	pod := server{
		ServerID:          ServerID{Namespace: "default", Name: "server1"},
		LRUCapacityBlocks: 2,
	}

	i.Add([]blockHash{blockHash(1), blockHash(2)}, pod)
	assert.Equal(t, 2, i.podToLRU[pod.ServerID].Len())

	// Capacity 2 without resize would evict on the third insert.
	pod.LRUCapacityBlocks = 3
	i.Add([]blockHash{blockHash(3)}, pod)
	assert.Equal(t, 3, i.podToLRU[pod.ServerID].Len(), "LRU should have been resized to 3")
	assert.Equal(t, 3, i.podToCapacity[pod.ServerID])

	// Shrinking evicts down to the new capacity.
	pod.LRUCapacityBlocks = 2
	i.Add([]blockHash{blockHash(4)}, pod)
	assert.Equal(t, 2, i.podToLRU[pod.ServerID].Len(), "LRU should have been resized down to 2")
}
