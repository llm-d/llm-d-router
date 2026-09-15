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

package sessionprefixcache

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// TestProducerResidencyAgainstModel replays random stores, removals, and
// clears across two endpoints, three ranks, and three groups, and checks every
// lookup against a reference residency model.
func TestProducerResidencyAgainstModel(t *testing.T) {
	p, _ := newTestProducer(t)
	addresses := []string{"10.0.0.1:8080", "10.0.0.2:8080"}
	dimensions := []*int{nil, ptr.To(0), ptr.To(1)}
	type location struct{ endpoint, rank, group int }
	resident := map[location]map[uint64]bool{}
	paths := [][]uint64{{10, 20, 30}, {10, 40, 50}, {60, 70}}
	random := rand.New(rand.NewPCG(42, 17))
	for step := range 1000 {
		loc := location{random.IntN(2), random.IntN(3), random.IntN(3)}
		source := kvevents.EventSource{Endpoint: addresses[loc.endpoint]}
		batch := kvevents.EventBatch{DataParallelRank: dimensions[loc.rank]}
		hashes := paths[random.IntN(len(paths))]
		switch random.IntN(5) {
		case 0, 1:
			batch.Events = []kvevents.GenericEvent{&kvevents.BlockStoredEvent{
				BlockHashes: hashes, BlockSize: 16, DeviceTier: "GPU", GroupIdx: dimensions[loc.group],
				KVCacheSpecKind: kvevents.KVCacheSpecKindFullAttention,
			}}
			if resident[loc] == nil {
				resident[loc] = map[uint64]bool{}
			}
			for _, hash := range hashes {
				resident[loc][hash] = true
			}
		case 2:
			hash := hashes[random.IntN(len(hashes))]
			batch.Events = []kvevents.GenericEvent{&kvevents.BlockRemovedEvent{
				BlockHashes: []uint64{hash}, DeviceTier: "GPU", GroupIdx: dimensions[loc.group],
			}}
			delete(resident[loc], hash)
		case 3, 4:
			batch.Events = []kvevents.GenericEvent{&kvevents.AllBlocksClearedEvent{}}
			for key := range resident {
				if key.endpoint == loc.endpoint {
					delete(resident, key)
				}
			}
		}
		observeBatch(t, p, nil, source, batch)
		for _, exact := range []bool{false, true} {
			total := random.IntN(65)
			lookup := attrsession.SessionCacheRequest{SessionID: "query", TotalTokens: total}
			for _, path := range paths {
				lookup.Prefixes = append(lookup.Prefixes, attrsession.SessionCachePrefix{
					BlockHashes: path, BlockSizeTokens: 16, Exact: exact,
				})
			}
			for endpoint, info := range matchOn(t, p, lookupRequest("query", lookup)) {
				want := 0
				for key, blocks := range resident {
					if key.endpoint != endpoint {
						continue
					}
					for _, path := range paths {
						matched := 0
						for _, hash := range path {
							if !blocks[hash] || (exact && matched+16 > total) {
								break
							}
							matched += 16
						}
						want = max(want, min(total, matched))
					}
				}
				require.Equal(t, want, info.MatchBlocks(), "step %d endpoint %d exact %t", step, endpoint, exact)
				if !exact {
					want = 0
				}
				require.Equal(t, want, info.CachedBlockCount())
			}
		}
	}
}

func TestProducerConcurrentEventsAndQueries(t *testing.T) {
	p, _ := newTestProducer(t)
	var wg sync.WaitGroup
	errors := make(chan error, 6)
	for _, address := range []string{"10.0.0.1:8080", "10.0.0.2:8080"} {
		wg.Go(func() {
			for step := range 200 {
				var event kvevents.GenericEvent = &kvevents.BlockStoredEvent{
					BlockHashes: []uint64{10, 20, 30}, BlockSize: 16, DeviceTier: "GPU",
				}
				switch step % 3 {
				case 1:
					event = &kvevents.BlockRemovedEvent{BlockHashes: []uint64{20}, DeviceTier: "GPU"}
				case 2:
					event = &kvevents.AllBlocksClearedEvent{}
				}
				if err := p.events.ProcessEvents(t.Context(), kvevents.EventSource{Endpoint: address},
					kvevents.EventBatch{Events: []kvevents.GenericEvent{event}}); err != nil {
					errors <- err
					return
				}
			}
		})
	}
	for reader := range 4 {
		wg.Go(func() {
			for range 200 {
				lookup := attrsession.SessionCacheRequest{SessionID: "query", TotalTokens: 47,
					Prefixes: []attrsession.SessionCachePrefix{{BlockHashes: []uint64{10, 20, 30}, BlockSizeTokens: 16, Exact: reader%2 == 0}}}
				endpoints := freshEndpoints()
				if err := p.Produce(t.Context(), lookupRequest("query", lookup), endpoints); err != nil {
					errors <- err
					return
				}
				for _, endpoint := range endpoints {
					value, _ := endpoint.Get(p.dk)
					info := value.(*attrprefix.PrefixCacheMatchInfo)
					cached := info.CachedBlockCount()
					if cached < 0 || cached > 32 || cached%16 != 0 || (reader%2 == 1 && cached != 0) ||
						info.MatchBlocks() < 0 || info.MatchBlocks() > 47 {
						errors <- fmt.Errorf("invalid concurrent match: %+v", info)
						return
					}
				}
			}
		})
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		require.NoError(t, err)
	}
	for _, address := range []string{"10.0.0.1:8080", "10.0.0.2:8080"} {
		require.NoError(t, p.events.Reset(t.Context(), address))
	}
	lookup := attrsession.SessionCacheRequest{SessionID: "final", TotalTokens: 48,
		Prefixes: []attrsession.SessionCachePrefix{{BlockHashes: []uint64{10, 20, 30}, BlockSizeTokens: 16, Exact: true}}}
	require.Zero(t, firstMatch(t, p, lookupRequest("final", lookup)).MatchBlocks())
}
