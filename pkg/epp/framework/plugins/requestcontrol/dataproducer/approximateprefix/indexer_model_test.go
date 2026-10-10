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

package approximateprefix

import (
	"fmt"
	"math/rand/v2"
	"testing"

	lru "github.com/hashicorp/golang-lru/v2"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
)

// referenceIndex models the indexer contract: one LRU per pod, filled
// tail-first, and a greedy match that counts every pod holding block i until
// no pod holds a block.
type referenceIndex struct {
	pods map[ServerID]*lru.Cache[blockHash, struct{}]
}

func (r *referenceIndex) add(hashes []blockHash, s server) {
	c, ok := r.pods[s.ServerID]
	if !ok {
		c, _ = lru.New[blockHash, struct{}](s.NumOfGPUBlocks)
		r.pods[s.ServerID] = c
	}
	for i := len(hashes) - 1; i >= 0; i-- {
		c.Add(hashes[i], struct{}{})
	}
}

func (r *referenceIndex) match(hashes []blockHash) map[ServerID]int {
	res := map[ServerID]int{}
	for _, h := range hashes {
		found := false
		for id, c := range r.pods {
			if c.Contains(h) {
				res[id]++
				found = true
			}
		}
		if !found {
			break
		}
	}
	return res
}

// TestProduceMatchesReferenceModel drives the producer and the reference
// model with the same random sequence of requests, pod capacities, and pod
// removals. After each query it checks every candidate's match length and
// predicted cached tokens. Requests carry one to three prompts whose tokens
// come from a three-token alphabet, so prompts share prefixes of varying
// depth. Prompts also repeat or extend earlier prompts, as multi-turn
// traffic does, so block sizes of 2 and 3 produce matched partial final
// blocks.
func TestProduceMatchesReferenceModel(t *testing.T) {
	disableMinBlockSizeClamp(t)
	for seed := range uint64(30) {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(seed, seed))
			blockSize := 1 + rng.IntN(3)
			cfg := config{BlockSizeTokens: blockSize, MaxPrefixBlocksToMatch: 1 << 20}
			p, err := newDataProducer(t.Context(), ApproxPrefixCachePluginType, cfg, testHandle())
			if err != nil {
				t.Fatal(err)
			}
			ref := &referenceIndex{pods: map[ServerID]*lru.Cache[blockHash, struct{}]{}}

			numPods := 1 + rng.IntN(12)
			pods := make([]server, numPods)
			endpoints := make([]fwksched.Endpoint, numPods)
			for i := range pods {
				id := k8stypes.NamespacedName{Namespace: "default", Name: fmt.Sprintf("pod-%d", i)}
				pods[i] = server{ServerID: ServerID(id), NumOfGPUBlocks: 4 + rng.IntN(60)}
				endpoints[i] = fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: id}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
			}
			var history [][]uint32
			randomTokens := func(n int) []uint32 {
				tokens := make([]uint32, n)
				for j := range tokens {
					tokens[j] = uint32(1 + rng.IntN(3))
				}
				return tokens
			}
			// A prompt repeats an earlier prompt, extends it with a new turn,
			// or is new.
			prompt := func() []uint32 {
				if len(history) > 0 {
					prev := history[rng.IntN(len(history))]
					switch rng.IntN(3) {
					case 0:
						return prev
					case 1:
						return append(append([]uint32{}, prev...), randomTokens(1+rng.IntN(24))...)
					}
				}
				tokens := randomTokens(1 + rng.IntN(96))
				history = append(history, tokens)
				return tokens
			}
			request := func() *fwksched.InferenceRequest {
				prompts := make([]fwkrh.PromptTokens, 1+rng.IntN(3))
				for i := range prompts {
					prompts[i] = fwkrh.PromptTokens{TokenIDs: prompt()}
				}
				return &fwksched.InferenceRequest{
					RequestID:   "q",
					TargetModel: "m",
					Body:        &fwkrh.InferenceRequestBody{TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: prompts}},
				}
			}
			hashesOf := func(req *fwksched.InferenceRequest) ([][]blockHash, []int) {
				return prefixhash.GetBlockHashesWithPromptTokens(t.Context(), req, blockSize, cfg.MaxPrefixBlocksToMatch)
			}
			key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType)

			for op := range 400 {
				switch r := rng.IntN(100); {
				case r < 65:
					s := pods[rng.IntN(numPods)]
					perPrompt, _ := hashesOf(request())
					for _, hashes := range perPrompt {
						p.indexerInst.Add(hashes, s)
						ref.add(hashes, s)
					}
				case r < 70:
					s := pods[rng.IntN(numPods)].ServerID
					p.indexerInst.RemovePod(s)
					delete(ref.pods, s)
				default:
					var candidates []fwksched.Endpoint
					for _, ep := range endpoints {
						if rng.IntN(2) == 0 {
							candidates = append(candidates, ep)
						}
					}
					req := request()
					if err := p.Produce(t.Context(), req, candidates); err != nil {
						t.Fatal(err)
					}
					state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
					if err != nil {
						t.Fatal(err)
					}

					wantBlocks, wantTokens := map[ServerID]int{}, map[ServerID]int{}
					perPrompt, promptTokens := hashesOf(req)
					for i, hashes := range perPrompt {
						for id, n := range ref.match(hashes) {
							wantBlocks[id] += n
							wantTokens[id] += min(n*blockSize, promptTokens[i])
						}
					}
					for _, ep := range candidates {
						id := ServerID(ep.GetMetadata().ID)
						info, ok := ep.Get(key)
						if !ok {
							t.Fatalf("op %d: no match info for %s", op, id)
						}
						if got := info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks(); got != wantBlocks[id] {
							t.Fatalf("op %d: pod %s matched %d blocks, reference %d", op, id, got, wantBlocks[id])
						}
						if got := state.PredictedCachedTokens[id]; got != wantTokens[id] {
							t.Fatalf("op %d: pod %s predicted %d cached tokens, reference %d", op, id, got, wantTokens[id])
						}
					}
				}
			}
		})
	}
}
