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
	"context"
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// BenchmarkProduceThenPreRequest includes rebuilding and cleaning up
// per-request prediction state.
func BenchmarkProduceThenPreRequest(b *testing.B) {
	prefixmetrics.Register()
	for _, candidates := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("candidates=%d", candidates), func(b *testing.B) {
			p, req, endpoints, result := newPredictionBenchmark(b, candidates)
			if err := p.PreRequest(b.Context(), req, result); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			for b.Loop() {
				if err := p.Produce(b.Context(), req, endpoints); err != nil {
					b.Fatal(err)
				}
				if err := p.PreRequest(b.Context(), req, result); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkRecordPrediction retains Produce's state to isolate recording costs.
func BenchmarkRecordPrediction(b *testing.B) {
	prefixmetrics.Register()
	for _, candidates := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("candidates=%d", candidates), func(b *testing.B) {
			p, req, _, result := newPredictionBenchmark(b, candidates)

			b.ReportAllocs()
			for b.Loop() {
				p.recordPrediction(b.Context(), req, result)
			}
		})
	}
}

func newPredictionBenchmark(b *testing.B, candidates int) (*Producer, *scheduling.InferenceRequest, []scheduling.Endpoint, *scheduling.SchedulingResult) {
	b.Helper()
	endpoints := make([]scheduling.Endpoint, candidates)
	matches := make(map[string]kvcache.PodMatch, candidates)
	for i := range endpoints {
		md := &datalayer.EndpointMetadata{
			ID:      datalayer.ID{Namespace: "default", Name: fmt.Sprintf("pod-%d", i)},
			Address: fmt.Sprintf("pod-%d", i),
			Port:    "8080",
		}
		endpoints[i] = scheduling.NewEndpoint(md, nil, nil)
		matches[md.Address+":"+md.Port] = kvcache.PodMatch{
			WeightedScore: 4, MatchedBlocks: 4,
			BlocksByTier: map[string]int{"gpu": 4},
		}
	}
	keys := make([]kvblock.BlockHash, 8)
	// Fixed indexer results isolate request processing from hashing and index lookup.
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return keys, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return matches, nil
		},
	}
	p := newProducerForProduceAndPreRequest(b.Context(), "bench-prediction", idx)
	req := tokenizedRequest("bench", len(keys)*testBlockSize)
	req.TargetModel = "bench-model"
	result := primaryWithScored("default", endpoints[0], endpoints...)
	if err := p.Produce(b.Context(), req, endpoints); err != nil {
		b.Fatal(err)
	}
	// Exclude metric-series initialization from steady-state measurements.
	p.recordPrediction(b.Context(), req, result)
	return p, req, endpoints, result
}
