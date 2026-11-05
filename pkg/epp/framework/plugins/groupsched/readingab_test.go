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

package groupsched

// THE EXACTNESS REPLAY: every cost the scheduler takes is the request's own prefill chain, on
// the wave's real prompt-size distribution.

import (
	"fmt"
	"sort"
	"testing"
)

// promptSizeQuantiles is the wave's own prompt-token distribution (p0..p100, step 5); mean of
// the full set is 74,493 tokens, p50 71,489, max 209,035.
var promptSizeQuantiles = []int{
	264, 2954, 20936, 31358, 41983, 46500, 50930, 56435, 62381, 66565, 71487,
	75933, 81134, 86547, 91884, 98033, 104611, 115559, 131899, 157181, 209035,
}

// readingABBlocks is the campaign engine's num_gpu_blocks (2.57M tokens of KV).
const readingABBlocks = 40183

func pct(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	i := int(p / 100 * float64(len(s)-1))
	if i < 0 {
		i = 0
	}
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

func TestAdmissionCostIsExactOnTheCorpus(t *testing.T) {
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	const engine = "engine-a"
	if err := s.SetNumBlocksFor(engine, 64, readingABBlocks); err != nil {
		t.Fatal(err)
	}
	g := s.groups[engine]

	var ratios []float64
	var partial, exact int
	const n = 420 // 20 draws per quantile bucket
	for i := 0; i < n; i++ {
		tokens := promptSizeQuantiles[(i*7)%len(promptSizeQuantiles)]
		ids := make([]uint32, tokens)
		for j := range ids {
			ids[j] = uint32(j%50257) + 1
		}
		chains := chainsOfTokens(ids, 64)
		wantBlocks := (tokens + 63) / 64
		if len(chains) != 1 || len(chains[0].Keys) != wantBlocks {
			t.Fatalf("request %d (%d tokens): %d chains of %d blocks, want 1 chain of %d",
				i, tokens, len(chains), len(chains[0].Keys), wantBlocks)
		}
		cost := g.index.Cost(chains)
		if cost != wantBlocks {
			t.Fatalf("request %d (%d tokens): cost %d blocks, want %d — the cost must be the exact chain",
				i, tokens, cost, wantBlocks)
		}
		if cold := g.index.ColdTokens(chains); cold != int64(tokens) {
			t.Fatalf("request %d (%d tokens): cold charge %d tokens, want %d", i, tokens, cold, tokens)
		}
		// and the seat charges exactly that
		s.mu.Lock()
		ok, seatedCost := g.index.TrySeat(fmt.Sprintf("m%03d", i), chains, readingABBlocks)
		used := g.index.UniqueBlocks()
		s.mu.Unlock()
		if !ok || seatedCost != cost || used != wantBlocks {
			t.Fatalf("request %d: seat ok=%v cost=%d used=%d, want true/%d/%d",
				i, ok, seatedCost, used, cost, wantBlocks)
		}
		// the next request is costed against an engine that now holds this chain
		if next := g.index.Cost(chains); next != 0 {
			t.Fatalf("request %d: a fully resident chain still costs %d", i, next)
		}
		s.mu.Lock()
		g.index.Unseat(fmt.Sprintf("m%03d", i))
		s.mu.Unlock()
		ratios = append(ratios, float64(cost)/float64(wantBlocks))
		if tokens%64 != 0 {
			partial++
		}
		exact++
	}
	t.Logf("EXACTNESS REPLAY — n=%d requests, %d with a partial tail block, engine %d blocks x 64 tokens",
		n, partial, readingABBlocks)
	t.Logf("  cost / true blocks: p05 %.3f  p50 %.3f  p95 %.3f   (1.000 = exact)",
		pct(ratios, 5), pct(ratios, 50), pct(ratios, 95))
	for i, r := range ratios {
		if r != 1 {
			t.Fatalf("request %d: cost %.4f of the true block count — the cost must be exact", i, r)
		}
	}
	if exact != n {
		t.Fatalf("%d of %d requests were exact", exact, n)
	}
}

// THE TRAILING PARTIAL BLOCK: the engine allocates a block for the last tokens of a prompt
// whether or not they fill it, so the chain counts it — and the valve's charge, which is in
// tokens, does not.
func TestPartialBlockIsCountedInTheChain(t *testing.T) {
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	ids := func(n int) []uint32 {
		out := make([]uint32, n)
		for i := range out {
			out[i] = uint32(i) + 1
		}
		return out
	}
	if got := len(chainsOfTokens(ids(65), 64)[0].Keys); got != 2 {
		t.Fatalf("65 tokens at 64/block: %d keys, want 2 (the tail block is allocated)", got)
	}
	if got := len(chainsOfTokens(ids(64), 64)[0].Keys); got != 1 {
		t.Fatalf("64 tokens at 64/block: %d keys, want 1", got)
	}
	if got := len(chainsOfTokens(nil, 64)); got != 0 {
		t.Fatalf("no tokens: %d chains, want 0 (no phantom block)", got)
	}
	_ = s
}
