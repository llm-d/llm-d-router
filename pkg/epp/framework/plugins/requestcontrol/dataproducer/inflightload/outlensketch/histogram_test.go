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

package outlensketch

import (
	"sync"
	"testing"
)

func TestBinOf(t *testing.T) {
	cases := []struct {
		osl  int64
		want int
	}{
		{0, 0}, {499, 0}, {500, 1}, {1999, 1}, {2000, 2}, {6999, 2},
		{7000, 3}, {19999, 3}, {20000, 4}, {1_000_000, 4},
	}
	for _, c := range cases {
		if got := binOf(c.osl); got != c.want {
			t.Errorf("binOf(%d) = %d, want %d", c.osl, got, c.want)
		}
	}
}

func TestQuantileBin(t *testing.T) {
	// 50% S, 20% M, 20% L, 8% XL, 2% XXL. Cumulative mass reaches 0.85 in L
	// (0.50+0.20+0.20=0.90), so the 0.85-quantile bin is L.
	counts := [numBins]float64{50, 20, 20, 8, 2}
	if got := quantileBin(counts, 100, 0.85); got != 2 {
		t.Errorf("quantileBin at 0.85 = %d, want 2 (L)", got)
	}
	// A lower quantile that the short mass alone satisfies returns S.
	if got := quantileBin(counts, 100, 0.5); got != 0 {
		t.Errorf("quantileBin at 0.50 = %d, want 0 (S)", got)
	}
	// All mass in the top bin.
	if got := quantileBin([numBins]float64{0, 0, 0, 0, 10}, 10, 0.85); got != 4 {
		t.Errorf("quantileBin all-XXL = %d, want 4", got)
	}
}

func TestColdAndWarmup(t *testing.T) {
	// Large half-life so decay is negligible and the warmup boundary is exact.
	h := newHistogramStore(1_000_000_000, 0.85, 5, 100)

	if _, ok := h.predict(42); ok {
		t.Fatal("unseen key must abstain")
	}

	for i := 0; i < 4; i++ {
		h.observe(42, 100) // S
	}
	if _, ok := h.predict(42); ok {
		t.Fatal("key below warmup (4 < 5) must abstain")
	}
	// Decay makes the effective count slightly below the raw observation count, so
	// a few more observations are needed to clear the warmup floor of 5.
	for i := 0; i < 6; i++ {
		h.observe(42, 100)
	}
	mag, ok := h.predict(42)
	if !ok {
		t.Fatal("warm key must predict")
	}
	if mag != binMidpoints[0] {
		t.Errorf("all-short key magnitude = %d, want %d (S midpoint)", mag, binMidpoints[0])
	}
}

func TestDecayTracksRecentDrift(t *testing.T) {
	// Short half-life: recent observations dominate, so a workload that flips from
	// short to long should move the prediction up to the long bins.
	h := newHistogramStore(5, 0.85, 5, 100)
	for i := 0; i < 50; i++ {
		h.observe(7, 100) // S
	}
	if mag, _ := h.predict(7); mag != binMidpoints[0] {
		t.Fatalf("after short burst magnitude = %d, want S", mag)
	}
	for i := 0; i < 50; i++ {
		h.observe(7, 25_000) // XXL
	}
	mag, ok := h.predict(7)
	if !ok || mag != binMidpoints[numBins-1] {
		t.Errorf("after long burst magnitude = %d (ok=%v), want XXL midpoint %d", mag, ok, binMidpoints[numBins-1])
	}
}

func TestLRUEviction(t *testing.T) {
	h := newHistogramStore(1_000_000_000, 0.85, 1, 2)
	h.observe(1, 100)
	h.observe(2, 100)
	// Touch key 1 so key 2 becomes the least-recently-used.
	h.observe(1, 100)
	h.observe(3, 100) // exceeds cap -> evicts the LRU (key 2)
	if h.numKeys() != 2 {
		t.Fatalf("numKeys = %d, want 2 (capped)", h.numKeys())
	}
	if _, ok := h.predict(2); ok {
		t.Error("key 2 should have been evicted as LRU")
	}
	if _, ok := h.predict(1); !ok {
		t.Error("key 1 was recently used and should survive")
	}
}

func TestConcurrentObservePredict(t *testing.T) {
	// Exercises the store under the race detector.
	h := newHistogramStore(200, 0.85, 5, 1000)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				key := uint64((i + w) % 50)
				h.observe(key, int64(100*(i%5+1)))
				h.predict(key)
			}
		}(w)
	}
	wg.Wait()
}
