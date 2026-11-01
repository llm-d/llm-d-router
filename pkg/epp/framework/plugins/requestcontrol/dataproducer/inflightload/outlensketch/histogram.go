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
	"math"
	"sync"
	"sync/atomic"
)

// Keyed, count-decaying histogram of observed output lengths. Each key owns a
// fixed-length count vector over the output-size bins; an observed completion
// decays every bin by a factor and increments the observed bin, so a count halves
// after a key-local half-life of N observations (count-based aging, no wall clock).
// A prediction reads the bin at a high quantile of the key's distribution, which
// over-provisions because under-calling a long request is the expensive error. A
// key below the warmup count abstains. Memory is bounded by an LRU cap that, with
// the small live key set, rarely fires.
//
// The output-size bins are data-driven (distribution valleys): S<500, M 500-2000,
// L 2000-7000, XL 7000-20000, XXL 20000+. A prediction maps the chosen bin to a
// representative token magnitude (bin midpoint; the open-ended top bin uses a fixed
// representative).
const numBins = 5

var (
	// binEdges are the lower bounds for bins M..XXL; an output below binEdges[i]
	// is in bin i, and the top bin is open-ended.
	binEdges = [numBins - 1]int64{500, 2000, 7000, 20000}
	// binMidpoints are the representative token magnitudes per bin.
	binMidpoints = [numBins]int64{250, 1250, 4500, 13500, 30000}
)

// binOf returns the bin index for an observed output length.
func binOf(osl int64) int {
	for i, edge := range binEdges {
		if osl < edge {
			return i
		}
	}
	return numBins - 1
}

// histEntry is one key's decaying count vector plus an LRU timestamp.
type histEntry struct {
	mu       sync.Mutex
	counts   [numBins]float64
	lastSeen atomic.Int64
}

// histogramStore holds the per-key histograms and the shared read/predict logic.
type histogramStore struct {
	lambda    float64 // per-observation decay factor, 0.5^(1/N)
	q         float64 // over-provision quantile
	warmupMin float64 // minimum effective count before a key is trusted
	maxKeys   int

	mu    sync.RWMutex
	store map[uint64]*histEntry
	clock atomic.Int64
}

func newHistogramStore(halfLife int, q, warmupMin float64, maxKeys int) *histogramStore {
	return &histogramStore{
		lambda:    math.Pow(0.5, 1.0/float64(halfLife)),
		q:         q,
		warmupMin: warmupMin,
		maxKeys:   maxKeys,
		store:     make(map[uint64]*histEntry),
	}
}

// observe folds an observed output length into the key's histogram: decay every
// bin, then increment the observed bin.
func (h *histogramStore) observe(key uint64, osl int64) {
	e := h.getOrCreate(key)
	e.mu.Lock()
	for b := range e.counts {
		e.counts[b] *= h.lambda
	}
	e.counts[binOf(osl)]++
	e.mu.Unlock()
	e.lastSeen.Store(h.clock.Add(1))
}

// predict returns the representative token magnitude for the key, or ok=false when
// the key is unseen or still below the warmup count (abstain).
func (h *histogramStore) predict(key uint64) (magnitude int64, ok bool) {
	h.mu.RLock()
	e := h.store[key]
	h.mu.RUnlock()
	if e == nil {
		return 0, false
	}

	e.mu.Lock()
	var total float64
	for _, c := range e.counts {
		total += c
	}
	if total < h.warmupMin {
		e.mu.Unlock()
		return 0, false
	}
	bin := quantileBin(e.counts, total, h.q)
	e.mu.Unlock()

	e.lastSeen.Store(h.clock.Add(1))
	return binMidpoints[bin], true
}

func (h *histogramStore) getOrCreate(key uint64) *histEntry {
	h.mu.RLock()
	e := h.store[key]
	h.mu.RUnlock()
	if e != nil {
		return e
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	if e = h.store[key]; e != nil {
		return e
	}
	if len(h.store) >= h.maxKeys {
		h.evictLRULocked()
	}
	e = &histEntry{}
	h.store[key] = e
	return e
}

// evictLRULocked drops the least-recently-used key. The caller holds h.mu. The scan
// is O(maxKeys) but fires only at the cap, which the small live key set rarely hits.
func (h *histogramStore) evictLRULocked() {
	var victim uint64
	var oldest int64
	first := true
	for k, e := range h.store {
		ls := e.lastSeen.Load()
		if first || ls < oldest {
			victim, oldest, first = k, ls, false
		}
	}
	if !first {
		delete(h.store, victim)
	}
}

func (h *histogramStore) numKeys() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.store)
}

// quantileBin returns the bin containing the q-quantile of the distribution: the
// lowest bin whose cumulative mass reaches q. Walking from the short end makes
// SHORT high-precision (returned only when at least q of the mass is short) and
// over-provisions otherwise.
func quantileBin(counts [numBins]float64, total, q float64) int {
	cum := 0.0
	for b := 0; b < numBins; b++ {
		cum += counts[b] / total
		if cum >= q {
			return b
		}
	}
	return numBins - 1
}
