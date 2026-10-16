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

// Package lorascore scores candidate pods for a LoRA request by the estimated
// cost of serving it there: the adapter miss itself, the eviction it causes,
// and whether every GPU slot is busy.
package lorascore

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore/residency"
)

// Weights are relative costs, in units of one load from storage. The defaults
// are placeholders until measured on the target GPUs.
type Weights struct {
	// CopyCost is a CPU-to-GPU copy (adapter in the CPU cache only).
	CopyCost float64
	// LoadCost is a load from storage (adapter in neither tier).
	LoadCost float64
	// BlockedCost is charged when every GPU slot is running or reported
	// active, so vLLM would skip the request until a slot drains.
	BlockedCost float64
	// PendingCost replaces the tier cost when an earlier request for the
	// adapter is already loading on the pod: the request waits for that load
	// rather than paying for its own.
	PendingCost float64
	// EvictionWeight scales the cost of the evictions an admission causes:
	// each victim's demand share times the cost of bringing it back, a copy
	// for a GPU adapter demoted to CPU and a load for one dropped to storage.
	EvictionWeight float64
}

// DefaultWeights returns the starting weights.
func DefaultWeights() Weights {
	return Weights{CopyCost: 0.2, LoadCost: 1, BlockedCost: 1, PendingCost: 0.1, EvictionWeight: 1}
}

// Validate reports an error if any weight is negative or not finite, which
// would turn costs, and so scores, into NaN or infinities.
func (w Weights) Validate() error {
	for name, v := range map[string]float64{
		"CopyCost": w.CopyCost, "LoadCost": w.LoadCost, "BlockedCost": w.BlockedCost,
		"PendingCost": w.PendingCost, "EvictionWeight": w.EvictionWeight,
	} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("lorascore: weight %s = %v, want finite and non-negative", name, v)
		}
	}
	return nil
}

// Score returns a score in [0,1] per pod for a request for adapter, min-max
// normalized over the candidates' costs: the cheapest pod scores 1, the
// costliest 0, and equal costs score every pod 1 so they do not discriminate.
// A base-model request scores every pod 1, leaving the choice to other scorers.
//
// isBase marks a base-model request. It must come from configuration (the
// pool's base model names), not from the adapter being absent from metrics,
// which is indistinguishable from a cold adapter.
func Score(m *residency.Model, d *Demand, adapter string, isBase bool, pods []string, w Weights, now time.Time) map[string]float64 {
	scores := make(map[string]float64, len(pods))
	if isBase {
		for _, p := range pods {
			scores[p] = 1
		}
		return scores
	}
	costs := make(map[string]float64, len(pods))
	minCost, maxCost := math.Inf(1), 0.0
	for _, p := range pods {
		c := cost(m.Admit(p, adapter), d, w, now)
		costs[p] = c
		minCost = math.Min(minCost, c)
		maxCost = math.Max(maxCost, c)
	}
	for _, p := range pods {
		if maxCost == minCost {
			scores[p] = 1
		} else {
			scores[p] = 1 - (costs[p]-minCost)/(maxCost-minCost)
		}
	}
	return scores
}

func cost(a residency.Admission, d *Demand, w Weights, now time.Time) float64 {
	c := 0.0
	switch {
	case a.Pending:
		c += w.PendingCost
	case a.Tier == residency.CPU:
		c += w.CopyCost
	case a.Tier == residency.Absent:
		c += w.LoadCost
	}
	if a.Blocked {
		c += w.BlockedCost
	}
	if a.GPUVictim != "" {
		c += w.EvictionWeight * d.Share(a.GPUVictim, now) * w.CopyCost
	}
	if a.CPUVictim != "" {
		c += w.EvictionWeight * d.Share(a.CPUVictim, now) * w.LoadCost
	}
	return c
}

// pruneBelow is the decayed rate, in requests per second, below which an
// adapter is forgotten. Its share of any total is then negligible, and
// forgetting it keeps the estimator from growing with every target model name
// a client sends.
const pruneBelow = 1e-6

// minPruneSize is the fewest observations between sweeps for decayed adapters.
const minPruneSize = 64

// Demand estimates each adapter's request rate as an exponentially decaying
// count, in requests per second. It is safe for concurrent use.
type Demand struct {
	mu   sync.Mutex
	tau  float64 // decay time constant in seconds
	rate map[string]float64
	last map[string]time.Time
	// total is the sum of every adapter's rate as of totalAt. All rates decay
	// alike, so Share reads it in O(1) instead of summing over client-chosen
	// names.
	total   float64
	totalAt time.Time
	// sinceSweep counts observations since the last sweep. A sweep runs once
	// it reaches nextSweep, the number of adapters the previous sweep kept (at
	// least minPruneSize), so sweeps cost O(log n) amortized. The threshold is
	// fixed at the sweep: compared with the live map size, it would never be
	// reached while every observation adds a new name.
	sinceSweep int
	nextSweep  int
}

// NewDemand returns an estimator whose estimate halves every halfLife, which
// must be positive.
func NewDemand(halfLife time.Duration) (*Demand, error) {
	if halfLife <= 0 {
		return nil, errors.New("lorascore: demand half-life must be positive")
	}
	return &Demand{
		tau:  halfLife.Seconds() / math.Ln2,
		rate: map[string]float64{},
		last: map[string]time.Time{},
	}, nil
}

// Observe records one request for adapter at time t.
func (d *Demand) Observe(adapter string, t time.Time) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if t.After(d.totalAt) {
		d.total = d.decayedTotal(t)
		d.totalAt = t
	}
	before := d.decayed(adapter, d.totalAt)
	d.rate[adapter] = d.decayed(adapter, t) + 1/d.tau
	if t.After(d.last[adapter]) {
		d.last[adapter] = t
	}
	d.total += d.decayed(adapter, d.totalAt) - before
	if d.sinceSweep++; d.sinceSweep >= max(d.nextSweep, minPruneSize) {
		d.prune(t)
	}
}

// prune forgets adapters whose rate at time t has decayed below pruneBelow,
// then recomputes the total. Callers hold d.mu.
func (d *Demand) prune(t time.Time) {
	for a := range d.rate {
		if d.decayed(a, t) < pruneBelow {
			delete(d.rate, a)
			delete(d.last, a)
		}
	}
	d.retotal(d.totalAt)
	d.sinceSweep, d.nextSweep = 0, len(d.rate)
}

// retotal recomputes the total as of t from the per-adapter rates, in sorted
// order so that the result does not depend on map iteration order. It also
// clears the rounding error that incremental updates accumulate. Callers hold
// d.mu.
func (d *Demand) retotal(t time.Time) {
	names := make([]string, 0, len(d.rate))
	for a := range d.rate {
		names = append(names, a)
	}
	sort.Strings(names)
	total := 0.0
	for _, a := range names {
		total += d.decayed(a, t)
	}
	d.total, d.totalAt = total, t
}

// decayedTotal returns the total decayed from totalAt to t. Callers hold d.mu.
func (d *Demand) decayedTotal(t time.Time) float64 {
	dt := t.Sub(d.totalAt).Seconds()
	if dt <= 0 {
		return d.total
	}
	return d.total * math.Exp(-dt/d.tau)
}

// Rate returns adapter's estimated request rate at time t.
func (d *Demand) Rate(adapter string, t time.Time) float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.decayed(adapter, t)
}

// Share returns adapter's fraction of the total estimated rate at time t, or 0
// when nothing has been observed.
func (d *Demand) Share(adapter string, t time.Time) float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	// Every adapter was last observed no later than totalAt, so the total is
	// known for any t at or after it. An earlier t, from a caller that read the
	// clock before a concurrent Observe, is read at totalAt instead, so that
	// the numerator and the total are taken at the same time and the shares
	// still sum to 1.
	if t.Before(d.totalAt) {
		t = d.totalAt
	}
	total := d.decayedTotal(t)
	if total <= 0 {
		return 0
	}
	return min(d.decayed(adapter, t)/total, 1)
}

func (d *Demand) decayed(adapter string, t time.Time) float64 {
	dt := t.Sub(d.last[adapter]).Seconds()
	if dt <= 0 {
		return d.rate[adapter]
	}
	return d.rate[adapter] * math.Exp(-dt/d.tau)
}
