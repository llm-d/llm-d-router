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
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore/residency"
)

// Weights are relative costs, in units of one load from storage. The defaults
// are placeholders until measured on the target GPUs.
type Weights struct {
	// CopyCost is a CPU-to-GPU copy (adapter in the CPU cache only).
	CopyCost float64 `json:"copyCost"`
	// LoadCost is a load from storage (adapter in neither tier).
	LoadCost float64 `json:"loadCost"`
	// BlockedCost is charged when every GPU slot is running or reported
	// active, so vLLM would skip the request until a slot drains.
	BlockedCost float64 `json:"blockedCost"`
	// PendingCost replaces the tier cost when an earlier request for the
	// adapter is already loading on the pod: the request waits for that load
	// rather than paying for its own.
	PendingCost float64 `json:"pendingCost"`
	// EvictionWeight scales the cost of the evictions an admission causes:
	// each victim's demand share times the cost of bringing it back, a copy
	// for a GPU adapter demoted to CPU and a load for one dropped to storage.
	EvictionWeight float64 `json:"evictionWeight"`
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

// Score returns a score in [0,1] per pod for a request, given each candidate
// pod's admission of the request's adapter. It is min-max normalized over the
// candidates' costs: the cheapest pod scores 1, the costliest 0, and equal
// costs score every pod 1 so they do not discriminate. A base-model request
// scores every pod 1, leaving the choice to other scorers.
//
// isBase marks a base-model request. It must come from configuration (the
// pool's base model names), not from the adapter being absent from metrics,
// which is indistinguishable from a cold adapter.
func Score(admissions map[string]residency.Admission, d *Demand, isBase bool, w Weights, now time.Time) map[string]float64 {
	scores := make(map[string]float64, len(admissions))
	if isBase {
		for p := range admissions {
			scores[p] = 1
		}
		return scores
	}
	victims := make([]string, 0, 2*len(admissions))
	for _, a := range admissions {
		victims = append(victims, a.GPUVictim, a.CPUVictim)
	}
	// One read of every victim's share, so that pods evicting the same adapter
	// are charged exactly the same amount.
	shares := d.Shares(victims, now)
	costs := make(map[string]float64, len(admissions))
	minCost, maxCost := math.Inf(1), 0.0
	for p, a := range admissions {
		c := cost(a, shares, w)
		costs[p] = c
		minCost = math.Min(minCost, c)
		maxCost = math.Max(maxCost, c)
	}
	for p, c := range costs {
		if maxCost == minCost {
			scores[p] = 1
		} else {
			scores[p] = 1 - (c-minCost)/(maxCost-minCost)
		}
	}
	return scores
}

func cost(a residency.Admission, shares map[string]float64, w Weights) float64 {
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
		c += w.EvictionWeight * shares[a.GPUVictim] * w.CopyCost
	}
	if a.CPUVictim != "" {
		c += w.EvictionWeight * shares[a.CPUVictim] * w.LoadCost
	}
	return c
}

// maxAdapters bounds the adapters Demand tracks. Target model names come from
// clients, so without a bound the estimator grows with every name sent.
const maxAdapters = 1024

// pruneBelow is the decayed rate, in requests per second, below which an
// adapter is forgotten. Its share of any total is then negligible.
const pruneBelow = 1e-6

type demand struct {
	rate float64
	last time.Time
}

// Demand estimates each adapter's request rate as an exponentially decaying
// count, in requests per second. It is safe for concurrent use.
type Demand struct {
	mu      sync.Mutex
	tau     float64 // decay time constant in seconds
	limit   int
	adapter map[string]*demand
}

// NewDemand returns an estimator whose estimate halves every halfLife, which
// must be positive.
func NewDemand(halfLife time.Duration) (*Demand, error) {
	if halfLife <= 0 {
		return nil, errors.New("lorascore: demand half-life must be positive")
	}
	return &Demand{tau: halfLife.Seconds() / math.Ln2, limit: maxAdapters, adapter: map[string]*demand{}}, nil
}

// Observe records one request for adapter at time t. When the estimator is
// full, a new adapter first drops every adapter decayed below pruneBelow and,
// if none has, the one with the lowest rate.
func (d *Demand) Observe(adapter string, t time.Time) {
	d.mu.Lock()
	defer d.mu.Unlock()
	e, ok := d.adapter[adapter]
	if !ok {
		if len(d.adapter) >= d.limit {
			d.evict(t)
		}
		e = &demand{last: t}
		d.adapter[adapter] = e
	}
	e.rate = d.decayed(e, t) + 1/d.tau
	if t.After(e.last) {
		e.last = t
	}
}

// evict makes room for one adapter. Callers hold d.mu.
func (d *Demand) evict(t time.Time) {
	lowest, lowestRate := "", math.Inf(1)
	for a, e := range d.adapter {
		r := d.decayed(e, t)
		if r < pruneBelow {
			delete(d.adapter, a)
			continue
		}
		if r < lowestRate || (r == lowestRate && a < lowest) {
			lowest, lowestRate = a, r
		}
	}
	if len(d.adapter) >= d.limit {
		delete(d.adapter, lowest)
	}
}

// rate returns adapter's estimated request rate at time t.
func (d *Demand) rate(adapter string, t time.Time) float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.decayed(d.adapter[adapter], t)
}

// share returns adapter's fraction of the total estimated rate at time t, or 0
// when nothing has been observed.
func (d *Demand) share(adapter string, t time.Time) float64 {
	return d.Shares([]string{adapter}, t)[adapter]
}

// Shares returns each named adapter's share at time t, all against one total.
// Empty names are skipped. Summing the total also forgets decayed adapters.
func (d *Demand) Shares(adapters []string, t time.Time) map[string]float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	total := 0.0
	for a, e := range d.adapter {
		r := d.decayed(e, t)
		if r < pruneBelow {
			delete(d.adapter, a)
			continue
		}
		total += r
	}
	out := make(map[string]float64, len(adapters))
	for _, a := range adapters {
		if a == "" {
			continue
		}
		if total > 0 {
			out[a] = min(d.decayed(d.adapter[a], t)/total, 1)
		} else {
			out[a] = 0
		}
	}
	return out
}

func (d *Demand) decayed(e *demand, t time.Time) float64 {
	if e == nil {
		return 0
	}
	dt := t.Sub(e.last).Seconds()
	if dt <= 0 {
		return e.rate
	}
	return e.rate * math.Exp(-dt/d.tau)
}
