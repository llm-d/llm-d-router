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

// Package residency estimates, from the router's side, which LoRA adapters a
// vLLM pod holds on GPU and in its CPU cache.
//
// It mirrors vLLM's V1 engine (checked at vllm ad0f67a7ec): both tiers are LRU
// caches that every engine step touches for each adapter in the scheduled
// batch, and evicting from the CPU cache also deactivates on GPU, so the GPU
// set is a subset of the CPU set. One recency order per pod therefore
// describes both tiers: the max_loras most recent adapters are on GPU, the
// max_cpu_loras most recent are in CPU memory, and anything older must be
// loaded from storage.
//
// The router does not see engine steps, so recency comes from evidence that an
// adapter executed: a request that started producing output, or a finished
// request that had started. A request that is only routed is pending and
// changes nothing, because vLLM schedules running requests first and skips a
// waiting request whose adapter would exceed max_loras. An executing adapter
// ranks above every idle one, since vLLM touches it each step.
package residency

import (
	"sort"
	"time"
)

// Tier is where an adapter is estimated to live on a pod.
type Tier int

const (
	// Absent means neither on GPU nor in the CPU cache: a load from storage.
	Absent Tier = iota
	// CPU means in the CPU cache only: a CPU-to-GPU copy.
	CPU
	// GPU means active in a GPU slot: no load.
	GPU
)

func (t Tier) String() string {
	switch t {
	case GPU:
		return "GPU"
	case CPU:
		return "CPU"
	default:
		return "Absent"
	}
}

// Admission is the estimated effect of serving one more request for an adapter
// on a pod, without changing the estimate.
type Admission struct {
	// Tier is where the adapter is now.
	Tier Tier
	// GPUVictim is the GPU adapter demoted to the CPU cache to free a slot,
	// or "".
	GPUVictim string
	// CPUVictim is the CPU-cache adapter dropped to storage to make room for
	// an absent adapter, or "". In vLLM's default max_cpu_loras == max_loras
	// it is also the adapter whose GPU slot is freed, so GPUVictim is "".
	CPUVictim string
	// Blocked reports that every GPU slot holds an adapter that is running or
	// reported active, so vLLM would skip the request until one drains.
	Blocked bool
	// Pending reports that an earlier request for the adapter is routed to
	// this pod but has not started, so its load is already under way and the
	// evictions it causes are not charged again.
	Pending bool
}

type entry struct {
	last    time.Time
	running int
}

// Pod is the estimate for one pod. It is not safe for concurrent use.
type Pod struct {
	gpu, cpu int
	entries  map[string]*entry
	pending  map[string]int
	// active is the latest running-or-queued snapshot (llm-d ActiveModels):
	// it protects GPU adapters from eviction but is not evidence of execution.
	active map[string]bool
}

// NewPod returns an empty estimate with the given GPU slots (max_loras) and CPU
// cache size (max_cpu_loras). A CPU size below the GPU size is raised to it, as
// vLLM requires max_cpu_loras >= max_loras.
func NewPod(gpuSlots, cpuSlots int) *Pod {
	p := &Pod{
		entries: map[string]*entry{},
		pending: map[string]int{},
		active:  map[string]bool{},
	}
	p.SetCapacity(gpuSlots, cpuSlots)
	return p
}

// SetCapacity changes the tier sizes, forgetting idle adapters that no longer fit.
func (p *Pod) SetCapacity(gpuSlots, cpuSlots int) {
	p.gpu = max(gpuSlots, 1)
	p.cpu = max(cpuSlots, p.gpu)
	p.prune()
}

// Routed records a request for adapter sent to this pod. It is pending until
// Started: it neither touches recency nor makes the adapter resident.
func (p *Pod) Routed(adapter string) {
	p.pending[adapter]++
}

// Started records evidence that a request for adapter executed at time t, such
// as its first response chunk.
func (p *Pod) Started(adapter string, t time.Time) {
	p.unpend(adapter)
	p.touch(adapter, t).running++
	p.prune()
}

// Finished records the end of a request for adapter at time t. started tells
// whether it had started; a request aborted before it reached the engine
// leaves no trace.
func (p *Pod) Finished(adapter string, t time.Time, started bool) {
	if !started {
		p.unpend(adapter)
		return
	}
	e := p.touch(adapter, t)
	if e.running > 0 {
		e.running--
	}
	p.prune()
}

// ObserveActive replaces the snapshot of adapters the pod reports running or
// queued (llm-d's ActiveModels). It does not touch recency or create
// residency, since a queued adapter need not be loaded; it only keeps adapters
// already estimated on GPU from being chosen as victims.
func (p *Pod) ObserveActive(adapters []string) {
	p.active = make(map[string]bool, len(adapters))
	for _, a := range adapters {
		p.active[a] = true
	}
}

// Order returns the tracked adapters, most recent first.
func (p *Pod) Order() []string {
	names := make([]string, 0, len(p.entries))
	for a := range p.entries {
		names = append(names, a)
	}
	sort.Slice(names, func(i, j int) bool { return p.before(names[i], names[j]) })
	return names
}

func (p *Pod) tierIn(order []string, adapter string) Tier {
	for i, a := range order {
		if a == adapter {
			switch {
			case i < p.gpu:
				return GPU
			case i < p.cpu:
				return CPU
			}
			return Absent
		}
	}
	return Absent
}

// Admit estimates the effect of serving a request for adapter now, following
// vLLM's order: an absent adapter is first loaded into the CPU cache, evicting
// its oldest entry if full (which also frees that entry's GPU slot), and is
// then activated, evicting the oldest GPU adapter if no slot is free.
func (p *Pod) Admit(adapter string) Admission {
	order := p.Order()
	a := Admission{Tier: p.tierIn(order, adapter)}
	if a.Tier == GPU {
		return a
	}
	a.Pending = p.pending[adapter] > 0
	if len(order) < p.gpu {
		return a
	}
	gpuVictim := p.oldestEvictable(order, p.gpu)
	if gpuVictim < 0 {
		a.Blocked = true
		return a
	}
	if a.Pending {
		return a
	}
	if a.Tier == Absent && len(order) >= p.cpu {
		if cpuVictim := p.oldestEvictable(order, p.cpu); cpuVictim >= 0 {
			a.CPUVictim = order[cpuVictim]
			if cpuVictim < p.gpu {
				return a // the dropped adapter's GPU slot is the free slot
			}
		}
	}
	a.GPUVictim = order[gpuVictim]
	return a
}

// oldestEvictable returns the index of the least recent evictable adapter among
// the first n of order, or -1. Executing adapters are never evictable. An
// adapter reported active is protected only while estimated on GPU (index <
// gpu), where it is plausibly running and so touched every step; one only in
// the CPU cache cannot be running, and vLLM's CPU LRU ignores queue state.
func (p *Pod) oldestEvictable(order []string, n int) int {
	for i := min(n, len(order)) - 1; i >= 0; i-- {
		if !p.executing(order[i]) && (!p.active[order[i]] || i >= p.gpu) {
			return i
		}
	}
	return -1
}

func (p *Pod) executing(adapter string) bool {
	e := p.entries[adapter]
	return e != nil && e.running > 0
}

func (p *Pod) unpend(adapter string) {
	if p.pending[adapter] > 1 {
		p.pending[adapter]--
	} else {
		delete(p.pending, adapter)
	}
}

func (p *Pod) touch(adapter string, t time.Time) *entry {
	e, ok := p.entries[adapter]
	if !ok {
		e = &entry{}
		p.entries[adapter] = e
	}
	if t.After(e.last) {
		e.last = t
	}
	return e
}

// before orders executing adapters first, then by recency, then by name so the
// order is deterministic.
func (p *Pod) before(a, b string) bool {
	if xa, xb := p.executing(a), p.executing(b); xa != xb {
		return xa
	}
	ea, eb := p.entries[a], p.entries[b]
	if !ea.last.Equal(eb.last) {
		return ea.last.After(eb.last)
	}
	return a < b
}

// prune forgets idle adapters ranked past the CPU cache. Executing adapters are
// kept even past it, since their requests have not finished.
func (p *Pod) prune() {
	for i, a := range p.Order() {
		if i >= p.cpu && !p.executing(a) {
			delete(p.entries, a)
		}
	}
}
