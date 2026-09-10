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

package loraloadstate

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/lora"
)

const (
	LoraLoadStateScorerType = "lora-load-state-scorer"
)

// Parameters tunes the score each residency tier receives and the two
// within-tier bonuses. The gaps between tiers encode the relative cost of
// serving the adapter from that state, which grows with adapter size: a large
// adapter makes a miss expensive and the free-slot tier should sit close to
// saturated. Unset fields keep the defaults. Pointers so an explicit 0.0 is
// distinguishable from unset.
type Parameters struct {
	// GPUResidentScore is given when the adapter occupies a GPU slot. Default 1.0.
	GPUResidentScore *float64 `json:"gpuResidentScore,omitempty"`
	// CPUResidentScore is given when the adapter is only in the host cache.
	// Activating a host-cache copy evicts a GPU resident and, measured on
	// Qwen3-32B, costs about as much as a load from disk, so the default sits
	// just above the free-slot tier. Default 0.7.
	CPUResidentScore *float64 `json:"cpuResidentScore,omitempty"`
	// FreeSlotScore is given when the adapter is not resident but a GPU slot is free. Default 0.6.
	FreeSlotScore *float64 `json:"freeSlotScore,omitempty"`
	// EvictableScore is given when the adapter is not resident, every GPU slot is
	// taken, and at least one unpinned resident has no request in flight, so
	// loading costs an eviction nobody is waiting on. Default 0.3.
	EvictableScore *float64 `json:"evictableScore,omitempty"`
	// SaturatedScore is given when the adapter is not resident and every GPU slot
	// holds a busy or pinned adapter. Default 0.0.
	SaturatedScore *float64 `json:"saturatedScore,omitempty"`
	// PlacementBonus is added to the one endpoint a rendezvous hash of the adapter
	// name selects, while the adapter does not occupy a GPU slot there, so the
	// first misses for an adapter converge on a single home. Default 0.03.
	PlacementBonus *float64 `json:"placementBonus,omitempty"`
	// HeadroomBonus scales with the endpoint's share of free GPU slots, breaking
	// ties within a tier toward the endpoint with the most room. Default 0.03.
	HeadroomBonus *float64 `json:"headroomBonus,omitempty"`
	// LoadHorizonSeconds prices a miss from the transition times the model
	// servers report instead of the fixed tier scores: an endpoint that would
	// take t seconds to make the adapter servable scores gpuResidentScore *
	// (1 - t/horizon), floored at saturatedScore. Endpoints that have not
	// reported a transition use the fleet mean; with no data anywhere the
	// fixed tiers apply. 0 disables it. Default 0.
	LoadHorizonSeconds *float64 `json:"loadHorizonSeconds,omitempty"`
}

type scoreTable struct {
	gpuResident    float64
	cpuResident    float64
	freeSlot       float64
	evictable      float64
	saturated      float64
	placementBonus float64
	headroomBonus  float64
	loadHorizon    float64
}

var defaultScores = scoreTable{
	gpuResident: 1.0, cpuResident: 0.7, freeSlot: 0.6, evictable: 0.3, saturated: 0.0,
	placementBonus: 0.03, headroomBonus: 0.03,
}

// budget is the score range reserved for the bonuses. Tiers are scaled into
// the remainder so a bonus can reorder endpoints within a tier but never
// across one.
func (t scoreTable) budget() float64 { return t.placementBonus + t.headroomBonus }

// valid reports whether every score is in [0, 1], the tiers are ordered
// gpuResident >= cpuResident >= freeSlot >= evictable >= saturated, and the
// bonuses fit under every non-zero gap between consecutive tiers.
func (t scoreTable) valid() bool {
	inRange := func(v float64) bool { return v >= 0 && v <= 1 }
	tiers := []float64{t.gpuResident, t.cpuResident, t.freeSlot, t.evictable, t.saturated}
	for _, v := range tiers {
		if !inRange(v) {
			return false
		}
	}
	if !inRange(t.placementBonus) || !inRange(t.headroomBonus) || t.budget() >= 1 || t.loadHorizon < 0 {
		return false
	}
	for i := 1; i < len(tiers); i++ {
		gap := tiers[i-1] - tiers[i]
		if gap < 0 {
			return false
		}
		if gap > 0 && gap*(1-t.budget()) <= t.budget() {
			return false
		}
	}
	return true
}

// scores applies the parameters over the defaults, falling back to the
// defaults as a set when the result is not valid.
func (p *Parameters) scores(ctx context.Context) scoreTable {
	if p == nil {
		return defaultScores
	}
	pick := func(v *float64, d float64) float64 {
		if v == nil {
			return d
		}
		return *v
	}
	candidate := scoreTable{
		gpuResident:    pick(p.GPUResidentScore, defaultScores.gpuResident),
		cpuResident:    pick(p.CPUResidentScore, defaultScores.cpuResident),
		freeSlot:       pick(p.FreeSlotScore, defaultScores.freeSlot),
		evictable:      pick(p.EvictableScore, defaultScores.evictable),
		saturated:      pick(p.SaturatedScore, defaultScores.saturated),
		placementBonus: pick(p.PlacementBonus, defaultScores.placementBonus),
		headroomBonus:  pick(p.HeadroomBonus, defaultScores.headroomBonus),
		loadHorizon:    pick(p.LoadHorizonSeconds, defaultScores.loadHorizon),
	}
	if !candidate.valid() {
		log.FromContext(ctx).Info("Ignoring lora-load-state-scorer parameters; scores must be in [0, 1], tiers ordered gpuResident >= cpuResident >= freeSlot >= evictable >= saturated, and the bonuses must fit under every tier gap, using defaults",
			"parameters", fmt.Sprintf("%+v", candidate))
		return defaultScores
	}
	return candidate
}

// compile-time type assertion
var (
	_ fwksched.Scorer          = &LoraLoadStateScorer{}
	_ fwkplugin.ConsumerPlugin = &LoraLoadStateScorer{}
)

// LoraLoadStateScorerFactory defines the factory function for LoraLoadStateScorer.
func LoraLoadStateScorerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", LoraLoadStateScorerType, err)
		}
	}
	ctx := context.Background()
	if handle != nil {
		ctx = handle.Context()
	}
	return NewLoraLoadStateScorer(ctx, &parameters).WithName(name), nil
}

// NewLoraLoadStateScorer initializes a new LoraLoadStateScorer and returns its pointer.
func NewLoraLoadStateScorer(ctx context.Context, params *Parameters) *LoraLoadStateScorer {
	return &LoraLoadStateScorer{
		typedName: fwkplugin.TypedName{Type: LoraLoadStateScorerType, Name: LoraLoadStateScorerType},
		scores:    params.scores(ctx),
	}
}

// LoraLoadStateScorer scores candidate endpoints by where the requested LoRA
// adapter's weights currently live on each model server: a GPU slot, the host
// cache, or nowhere. It reads adapter residency rather than request activity,
// so an idle adapter still attracts its own traffic.
type LoraLoadStateScorer struct {
	typedName fwkplugin.TypedName
	scores    scoreTable
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *LoraLoadStateScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *LoraLoadStateScorer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Consumes declares the scorer reads the per-pod resident adapter set, GPU
// slot occupancy and in-flight adapter set from the endpoint's Metrics
// struct, published by the core-metrics-extractor.
func (s *LoraLoadStateScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.LoadedModelsKey, metrics.MetricsExtractorType):        map[string]fwkdl.LoraLoadState{},
			fwkplugin.NewDataKey(metrics.GPULoadedModelsKey, metrics.MetricsExtractorType):     int(0),
			fwkplugin.NewDataKey(metrics.BaseModelKey, metrics.MetricsExtractorType):           string(""),
			fwkplugin.NewDataKey(metrics.ActiveModelsKey, metrics.MetricsExtractorType):        map[string]int{},
			fwkplugin.NewDataKey(metrics.LoraLoadSecondsKey, metrics.MetricsExtractorType):     float64(0),
			fwkplugin.NewDataKey(metrics.LoraActivateSecondsKey, metrics.MetricsExtractorType): float64(0),
		},
	}
}

// WithName sets the name of the scorer.
func (s *LoraLoadStateScorer) WithName(name string) *LoraLoadStateScorer {
	s.typedName.Name = name
	return s
}

// Score ranks endpoints by the cost of serving the target adapter there.
// Endpoints whose model server reports no residency (LoadedModels nil) fall
// through to the capacity tiers, which score the same for every such
// endpoint and so leave the decision to the other scorers.
func (s *LoraLoadStateScorer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	scale := 1 - s.scores.budget()
	if lora.IsBaseModelRequest(request, endpoints) {
		for _, endpoint := range endpoints {
			scores[endpoint] = baseModelHeadroom(endpoint.GetMetrics()) * scale
		}
		return scores
	}
	preferred := rendezvous(request.TargetModel, endpoints)
	costs := s.transitionCosts(endpoints)

	for _, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		r := classify(m, request.TargetModel)
		score := s.tierScore(r, costs[endpoint]) * scale
		if endpoint == preferred && r != gpuResident {
			score += s.scores.placementBonus
		}
		if m.MaxActiveModels > 0 && m.GPULoadedModels < m.MaxActiveModels {
			score += s.scores.headroomBonus * float64(m.MaxActiveModels-m.GPULoadedModels) / float64(m.MaxActiveModels)
		}
		scores[endpoint] = score
	}

	return scores
}

// residency is where an endpoint stands relative to one adapter, from
// cheapest to most expensive to serve it there.
type residency int

const (
	gpuResident residency = iota
	cpuResident
	freeSlot
	evictable
	saturated
)

// classify places an endpoint in a residency tier for the adapter.
func classify(m *fwkdl.Metrics, adapter string) residency {
	state, resident := m.LoadedModels[adapter]
	switch {
	case resident && state.Level == fwkdl.LoraLoadLevelGPU:
		return gpuResident
	case resident:
		return cpuResident
	case m.GPULoadedModels < m.MaxActiveModels:
		return freeSlot
	case lora.HasIdleGPUAdapter(m):
		return evictable
	default:
		return saturated
	}
}

// tierScore is the score for a residency tier: the fixed table entry, or,
// when a load horizon is set and the endpoint's transition times are known,
// gpuResident * (1 - t/horizon) floored at saturated, where t is the time to
// make the adapter servable. Evicting keeps its fixed ratio below a free
// slot so the tiers stay ordered.
func (s *LoraLoadStateScorer) tierScore(r residency, cost transitionCost) float64 {
	t := s.scores
	priced := cost.known && t.loadHorizon > 0
	price := func(seconds float64) float64 {
		return max(t.saturated, t.gpuResident*max(0, 1-seconds/t.loadHorizon))
	}
	switch r {
	case gpuResident:
		return t.gpuResident
	case cpuResident:
		if priced {
			return price(cost.activate)
		}
		return t.cpuResident
	case freeSlot:
		if priced {
			return price(cost.load + cost.activate)
		}
		return t.freeSlot
	case evictable:
		if priced && t.freeSlot > 0 {
			return price(cost.load+cost.activate) * t.evictable / t.freeSlot
		}
		return t.evictable
	default:
		return t.saturated
	}
}

// transitionCost is what an endpoint would spend making a non-resident
// adapter servable, in seconds, from the transition times it reports.
type transitionCost struct {
	load, activate float64
	known          bool
}

// transitionCosts returns a cost per endpoint when a load horizon is set.
// An endpoint that has not reported a transition borrows the fleet mean, so
// a fresh pod is priced like its peers rather than by the fixed tiers; with
// no reports anywhere every cost is unknown and the fixed tiers apply.
func (s *LoraLoadStateScorer) transitionCosts(endpoints []fwksched.Endpoint) map[fwksched.Endpoint]transitionCost {
	costs := make(map[fwksched.Endpoint]transitionCost, len(endpoints))
	if s.scores.loadHorizon <= 0 {
		return costs
	}
	var loadSum, activateSum float64
	var loadN, activateN int
	for _, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		if m.LoraLoadSeconds > 0 {
			loadSum += m.LoraLoadSeconds
			loadN++
		}
		if m.LoraActivateSeconds > 0 {
			activateSum += m.LoraActivateSeconds
			activateN++
		}
	}
	if loadN == 0 && activateN == 0 {
		return costs
	}
	fleet := transitionCost{known: true}
	if loadN > 0 {
		fleet.load = loadSum / float64(loadN)
	}
	if activateN > 0 {
		fleet.activate = activateSum / float64(activateN)
	}
	for _, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		cost := fleet
		if m.LoraLoadSeconds > 0 {
			cost.load = m.LoraLoadSeconds
		}
		if m.LoraActivateSeconds > 0 {
			cost.activate = m.LoraActivateSeconds
		}
		costs[endpoint] = cost
	}
	return costs
}

// baseModelHeadroom scores an endpoint for a request that needs no adapter:
// the share of GPU slots not serving an adapter right now, so base-model
// traffic drifts away from pods that are batching LoRA work. A resident
// adapter with nothing in flight costs the base model nothing, so only busy
// residents count. Once every endpoint is busy the term is constant and the
// other scorers decide.
func baseModelHeadroom(m *fwkdl.Metrics) float64 {
	if m.MaxActiveModels <= 0 {
		return 1
	}
	return float64(max(m.MaxActiveModels-lora.BusyGPUAdapters(m), 0)) / float64(m.MaxActiveModels)
}

// rendezvous picks the endpoint with the highest hash of (adapter, endpoint
// id), which is stable across calls and moves only the adapters that hashed
// to an endpoint that left.
func rendezvous(adapter string, endpoints []fwksched.Endpoint) fwksched.Endpoint {
	var best fwksched.Endpoint
	var bestHash uint64
	for _, endpoint := range endpoints {
		h := fnv.New64a()
		_, _ = h.Write([]byte(adapter))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(endpoint.GetMetadata().ID.String()))
		if sum := h.Sum64(); best == nil || sum > bestHash {
			best, bestHash = endpoint, sum
		}
	}
	return best
}
