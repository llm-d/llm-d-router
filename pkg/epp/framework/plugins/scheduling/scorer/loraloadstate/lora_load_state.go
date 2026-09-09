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

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

const (
	LoraLoadStateScorerType = "lora-load-state-scorer"
)

// Parameters tunes the score each residency tier receives. The gaps between
// tiers encode the relative cost of serving the adapter from that state, which
// grows with adapter size: a large adapter makes a miss expensive and the
// free-slot tier should sit close to saturated. Unset fields keep the defaults.
// Pointers so an explicit 0.0 is distinguishable from unset.
type Parameters struct {
	// GPUResidentScore is given when the adapter occupies a GPU slot. Default 1.0.
	GPUResidentScore *float64 `json:"gpuResidentScore,omitempty"`
	// CPUResidentScore is given when the adapter is only in the host cache. Default 0.8.
	CPUResidentScore *float64 `json:"cpuResidentScore,omitempty"`
	// FreeSlotScore is given when the adapter is not resident but a GPU slot is free. Default 0.6.
	FreeSlotScore *float64 `json:"freeSlotScore,omitempty"`
	// SaturatedScore is given when the adapter is not resident and every GPU slot is taken. Default 0.0.
	SaturatedScore *float64 `json:"saturatedScore,omitempty"`
}

type tierScores struct {
	gpuResident float64
	cpuResident float64
	freeSlot    float64
	saturated   float64
}

var defaultTierScores = tierScores{gpuResident: 1.0, cpuResident: 0.8, freeSlot: 0.6, saturated: 0.0}

// tierScores applies the parameters over the defaults. Scores outside [0, 1]
// or that break gpu >= cpu >= freeSlot >= saturated are rejected as a set and
// the defaults are used.
func (p *Parameters) tierScores(ctx context.Context) tierScores {
	scores := defaultTierScores
	if p == nil {
		return scores
	}
	pick := func(v *float64, d float64) float64 {
		if v == nil {
			return d
		}
		return *v
	}
	candidate := tierScores{
		gpuResident: pick(p.GPUResidentScore, scores.gpuResident),
		cpuResident: pick(p.CPUResidentScore, scores.cpuResident),
		freeSlot:    pick(p.FreeSlotScore, scores.freeSlot),
		saturated:   pick(p.SaturatedScore, scores.saturated),
	}
	inRange := func(v float64) bool { return v >= 0 && v <= 1 }
	ordered := candidate.gpuResident >= candidate.cpuResident &&
		candidate.cpuResident >= candidate.freeSlot &&
		candidate.freeSlot >= candidate.saturated
	if !inRange(candidate.gpuResident) || !inRange(candidate.cpuResident) ||
		!inRange(candidate.freeSlot) || !inRange(candidate.saturated) || !ordered {
		log.FromContext(ctx).Info("Ignoring lora-load-state-scorer tier scores; each must be in [0, 1] with gpuResident >= cpuResident >= freeSlot >= saturated, using defaults",
			"gpuResidentScore", candidate.gpuResident, "cpuResidentScore", candidate.cpuResident,
			"freeSlotScore", candidate.freeSlot, "saturatedScore", candidate.saturated)
		return scores
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
		scores:    params.tierScores(ctx),
	}
}

// LoraLoadStateScorer scores candidate endpoints by where the requested LoRA
// adapter's weights currently live on each model server: a GPU slot, the host
// cache, or nowhere. It reads adapter residency rather than request activity,
// so an idle adapter still attracts its own traffic.
type LoraLoadStateScorer struct {
	typedName fwkplugin.TypedName
	scores    tierScores
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *LoraLoadStateScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *LoraLoadStateScorer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Consumes declares the scorer reads the per-pod resident adapter set and GPU
// slot occupancy from the endpoint's Metrics struct, published by the
// core-metrics-extractor.
func (s *LoraLoadStateScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.LoadedModelsKey, metrics.MetricsExtractorType):    map[string]fwkdl.LoraLoadState{},
			fwkplugin.NewDataKey(metrics.GPULoadedModelsKey, metrics.MetricsExtractorType): int(0),
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

	for _, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		state, resident := m.LoadedModels[request.TargetModel]

		switch {
		case resident && state.Level == fwkdl.LoraLoadLevelGPU:
			scores[endpoint] = s.scores.gpuResident
		case resident:
			scores[endpoint] = s.scores.cpuResident
		case m.GPULoadedModels < m.MaxActiveModels:
			scores[endpoint] = s.scores.freeSlot
		default:
			scores[endpoint] = s.scores.saturated
		}
	}

	return scores
}
