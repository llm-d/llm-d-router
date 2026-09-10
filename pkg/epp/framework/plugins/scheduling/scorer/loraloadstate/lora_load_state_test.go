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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

var (
	gpu    = fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelGPU}
	cpu    = fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelCPU}
	pinned = fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelGPU, Pinned: true}
)

func endpoint(name string, m *fwkdl.Metrics) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, m, nil)
}

func score(t *testing.T, params *Parameters, target string, endpoints ...fwksched.Endpoint) map[string]float64 {
	t.Helper()
	scores := NewLoraLoadStateScorer(context.Background(), params).Score(context.Background(), &fwksched.InferenceRequest{TargetModel: target}, endpoints)
	require.Len(t, scores, len(endpoints))
	byName := map[string]float64{}
	for ep, s := range scores {
		byName[ep.GetMetadata().ID.Name] = s
		assert.GreaterOrEqual(t, s, 0.0)
		assert.LessOrEqual(t, s, 1.0)
	}
	return byName
}

func TestTiers(t *testing.T) {
	full := map[string]int{} // no in-flight requests
	tests := []struct {
		name     string
		metrics  *fwkdl.Metrics
		expected float64
	}{
		{"gpu resident", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"target": gpu, "other": gpu}, GPULoadedModels: 2, MaxActiveModels: 2, ActiveModels: full}, 1.0},
		{"cpu resident, slots full", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"target": cpu, "a": gpu, "b": gpu}, GPULoadedModels: 2, MaxActiveModels: 2, ActiveModels: full}, 0.7},
		{"not resident, free slot", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"other": gpu}, GPULoadedModels: 1, MaxActiveModels: 2, ActiveModels: full}, 0.6},
		{"saturated, a resident is idle", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"a": gpu, "b": gpu}, GPULoadedModels: 2, MaxActiveModels: 2, ActiveModels: map[string]int{"a": 1}}, 0.3},
		{"saturated, every resident busy", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"a": gpu, "b": gpu}, GPULoadedModels: 2, MaxActiveModels: 2, ActiveModels: map[string]int{"a": 1, "b": 1}}, 0.0},
		{"saturated, idle resident is pinned", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"a": pinned, "b": gpu}, GPULoadedModels: 2, MaxActiveModels: 2, ActiveModels: map[string]int{"b": 1}}, 0.0},
		{"saturated, idle resident only in cpu cache", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"a": gpu, "c": cpu}, GPULoadedModels: 1, MaxActiveModels: 1, ActiveModels: map[string]int{"a": 1}}, 0.0},
		{"residency not reported, capacity known", &fwkdl.Metrics{ActiveModels: map[string]int{"target": 1}, MaxActiveModels: 2}, 0.6},
		{"residency not reported, capacity unknown", &fwkdl.Metrics{ActiveModels: map[string]int{"target": 1}}, 0.0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := score(t, nil, "target", endpoint("pod", test.metrics))
			assert.InDelta(t, test.expected, got["pod"], 0.0001)
		})
	}
}

func TestIdleResidentBeatsBusyOneAndFreeSlotBeatsBoth(t *testing.T) {
	got := score(t, nil, "target",
		endpoint("free", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"a": gpu}, GPULoadedModels: 1, MaxActiveModels: 2, ActiveModels: map[string]int{"a": 1}}),
		endpoint("idle-victim", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"a": gpu, "b": gpu}, GPULoadedModels: 2, MaxActiveModels: 2, ActiveModels: map[string]int{"a": 1}}),
		endpoint("all-busy", &fwkdl.Metrics{LoadedModels: map[string]fwkdl.LoraLoadState{"a": gpu, "b": gpu}, GPULoadedModels: 2, MaxActiveModels: 2, ActiveModels: map[string]int{"a": 1, "b": 1}}),
	)
	assert.Greater(t, got["free"], got["idle-victim"])
	assert.Greater(t, got["idle-victim"], got["all-busy"])
}

func TestParameters(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	tests := []struct {
		name     string
		params   *Parameters
		expected scoreTable
	}{
		{"nil uses defaults", nil, defaultScores},
		{"empty uses defaults", &Parameters{}, defaultScores},
		{
			"large adapter: a miss is nearly as bad as saturation",
			&Parameters{CPUResidentScore: f(0.9), FreeSlotScore: f(0.15), EvictableScore: f(0.1), SaturatedScore: f(0.0)},
			scoreTable{gpuResident: 1.0, cpuResident: 0.9, freeSlot: 0.15, evictable: 0.1, saturated: 0.0},
		},
		{"partial override keeps the other defaults", &Parameters{FreeSlotScore: f(0.5)}, scoreTable{1.0, 0.7, 0.5, 0.3, 0.0, 0}},
		{"out of range falls back as a set", &Parameters{GPUResidentScore: f(1.5), FreeSlotScore: f(0.1)}, defaultScores},
		{"tier order violation falls back as a set", &Parameters{CPUResidentScore: f(0.2), FreeSlotScore: f(0.5)}, defaultScores},
		{"equal tiers are allowed", &Parameters{EvictableScore: f(0.0)}, scoreTable{1.0, 0.7, 0.6, 0.0, 0.0, 0}},
		{"load horizon is kept", &Parameters{LoadHorizonSeconds: f(2)}, scoreTable{1.0, 0.7, 0.6, 0.3, 0.0, 2}},
		{"negative load horizon falls back as a set", &Parameters{LoadHorizonSeconds: f(-1)}, defaultScores},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.expected, NewLoraLoadStateScorer(context.Background(), test.params).scores)
		})
	}
}

func TestFactory(t *testing.T) {
	decoder := json.NewDecoder(strings.NewReader(`{"cpuResidentScore": 0.95, "freeSlotScore": 0.2, "evictableScore": 0.1, "saturatedScore": 0.0}`))
	plugin, err := LoraLoadStateScorerFactory("big-adapters", decoder, nil)
	require.NoError(t, err)
	scorer := plugin.(*LoraLoadStateScorer)
	assert.Equal(t, scoreTable{1.0, 0.95, 0.2, 0.1, 0.0, 0}, scorer.scores)
	assert.Equal(t, fwkplugin.TypedName{Type: LoraLoadStateScorerType, Name: "big-adapters"}, scorer.TypedName())
	assert.Equal(t, fwksched.Affinity, scorer.Category())

	deps := scorer.Consumes()
	assert.Empty(t, deps.Optional)
	for _, key := range []string{metrics.LoadedModelsKey, metrics.GPULoadedModelsKey, metrics.ActiveModelsKey} {
		assert.Contains(t, deps.Required, fwkplugin.NewDataKey(key, metrics.MetricsExtractorType))
	}

	_, err = LoraLoadStateScorerFactory("bad", json.NewDecoder(strings.NewReader(`{"freeSlotScore": "high"}`)), nil)
	assert.Error(t, err)
}

func TestBaseModelRequestPrefersEndpointsNotServingAdapters(t *testing.T) {
	resident := &fwkdl.Metrics{BaseModel: "base", MaxActiveModels: 2, GPULoadedModels: 2, LoadedModels: map[string]fwkdl.LoraLoadState{"target": gpu, "x": gpu}, ActiveModels: map[string]int{"target": 1, "x": 1}}
	empty := &fwkdl.Metrics{BaseModel: "base", MaxActiveModels: 2, GPULoadedModels: 0, LoadedModels: map[string]fwkdl.LoraLoadState{}}
	idle := &fwkdl.Metrics{BaseModel: "base", MaxActiveModels: 2, GPULoadedModels: 2, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu, "y": gpu}}
	half := &fwkdl.Metrics{BaseModel: "base", MaxActiveModels: 2, GPULoadedModels: 2, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu, "y": gpu}, ActiveModels: map[string]int{"x": 1}}
	full := &fwkdl.Metrics{BaseModel: "base", MaxActiveModels: 2, GPULoadedModels: 2, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu, "y": gpu}, ActiveModels: map[string]int{"x": 1, "y": 1}}
	got := score(t, nil, "base", endpoint("resident", resident), endpoint("empty", empty), endpoint("idle", idle), endpoint("half", half), endpoint("full", full))
	assert.Equal(t, 1.0, got["empty"])
	assert.Equal(t, 1.0, got["idle"], "resident but idle adapters cost the base model nothing")
	assert.Equal(t, 0.5, got["half"])
	assert.Equal(t, 0.0, got["full"])
	assert.Equal(t, got["full"], got["resident"], "which adapters are busy is irrelevant to a base-model request")

	got = score(t, nil, "target", endpoint("resident", resident), endpoint("empty", empty), endpoint("full", full))
	assert.Greater(t, got["resident"], got["empty"])
	assert.Greater(t, got["empty"], got["full"])
}

func TestBaseModelRequestIsNeutralWhenEveryEndpointIsFull(t *testing.T) {
	a := &fwkdl.Metrics{BaseModel: "base", MaxActiveModels: 2, GPULoadedModels: 2, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu, "y": gpu}, ActiveModels: map[string]int{"x": 1, "y": 1}}
	b := &fwkdl.Metrics{BaseModel: "base", MaxActiveModels: 2, GPULoadedModels: 2, LoadedModels: map[string]fwkdl.LoraLoadState{"p": gpu, "q": gpu, "r": cpu}, ActiveModels: map[string]int{"p": 1, "q": 1, "r": 1}}
	unreported := &fwkdl.Metrics{BaseModel: "base"}
	got := score(t, nil, "base", endpoint("a", a), endpoint("b", b))
	assert.Equal(t, got["a"], got["b"])
	got = score(t, nil, "base", endpoint("a", a), endpoint("unreported", unreported))
	assert.Greater(t, got["unreported"], got["a"], "an endpoint with no adapter slots is the best place for base traffic")
}

func TestNoEndpoints(t *testing.T) {
	assert.Empty(t, score(t, nil, "target"))
}

func TestLoadHorizonPricesMissesFromObservedTransitionTimes(t *testing.T) {
	horizon := 2.0
	params := &Parameters{LoadHorizonSeconds: &horizon}
	// Every endpoint has one free slot; they differ only in how long a load
	// takes there and in whether the adapter is already in the host cache.
	slowDisk := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 1, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu}, LoraLoadSeconds: 1.0, LoraActivateSeconds: 0.2}
	fastDisk := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 1, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu}, LoraLoadSeconds: 0.2, LoraActivateSeconds: 0.2}
	hostCached := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 1, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu, "target": cpu}, LoraLoadSeconds: 1.0, LoraActivateSeconds: 0.2}
	fresh := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 1, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu}}
	got := score(t, params, "target", endpoint("slow", slowDisk), endpoint("fast", fastDisk), endpoint("host", hostCached), endpoint("fresh", fresh))

	assert.InDelta(t, 1-1.2/horizon, got["slow"], 1e-9, "load + activate over the horizon")
	assert.InDelta(t, 1-0.4/horizon, got["fast"], 1e-9)
	assert.InDelta(t, 1-0.2/horizon, got["host"], 1e-9, "host-cached pays activation only")
	// The fresh pod borrows the fleet means: load (1.0+0.2+1.0)/3, activate 0.2.
	assert.InDelta(t, 1-(2.2/3+0.2)/horizon, got["fresh"], 1e-9)
	assert.Greater(t, got["host"], got["fast"])
	assert.Greater(t, got["fast"], got["fresh"])
	assert.Greater(t, got["fresh"], got["slow"])
}

func TestLoadHorizonFloorsAtSaturatedAndKeepsEvictableBelowFreeSlot(t *testing.T) {
	horizon := 1.0
	params := &Parameters{LoadHorizonSeconds: &horizon}
	glacial := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 1, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu}, LoraLoadSeconds: 5, LoraActivateSeconds: 1}
	freeSlot := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 1, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu}, LoraLoadSeconds: 0.2, LoraActivateSeconds: 0.2}
	evictable := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 2, LoadedModels: map[string]fwkdl.LoraLoadState{"x": gpu, "y": gpu}, LoraLoadSeconds: 0.2, LoraActivateSeconds: 0.2}
	got := score(t, params, "target", endpoint("glacial", glacial), endpoint("free", freeSlot), endpoint("evictable", evictable))
	assert.Equal(t, defaultScores.saturated, got["glacial"])
	assert.InDelta(t, 0.6, got["free"], 1e-9)
	assert.InDelta(t, 0.6*defaultScores.evictable/defaultScores.freeSlot, got["evictable"], 1e-9)
}

func TestLoadHorizonWithoutAnyObservationUsesFixedTiers(t *testing.T) {
	horizon := 2.0
	params := &Parameters{LoadHorizonSeconds: &horizon}
	free := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 0, LoadedModels: map[string]fwkdl.LoraLoadState{}}
	host := &fwkdl.Metrics{MaxActiveModels: 2, GPULoadedModels: 0, LoadedModels: map[string]fwkdl.LoraLoadState{"target": cpu}}
	got := score(t, params, "target", endpoint("free", free), endpoint("host", host))
	assert.Equal(t, defaultScores.freeSlot, got["free"])
	assert.Equal(t, defaultScores.cpuResident, got["host"])
}
