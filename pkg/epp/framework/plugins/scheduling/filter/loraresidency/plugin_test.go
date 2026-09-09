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

package loraresidency

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var gpu = fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelGPU}

type ep struct {
	name     string
	resident bool
	queue    int
	kv       float64
}

func fleet(eps ...ep) []fwksched.Endpoint {
	out := make([]fwksched.Endpoint, 0, len(eps))
	for _, e := range eps {
		loaded := map[string]fwkdl.LoraLoadState{}
		if e.resident {
			loaded["target"] = gpu
		}
		out = append(out, fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: e.name}},
			&fwkdl.Metrics{LoadedModels: loaded, WaitingQueueSize: e.queue, KVCacheUsagePercent: e.kv},
			nil))
	}
	return out
}

func names(eps []fwksched.Endpoint) []string {
	out := make([]string, 0, len(eps))
	for _, e := range eps {
		out = append(out, e.GetMetadata().ID.Name)
	}
	return out
}

func run(t *testing.T, cfg Config, eps []fwksched.Endpoint) []string {
	t.Helper()
	resetMetrics()
	t.Cleanup(resetMetrics)
	return names(New("test", cfg).Filter(context.Background(), &fwksched.InferenceRequest{TargetModel: "target"}, eps))
}

func outcomeCount(outcome string) float64 {
	return testutil.ToFloat64(filterDecisions.WithLabelValues("test", outcome))
}

func TestFilter(t *testing.T) {
	cfg := DefaultConfig
	tests := []struct {
		name    string
		cfg     Config
		eps     []ep
		want    []string
		outcome string
	}{
		{
			name:    "no home: everything passes so the scorer picks the first home",
			cfg:     cfg,
			eps:     []ep{{"a", false, 0, 0}, {"b", false, 0, 0}},
			want:    []string{"a", "b"},
			outcome: outcomeNoHome,
		},
		{
			name:    "a home with room: only homes pass",
			cfg:     cfg,
			eps:     []ep{{"home", true, 2, 0.5}, {"cold", false, 0, 0}, {"home2", true, 20, 0.9}},
			want:    []string{"home", "home2"},
			outcome: outcomeSticky,
		},
		{
			name:    "every home saturated by queue: one new copy on an endpoint with room",
			cfg:     cfg,
			eps:     []ep{{"home", true, 9, 0.1}, {"cold", false, 0, 0}, {"busy", false, 30, 0.1}},
			want:    []string{"cold"},
			outcome: outcomeSpread,
		},
		{
			name:    "every home saturated by kv cache: spread",
			cfg:     cfg,
			eps:     []ep{{"home", true, 0, 0.95}, {"cold", false, 0, 0.2}},
			want:    []string{"cold"},
			outcome: outcomeSpread,
		},
		{
			name:    "cap reached: stay on saturated homes",
			cfg:     Config{MaxReplicas: 2, QueueThreshold: 8, KVCacheThreshold: 0.8},
			eps:     []ep{{"h1", true, 9, 0}, {"h2", true, 9, 0}, {"cold", false, 0, 0}},
			want:    []string{"h1", "h2"},
			outcome: outcomeCapBlocked,
		},
		{
			name:    "cap not yet reached: spread allowed",
			cfg:     Config{MaxReplicas: 3, QueueThreshold: 8, KVCacheThreshold: 0.8},
			eps:     []ep{{"h1", true, 9, 0}, {"h2", true, 9, 0}, {"cold", false, 0, 0}},
			want:    []string{"cold"},
			outcome: outcomeSpread,
		},
		{
			name:    "whole fleet saturated: stay home rather than load under pressure",
			cfg:     cfg,
			eps:     []ep{{"home", true, 9, 0}, {"other", false, 9, 0}},
			want:    []string{"home"},
			outcome: outcomeFleetSaturated,
		},
		{
			name:    "saturation checks disabled: homes are always sticky",
			cfg:     Config{},
			eps:     []ep{{"home", true, 999, 1.0}, {"cold", false, 0, 0}},
			want:    []string{"home"},
			outcome: outcomeSticky,
		},
		{
			name:    "single candidate is passed through",
			cfg:     cfg,
			eps:     []ep{{"only", false, 0, 0}},
			want:    []string{"only"},
			outcome: outcomeNotApplicable,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := run(t, test.cfg, fleet(test.eps...))
			assert.Equal(t, test.want, got)
			assert.Equal(t, 1.0, outcomeCount(test.outcome))
		})
	}
}

func TestFilterWithoutTargetModelPassesThrough(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)
	eps := fleet(ep{"a", true, 0, 0}, ep{"b", false, 0, 0})
	got := New("test", DefaultConfig).Filter(context.Background(), &fwksched.InferenceRequest{}, eps)
	assert.Equal(t, []string{"a", "b"}, names(got))
	assert.Equal(t, 1.0, outcomeCount(outcomeNotApplicable))
}

func TestFactory(t *testing.T) {
	plugin, err := Factory("lora-homes", json.NewDecoder(strings.NewReader(`{"maxReplicas": 2, "queueThreshold": 4, "kvCacheThreshold": 0.9}`)), nil)
	require.NoError(t, err)
	p := plugin.(*Plugin)
	assert.Equal(t, Config{MaxReplicas: 2, QueueThreshold: 4, KVCacheThreshold: 0.9}, p.config)
	assert.Equal(t, PluginType, p.TypedName().Type)
	assert.Equal(t, "lora-homes", p.TypedName().Name)
	assert.Len(t, p.Consumes().Required, 3)

	plugin, err = Factory("defaults", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, DefaultConfig, plugin.(*Plugin).config)

	for _, bad := range []string{`{"maxReplicas": -1}`, `{"queueThreshold": -1}`, `{"kvCacheThreshold": 1.5}`, `{"maxReplicas": "two"}`} {
		_, err := Factory("bad", json.NewDecoder(strings.NewReader(bad)), nil)
		assert.Error(t, err, bad)
	}
}
