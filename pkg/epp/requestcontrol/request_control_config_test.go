/*
Copyright 2026 llm-d Authors.

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

package requestcontrol

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// orderTestData is the payload type exchanged between the producer and consumer
// mocks below. buildDAG compares produced and consumed types, so both sides must
// declare the same one.
type orderTestData struct{}

func (d *orderTestData) Clone() fwkdl.Cloneable { return &orderTestData{} }

// hookPlugin implements every requestcontrol extension point plus Produces and
// Consumes, so a single instance lands on all seven Config hook lists and is ranked
// by the data-dependency DAG. Leaving produces or consumes nil makes the instance
// a pure consumer or a pure producer respectively.
type hookPlugin struct {
	name     string
	produces map[fwkplugin.DataKey]any
	consumes map[fwkplugin.DataKey]any
}

func (p *hookPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: p.name, Type: "mock"}
}
func (p *hookPlugin) Produces() map[fwkplugin.DataKey]any { return p.produces }
func (p *hookPlugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: p.consumes}
}
func (p *hookPlugin) Produce(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	return nil
}
func (p *hookPlugin) PreRequest(_ context.Context, _ *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	return nil
}
func (p *hookPlugin) RequestHeader(_ context.Context, _ *fwksched.InferenceRequest) error { return nil }
func (p *hookPlugin) Screen(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	return endpoints
}
func (p *hookPlugin) ResponseHeader(_ context.Context, _ *fwksched.InferenceRequest, _ *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
}
func (p *hookPlugin) ResponseBody(_ context.Context, _ *fwksched.InferenceRequest, _ *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
}
func (p *hookPlugin) Admit(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	return nil
}

// unrankedPlugin implements a hook but is neither a producer nor a consumer, so
// the data-dependency DAG never ranks it. Ordering must not drop it.
type unrankedPlugin struct {
	name string
}

func (p *unrankedPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: p.name, Type: "mock"}
}
func (p *unrankedPlugin) PreRequest(_ context.Context, _ *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	return nil
}

// Compile-time proof that the mocks still satisfy every extension point. Without
// these, an interface change upstream would silently drop a mock from a hook list
// and leave the matrix below asserting on empty slices.
var (
	_ fwkrc.RequestHeaderProcessor  = &hookPlugin{}
	_ fwkrc.Screener                = &hookPlugin{}
	_ fwkrc.Admitter                = &hookPlugin{}
	_ fwkrc.DataProducer            = &hookPlugin{}
	_ fwkrc.PreRequest              = &hookPlugin{}
	_ fwkrc.ResponseHeaderProcessor = &hookPlugin{}
	_ fwkrc.ResponseBodyProcessor   = &hookPlugin{}
	_ fwkrc.PreRequest              = &unrankedPlugin{}
)

func names[T interface{ TypedName() fwkplugin.TypedName }](plugins []T) []string {
	result := make([]string, 0, len(plugins))
	for _, p := range plugins {
		result = append(result, p.TypedName().String())
	}
	return result
}

// TestConfig_OrderPlugins_AllHookLists drives the ordering path the runner uses
// (parseConfigurationPhaseTwo: AddPlugins over the handle's plugins, then apply the
// data-dependency order) and asserts every extension point honours the DAG, not
// just Produce.
func TestConfig_OrderPlugins_AllHookLists(t *testing.T) {
	key := fwkplugin.NewDataKey("orderTestData", "mock")
	producer := &hookPlugin{name: "P", produces: map[fwkplugin.DataKey]any{key: &orderTestData{}}}
	consumer := &hookPlugin{name: "C", consumes: map[fwkplugin.DataKey]any{key: &orderTestData{}}}
	unranked := &unrankedPlugin{name: "U"}

	cfg := NewConfig()
	// Deliberately consumer-first: this is the order GetAllPlugins may hand over,
	// since it ranges a map.
	cfg.AddPlugins(consumer, unranked, producer)

	sorted, err := datalayer.ValidateAndOrderDataDependencies([]fwkplugin.Plugin{consumer, unranked, producer})
	require.NoError(t, err)
	require.Equal(t, []string{"P/mock", "C/mock"}, sorted,
		"DAG must rank the producer before the consumer and must not rank the unranked plugin")

	cfg.OrderPlugins(sorted)

	testCases := []struct {
		name string
		got  []string
		want []string
	}{
		{
			// The regression from #1856: a consumer's PreRequest ran before the
			// producer's, depending on map order.
			name: "preRequest orders producer before consumer and keeps unranked plugins",
			got:  names(cfg.preRequestPlugins),
			want: []string{"P/mock", "C/mock", "U/mock"},
		},
		{
			name: "requestHeader orders producer before consumer",
			got:  names(cfg.requestHeaderPlugins),
			want: []string{"P/mock", "C/mock"},
		},
		{
			// Added by #2259 after this change was first written; the extension
			// point is covered by the same rule as the others.
			name: "screeners order producer before consumer",
			got:  names(cfg.screeners),
			want: []string{"P/mock", "C/mock"},
		},
		{
			name: "responseReceived orders producer before consumer",
			got:  names(cfg.responseReceivedPlugins),
			want: []string{"P/mock", "C/mock"},
		},
		{
			name: "responseStreaming orders producer before consumer",
			got:  names(cfg.responseStreamingPlugins),
			want: []string{"P/mock", "C/mock"},
		},
		{
			name: "admission orders producer before consumer",
			got:  names(cfg.admissionPlugins),
			want: []string{"P/mock", "C/mock"},
		},
		{
			// Already correct before this change; guards against regressing it.
			name: "dataProducer orders producer before consumer",
			got:  names(cfg.dataProducerPlugins),
			want: []string{"P/mock", "C/mock"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.got)
		})
	}
}

// TestConfig_OrderPlugins_EdgeCases covers the boundaries the DAG path can hand to
// OrderPlugins: nothing configured, nothing ranked, and names for plugins that are
// not on a given list.
func TestConfig_OrderPlugins_EdgeCases(t *testing.T) {
	key := fwkplugin.NewDataKey("orderTestData", "mock")

	testCases := []struct {
		name        string
		plugins     []fwkplugin.Plugin
		sortedNames []string
		wantPreReq  []string
	}{
		{
			name:        "empty config with empty order",
			plugins:     nil,
			sortedNames: nil,
			wantPreReq:  []string{},
		},
		{
			name:        "no plugin is ranked: unranked plugins are kept, sorted by name",
			plugins:     []fwkplugin.Plugin{&unrankedPlugin{name: "U2"}, &unrankedPlugin{name: "U1"}},
			sortedNames: nil,
			wantPreReq:  []string{"U1/mock", "U2/mock"},
		},
		{
			name:        "order contains names absent from the config",
			plugins:     []fwkplugin.Plugin{&unrankedPlugin{name: "U1"}},
			sortedNames: []string{"ghost/mock", "U1/mock"},
			wantPreReq:  []string{"U1/mock"},
		},
		{
			name: "ranked and unranked mixed: ranked first, unranked appended by name",
			plugins: []fwkplugin.Plugin{
				&unrankedPlugin{name: "U2"},
				&hookPlugin{name: "C", consumes: map[fwkplugin.DataKey]any{key: &orderTestData{}}},
				&unrankedPlugin{name: "U1"},
				&hookPlugin{name: "P", produces: map[fwkplugin.DataKey]any{key: &orderTestData{}}},
			},
			sortedNames: []string{"P/mock", "C/mock"},
			wantPreReq:  []string{"P/mock", "C/mock", "U1/mock", "U2/mock"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := NewConfig()
			cfg.AddPlugins(tc.plugins...)
			cfg.OrderPlugins(tc.sortedNames)
			assert.Equal(t, tc.wantPreReq, names(cfg.preRequestPlugins))
		})
	}
}

// TestConfig_OrderPlugins_InputOrderIndependent is the core #1856/#2040 invariant:
// AddPlugins receives plugins in Go's randomized map-iteration order (from
// GetAllPlugins), so the ordered hook lists must be a pure function of the DAG and
// the plugin names, not of the order plugins were handed over. Each fixed permutation
// below stands in for one possible map-iteration order and must produce the same
// result.
func TestConfig_OrderPlugins_InputOrderIndependent(t *testing.T) {
	key := fwkplugin.NewDataKey("orderTestData", "mock")
	// Base set indexed for permutation: 0=P (producer), 1=C (consumer), 2=U1, 3=U2.
	newSet := func() []fwkplugin.Plugin {
		return []fwkplugin.Plugin{
			&hookPlugin{name: "P", produces: map[fwkplugin.DataKey]any{key: &orderTestData{}}},
			&hookPlugin{name: "C", consumes: map[fwkplugin.DataKey]any{key: &orderTestData{}}},
			&unrankedPlugin{name: "U1"},
			&unrankedPlugin{name: "U2"},
		}
	}
	// Producer before consumer (DAG); unranked appended by name. Identical for every
	// input order.
	wantPreReq := []string{"P/mock", "C/mock", "U1/mock", "U2/mock"}
	wantRanked := []string{"P/mock", "C/mock"} // lists the unranked plugins do not join

	permutations := [][]int{
		{0, 1, 2, 3},
		{3, 2, 1, 0},
		{1, 3, 0, 2},
		{2, 0, 3, 1},
	}
	for _, perm := range permutations {
		t.Run(fmt.Sprintf("input order %v", perm), func(t *testing.T) {
			base := newSet()
			shuffled := make([]fwkplugin.Plugin, len(perm))
			for i, idx := range perm {
				shuffled[i] = base[idx]
			}

			cfg := NewConfig()
			cfg.AddPlugins(shuffled...)
			sorted, err := datalayer.ValidateAndOrderDataDependencies(shuffled)
			require.NoError(t, err)
			cfg.OrderPlugins(sorted)

			assert.Equal(t, wantPreReq, names(cfg.preRequestPlugins), "preRequest")
			assert.Equal(t, wantRanked, names(cfg.requestHeaderPlugins), "requestHeader")
			assert.Equal(t, wantRanked, names(cfg.screeners), "screeners")
			assert.Equal(t, wantRanked, names(cfg.admissionPlugins), "admission")
			assert.Equal(t, wantRanked, names(cfg.responseReceivedPlugins), "responseReceived")
			assert.Equal(t, wantRanked, names(cfg.responseStreamingPlugins), "responseStreaming")
			assert.Equal(t, wantRanked, names(cfg.dataProducerPlugins), "dataProducer")
		})
	}
}
