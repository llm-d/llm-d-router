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

package prefixcacheaffinity_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/p2psource"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity"
)

// fakeProducer stands in for a data producer the filter reads.
type fakeProducer struct {
	name     string
	produces map[plugin.DataKey]any
}

func (f *fakeProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "fake-producer", Name: f.name}
}
func (f *fakeProducer) Produces() map[plugin.DataKey]any { return f.produces }
func (f *fakeProducer) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

var _ fwkrh.DataProducer = &fakeProducer{}

func affinityFilter(t *testing.T, name, params string) plugin.Plugin {
	t.Helper()
	p, err := prefixcacheaffinity.Factory(name, json.NewDecoder(strings.NewReader(params)), nil)
	require.NoError(t, err)
	return p
}

// A paired filter reads the p2p-source-producer's PreRequest decision. The
// startup validation must accept that dependency under either TTFT source, with
// or without the producer, and order the producer first.
func TestStartupValidationWithP2PPairing(t *testing.T) {
	prefix := &fakeProducer{name: "prefix", produces: map[plugin.DataKey]any{
		attrprefix.PrefixCacheMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{}}}
	inFlight := &fakeProducer{name: "in-flight", produces: map[plugin.DataKey]any{
		attrconcurrency.InFlightLoadDataKey: attrconcurrency.InFlightLoad{}}}
	latency := &fakeProducer{name: "latency", produces: map[plugin.DataKey]any{
		attrlatency.LatencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfo{}}}
	p2p := p2psource.New("p2p", p2psource.Config{})

	const paired = `{"p2pSourceProducerName": "p2p", "maxTTFTPenaltyMs": 100}`
	tests := []struct {
		name    string
		plugins func(t *testing.T) []plugin.Plugin
		// explicit request-handling phase for the P2P producer
		requestHandlingP2P bool
	}{
		{
			name: "prefill throughput TTFT",
			plugins: func(t *testing.T) []plugin.Plugin {
				return []plugin.Plugin{prefix, inFlight, p2p, affinityFilter(t, "affinity", paired)}
			},
		},
		{
			name: "latency predictor TTFT",
			plugins: func(t *testing.T) []plugin.Plugin {
				return []plugin.Plugin{prefix, latency, p2p,
					affinityFilter(t, "affinity", `{"p2pSourceProducerName": "p2p", "maxTTFTPenaltyMs": 100, "ttftSource": "latencyPredictor"}`)}
			},
		},
		{
			name: "P2P producer in the request-handling phase",
			plugins: func(t *testing.T) []plugin.Plugin {
				return []plugin.Plugin{prefix, inFlight, p2p, affinityFilter(t, "affinity", paired)}
			},
			requestHandlingP2P: true,
		},
		{
			name: "paired without a p2p-source-producer",
			plugins: func(t *testing.T) []plugin.Plugin {
				return []plugin.Plugin{prefix, inFlight, affinityFilter(t, "affinity", paired)}
			},
		},
		{
			name: "pairing off",
			plugins: func(t *testing.T) []plugin.Plugin {
				return []plugin.Plugin{prefix, inFlight, p2p, affinityFilter(t, "affinity", `{"maxTTFTPenaltyMs": 100}`)}
			},
		},
		{
			name: "two paired instances",
			plugins: func(t *testing.T) []plugin.Plugin {
				return []plugin.Plugin{prefix, inFlight, p2p,
					affinityFilter(t, "affinity-prefill", paired), affinityFilter(t, "affinity-decode", paired)}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plugins := test.plugins(t)
			var explicit map[string]fwkrh.DataProducerPhase
			if test.requestHandlingP2P {
				explicit = map[string]fwkrh.DataProducerPhase{"p2p": fwkrh.DataProducerPhaseRequestHandling}
			}
			phases := datalayer.ResolveDataProducerPhases(plugins, explicit)
			order, err := datalayer.ValidateAndOrderDataDependencies(plugins, datalayer.WithDataProducerPhases(phases))
			require.NoError(t, err)

			p2pAt := slices.Index(order, p2p.TypedName().String())
			for _, p := range plugins {
				if p.TypedName().Type != prefixcacheaffinity.PluginType || p2pAt < 0 {
					continue
				}
				// Only a paired filter declares an output.
				if len(p.(plugin.ProducerPlugin).Produces()) == 0 {
					continue
				}
				require.Less(t, p2pAt, slices.Index(order, p.TypedName().String()),
					"the p2p-source-producer runs before the paired filter")
			}
		})
	}
}
