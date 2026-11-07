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

package prefixcacheaffinity

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// loadOverrideInput breaks stickiness: with 1000 tokens/s, endpoint "a" (sticky)
// estimates 500 ms and "b" 50 ms, over the 100 ms penalty.
func loadOverrideInput() (Config, []fwksched.Endpoint) {
	return Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 100, TTFTSource: TTFTSourcePrefillThroughput,
			PeakPrefillThroughput: 1000, P2PSourceProducerName: "p2p"},
		[]fwksched.Endpoint{makeEndpoint("a", 90, 10, 500), makeEndpoint("b", 10, 10, 50)}
}

func newTestPairedPlugin(config Config) *pairedPlugin {
	return &pairedPlugin{Plugin: newTestPlugin(config)}
}

func factoryPlugin(t *testing.T, name, params string) fwkplugin.Plugin {
	t.Helper()
	p, err := Factory(name, json.NewDecoder(strings.NewReader(params)), nil)
	require.NoError(t, err)
	return p
}

func publishP2PDecision(request *fwksched.InferenceRequest, outcome, role string) {
	request.PutAttribute(attrprefix.P2PSourceDecisionDataKey.WithNonEmptyProducerName("p2p"),
		attrprefix.P2PSourceDecision{Outcome: outcome, EndpointRole: role})
}

// PreRequest counts each filtered request once, paired with the P2P decision
// for the same request, and never relabels a missing or repeated decision as a
// real outcome.
func TestPreRequestPairsAffinityOutcomeWithP2PDecision(t *testing.T) {
	type series struct{ role, affinity, p2p string }
	tests := []struct {
		name string
		run  func(p *pairedPlugin, request *fwksched.InferenceRequest)
		want *series // nil: nothing recorded
	}{
		{
			name: "load override followed by a pull",
			run: func(p *pairedPlugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
				publishP2PDecision(request, "pulled", "prefill")
			},
			want: &series{"prefill", outcomeLoadOverride, "pulled"},
		},
		{
			// An aggregated request: the P2P decision carries endpoint_role decode.
			name: "load override that stayed on the cache holder",
			run: func(p *pairedPlugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
				publishP2PDecision(request, "self", "decode")
			},
			want: &series{"decode", outcomeLoadOverride, "self"},
		},
		{
			name: "no P2P decision published",
			run: func(p *pairedPlugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
			},
			want: &series{"", outcomeLoadOverride, p2pOutcomeUnreported},
		},
		{
			name: "request this instance did not filter",
			run: func(_ *pairedPlugin, request *fwksched.InferenceRequest) {
				publishP2PDecision(request, "pulled", "prefill")
			},
		},
		{
			name: "filtered twice for one request",
			run: func(p *pairedPlugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
				p.Filter(context.Background(), request, endpoints[:1])
				publishP2PDecision(request, "pulled", "prefill")
			},
			want: &series{"prefill", outcomeMultiple, "pulled"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)

			config, _ := loadOverrideInput()
			p := newTestPairedPlugin(config)
			request := &fwksched.InferenceRequest{RequestID: "req", Headers: map[string]string{}}
			test.run(p, request)
			require.NoError(t, p.PreRequest(context.Background(), request, nil))

			if test.want == nil {
				assert.Equal(t, 0, testutil.CollectAndCount(p2pDecisions))
			} else {
				require.Equal(t, 1, testutil.CollectAndCount(p2pDecisions), "one paired series")
				assert.Equal(t, float64(1), testutil.ToFloat64(
					p2pDecisions.WithLabelValues("test", test.want.role, test.want.affinity, test.want.p2p)))
			}
		})
	}
}

// Only a paired filter has a PreRequest hook and declares the pairing data; an
// unpaired filter is the plain Filter and stores nothing on the request.
func TestFactoryAddsPreRequestOnlyWhenPaired(t *testing.T) {
	paired := factoryPlugin(t, "affinity", `{"p2pSourceProducerName": "p2p"}`)
	require.Implements(t, (*fwkrh.PreRequest)(nil), paired)
	require.Implements(t, (*fwkplugin.ProducerPlugin)(nil), paired)
	assert.Contains(t, paired.(fwkplugin.ConsumerPlugin).Consumes().Optional,
		attrprefix.P2PSourceDecisionDataKey.WithNonEmptyProducerName("p2p"))

	unpaired := factoryPlugin(t, "affinity", `{}`)
	assert.NotImplements(t, (*fwkrh.PreRequest)(nil), unpaired)
	assert.NotImplements(t, (*fwkplugin.ProducerPlugin)(nil), unpaired)
	assert.Empty(t, unpaired.(fwkplugin.ConsumerPlugin).Consumes().Optional)

	_, endpoints := loadOverrideInput()
	request := &fwksched.InferenceRequest{RequestID: "req", Headers: map[string]string{}}
	unpaired.(fwksched.Filter).Filter(context.Background(), request, endpoints)
	_, stored := fwksched.ReadRequestAttribute[affinityOutcome](request, unpaired.(*Plugin).outcomeDataKey)
	assert.False(t, stored, "an unpaired filter stores no outcome on the request")
}

// Two paired instances filtering the same request, as a prefill and a decode
// profile do, each count their own outcome.
func TestPairedInstancesKeepTheirOwnOutcome(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	const params = `{"p2pSourceProducerName": "p2p", "maxTTFTPenaltyMs": 100, "peakPrefillThroughput": 1000}`
	prefill := factoryPlugin(t, "affinity-prefill", params)
	decode := factoryPlugin(t, "affinity-decode", params)
	request := &fwksched.InferenceRequest{RequestID: "req", Headers: map[string]string{}}

	_, overloaded := loadOverrideInput()
	prefill.(fwksched.Filter).Filter(context.Background(), request, overloaded)
	decode.(fwksched.Filter).Filter(context.Background(), request,
		[]fwksched.Endpoint{makeEndpoint("c", 90, 10, 50), makeEndpoint("d", 10, 10, 50)})
	publishP2PDecision(request, "pulled", "prefill")
	for _, p := range []fwkplugin.Plugin{prefill, decode} {
		require.NoError(t, p.(fwkrh.PreRequest).PreRequest(context.Background(), request, nil))
	}

	require.Equal(t, 2, testutil.CollectAndCount(p2pDecisions), "one series per instance")
	assert.Equal(t, float64(1), testutil.ToFloat64(
		p2pDecisions.WithLabelValues("affinity-prefill", "prefill", outcomeLoadOverride, "pulled")))
	assert.Equal(t, float64(1), testutil.ToFloat64(
		p2pDecisions.WithLabelValues("affinity-decode", "prefill", outcomeSticky, "pulled")))
}
