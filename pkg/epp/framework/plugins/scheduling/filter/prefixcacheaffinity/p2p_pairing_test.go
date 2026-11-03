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
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
		run  func(p *Plugin, request *fwksched.InferenceRequest)
		pair bool
		want *series // nil: nothing recorded
	}{
		{
			name: "load override followed by a pull",
			run: func(p *Plugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
				publishP2PDecision(request, "pulled", "prefill")
			},
			pair: true,
			want: &series{"prefill", outcomeLoadOverride, "pulled"},
		},
		{
			// An aggregated request: the P2P decision carries endpoint_role decode.
			name: "load override that stayed on the cache holder",
			run: func(p *Plugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
				publishP2PDecision(request, "self", "decode")
			},
			pair: true,
			want: &series{"decode", outcomeLoadOverride, "self"},
		},
		{
			name: "no P2P decision published",
			run: func(p *Plugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
			},
			pair: true,
			want: &series{"", outcomeLoadOverride, p2pOutcomeUnreported},
		},
		{
			name: "request this instance did not filter",
			run: func(_ *Plugin, request *fwksched.InferenceRequest) {
				publishP2PDecision(request, "pulled", "prefill")
			},
			pair: true,
		},
		{
			name: "filtered twice for one request",
			run: func(p *Plugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
				p.Filter(context.Background(), request, endpoints[:1])
				publishP2PDecision(request, "pulled", "prefill")
			},
			pair: true,
			want: &series{"prefill", outcomeMultiple, "pulled"},
		},
		{
			name: "pairing off",
			run: func(p *Plugin, request *fwksched.InferenceRequest) {
				_, endpoints := loadOverrideInput()
				p.Filter(context.Background(), request, endpoints)
				publishP2PDecision(request, "pulled", "prefill")
			},
			pair: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)

			config, _ := loadOverrideInput()
			if !test.pair {
				config.P2PSourceProducerName = ""
			}
			p := newTestPlugin(config)
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
			if !test.pair {
				_, stored := fwksched.ReadRequestAttribute[affinityOutcome](request, p.outcomeDataKey)
				assert.False(t, stored, "an unpaired filter stores no outcome on the request")
			}
		})
	}
}

func TestDeclarationsFollowPairing(t *testing.T) {
	config, _ := loadOverrideInput()
	paired := newTestPlugin(config)
	assert.Contains(t, paired.Produces(), paired.outcomeDataKey)
	assert.Contains(t, paired.Consumes().Optional, attrprefix.P2PSourceDecisionDataKey.WithNonEmptyProducerName("p2p"))

	config.P2PSourceProducerName = ""
	unpaired := newTestPlugin(config)
	assert.Empty(t, unpaired.Produces())
	assert.Empty(t, unpaired.Consumes().Optional)
}
