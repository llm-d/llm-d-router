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

package p2psource

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry), "re-registering the same collectors is tolerated")
	require.NoError(t, registerMetrics(nil), "a nil registerer is a no-op")
}

// Each PreRequest records exactly one decision outcome, the same one its
// decision log line names, under the endpoint_role the prefix-cache metrics use,
// and only a pull records predicted pulled tokens.
func TestPreRequestRecordsDecisionOutcome(t *testing.T) {
	tests := []struct {
		name          string
		config        Config
		request       func(p *Producer) *scheduling.InferenceRequest
		result        func(p *Producer) *scheduling.SchedulingResult
		outcome       string
		role          string
		pulledTokens  int // 0 when no pull is recorded
		wantPreReqErr bool
	}{
		{
			name:    "pulled",
			config:  Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest { return preRequest(p, "req-pull", peer("10.0.0.2", 48)) },
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1))
			},
			outcome: "pulled",
			role:    "decode",
			// 48 tokens on the source, one 16-token block on the computing pod.
			pulledTokens: 32,
		},
		{
			name:    "pulled for a disaggregated request",
			config:  Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest { return preRequest(p, "req-pd", peer("10.0.0.3", 48)) },
			result: func(p *Producer) *scheduling.SchedulingResult {
				return &scheduling.SchedulingResult{
					PrimaryProfileName: "decode",
					ProfileResults: map[string]*scheduling.ProfileRunResult{
						"decode":  {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-d", "10.0.0.4", 3)}},
						"prefill": {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-p", "10.0.0.2", 1)}},
					},
				}
			},
			outcome:      "pulled",
			role:         "prefill",
			pulledTokens: 32,
		},
		{
			// The role follows the prefix-cache metrics, which read only the
			// "prefill" profile, not prefillProfileName.
			name:    "pulled with a renamed computing profile",
			config:  Config{MinCachedTokenDelta: 1, PrefillProfileName: "default"},
			request: func(p *Producer) *scheduling.InferenceRequest { return preRequest(p, "req-agg", peer("10.0.0.2", 48)) },
			result: func(p *Producer) *scheduling.SchedulingResult {
				return &scheduling.SchedulingResult{
					PrimaryProfileName: "default",
					ProfileResults: map[string]*scheduling.ProfileRunResult{
						"default": {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-a", "10.0.0.1", 1)}},
					},
				}
			},
			outcome:      "pulled",
			role:         "decode",
			pulledTokens: 32,
		},
		{
			name:    "self",
			config:  Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest { return preRequest(p, "req-self", peer("10.0.0.1", 32)) },
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 2))
			},
			outcome: "self",
			role:    "decode",
		},
		{
			name:    "below floor",
			config:  Config{MinCachedTokenDelta: 17},
			request: func(p *Producer) *scheduling.InferenceRequest { return preRequest(p, "req-low", peer("10.0.0.2", 32)) },
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1))
			},
			outcome: "below-floor",
			role:    "decode",
		},
		{
			name:    "no best match",
			config:  Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest { return preRequest(p, "req-none", nil) },
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 2))
			},
			outcome: "no-best-match",
			role:    "decode",
		},
		{
			name:   "no endpoint",
			config: Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest {
				return preRequest(p, "req-empty", peer("10.0.0.2", 48))
			},
			result: func(*Producer) *scheduling.SchedulingResult {
				return &scheduling.SchedulingResult{
					PrimaryProfileName: "decode",
					ProfileResults:     map[string]*scheduling.ProfileRunResult{"decode": {}},
				}
			},
			outcome: "no-endpoint",
			role:    "",
		},
		{
			name:   "decode request",
			config: Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest {
				req := preRequest(p, "req-decode", peer("10.0.0.2", 48))
				req.Body = bodyWith(map[string]any{
					reqcommon.FieldKVTransferParams: map[string]any{
						reqcommon.FieldRemotePrefiller: map[string]any{reqcommon.FieldKVRequestID: "K"},
					},
				})
				return req
			},
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1))
			},
			outcome: "decode-request",
			role:    "decode",
		},
		{
			name:   "unparsed body",
			config: Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest {
				req := preRequest(p, "req-raw", peer("10.0.0.2", 48))
				req.Body = &fwkrequest.InferenceRequestBody{Payload: fwkrequest.RawPayload([]byte(`{}`))}
				return req
			},
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1))
			},
			outcome: "unparsed-body",
			role:    "decode",
		},
		{
			name:    "declined by the cost model",
			config:  Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()},
			request: func(p *Producer) *scheduling.InferenceRequest { return costRequest(p, "req-busy", 8192, 0) },
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 1))
			},
			outcome: "declined",
			role:    "decode",
		},
		{
			name:   "prefill port out of range",
			config: Config{MinCachedTokenDelta: 1, P2PConnectorPort: 65530},
			request: func(*Producer) *scheduling.InferenceRequest {
				return &scheduling.InferenceRequest{RequestID: "req-port", Headers: map[string]string{}, Body: bodyWith(map[string]any{})}
			},
			result: func(*Producer) *scheduling.SchedulingResult {
				return prefillOnly(scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
					ID: k8stypes.NamespacedName{Name: "pod-p"}, Address: "10.0.0.2", Port: "8080", RankIndex: 7,
				}, nil, nil))
			},
			outcome:       "prefill-port-invalid",
			role:          "prefill",
			wantPreReqErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)

			p := New("test", test.config)
			err := p.PreRequest(utils.NewTestContext(t), test.request(p), test.result(p))
			if test.wantPreReqErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			require.Equal(t, 1, testutil.CollectAndCount(sourceDecisions), "one decision series")
			assert.Equal(t, float64(1), testutil.ToFloat64(sourceDecisions.WithLabelValues("test", test.role, test.outcome)))

			observed := pulledTokensObserved(t)
			if test.pulledTokens == 0 {
				assert.Empty(t, observed, "predicted pulled tokens observations")
				return
			}
			assert.Equal(t, map[string][2]float64{test.role: {1, float64(test.pulledTokens)}}, observed,
				"predicted pulled tokens count and sum by endpoint_role")
		})
	}
}

// pulledTokensObserved returns the observation count and sum of the predicted
// pulled tokens histogram, by endpoint_role.
func pulledTokensObserved(t *testing.T) map[string][2]float64 {
	t.Helper()
	registry := prometheus.NewRegistry()
	require.NoError(t, registry.Register(predictedPulledTokens))
	families, err := registry.Gather()
	require.NoError(t, err)
	observed := map[string][2]float64{}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			h := metric.GetHistogram()
			observed[labels["endpoint_role"]] = [2]float64{float64(h.GetSampleCount()), h.GetSampleSum()}
		}
	}
	return observed
}
