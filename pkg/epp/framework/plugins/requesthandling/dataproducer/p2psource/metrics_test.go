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
	"context"
	"encoding/json"
	"maps"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
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
// decision log line names, under the endpoint_role the prefix-cache metrics use.
// Only a pull records predicted pulled tokens, and only a declined or
// below-floor decision with a positive gap records unpulled tokens.
func TestPreRequestRecordsDecisionOutcome(t *testing.T) {
	tests := []struct {
		name           string
		config         Config
		request        func(p *Producer) *scheduling.InferenceRequest
		result         func(p *Producer) *scheduling.SchedulingResult
		outcome        string
		role           string
		pulledTokens   int // 0 when no pull is recorded
		unpulledTokens int // 0 when no unpulled gap is recorded
		wantPreReqErr  bool
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
					PrefillProfileName: "prefill",
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
			// The pull is for the prefill-h200 endpoint: against the decode
			// endpoint's 48 cached tokens it would be below the floor. The role
			// follows the prefill profile the scheduling result names.
			name:   "pulled for a disaggregated request with a renamed prefill profile",
			config: Config{MinCachedTokenDelta: 1, PrefillProfileName: "prefill-h200"},
			request: func(p *Producer) *scheduling.InferenceRequest {
				return preRequest(p, "req-pd-h200", peer("10.0.0.3", 48))
			},
			result: func(p *Producer) *scheduling.SchedulingResult {
				return &scheduling.SchedulingResult{
					PrimaryProfileName: "decode",
					PrefillProfileName: "prefill-h200",
					ProfileResults: map[string]*scheduling.ProfileRunResult{
						"decode":       {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-d", "10.0.0.4", 3)}},
						"prefill-h200": {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-p", "10.0.0.2", 1)}},
					},
				}
			},
			outcome:      "pulled",
			role:         "prefill",
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
			// 32 tokens on the source, one 16-token block on the computing pod.
			unpulledTokens: 16,
		},
		{
			// The computing pod holds more than the source: nothing is left
			// behind.
			name:   "below floor without a cached advantage",
			config: Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest {
				return preRequest(p, "req-ahead", peer("10.0.0.2", 16))
			},
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 2))
			},
			outcome: "below-floor",
			role:    "decode",
		},
		{
			// The source and the computing pod hold the same prefix.
			name:   "below floor with an equal cache",
			config: Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest {
				return preRequest(p, "req-equal", peer("10.0.0.2", 32))
			},
			result: func(p *Producer) *scheduling.SchedulingResult {
				return decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 2))
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
			name:   "no endpoint metadata",
			config: Config{MinCachedTokenDelta: 1},
			request: func(p *Producer) *scheduling.InferenceRequest {
				return preRequest(p, "req-nometa", peer("10.0.0.2", 48))
			},
			result: func(*Producer) *scheduling.SchedulingResult {
				return decodeOnly(scheduling.NewEndpointFromSnapshot(nil, nil, nil))
			},
			outcome: "no-endpoint",
			role:    "decode",
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
			outcome:        "declined",
			role:           "decode",
			unpulledTokens: 8192,
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
			request := test.request(p)
			err := p.PreRequest(utils.NewTestContext(t), request, test.result(p))
			if test.wantPreReqErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			require.Equal(t, 1, testutil.CollectAndCount(sourceDecisions), "one decision series")
			assert.Equal(t, float64(1), testutil.ToFloat64(sourceDecisions.WithLabelValues("test", test.role, test.outcome)))
			published, ok := scheduling.ReadRequestAttribute[attrprefix.P2PSourceDecision](request, p.decisionDataKey)
			require.True(t, ok, "the decision is published on the request")
			assert.Equal(t, attrprefix.P2PSourceDecision{Outcome: test.outcome, EndpointRole: test.role}, published)

			unpulled := histogramObserved(t, unpulledTokens, "outcome")
			if test.unpulledTokens == 0 {
				assert.Empty(t, unpulled, "unpulled tokens observations")
			} else {
				assert.Equal(t, map[string][2]float64{test.outcome: {1, float64(test.unpulledTokens)}}, unpulled,
					"unpulled tokens count and sum by outcome")
			}

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

func TestCostMarginBucket(t *testing.T) {
	for _, test := range []struct {
		marginMs float64
		want     string
	}{
		{-1500, "-inf..-1s"},
		{-1000, "-1s..-250ms"},
		{-250, "-250ms..0"},
		{0, "-250ms..0"}, // gain == cost declines
		{0.1, "0..250ms"},
		{250, "250ms..1s"},
		{1000, "1s..+inf"},
	} {
		assert.Equal(t, test.want, costMarginBucket(test.marginMs), "margin %v ms", test.marginMs)
	}
}

// A pull whose gain equals its cost is declined: the producer leaves the body
// and headers untouched, records the gap as unpulled, and labels the margin
// -250ms..0.
func TestCostModelTieDeclines(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	// gain 4000*(1.5-1)/1000 = 2 ms, cost 2 ms; both exact in float64.
	cfg := `{"ttftByDecision": true, "costModel": {"prefillMicrosecondsPerToken": 1.5, "transferMicrosecondsPerToken": 1,
		"transferFixedMs": 2}}`
	recorder := newTTFTRecorder(t, cfg)
	request := costRequest(recorder.Producer, "req-tie", 4000, 0)
	require.NoError(t, recorder.PreRequest(utils.NewTestContext(t), request,
		decodeOnly(queuedEndpoint(recorder.Producer, "pod-a", "10.0.0.1", 0, 0))))

	assert.Equal(t, float64(1), testutil.ToFloat64(sourceDecisions.WithLabelValues("p2p", "decode", "declined")))
	assertNoSource(t, request)
	assert.Equal(t, map[string][2]float64{"declined": {1, 4000}}, histogramObserved(t, unpulledTokens, "outcome"))

	received := time.Unix(1000, 0)
	recorder.ResponseBody(utils.NewTestContext(t), request, &fwkrh.Response{
		EndOfStream: true, Streaming: true, TerminationCause: fwkrh.TerminationCauseNatural,
		ReceivedTimestamp: received, FirstTokenTimestamp: received.Add(time.Second),
	}, nil)
	assert.Equal(t, map[string][2]float64{"declined,-250ms..0": {1, 1}},
		histogramObserved(t, requestTTFT, "outcome", "cost_margin"))
}

// Only ttftByDecision adds the response-body hook.
func TestFactoryAddsResponseBodyOnlyWithTTFTByDecision(t *testing.T) {
	plain, err := PluginFactory("p2p", json.NewDecoder(strings.NewReader(`{}`)), nil)
	require.NoError(t, err)
	_, hooks := plain.(fwkrh.ResponseBodyProcessor)
	assert.False(t, hooks, "a producer without ttftByDecision registers no response-body hook")

	var recorder plugin.Plugin = newTTFTRecorder(t, `{"ttftByDecision": true}`)
	_, hooks = recorder.(fwkrh.ResponseBodyProcessor)
	assert.True(t, hooks, "ttftByDecision registers the response-body hook")
	_, preRequest := recorder.(fwkrh.PreRequest)
	assert.True(t, preRequest, "the wrapped producer keeps its PreRequest")
}

// With ttftByDecision, the end of a streaming response records its TTFT under
// the decision and cost margin its PreRequest made, once, and releases the
// stored decision.
func TestTTFTByDecision(t *testing.T) {
	received := time.Unix(1000, 0)
	streamEnd := func(streaming bool) *fwkrh.Response {
		return &fwkrh.Response{
			EndOfStream: true, Streaming: streaming, TerminationCause: fwkrh.TerminationCauseNatural,
			ReceivedTimestamp: received, FirstTokenTimestamp: received.Add(1500 * time.Millisecond),
		}
	}
	disconnected := streamEnd(true)
	disconnected.TerminationCause = fwkrh.TerminationCauseClientDisconnect
	errored := streamEnd(true)
	errored.TerminationCause = fwkrh.TerminationCauseError
	evicted := streamEnd(true)
	evicted.TerminationCause = fwkrh.TerminationCauseEvicted
	firstByteAtReceipt := streamEnd(true)
	firstByteAtReceipt.FirstTokenTimestamp = received
	tests := []struct {
		name     string
		waiting  int // computing pod's waiting queue: 1 adds requeueMs and declines
		chunks   []*fwkrh.Response
		labels   []string // outcome, cost_margin; nil when nothing is recorded
		skipPreq bool
		rawBody  bool
	}{
		{
			// gain 8192*(19-3.9)/1000 = 123.7 ms, cost 15 + 500 ms requeue.
			name:    "declined",
			waiting: 1,
			chunks:  []*fwkrh.Response{{StartOfStream: true}, streamEnd(true)},
			labels:  []string{"declined", "-1s..-250ms"},
		},
		{
			// gain 123.7 ms, cost 15 ms.
			name:   "pulled",
			chunks: []*fwkrh.Response{{StartOfStream: true}, streamEnd(true)},
			labels: []string{"pulled", "0..250ms"},
		},
		{
			// The margin is set before the body rejects the source.
			name:    "unparsed body",
			rawBody: true,
			chunks:  []*fwkrh.Response{{StartOfStream: true}, streamEnd(true)},
			labels:  []string{"unparsed-body", "0..250ms"},
		},
		{
			name:   "non-streaming response",
			chunks: []*fwkrh.Response{streamEnd(false)},
		},
		{
			// The EPP's request TTFT skips a stream it tears down.
			name:   "disconnect after the first byte",
			chunks: []*fwkrh.Response{{StartOfStream: true}, disconnected},
		},
		{
			name:   "stream error",
			chunks: []*fwkrh.Response{{StartOfStream: true}, errored},
		},
		{
			name:   "stream evicted",
			chunks: []*fwkrh.Response{{StartOfStream: true}, evicted},
		},
		{
			name:   "first byte not after receipt",
			chunks: []*fwkrh.Response{{StartOfStream: true}, firstByteAtReceipt},
		},
		{
			name:   "stream still open",
			chunks: []*fwkrh.Response{{StartOfStream: true}},
		},
		{
			name:     "no stored decision",
			chunks:   []*fwkrh.Response{streamEnd(true)},
			skipPreq: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)

			cfg := `{"ttftByDecision": true, "costModel": {"prefillMicrosecondsPerToken": 19, "transferMicrosecondsPerToken": 3.9,
				"transferFixedMs": 15, "requeueMs": 500}}`
			recorder := newTTFTRecorder(t, cfg)
			request := costRequest(recorder.Producer, "req-1", 8192, 0)
			if test.rawBody {
				request.Body = &fwkrequest.InferenceRequestBody{Payload: fwkrequest.RawPayload([]byte(`{}`))}
			}
			if !test.skipPreq {
				require.NoError(t, recorder.PreRequest(utils.NewTestContext(t), request,
					decodeOnly(queuedEndpoint(recorder.Producer, "pod-a", "10.0.0.1", 0, test.waiting))))
			}
			for _, chunk := range test.chunks {
				recorder.ResponseBody(utils.NewTestContext(t), request, chunk, nil)
			}

			observed := histogramObserved(t, requestTTFT, "outcome", "cost_margin")
			if test.labels == nil {
				assert.Empty(t, observed, "TTFT observations")
			} else {
				assert.Equal(t, map[string][2]float64{strings.Join(test.labels, ","): {1, 1.5}}, observed,
					"TTFT count and sum by outcome and cost_margin")
			}

			_, err := plugin.ReadPluginStateKey[*decisionState](recorder.pluginState, "req-1", decisionStateKey)
			if test.name == "stream still open" {
				assert.NoError(t, err, "the decision stays until the stream ends")
			} else {
				assert.Error(t, err, "the decision is released at the end of the stream")
			}
		})
	}
}

// A stream that outlasts the staleness threshold keeps its decision while its
// stream ctx is alive, so its end still records TTFT. A decision whose stream
// ctx ends without an end-of-stream call is reclaimed by the janitor.
func TestTTFTByDecisionSurvivesStateExpiry(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recorder := newTTFTRecorder(t, `{"ttftByDecision": true}`)
	janitorCtx, stopJanitor := context.WithCancel(context.Background())
	t.Cleanup(stopJanitor)
	recorder.pluginState = plugin.NewPluginState(janitorCtx,
		plugin.WithStalenessThreshold(50*time.Millisecond), plugin.WithCleanupInterval(10*time.Millisecond))
	preRequestOn := func(ctx context.Context, id string) *scheduling.InferenceRequest {
		request := preRequest(recorder.Producer, id, peer("10.0.0.2", 48))
		require.NoError(t, recorder.PreRequest(ctx, request, decodeOnly(endpoint(recorder.Producer, "pod-a", "10.0.0.1", 1))))
		return request
	}

	streamCtx, endStream := context.WithCancel(utils.NewTestContext(t))
	t.Cleanup(endStream)
	long := preRequestOn(streamCtx, "req-long")
	recorder.ResponseBody(streamCtx, long, &fwkrh.Response{StartOfStream: true}, nil)
	// Several janitor sweeps past the threshold, no chunks.
	time.Sleep(150 * time.Millisecond)
	received := time.Unix(1000, 0)
	recorder.ResponseBody(streamCtx, long, &fwkrh.Response{
		EndOfStream: true, Streaming: true, TerminationCause: fwkrh.TerminationCauseNatural,
		ReceivedTimestamp: received, FirstTokenTimestamp: received.Add(time.Second),
	}, nil)
	assert.Equal(t, map[string][2]float64{"pulled,none": {1, 1}}, histogramObserved(t, requestTTFT, "outcome", "cost_margin"),
		"the long stream records its TTFT")

	leakedCtx, endLeaked := context.WithCancel(utils.NewTestContext(t))
	preRequestOn(leakedCtx, "req-leaked")
	endLeaked()
	// LastAccessTime does not refresh the entry, so polling cannot keep it alive.
	require.Eventually(t, func() bool {
		_, ok := recorder.pluginState.LastAccessTime("req-leaked")
		return !ok
	}, time.Second, 10*time.Millisecond, "the janitor reclaims the decision of an ended stream")
}

// newTTFTRecorder builds a ttftByDecision producer through PluginFactory on a
// handle whose context ends with the test, so the plugin-state janitor stops.
func newTTFTRecorder(t *testing.T, cfg string) *ttftRecorder {
	t.Helper()
	built, err := PluginFactory("p2p", json.NewDecoder(strings.NewReader(cfg)), plugin.NewEppHandle(t.Context(), nil))
	require.NoError(t, err)
	return built.(*ttftRecorder)
}

// histogramObserved returns the observation count and sum of a histogram
// vector, keyed by the comma-joined values of the given labels.
func histogramObserved(t *testing.T, histogram *prometheus.HistogramVec, keyLabels ...string) map[string][2]float64 {
	t.Helper()
	registry := prometheus.NewRegistry()
	require.NoError(t, registry.Register(histogram))
	families, err := registry.Gather()
	require.NoError(t, err)
	observed := map[string][2]float64{}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := labelMap(metric)
			key := make([]string, 0, len(keyLabels))
			for _, name := range keyLabels {
				key = append(key, labels[name])
			}
			h := metric.GetHistogram()
			observed[strings.Join(key, ",")] = [2]float64{float64(h.GetSampleCount()), h.GetSampleSum()}
		}
	}
	return observed
}

// PluginFactory registers every collector on the handle's registry, so a
// deployment exports them.
func TestPluginFactoryExportsMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	handle := plugin.NewEppHandle(t.Context(),
		func() []k8stypes.NamespacedName { return nil }, plugin.WithMetricsRecorder(registry))
	cfg := `{"ttftByDecision": true, "minCachedTokenDelta": 32}`
	built, err := PluginFactory("p2p", json.NewDecoder(strings.NewReader(cfg)), handle)
	require.NoError(t, err)
	recorder := built.(*ttftRecorder)
	p := recorder.Producer
	// Against one 16-token block on the computing pod, 64 source tokens clear
	// the floor and 32 leave a 16-token gap below it.
	pull := preRequest(p, "req-pull", peer("10.0.0.2", 64))
	require.NoError(t, recorder.PreRequest(utils.NewTestContext(t), pull, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1))))
	require.NoError(t, recorder.PreRequest(utils.NewTestContext(t), preRequest(p, "req-low", peer("10.0.0.2", 32)),
		decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1))))
	received := time.Unix(1000, 0)
	recorder.ResponseBody(utils.NewTestContext(t), pull, &fwkrh.Response{
		EndOfStream: true, Streaming: true, TerminationCause: fwkrh.TerminationCauseNatural,
		ReceivedTimestamp: received, FirstTokenTimestamp: received.Add(time.Second),
	}, nil)

	families, err := registry.Gather()
	require.NoError(t, err)
	byName := map[string]*dto.MetricFamily{}
	for _, family := range families {
		byName[family.GetName()] = family
	}
	decisions := byName["llm_d_epp_p2p_source_decisions_total"]
	require.NotNil(t, decisions, "decision counter exported")
	assert.Len(t, decisions.GetMetric(), 2, "one series per outcome")

	for _, want := range []struct {
		name   string
		labels map[string]string
		sum    float64
	}{
		{"llm_d_epp_p2p_source_predicted_pulled_tokens", map[string]string{}, 48},
		{"llm_d_epp_p2p_source_unpulled_tokens", map[string]string{"outcome": "below-floor"}, 16},
		{"llm_d_epp_p2p_source_request_ttft_seconds", map[string]string{"outcome": "pulled", "cost_margin": "none"}, 1},
	} {
		family := byName[want.name]
		require.NotNil(t, family, "%s exported", want.name)
		require.Len(t, family.GetMetric(), 1, want.name)
		labels := map[string]string{"plugin_name": "p2p", "endpoint_role": "decode"}
		maps.Copy(labels, want.labels)
		assert.Equal(t, labels, labelMap(family.GetMetric()[0]), want.name)
		assert.Equal(t, uint64(1), family.GetMetric()[0].GetHistogram().GetSampleCount(), want.name)
		assert.Equal(t, want.sum, family.GetMetric()[0].GetHistogram().GetSampleSum(), want.name)
	}
}

func labelMap(metric *dto.Metric) map[string]string {
	labels := map[string]string{}
	for _, label := range metric.GetLabel() {
		labels[label.GetName()] = label.GetValue()
	}
	return labels
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
