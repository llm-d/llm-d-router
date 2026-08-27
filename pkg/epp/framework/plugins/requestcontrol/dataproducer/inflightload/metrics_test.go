/*
Copyright 2026 The Kubernetes Authors.

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

package inflightload

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func TestRegisterMetrics_Idempotent(t *testing.T) {
	reg := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(reg))
	require.NoError(t, registerMetrics(reg))
}

func TestRegisterMetrics_NilRegisterer(t *testing.T) {
	require.Error(t, registerMetrics(nil))
}

func TestSplitNamespacedName(t *testing.T) {
	cases := []struct {
		id, wantName, wantNS string
	}{
		{"ns/ep", "ep", "ns"},
		{"bare", "bare", ""},
		{"ns/ep/extra", "ep/extra", "ns"},
	}
	for _, c := range cases {
		name, ns := splitNamespacedName(c.id)
		if name != c.wantName || ns != c.wantNS {
			t.Errorf("splitNamespacedName(%q) = (%q,%q), want (%q,%q)", c.id, name, ns, c.wantName, c.wantNS)
		}
	}
}

func TestInflightGauges_Lifecycle(t *testing.T) {
	inflightRequests.Reset()
	inflightTokens.Reset()
	producer := newTestProducer(t)
	ctx := context.Background()

	expect := func(t *testing.T, requests, tokens int64) {
		t.Helper()
		expectedRequests := fmt.Sprintf(`
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} %d
`, requests)
		require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequests), "llm_d_epp_inflight_requests"))
		expectedTokens := fmt.Sprintf(`
# HELP llm_d_epp_inflight_tokens [ALPHA] Current number of in-flight tokens per endpoint (uncached prompt tokens, optionally plus estimated output), as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_tokens gauge
llm_d_epp_inflight_tokens{endpoint_name="ep1",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} %d
`, tokens)
		require.NoError(t, promtestutil.CollectAndCompare(inflightTokens, strings.NewReader(expectedTokens), "llm_d_epp_inflight_tokens"))
	}

	// Admission: 4 input + UnknownOutputTokens estimated output (no outlen-bucket
	// attribute in this unit test) = 4+UnknownOutputTokens tokens, 1 request.
	req := makeTokenRequest("req-gauge-lifecycle", 4)
	res := makeSchedulingResult("ep1")
	err := producer.PreRequest(ctx, req, res)
	require.NoError(t, err)
	expect(t, 1, 4+UnknownOutputTokens)

	// Completion: series stay present at zero.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	expect(t, 0, 0)

	// Endpoint removal cleans up internal trackers, while gauge series remain at zero.
	producer.DeleteEndpoint(fullEndpointName("ep1"))
	expect(t, 0, 0)
	require.Equal(t, int64(0), producer.GetRequests(fullEndpointName("ep1")))
	require.Equal(t, int64(0), producer.GetTokens(fullEndpointName("ep1")))
}

func TestInflightGauges_CustomFairnessAndPriority(t *testing.T) {
	inflightRequests.Reset()
	inflightTokens.Reset()
	producer := newTestProducer(t)
	ctx := context.Background()

	req := makeTokenRequest("req-fairness-prio", 4)
	req.FairnessID = "custom-tenant"
	req.Objectives.Priority = 3
	res := makeSchedulingResult("ep1")

	err := producer.PreRequest(ctx, req, res)
	require.NoError(t, err)

	expectedRequests := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",fairness_id="custom-tenant",namespace="default",priority="3",producer_name="inflight-load-producer"} 1
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequests), "llm_d_epp_inflight_requests"))

	// 1004 = 4 input tokens + UnknownOutputTokens (no outlen-bucket attribute set).
	expectedTokens := `
# HELP llm_d_epp_inflight_tokens [ALPHA] Current number of in-flight tokens per endpoint (uncached prompt tokens, optionally plus estimated output), as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_tokens gauge
llm_d_epp_inflight_tokens{endpoint_name="ep1",fairness_id="custom-tenant",namespace="default",priority="3",producer_name="inflight-load-producer"} 1004
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightTokens, strings.NewReader(expectedTokens), "llm_d_epp_inflight_tokens"))

	// Cleanup on completion
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)

	expectedRequestsZero := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",fairness_id="custom-tenant",namespace="default",priority="3",producer_name="inflight-load-producer"} 0
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequestsZero), "llm_d_epp_inflight_requests"))
}

func TestInflightGauges_EarlyTokenRelease(t *testing.T) {
	inflightRequests.Reset()
	inflightTokens.Reset()
	producer := newTestProducer(t)
	producer.addEstimatedOutputTokens = false
	ctx := context.Background()

	req := makeTokenRequest("req-early-tokens", 4)
	res := makeSchedulingResult("ep1")

	err := producer.PreRequest(ctx, req, res)
	require.NoError(t, err)

	// Admitted: 1 request, 4 tokens.
	expectedRequests := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 1
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequests), "llm_d_epp_inflight_requests"))

	expectedTokens := `
# HELP llm_d_epp_inflight_tokens [ALPHA] Current number of in-flight tokens per endpoint (uncached prompt tokens, optionally plus estimated output), as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_tokens gauge
llm_d_epp_inflight_tokens{endpoint_name="ep1",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 4
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightTokens, strings.NewReader(expectedTokens), "llm_d_epp_inflight_tokens"))

	// StartOfStream chunk arrives: tokens released early, request still in flight.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true}, nil)

	expectedTokensReleased := `
# HELP llm_d_epp_inflight_tokens [ALPHA] Current number of in-flight tokens per endpoint (uncached prompt tokens, optionally plus estimated output), as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_tokens gauge
llm_d_epp_inflight_tokens{endpoint_name="ep1",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 0
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightTokens, strings.NewReader(expectedTokensReleased), "llm_d_epp_inflight_tokens"))
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequests), "llm_d_epp_inflight_requests"))

	// EndOfStream chunk arrives: request count released.
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)

	expectedRequestsReleased := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 0
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequestsReleased), "llm_d_epp_inflight_requests"))
}

func TestInflightGauges_DisaggEarlyPrefillRelease(t *testing.T) {
	inflightRequests.Reset()
	inflightTokens.Reset()
	producer := newTestProducer(t)
	ctx := context.Background()
	podA := "pod-a"
	podB := "pod-b"

	req := makeTokenRequest("req-disagg", 4) // 4 input + 6 output = 10 tokens
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(podA)}},
			"decode":  {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(podB)}},
		},
	}

	err := producer.PreRequest(ctx, req, res)
	require.NoError(t, err)

	expectedRequestsBoth := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="pod-a",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 1
llm_d_epp_inflight_requests{endpoint_name="pod-b",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 1
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequestsBoth), "llm_d_epp_inflight_requests"))

	// First chunk arrives: prefill endpoint released, decode endpoint stays busy.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true}, nil)

	expectedRequestsPrefillReleased := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="pod-a",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 0
llm_d_epp_inflight_requests{endpoint_name="pod-b",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 1
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequestsPrefillReleased), "llm_d_epp_inflight_requests"))

	// EndOfStream: decode endpoint released.
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)

	expectedRequestsAllReleased := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="pod-a",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 0
llm_d_epp_inflight_requests{endpoint_name="pod-b",fairness_id="",namespace="default",priority="0",producer_name="inflight-load-producer"} 0
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequestsAllReleased), "llm_d_epp_inflight_requests"))
}

func TestInflightGauges_MultiProducer(t *testing.T) {
	inflightRequests.Reset()
	inflightTokens.Reset()

	ctx := context.Background()

	p1, err := InFlightLoadProducerFactory("producer-1", nil, testutils.NewTestHandle(ctx))
	require.NoError(t, err)
	p2, err := InFlightLoadProducerFactory("producer-2", nil, testutils.NewTestHandle(ctx))
	require.NoError(t, err)

	producer1 := p1.(*InFlightLoadProducer)
	producer2 := p2.(*InFlightLoadProducer)

	req1 := makeTokenRequest("req-1", 4)
	req1.FairnessID = "flow-1"
	req1.Objectives.Priority = 1
	res1 := makeSchedulingResult("ep1")

	req2 := makeTokenRequest("req-2", 8)
	req2.FairnessID = "flow-2"
	req2.Objectives.Priority = 2
	res2 := makeSchedulingResult("ep2")

	require.NoError(t, producer1.PreRequest(ctx, req1, res1))
	require.NoError(t, producer2.PreRequest(ctx, req2, res2))

	expectedRequests := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",fairness_id="flow-1",namespace="default",priority="1",producer_name="producer-1"} 1
llm_d_epp_inflight_requests{endpoint_name="ep2",fairness_id="flow-2",namespace="default",priority="2",producer_name="producer-2"} 1
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequests), "llm_d_epp_inflight_requests"))

	// Complete req-1 on producer-1 -> series returns to 0
	req1.SchedulingResult = res1
	producer1.ResponseBody(ctx, req1, &requestcontrol.Response{EndOfStream: true}, nil)

	expectedRequestsAfterComplete := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",fairness_id="flow-1",namespace="default",priority="1",producer_name="producer-1"} 0
llm_d_epp_inflight_requests{endpoint_name="ep2",fairness_id="flow-2",namespace="default",priority="2",producer_name="producer-2"} 1
`
	require.NoError(t, promtestutil.CollectAndCompare(inflightRequests, strings.NewReader(expectedRequestsAfterComplete), "llm_d_epp_inflight_requests"))
}

func TestAddedTokensEntry_Clone(t *testing.T) {
	var nilEntry *addedTokensEntry
	require.Nil(t, nilEntry.Clone())

	entry := &addedTokensEntry{
		endpointName: "ep1",
		namespace:    "default",
		producerName: "prod",
		fairnessID:   "flow",
		priority:     "1",
	}
	entry.tokens.Store(15)
	entry.requests.Store(1)

	cloned := entry.Clone().(*addedTokensEntry)
	require.Equal(t, entry.endpointName, cloned.endpointName)
	require.Equal(t, entry.namespace, cloned.namespace)
	require.Equal(t, entry.producerName, cloned.producerName)
	require.Equal(t, entry.fairnessID, cloned.fairnessID)
	require.Equal(t, entry.priority, cloned.priority)
	require.Equal(t, int64(15), cloned.tokens.Load())
	require.Equal(t, int32(1), cloned.requests.Load())
}

// The plugin's fairness_id label shares the cardinality cap, so a flood of distinct client IDs
// collapses to the bounded set plus one overflow series instead of one series per ID.
func TestInflightGauges_FairnessIDCardinalityBounded(t *testing.T) {
	const testCap = 3
	inflightRequests.Reset()
	inflightTokens.Reset()
	metricsutil.SetFairnessIDLabelLimit(testCap)
	t.Cleanup(func() {
		metricsutil.SetFairnessIDLabelLimit(metricsutil.DefaultFairnessIDLabelLimit)
		inflightRequests.Reset()
		inflightTokens.Reset()
	})

	producer := newTestProducer(t)
	ctx := context.Background()

	// All requests target the same endpoint, so fairness_id is the only varying label.
	for i := 0; i < 100; i++ {
		req := makeTokenRequest(fmt.Sprintf("req-%d", i), 4)
		req.FairnessID = fmt.Sprintf("tenant-%d", i)
		require.NoError(t, producer.PreRequest(ctx, req, makeSchedulingResult("ep1")))
	}

	require.Equal(t, testCap+1, promtestutil.CollectAndCount(inflightRequests),
		"100 distinct fairness IDs must collapse to cap+overflow series")
	require.Equal(t, testCap+1, promtestutil.CollectAndCount(inflightTokens),
		"the tokens gauge must be bounded identically")
}
