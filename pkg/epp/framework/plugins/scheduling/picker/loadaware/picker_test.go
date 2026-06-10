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

package loadaware

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newTestPicker creates a LoadAwarePicker with nil handle (no external metrics registration).
func newTestPicker(t *testing.T) *LoadAwarePicker {
	t.Helper()
	p, err := newLoadAwarePicker("test-picker", nil)
	if err != nil {
		t.Fatalf("newLoadAwarePicker: %v", err)
	}
	return p
}

func makeEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		&fwkdl.Metrics{},
		nil,
	)
}

func makeScoredEndpoint(ep fwksched.Endpoint, score float64) *fwksched.ScoredEndpoint {
	return &fwksched.ScoredEndpoint{Endpoint: ep, Score: score}
}

// TestPick_SingleEndpoint verifies that a single-endpoint pool always returns that endpoint.
func TestPick_SingleEndpoint(t *testing.T) {
	p := newTestPicker(t)
	ep := makeEndpoint("ep1")
	result := p.Pick(context.Background(), []*fwksched.ScoredEndpoint{makeScoredEndpoint(ep, 1.0)})
	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected 1 target endpoint, got %d", len(result.TargetEndpoints))
	}
	if result.TargetEndpoints[0].GetMetadata().NamespacedName.Name != "ep1" {
		t.Errorf("expected ep1, got %s", result.TargetEndpoints[0].GetMetadata().NamespacedName.Name)
	}
}

// TestPick_EmptyPool verifies that an empty pool returns an empty result without panicking.
func TestPick_EmptyPool(t *testing.T) {
	p := newTestPicker(t)
	result := p.Pick(context.Background(), nil)
	if len(result.TargetEndpoints) != 0 {
		t.Fatalf("expected 0 target endpoints, got %d", len(result.TargetEndpoints))
	}
}

// TestPick_BestScoreSelected verifies that the endpoint with the highest score is chosen
// when no concentration or capacity penalties apply.
func TestPick_BestScoreSelected(t *testing.T) {
	p := newTestPicker(t)
	ep1 := makeEndpoint("ep1")
	ep2 := makeEndpoint("ep2")
	ep3 := makeEndpoint("ep3")
	result := p.Pick(context.Background(), []*fwksched.ScoredEndpoint{
		makeScoredEndpoint(ep1, 0.3),
		makeScoredEndpoint(ep2, 0.9),
		makeScoredEndpoint(ep3, 0.5),
	})
	if result.TargetEndpoints[0].GetMetadata().NamespacedName.Name != "ep2" {
		t.Errorf("expected ep2 (highest score), got %s", result.TargetEndpoints[0].GetMetadata().NamespacedName.Name)
	}
}

// TestConcentrationFactor_NoPenaltyAtFairShare verifies factor = 1.0 when picks == expectedShare.
func TestConcentrationFactor_NoPenaltyAtFairShare(t *testing.T) {
	p := newTestPicker(t)
	st := &endpointLiveState{}
	// 3 picks, pool=3, expectedShare=1; actual=1 → factor=1.0
	st.pickBuckets.Inc()
	f := p.concentrationFactor(st, 1.0, 3)
	if f != 1.0 {
		t.Errorf("expected 1.0, got %f", f)
	}
}

// TestConcentrationFactor_PenaltyAtDoubleShare verifies factor = 0.5 when actual = 2 * expected.
func TestConcentrationFactor_PenaltyAtDoubleShare(t *testing.T) {
	p := newTestPicker(t)
	st := &endpointLiveState{}
	st.pickBuckets.Inc()
	st.pickBuckets.Inc() // actual = 2
	f := p.concentrationFactor(st, 1.0, 3)
	if math.Abs(f-0.5) > 1e-9 {
		t.Errorf("expected 0.5, got %f", f)
	}
}

// TestConcentrationFactor_SingleEndpointPool verifies factor = 1.0 for pool of 1.
func TestConcentrationFactor_SingleEndpointPool(t *testing.T) {
	p := newTestPicker(t)
	st := &endpointLiveState{}
	for range 100 {
		st.pickBuckets.Inc()
	}
	f := p.concentrationFactor(st, 0, 1)
	if f != 1.0 {
		t.Errorf("expected 1.0 for single-endpoint pool, got %f", f)
	}
}

// TestCapacityFactor_ColdStart verifies factor = 1.0 before any EMA data.
func TestCapacityFactor_ColdStart(t *testing.T) {
	p := newTestPicker(t)
	ep := makeEndpoint("ep1")
	scored := makeScoredEndpoint(ep, 1.0)
	st := &endpointLiveState{}
	f := p.capacityFactor(scored, st)
	if f != 1.0 {
		t.Errorf("expected 1.0 (cold start), got %f", f)
	}
}

// TestCapacityFactor_HalfLoad verifies factor ≈ 0.5 when committed = 0.5 * maxConcurrency.
func TestCapacityFactor_HalfLoad(t *testing.T) {
	p := newTestPicker(t)

	// maxConcurrency = capacityEMA * avgLatency / avgTokensPerReq = 100 * 1.0 / 10 = 10
	st := &endpointLiveState{
		capacityEMA:     100.0,
		avgLatency:      1.0,
		avgTokensPerReq: 10.0,
	}
	// committedRequests = 5 (50% of maxConcurrency=10), pendingRequests=0 → factor = 0.5
	metrics := &fwkdl.Metrics{RunningRequestsSize: 5}
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "ep"}},
		metrics,
		nil,
	)
	scored := makeScoredEndpoint(ep, 1.0)
	f := p.capacityFactor(scored, st)
	if math.Abs(f-0.5) > 1e-9 {
		t.Errorf("expected 0.5, got %f", f)
	}
}

// TestCapacityFactor_Overloaded verifies factor = 0 when load >= maxConcurrency.
func TestCapacityFactor_Overloaded(t *testing.T) {
	p := newTestPicker(t)
	// maxConcurrency = 10, committedRequests = 12 → factor = 0
	st := &endpointLiveState{
		capacityEMA:     100.0,
		avgLatency:      1.0,
		avgTokensPerReq: 10.0,
	}
	metrics := &fwkdl.Metrics{RunningRequestsSize: 12}
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "ep"}},
		metrics,
		nil,
	)
	scored := makeScoredEndpoint(ep, 1.0)
	f := p.capacityFactor(scored, st)
	if f != 0.0 {
		t.Errorf("expected 0.0 (overloaded), got %f", f)
	}
}

// TestPreRequest_DecrementsPending verifies the pending counter drops after PreRequest.
func TestPreRequest_DecrementsPending(t *testing.T) {
	p := newTestPicker(t)
	ep := makeEndpoint("ep1")
	scored := []*fwksched.ScoredEndpoint{makeScoredEndpoint(ep, 1.0)}

	// Pick once -- this increments pending to 1.
	p.Pick(context.Background(), scored)

	key := ep.GetMetadata().NamespacedName.String()
	p.mu.RLock()
	st := p.state[key]
	p.mu.RUnlock()
	if st == nil {
		t.Fatal("state not created after Pick")
	}
	if st.pendingRequests.Load() != 1 {
		t.Fatalf("expected pending=1 after Pick, got %d", st.pendingRequests.Load())
	}

	// Call PreRequest -- should decrement to 0.
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{ep}},
		},
	}
	req := &fwksched.InferenceRequest{RequestID: "req-1"}
	p.PreRequest(context.Background(), req, result)

	if st.pendingRequests.Load() != 0 {
		t.Errorf("expected pending=0 after PreRequest, got %d", st.pendingRequests.Load())
	}
}

// TestResponseBody_UpdatesEMA verifies capacityEMA is populated after an EndOfStream response.
func TestResponseBody_UpdatesEMA(t *testing.T) {
	p := newTestPicker(t)

	ep := makeEndpoint("ep1")
	epMeta := ep.GetMetadata()

	// Seed the request cache manually (simulating PreRequest).
	p.requestCache.Set("req-1", &requestRecord{
		dispatchTime: time.Now().Add(-500 * time.Millisecond),
	}, ttlcache.DefaultTTL)

	req := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: []uint32{1, 2, 3, 4, 5}},
		},
	}
	resp := &fwkrc.Response{
		EndOfStream: true,
		Usage: fwkrh.Usage{
			CompletionTokens: 10,
		},
	}

	p.ResponseBody(context.Background(), req, resp, epMeta)

	p.mu.RLock()
	st := p.state[epMeta.NamespacedName.String()]
	p.mu.RUnlock()
	if st == nil {
		t.Fatal("state not created after ResponseBody")
	}
	if st.capacityEMA <= 0 {
		t.Errorf("expected capacityEMA > 0, got %f", st.capacityEMA)
	}
	if st.avgLatency <= 0 {
		t.Errorf("expected avgLatency > 0, got %f", st.avgLatency)
	}
	if st.avgTokensPerReq != 15 { // 5 input + 10 output
		t.Errorf("expected avgTokensPerReq=15, got %f", st.avgTokensPerReq)
	}
}

// TestResponseBody_SkipsNonEndOfStream verifies that intermediate chunks do not update the EMA.
func TestResponseBody_SkipsNonEndOfStream(t *testing.T) {
	p := newTestPicker(t)
	ep := makeEndpoint("ep1")
	epMeta := ep.GetMetadata()

	req := &fwksched.InferenceRequest{RequestID: "req-1"}
	resp := &fwkrc.Response{
		EndOfStream: false,
		Usage:       fwkrh.Usage{CompletionTokens: 5},
	}

	p.ResponseBody(context.Background(), req, resp, epMeta)

	p.mu.RLock()
	st := p.state[epMeta.NamespacedName.String()]
	p.mu.RUnlock()
	if st != nil && st.capacityEMA != 0 {
		t.Errorf("unexpected EMA update on non-EndOfStream call")
	}
}

// TestResponseBody_SkipsZeroOutputTokens verifies errors/disconnects (0 output tokens) are ignored.
func TestResponseBody_SkipsZeroOutputTokens(t *testing.T) {
	p := newTestPicker(t)
	ep := makeEndpoint("ep1")
	epMeta := ep.GetMetadata()

	p.requestCache.Set("req-1", &requestRecord{
		dispatchTime: time.Now().Add(-100 * time.Millisecond),
	}, ttlcache.DefaultTTL)

	req := &fwksched.InferenceRequest{RequestID: "req-1"}
	resp := &fwkrc.Response{EndOfStream: true, Usage: fwkrh.Usage{CompletionTokens: 0}}

	p.ResponseBody(context.Background(), req, resp, epMeta)

	p.mu.RLock()
	st := p.state[epMeta.NamespacedName.String()]
	p.mu.RUnlock()
	if st != nil && st.capacityEMA != 0 {
		t.Errorf("unexpected EMA update when CompletionTokens=0")
	}
}

// TestBucketedCounter_BasicIncAndCount verifies basic increment and count.
func TestBucketedCounter_BasicIncAndCount(t *testing.T) {
	var b bucketedCounter
	if b.Count() != 0 {
		t.Errorf("expected 0, got %d", b.Count())
	}
	b.Inc()
	b.Inc()
	b.Inc()
	if b.Count() != 3 {
		t.Errorf("expected 3, got %d", b.Count())
	}
}

// TestPick_PendingIncrementedAfterPick verifies Pick increments the pending counter.
func TestPick_PendingIncrementedAfterPick(t *testing.T) {
	p := newTestPicker(t)
	ep := makeEndpoint("ep1")
	p.Pick(context.Background(), []*fwksched.ScoredEndpoint{makeScoredEndpoint(ep, 1.0)})

	key := ep.GetMetadata().NamespacedName.String()
	p.mu.RLock()
	st := p.state[key]
	p.mu.RUnlock()
	if st == nil || st.pendingRequests.Load() != 1 {
		t.Errorf("expected pendingRequests=1, got %v", st)
	}
}
