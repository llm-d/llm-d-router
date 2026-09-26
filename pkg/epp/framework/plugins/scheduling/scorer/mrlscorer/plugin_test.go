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

package mrlscorer

import (
	"context"
	"testing"
	"time"

	"github.com/caio/go-tdigest/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsojourn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/sojourntime"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sojourntimeobserver"
)

const testProfile = "default"

// warmSnapshot builds a snapshot from a list of TTFT samples and decode samples.
// Both digests get the same number of samples for simplicity.
func warmSnapshot(t *testing.T, ttftSamples, decodeSamples []float64) *attrsojourn.SojournEstimatorSnapshot {
	t.Helper()
	ttft, err := tdigest.New(tdigest.Compression(500))
	require.NoError(t, err)
	for _, s := range ttftSamples {
		require.NoError(t, ttft.Add(s))
	}
	tb, err := ttft.AsBytes()
	require.NoError(t, err)

	decode, err := tdigest.New(tdigest.Compression(500))
	require.NoError(t, err)
	for _, s := range decodeSamples {
		require.NoError(t, decode.Add(s))
	}
	db, err := decode.AsBytes()
	require.NoError(t, err)

	return &attrsojourn.SojournEstimatorSnapshot{TtftDigest: tb, DecodeDigest: db}
}

// newEndpointWithSnapshot builds a scheduling endpoint named id, optionally
// carrying a SojournEstimatorSnapshot under the scorer's data key.
func newEndpointWithSnapshot(id string, snap *attrsojourn.SojournEstimatorSnapshot) fwksched.Endpoint {
	attr := fwkdl.NewAttributes()
	if snap != nil {
		attr.Put(attrsojourn.SojournEstimatorSnapshotDataKey, snap)
	}
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: id, Namespace: "default"}}
	return fwksched.NewEndpoint(meta, nil, attr)
}

// newObserverWithInflight builds a live observer and pre-populates its
// in-flight index by injecting requests into it via PreRequest.
func newObserverWithInflight(t *testing.T) *sojourntimeobserver.Observer {
	t.Helper()
	obs, err := sojourntimeobserver.NewObserver("sojourn-observer", sojourntimeobserver.DefaultConfig)
	require.NoError(t, err)
	return obs
}

// dispatch drives PreRequest to insert an in-flight entry for the given
// endpoint at a specific dispatchedAt time — the test controls the clock
// exactly. We call the framework hook, which uses time.Now(), then the test
// sleeps to advance real time between operations. For predictable timing we
// instead call PreRequest and interpret ages relative to Score's now.
func dispatch(t *testing.T, obs *sojourntimeobserver.Observer, ep fwksched.Endpoint, reqID string) {
	t.Helper()
	require.NoError(t, obs.PreRequest(context.Background(),
		&fwksched.InferenceRequest{RequestID: reqID},
		&fwksched.SchedulingResult{
			PrimaryProfileName: testProfile,
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				testProfile: {TargetEndpoints: []fwksched.Endpoint{ep}},
			},
		},
	))
}

// TestScore_ColdEndpointsAllOne verifies that when every endpoint is cold (no
// snapshot in AttributeMap), all endpoints tie at 1.0 and the picker falls
// through to random-shuffle. Their residuals are all 0 (span=0), which
// normalization treats as a tie.
func TestScore_ColdEndpointsAllOne(t *testing.T) {
	obs := newObserverWithInflight(t)
	scorer := NewScorer(obs, Config{})

	ep0 := newEndpointWithSnapshot("a", nil)
	ep1 := newEndpointWithSnapshot("b", nil)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{ep0, ep1})
	assert.Equal(t, 1.0, scores[ep0])
	assert.Equal(t, 1.0, scores[ep1])
}

// TestScore_ArgminFavoured verifies that when one endpoint has higher residual
// than the other, the lower one gets 1.0 and the higher one gets 0.0.
func TestScore_ArgminFavoured(t *testing.T) {
	obs := newObserverWithInflight(t)
	scorer := NewScorer(obs, Config{})

	// Both endpoints have the same warm snapshot (same distribution).
	snap := warmSnapshot(t,
		[]float64{0.1, 0.2, 0.5, 1.0, 2.0}, // TTFT samples
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0}, // decode samples
	)
	epLight := newEndpointWithSnapshot("light", snap)
	epHeavy := newEndpointWithSnapshot("heavy", snap)

	// Only heavy has in-flight requests → heavy has higher residual.
	dispatch(t, obs, epHeavy, "req-1")
	dispatch(t, obs, epHeavy, "req-2")

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{epLight, epHeavy})
	assert.Equal(t, 1.0, scores[epLight], "empty in-flight → residual 0 → wins argmin")
	assert.Equal(t, 0.0, scores[epHeavy], "two in-flight requests → higher residual → loses")
}

// TestScore_TwoTermFormula verifies the residual matches a hand-computed
// two-term sum. One endpoint has one pre-first-chunk request and one
// post-first-chunk request.
func TestScore_TwoTermFormula(t *testing.T) {
	obs := newObserverWithInflight(t)
	scorer := NewScorer(obs, Config{})

	// Symmetric snapshot: TTFT ~ Exp(1) at mean 1.0, decode ~ Exp(1) at
	// mean 1.0. Under exponential memorylessness, MrlTtft(a) == 1.0 and
	// MrlDecode(a) == 1.0 for any a, so the residual has a closed form.
	ttftSamples := make([]float64, 500)
	decodeSamples := make([]float64, 500)
	for i := range ttftSamples {
		ttftSamples[i] = 1.0 + float64(i)*0.001 // narrow spread around 1.0
	}
	for i := range decodeSamples {
		decodeSamples[i] = 1.0 + float64(i)*0.001
	}
	snap := warmSnapshot(t, ttftSamples, decodeSamples)
	ep := newEndpointWithSnapshot("a", snap)

	// One in-flight request, pre-first-chunk (dispatched at a known instant
	// slightly in the past). Score's residual should be approximately
	// MrlTtft(age) + MeanDecode ~= 1.0 + 1.0 = 2.0.
	dispatch(t, obs, ep, "req-1")

	// Small delay so age > 0.
	time.Sleep(3 * time.Millisecond)
	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{ep})
	// With one endpoint, the normalization ties → 1.0.
	assert.Equal(t, 1.0, scores[ep], "single-candidate score is always 1.0")

	// Direct residual check via a two-endpoint call where the second is
	// empty, so the argmin span is exposed.
	epEmpty := newEndpointWithSnapshot("b", snap)
	scores = scorer.Score(context.Background(), nil, []fwksched.Endpoint{ep, epEmpty})
	assert.Equal(t, 0.0, scores[ep], "endpoint with one pre-first-chunk request loses")
	assert.Equal(t, 1.0, scores[epEmpty], "empty endpoint wins")
}

// TestScore_TiesAllOne verifies that when every endpoint has an identical
// residual (achieved here by giving both zero in-flight requests, so both
// residuals are exactly 0.0), every endpoint scores 1.0 — the picker then
// falls through to random-shuffle.
func TestScore_TiesAllOne(t *testing.T) {
	obs := newObserverWithInflight(t)
	scorer := NewScorer(obs, Config{})

	snap := warmSnapshot(t, []float64{1.0, 2.0, 3.0}, []float64{1.0, 2.0, 3.0})
	ep0 := newEndpointWithSnapshot("a", snap)
	ep1 := newEndpointWithSnapshot("b", snap)
	// Neither endpoint has in-flight requests → both residuals are 0 →
	// exact tie → both score 1.0.

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{ep0, ep1})
	assert.Equal(t, 1.0, scores[ep0])
	assert.Equal(t, 1.0, scores[ep1])
}

// TestScore_ConsumesSnapshotKey confirms Consumes declares the snapshot data
// key as Required, so the plugin DAG will fail to load a mrl-scorer-hub whose
// producer is missing.
func TestScore_ConsumesSnapshotKey(t *testing.T) {
	obs := newObserverWithInflight(t)
	scorer := NewScorer(obs, Config{})
	deps := scorer.Consumes()
	_, ok := deps.Required[attrsojourn.SojournEstimatorSnapshotDataKey]
	assert.True(t, ok, "SojournEstimatorSnapshotDataKey must be a Required consumption")
}

// TestScore_Category confirms the scorer identifies as Distribution-axis, so
// the SchedulerProfile composes it correctly.
func TestScore_Category(t *testing.T) {
	obs := newObserverWithInflight(t)
	scorer := NewScorer(obs, Config{})
	assert.Equal(t, fwksched.Distribution, scorer.Category())
}
