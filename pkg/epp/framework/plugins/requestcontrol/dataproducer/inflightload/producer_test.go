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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func newTestProducer(t testing.TB) *InFlightLoadProducer {
	params := Config{AddEstimatedOutputTokens: true}
	raw, err := json.Marshal(params)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	decoder := json.NewDecoder(bytes.NewReader(raw))
	p, err := InFlightLoadProducerFactory("inflight-load-producer", decoder, testutils.NewTestHandle(ctx))
	require.NoError(t, err)
	return p.(*InFlightLoadProducer)
}

func TestInFlightLoadProducer_Consumes(t *testing.T) {
	t.Parallel()

	deps := newTestProducer(t).Consumes()

	// TokenizedPrompt is required so the data-layer DAG auto-creates a
	// token-producer and orders it ahead of this producer; without it the input
	// token estimate silently reads zero.
	require.Contains(t, deps.Required, tokenproducer.TokenizedPromptDataKey)
	// PrefixCacheMatchInfo is optional: consumed for the cached-prefix discount.
	// With no prefixMatchInfoProducerName set, it defaults to the approximate
	// producer's key.
	require.Contains(t, deps.Optional, attrprefix.PrefixCacheMatchInfoDataKey)
	require.NotContains(t, deps.Required, attrprefix.PrefixCacheMatchInfoDataKey)
}

// prefixMatchInfoProducerName selects which prefix producer (approximate or
// precise) feeds the cached-prefix discount, by both the optional dependency key
// and the runtime read.
func TestInFlightLoadProducer_PrefixMatchInfoProducerName(t *testing.T) {
	t.Parallel()

	const preciseName = "precise-prefix-cache-producer"
	raw, err := json.Marshal(Config{PrefixMatchInfoProducerName: preciseName})
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	p, err := InFlightLoadProducerFactory("inflight-load-producer",
		json.NewDecoder(bytes.NewReader(raw)), testutils.NewTestHandle(ctx))
	require.NoError(t, err)
	producer := p.(*InFlightLoadProducer)

	preciseKey := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(preciseName)

	// The optional dependency points at the configured precise producer, not approx.
	require.Contains(t, producer.Consumes().Optional, preciseKey)
	require.NotContains(t, producer.Consumes().Optional, attrprefix.PrefixCacheMatchInfoDataKey)

	// The discount reads PrefixCacheMatchInfo from the configured producer's key
	// (indexed 2*4=8, matched 1*4=4 -> uncached 4).
	hit := newStubSchedulingEndpoint("ep-hit")
	hit.Put(preciseKey.String(), attrprefix.NewPrefixCacheMatchInfo(1, 2, 4))
	require.Equal(t, int64(4), producer.estimateRequestTokens(hit, nil, 5))

	// Data under the approx (default) key is ignored, so it falls back to inputTokens.
	miss := newStubSchedulingEndpoint("ep-miss")
	miss.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(1, 2, 4))
	require.Equal(t, int64(5), producer.estimateRequestTokens(miss, nil, 5))
}

func TestInFlightLoadProducer_Produce(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)

	endpointName := "test-endpoint"
	endpoint := newStubSchedulingEndpoint(endpointName)
	endpoints := []fwksched.Endpoint{endpoint}

	// 1. Produce with nil request -> should not put anything
	err := producer.Produce(context.Background(), nil, endpoints)
	require.NoError(t, err)
	_, ok := endpoint.Get(producer.uncachedRequestTokensDk.String())
	require.False(t, ok)

	// 2. Produce with request -> should put UncachedRequestTokens
	req := makeTokenRequest("req1", 4) // 4 input tokens -> 10 total (with output)
	err = producer.Produce(context.Background(), req, endpoints)

	require.NoError(t, err)

	val, ok := endpoint.Get(producer.uncachedRequestTokensDk.String())
	require.True(t, ok)
	uncached := val.(*attrconcurrency.UncachedRequestTokens)
	require.Equal(t, int64(10), uncached.Tokens)

	// Verify that InFlightLoad was NOT put/overwritten by Produce
	_, ok = endpoint.Get(producer.dk.String())
	require.False(t, ok, "InFlightLoad should not be populated by Produce")
}

func TestInFlightLoadProducer_ProduceUsesLiteralCachedBlocks(t *testing.T) {
	t.Parallel()

	// Initialize our load producer.
	producer := newTestProducer(t)
	// Disable output token estimation so we can test input tokens cleanly (output = 0).
	producer.addEstimatedOutputTokens = false

	// Create a stub endpoint representing our model-serving pod.
	endpoint := newStubSchedulingEndpoint("precise-cache-endpoint")
	// Inject mock prefix cache information onto this endpoint.
	// matchBlocks = 1 (weighted), cachedBlockCount = 2 (literal), totalBlocks = 2, blockSizeTokens = 4.
	endpoint.Put(
		attrprefix.PrefixCacheMatchInfoDataKey.String(),
		attrprefix.NewPrefixCacheMatchInfo(1, 2, 4).WithCachedBlockCount(2),
	)

	// Trigger the Produce logic with an 8-token request.
	err := producer.Produce(context.Background(), makeTokenRequest("req-precise", 8), []fwksched.Endpoint{endpoint})
	require.NoError(t, err)

	// Assert the produced output is exactly 0 uncached tokens.
	value, ok := endpoint.Get(producer.uncachedRequestTokensDk.String())
	require.True(t, ok)
	require.Equal(t, int64(0), value.(*attrconcurrency.UncachedRequestTokens).Tokens)
}

func TestInFlightLoadProducer_Extract(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	endpointName := "test-endpoint"
	endpointID := fullEndpointName(endpointName)

	endpoint := newStubSchedulingEndpoint(endpointName)
	ctx := context.Background()

	// Simulate Add event
	err := producer.Extract(ctx, datalayer.EndpointEvent{
		Type:     datalayer.EventAddOrUpdate,
		Endpoint: endpoint,
	})
	require.NoError(t, err)

	// Verify dynamic attribute is registered
	key := producer.dk.String()
	val, ok := endpoint.Get(key)
	require.True(t, ok)

	// Verify initial values (should be 0)
	load := val.(*attrconcurrency.InFlightLoad)
	require.Equal(t, int64(0), load.Requests)
	require.Equal(t, int64(0), load.Tokens)

	// Update trackers
	producer.requestTracker.add(endpointID, 5)
	producer.tokenTracker.add(endpointID, 500)

	// Verify values are updated dynamically without calling Produce
	val2, ok := endpoint.Get(key)
	require.True(t, ok)
	load2 := val2.(*attrconcurrency.InFlightLoad)
	require.Equal(t, int64(5), load2.Requests)
	require.Equal(t, int64(500), load2.Tokens)
}

func TestInFlightLoadProducer_Lifecycle(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "lifecycle-endpoint"
	endpointID := fullEndpointName(endpointName)

	// 1. PreRequest (Inc)
	req := makeTokenRequest("req1", 4) // 4 input + 6 output = 10 tokens
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)

	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(10), producer.tokenTracker.get(endpointID))

	// 2. ResponseBody EndOfStream (Dec)
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)

	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

func TestInFlightLoadProducer_MultiPodLifecycle(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	podA := "pod-a"
	podB := "pod-b"
	idA := fullEndpointName(podA)
	idB := fullEndpointName(podB)

	// 1. Dispatch to PodA (Prefill) and PodB (Decode)
	req := makeTokenRequest("multi-req", 4) // 4 input + 6 output = 10 tokens
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(podA)}},
			"decode":  {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(podB)}},
		},
	}

	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(1), producer.requestTracker.get(idA))
	require.Equal(t, int64(1), producer.requestTracker.get(idB))

	// 2. First Chunk arrives (Early Prefill Release)
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: false, StartOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(idA), "PodA should be released after first chunk")
	require.Equal(t, int64(1), producer.requestTracker.get(idB), "PodB should still be busy")

	// 3. Final Chunk arrives (Full Cleanup)
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(idA), "PodA should stay clean")
	require.Equal(t, int64(0), producer.requestTracker.get(idB), "PodB should now be released")
}

func TestInFlightLoadProducer_NotificationCleanup(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "deleted-endpoint"
	endpointID := fullEndpointName(endpointName)

	// Seed load
	producer.requestTracker.add(endpointID, 10)
	producer.tokenTracker.add(endpointID, 1000)

	// Simulate Delete Notification (Endpoint)
	eventEndpoint := datalayer.EndpointEvent{
		Type:     datalayer.EventDelete,
		Endpoint: newStubSchedulingEndpoint(endpointName),
	}

	err := producer.Extract(ctx, eventEndpoint)
	require.NoError(t, err)

	// Verify Cleanup
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

func TestInFlightLoadProducer_DumpState(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
	}
	podA := fullEndpointName("pod-a")
	podB := fullEndpointName("pod-b")

	producer.requestTracker.add(podB, 2)
	producer.tokenTracker.add(podA, 10)
	producer.tokenTracker.add(podB, 20)

	payload, err := producer.DumpState()
	require.NoError(t, err)

	var state inFlightLoadState
	require.NoError(t, json.Unmarshal(payload, &state))
	require.Equal(t, inFlightLoadState{
		Endpoints: []endpointInFlightLoadState{
			{Endpoint: podB, Requests: 2, Tokens: 20},
			{Endpoint: podA, Requests: 0, Tokens: 10},
		},
		TotalEndpoints: 2,
		MaxEndpoints:   maxDebugDumpEndpoints,
	}, state)
}

func TestInFlightLoadProducer_DumpStateCapsEndpoints(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
	}
	for i := range maxDebugDumpEndpoints + 5 {
		endpointID := fullEndpointName(fmt.Sprintf("pod-%03d", i))
		producer.requestTracker.add(endpointID, int64(i))
	}

	payload, err := producer.DumpState()
	require.NoError(t, err)

	var state inFlightLoadState
	require.NoError(t, json.Unmarshal(payload, &state))
	require.True(t, state.Truncated)
	require.Equal(t, maxDebugDumpEndpoints+5, state.TotalEndpoints)
	require.Equal(t, maxDebugDumpEndpoints, state.MaxEndpoints)
	require.Len(t, state.Endpoints, maxDebugDumpEndpoints)
	require.Equal(t, fullEndpointName("pod-104"), state.Endpoints[0].Endpoint)
	require.Equal(t, int64(104), state.Endpoints[0].Requests)
}

// TestInFlightLoadProducer_FlapDoesNotUnderflow asserts that a request's release lands on the exact
// counter instance it incremented, even after the endpoint is deleted and recreated under the same
// NamespacedName. A release that predates the flap hits the orphaned counter, leaving the live
// counter accurate and never negative. Cases cover the request and token counters across both the
// EndOfStream eviction path (OnEvicted) and the StartOfStream early-release path (releaseTokensEarly).
func TestInFlightLoadProducer_FlapDoesNotUnderflow(t *testing.T) {
	t.Parallel()

	requests := func(p *InFlightLoadProducer, id string) int64 { return p.requestTracker.get(id) }
	tokens := func(p *InFlightLoadProducer, id string) int64 { return p.tokenTracker.get(id) }

	tests := []struct {
		name                     string
		addEstimatedOutputTokens bool
		release                  requestcontrol.Response // chunk that triggers the release under test
		read                     func(*InFlightLoadProducer, string) int64
		inputA, inputB           int   // input tokens for requests A and B
		liveAfterA, liveAfterB   int64 // live counter after A's, then B's, PreRequest
		liveAfterReleaseA        int64 // live counter after A releases, while B is still in flight
	}{
		{
			name:    "request counter, EndOfStream eviction",
			release: requestcontrol.Response{EndOfStream: true},
			read:    requests,
			inputA:  4, inputB: 4,
			liveAfterA: 1, liveAfterB: 1, liveAfterReleaseA: 1,
		},
		{
			name:                     "token counter, EndOfStream eviction",
			addEstimatedOutputTokens: true, // 4 input + 6 estimated output = 10 tokens per request
			release:                  requestcontrol.Response{EndOfStream: true},
			read:                     tokens,
			inputA:                   4, inputB: 4,
			liveAfterA: 10, liveAfterB: 10, liveAfterReleaseA: 10,
		},
		{
			name:    "token counter, StartOfStream early release",
			release: requestcontrol.Response{StartOfStream: true}, // output excluded: tokens == input
			read:    tokens,
			inputA:  4, inputB: 6,
			liveAfterA: 4, liveAfterB: 6, liveAfterReleaseA: 6,
		},
	}

	const endpointName = "flap-endpoint"
	extract := func(t *testing.T, p *InFlightLoadProducer, eventType datalayer.EventType, ep datalayer.Endpoint) {
		require.NoError(t, p.Extract(context.Background(), datalayer.EndpointEvent{Type: eventType, Endpoint: ep}))
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			producer := newTestProducer(t)
			producer.addEstimatedOutputTokens = tc.addEstimatedOutputTokens
			ctx := context.Background()
			endpointID := fullEndpointName(endpointName)

			// E joins. The same Endpoint object is reused for its delete so the registeredEndpoints
			// guard allows cleanup (the datalayer's same-pointer add/delete contract).
			epA := newStubSchedulingEndpoint(endpointName)
			extract(t, producer, datalayer.EventAddOrUpdate, epA)
			reqA := makeTokenRequest("req-A", tc.inputA)
			resA := makeSchedulingResult(endpointName)
			producer.PreRequest(ctx, reqA, resA)
			require.Equal(t, tc.liveAfterA, tc.read(producer, endpointID))

			// E flaps and rejoins under the same NamespacedName; B recreates a fresh counter instance.
			extract(t, producer, datalayer.EventDelete, epA)
			require.Equal(t, int64(0), tc.read(producer, endpointID), "counter dropped on delete")
			epB := newStubSchedulingEndpoint(endpointName)
			extract(t, producer, datalayer.EventAddOrUpdate, epB)
			reqB := makeTokenRequest("req-B", tc.inputB)
			resB := makeSchedulingResult(endpointName)
			producer.PreRequest(ctx, reqB, resB)
			require.Equal(t, tc.liveAfterB, tc.read(producer, endpointID), "B recreated the counter")

			// A releases. It must hit the orphaned counter, not B's live one.
			reqA.SchedulingResult = resA
			producer.ResponseBody(ctx, reqA, &tc.release, nil)
			require.Equal(t, tc.liveAfterReleaseA, tc.read(producer, endpointID),
				"A's release must not discount B's live counter")

			// B releases. The live counter settles at 0 and never underflows.
			reqB.SchedulingResult = resB
			producer.ResponseBody(ctx, reqB, &tc.release, nil)
			require.Equal(t, int64(0), tc.read(producer, endpointID), "counter must settle at 0, never negative")
		})
	}
}

// TestInFlightLoadProducer_CrashWithHighLoadDoesNotUnderflow models a pod that crashes while
// holding many in-flight requests: its counter is dropped, the pod rejoins, and the crashed
// requests drain late. Their releases hit the orphaned counter, so the live counter holds only
// post-crash load and never goes negative.
func TestInFlightLoadProducer_CrashWithHighLoadDoesNotUnderflow(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "crash-endpoint"
	endpointID := fullEndpointName(endpointName)

	const inFlight = 8

	// The pod joins and accumulates several in-flight requests.
	epCrashed := newStubSchedulingEndpoint(endpointName)
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type:     datalayer.EventAddOrUpdate,
		Endpoint: epCrashed,
	}))
	reqs := make([]*fwksched.InferenceRequest, inFlight)
	results := make([]*fwksched.SchedulingResult, inFlight)
	for i := 0; i < inFlight; i++ {
		reqs[i] = makeTokenRequest(fmt.Sprintf("crash-req-%d", i), 4)
		results[i] = makeSchedulingResult(endpointName)
		producer.PreRequest(ctx, reqs[i], results[i])
	}
	require.Equal(t, int64(inFlight), producer.requestTracker.get(endpointID))

	// The pod crashes: the endpoint is removed, dropping the counter that held
	// all in-flight requests.
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type:     datalayer.EventDelete,
		Endpoint: epCrashed,
	}))

	// The pod restarts and rejoins; a fresh request lands on a new counter.
	epNew := newStubSchedulingEndpoint(endpointName)
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type:     datalayer.EventAddOrUpdate,
		Endpoint: epNew,
	}))
	reqNew := makeTokenRequest("post-crash-req", 4)
	resNew := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, reqNew, resNew)
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))

	// The crashed requests drain late. Each release must hit the orphaned counter, so the live
	// counter holds only the post-crash request throughout.
	for i := 0; i < inFlight; i++ {
		reqs[i].SchedulingResult = results[i]
		producer.ResponseBody(ctx, reqs[i], &requestcontrol.Response{EndOfStream: true}, nil)
		require.Equal(t, int64(1), producer.requestTracker.get(endpointID),
			"crashed request's release must hit the orphan, not the live counter")
	}

	// Only the post-crash request remains in flight.
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))

	reqNew.SchedulingResult = resNew
	producer.ResponseBody(ctx, reqNew, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
}

// TestInFlightLoadProducer_StaleDeleteIgnored verifies the registeredEndpoints
// guard: a delete carrying a different Endpoint object than the one currently
// registered (an out-of-order delete for an already-replaced pod) must NOT drop
// the live counter, while the matching delete still cleans up.
func TestInFlightLoadProducer_StaleDeleteIgnored(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "stale-delete-endpoint"
	endpointID := fullEndpointName(endpointName)

	// The current generation of the endpoint is registered and carries load.
	current := newStubSchedulingEndpoint(endpointName)
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type:     datalayer.EventAddOrUpdate,
		Endpoint: current,
	}))
	producer.requestTracker.add(endpointID, 3)

	// A stale delete for a previous, already-replaced object (different pointer,
	// same NamespacedName) arrives out of order. It must be ignored.
	stale := newStubSchedulingEndpoint(endpointName)
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type:     datalayer.EventDelete,
		Endpoint: stale,
	}))
	require.Equal(t, int64(3), producer.requestTracker.get(endpointID),
		"stale delete for a replaced endpoint must not drop the live counter")

	// The matching delete (the registered object) does clean up.
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type:     datalayer.EventDelete,
		Endpoint: current,
	}))
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
}

func TestInFlightLoadProducer_ConcurrencyStress(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "stress-endpoint"
	endpointID := fullEndpointName(endpointName)

	const (
		numGoroutines = 50
		opsPerRoutine = 100
	)

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(g int) {
			defer wg.Done()
			for j := 0; j < opsPerRoutine; j++ {
				reqID := fmt.Sprintf("req-%d-%d", g, j)
				res := makeSchedulingResult(endpointName)
				req := &fwksched.InferenceRequest{RequestID: reqID, SchedulingResult: res}

				producer.PreRequest(ctx, req, res)
				producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
			}
		}(i)
	}

	wg.Wait()

	require.Equal(t, int64(0), producer.requestTracker.get(endpointID), "request count drift detected")
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID), "token count drift detected")
}

// --- Helpers ---

func fullEndpointName(name string) string {
	return types.NamespacedName{Name: name, Namespace: "default"}.String()
}

func makeSchedulingResult(endpointName string) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {
				TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(endpointName)},
			},
		},
	}
}

type stubSchedulingEndpoint struct {
	fwksched.Endpoint
	metadata *datalayer.EndpointMetadata
	attr     datalayer.AttributeMap
}

func newStubSchedulingEndpoint(name string) *stubSchedulingEndpoint {
	return &stubSchedulingEndpoint{
		metadata: &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
		attr:     datalayer.NewAttributes(),
	}
}

func (f *stubSchedulingEndpoint) GetMetadata() *datalayer.EndpointMetadata   { return f.metadata }
func (f *stubSchedulingEndpoint) UpdateMetadata(*datalayer.EndpointMetadata) {}
func (f *stubSchedulingEndpoint) GetMetrics() *datalayer.Metrics             { return nil }
func (f *stubSchedulingEndpoint) UpdateMetrics(*datalayer.Metrics)           {}
func (f *stubSchedulingEndpoint) GetAttributes() datalayer.AttributeMap      { return f.attr }
func (f *stubSchedulingEndpoint) String() string                             { return "" }
func (f *stubSchedulingEndpoint) Put(key string, val datalayer.Cloneable)    { f.attr.Put(key, val) }
func (f *stubSchedulingEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	return f.attr.Get(key)
}
func (f *stubSchedulingEndpoint) Keys() []string { return f.attr.Keys() }

// makeTokenRequest builds a request whose tokenized prompt carries inputTokens token IDs,
// which is what the estimator reads to derive the input token count.
func makeTokenRequest(requestID string, inputTokens int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: requestID,
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{make([]uint32, inputTokens)},
			},
		},
	}
}

// TestInFlightLoadProducer_ExcludeOutputTokens_StartOfStreamRelease verifies that when
// AddEstimatedOutputTokens is false, token counters are released as soon as the first chunk
// arrives (StartOfStream), while request counters are released only on EndOfStream.
func TestInFlightLoadProducer_ExcludeOutputTokens_StartOfStreamRelease(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	producer.addEstimatedOutputTokens = false
	ctx := context.Background()
	endpointName := "exclude-output-endpoint"
	endpointID := fullEndpointName(endpointName)

	// 4 input tokens. Output tokens are excluded.
	req := makeTokenRequest("req-no-output", 4)
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(4), producer.tokenTracker.get(endpointID), "only input tokens should be tracked")

	// First chunk arrives: tokens released, request still in flight.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true}, nil)
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID), "request counter should still be held")
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID), "tokens should be released at StartOfStream")

	// EndOfStream releases the request counter.
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

// TestInFlightLoadProducer_ExcludeOutputTokens_SingleChunk verifies that a single-chunk
// response (StartOfStream && EndOfStream both true) releases both tokens and the request.
func TestInFlightLoadProducer_ExcludeOutputTokens_SingleChunk(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	producer.addEstimatedOutputTokens = false
	ctx := context.Background()
	endpointName := "single-chunk-endpoint"
	endpointID := fullEndpointName(endpointName)

	req := makeTokenRequest("req-single", 4)
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(4), producer.tokenTracker.get(endpointID))

	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true, EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

// TestInFlightLoadProducer_PrefixCacheDiscount verifies that when PrefixCacheMatchInfo
// is published on the endpoint, the literal cached prefix is excluded from the tracked
// input tokens, and that release subtracts the same (discounted) amount.
func TestInFlightLoadProducer_PrefixCacheDiscount(t *testing.T) {
	t.Parallel()

	// Setup the tracker database and context.
	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "prefix-cache-endpoint"
	endpointID := fullEndpointName(endpointName)

	// 8 input tokens. Output = 8 * 1.5 = 12.
	// With block_size=4 and total=2 blocks, the tier-weighted match is 1 block,
	// but both blocks are cached. The tracked total therefore contains only output.
	endpoint := newStubSchedulingEndpoint(endpointName)
	endpoint.Put(
		attrprefix.PrefixCacheMatchInfoDataKey.String(),
		attrprefix.NewPrefixCacheMatchInfo(1, 2, 4).WithCachedBlockCount(2),
	)

	// Create an 8-token input request (with 12 estimated output tokens).
	req := makeTokenRequest("req-prefix", 8)
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}

	// Simulate dispatch (PreRequest Phase) adding request weight to trackers.
	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(12), producer.tokenTracker.get(endpointID),
		"only output tokens should be tracked for a fully cached input")

	// Release uses the exact stored value, returning to zero.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID),
		"release should subtract the same discounted amount that was added")
}

// TestInFlightLoadProducer_PrefixCacheDiscount_PerEndpoint verifies that two profiles
// targeting different endpoints with different prefix-cache match levels each get their
// own discounted token amount, and that both counters return to zero after release.
func TestInFlightLoadProducer_PrefixCacheDiscount_PerEndpoint(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	podA := "pod-a-cached"
	podB := "pod-b-uncached"
	idA := fullEndpointName(podA)
	idB := fullEndpointName(podB)

	// 8 input tokens, output 12.
	epA := newStubSchedulingEndpoint(podA)
	epA.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(2, 2, 4)) // fully cached
	epB := newStubSchedulingEndpoint(podB)
	epB.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(0, 2, 4)) // none cached

	req := makeTokenRequest("req-multi-cache", 8)
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{epA}},
			"decode":  {TargetEndpoints: []fwksched.Endpoint{epB}},
		},
	}

	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(0+12), producer.tokenTracker.get(idA), "fully cached: only output tokens")
	require.Equal(t, int64(8+12), producer.tokenTracker.get(idB), "uncached: input + output")

	// Drive the response lifecycle: StartOfStream releases prefill, EndOfStream releases decode.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true}, nil)
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.tokenTracker.get(idA))
	require.Equal(t, int64(0), producer.tokenTracker.get(idB))
	require.Equal(t, int64(0), producer.requestTracker.get(idA))
	require.Equal(t, int64(0), producer.requestTracker.get(idB))
}

// TestInFlightLoadProducer_BalancedAddRelease_MultipleProfilesSameEndpoint verifies that
// when multiple profiles target the same endpoint, each contributes to the counters
// independently and each release subtracts the exact added amount, returning counters
// to their pre-request baseline.
func TestInFlightLoadProducer_BalancedAddRelease_MultipleProfilesSameEndpoint(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "shared-endpoint"
	endpointID := fullEndpointName(endpointName)

	// 4 input tokens, 6 output, total 10 tokens per profile.
	// Two profiles both targeting the same endpoint => 2 requests, 20 tokens.
	req := makeTokenRequest("req-shared", 4)
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(endpointName)}},
			"decode":  {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(endpointName)}},
		},
	}

	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(2), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(20), producer.tokenTracker.get(endpointID))

	// StartOfStream releases the prefill profile only (1 request, 10 tokens).
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true}, nil)
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(10), producer.tokenTracker.get(endpointID))

	// EndOfStream releases the remaining (decode) profile.
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID),
		"counters must return to zero with no drift across profiles")
}

// TestInFlightLoadProducer_ExcludeOutputTokens_EndOfStreamWithoutStart verifies the
// safety net for non-streaming or error paths: when addEstimatedOutputTokens=false and
// ResponseBody delivers EndOfStream without ever seeing StartOfStream, the token
// counter and request counter must both drain (tokens are normally released at
// StartOfStream, so a missing StartOfStream would otherwise leak them).
func TestInFlightLoadProducer_ExcludeOutputTokens_EndOfStreamWithoutStart(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	producer.addEstimatedOutputTokens = false
	ctx := context.Background()
	endpointName := "no-start-endpoint"
	endpointID := fullEndpointName(endpointName)

	req := makeTokenRequest("req-no-start", 4) // 4 input tokens
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint(endpointName)}},
		},
	}

	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(4), producer.tokenTracker.get(endpointID))

	// EndOfStream only (no StartOfStream): both counters must drain.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID),
		"tokens must be released on EndOfStream even if StartOfStream was never seen")

	// PluginState entry should be gone too (no leak).
	key := fwkplugin.StateKey(addedTokensKey(endpointID, "default"))
	_, err := producer.PluginState.Read(req.RequestID, key)
	require.ErrorIs(t, err, fwkplugin.ErrNotFound, "PluginState entry must be released")
}

// TestInFlightLoadProducer_Eviction verifies that global counters are rolled back
// when a request is explicitly deleted from PluginState (simulating either
// end-of-stream cleanup or janitor reaping).
func TestInFlightLoadProducer_Eviction(t *testing.T) {
	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "eviction-endpoint"
	endpointID := fullEndpointName(endpointName)

	// 1. PreRequest: Adds load
	req := makeTokenRequest("req-eviction", 4) // 4 input + 6 output = 10 tokens
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)

	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(10), producer.tokenTracker.get(endpointID))

	// 2. Explicitly delete the request (simulates what the janitor or EOS cleanup does).
	producer.PluginState.Delete(req.RequestID)

	// 3. Verify counters rolled back automatically via OnEvicted callback
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID), "request counter should have rolled back via Eviction")
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID), "token counter should have rolled back via Eviction")
}

// TestInFlightLoadProducer_Touch verifies that intermediate chunks extend the
// request's lifetime in PluginState.
func TestInFlightLoadProducer_Touch(t *testing.T) {
	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "touch-endpoint"

	req := makeTokenRequest("req-touch", 4)
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)

	// Get initial access time
	t1, ok := producer.PluginState.LastAccessTime(req.RequestID)
	require.True(t, ok)

	// Simulate intermediate chunks until access time is updated.
	// We use Eventually to handle coarse timer resolution or busy CI runners.
	require.Eventually(t, func() bool {
		req.SchedulingResult = res
		producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: false, StartOfStream: false}, nil)

		t2, ok := producer.PluginState.LastAccessTime(req.RequestID)
		return ok && t2.After(t1)
	}, 1*time.Second, 10*time.Millisecond, "Touch should have extended the lifetime")
}

// TestInFlightLoadProducer_LateResponseAfterReap verifies that if a ResponseBody
// arrives after the janitor has already reaped the request, we do NOT double-decrement.
func TestInFlightLoadProducer_LateResponseAfterReap(t *testing.T) {
	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "late-endpoint"
	endpointID := fullEndpointName(endpointName)

	req := makeTokenRequest("req-late", 4) // 4 input + 6 output = 10 tokens
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)

	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(10), producer.tokenTracker.get(endpointID))

	// Simulate janitor reap
	producer.PluginState.Delete(req.RequestID)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID), "counters should be 0 after reap")

	// Late ResponseBody arrives
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)

	// Verify no double-decrement
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID), "counters should NOT go negative")
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

func TestInFlightLoadProducer_AtomicTokenRelease_Concurrent(t *testing.T) {
	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "race-endpoint"
	endpointID := fullEndpointName(endpointName)

	req := makeTokenRequest("req-race", 4) // 4 input + 6 output = 10 tokens
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(10), producer.tokenTracker.get(endpointID))

	// Fire releaseTokensEarly and an explicit Delete concurrently. Whichever
	// wins the Swap does the -10; the other is a no-op. Net must be 0.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		producer.releaseTokensEarly(res.ProfileResults["default"].TargetEndpoints[0], req, "default")
	}()
	go func() {
		defer wg.Done()
		producer.PluginState.Delete(req.RequestID)
	}()
	wg.Wait()

	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
}

func TestUncachedInputTokens(t *testing.T) {
	t.Parallel()

	const key = "prefix-match-info"
	tests := []struct {
		name        string
		inputTokens int64
		info        datalayer.Cloneable
		noEndpoint  bool
		want        int64
	}{
		// Case: Missing/nil components should default back to treating the entire prompt as uncached.
		{name: "nil endpoint", inputTokens: 8, noEndpoint: true, want: 8},
		{name: "negative input without endpoint", inputTokens: -1, noEndpoint: true, want: 0},
		{name: "missing prefix information", inputTokens: 8, want: 8},
		{name: "wrong prefix information type", inputTokens: 8, info: &attrconcurrency.InFlightLoad{}, want: 8},
		// Case: Literal count differs from weighted match (our big fix for offloaded caches).
		{
			name:        "literal count differs from weighted match",
			inputTokens: 4096,
			info: attrprefix.NewPrefixCacheMatchInfo(204, 256, 16).
				WithCachedBlockCount(256),
			want: 0,
		},
		// Case: Partially cached precise prefix.
		{
			name:        "partially cached precise prefix",
			inputTokens: 4096,
			info: attrprefix.NewPrefixCacheMatchInfo(128, 256, 16).
				WithCachedBlockCount(192),
			want: 1024,
		},
		// Case: Unindexed tail (prompt length exceeds maximum indexed range).
		{
			name:        "unindexed tail",
			inputTokens: 4096,
			info: attrprefix.NewPrefixCacheMatchInfo(128, 128, 16).
				WithCachedBlockCount(128),
			want: 2048,
		},
		{
			name:        "partial final block",
			inputTokens: 10,
			info:        attrprefix.NewPrefixCacheMatchInfo(1, 2, 4),
			want:        6,
		},
		// Case: Backward compatibility test (defaults cached count to match blocks).
		{
			name:        "approximate producer compatibility",
			inputTokens: 4096,
			info:        attrprefix.NewPrefixCacheMatchInfo(128, 256, 16),
			want:        2048,
		},
		{
			name:        "invalid block size",
			inputTokens: 8,
			info:        attrprefix.NewPrefixCacheMatchInfo(1, 2, 0),
			want:        8,
		},
		// Case: Clamped bounds safety check.
		{
			name:        "cached count above total blocks",
			inputTokens: 8,
			info: attrprefix.NewPrefixCacheMatchInfo(2, 2, 4).
				WithCachedBlockCount(3),
			want: 0,
		},
		{
			name:        "prefix count trusted over smaller estimate",
			inputTokens: 5,
			info:        attrprefix.NewPrefixCacheMatchInfo(1, 2, 4),
			want:        4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var endpoint fwksched.Endpoint
			if !tt.noEndpoint {
				stub := newStubSchedulingEndpoint("test-endpoint")
				if tt.info != nil {
					stub.Put(key, tt.info)
				}
				endpoint = stub
			}

			require.Equal(t, tt.want, uncachedInputTokens(endpoint, tt.inputTokens, key))
		})
	}
}

func TestInFlightLoadProducer_PanicSafety(t *testing.T) {
	producer := newTestProducer(t)
	ctx := context.Background()

	t.Run("ExtractEndpoint", func(t *testing.T) {
		// 1. Nil Endpoint
		require.NotPanics(t, func() {
			_ = producer.Extract(ctx, datalayer.EndpointEvent{Type: datalayer.EventDelete, Endpoint: nil})
		})

		// 2. Nil Metadata
		stub := newStubSchedulingEndpoint("nil-meta")
		stub.metadata = nil
		require.NotPanics(t, func() {
			_ = producer.Extract(ctx, datalayer.EndpointEvent{Type: datalayer.EventDelete, Endpoint: stub})
		})
	})

	t.Run("Produce", func(t *testing.T) {
		// 1. Nil Endpoints slice
		require.NotPanics(t, func() {
			_ = producer.Produce(ctx, nil, nil)
		})

		// 2. Slice with nil endpoint
		require.NotPanics(t, func() {
			_ = producer.Produce(ctx, nil, []fwksched.Endpoint{nil})
		})

		// 3. Endpoint with nil metadata
		stub := newStubSchedulingEndpoint("nil-meta")
		stub.metadata = nil
		require.NotPanics(t, func() {
			_ = producer.Produce(ctx, nil, []fwksched.Endpoint{stub})
		})
	})

	t.Run("PreRequest", func(t *testing.T) {
		// 1. Nil Result
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, nil, nil)
		})

		// 2. Nil Request, non-nil Result
		res := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint("ep1")}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, nil, res)
		})
		require.Equal(t, int64(0), producer.requestTracker.get(fullEndpointName("ep1")), "should not increment counters without request")

		// 3. Empty ProfileResults
		resEmpty := &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{}}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{}, resEmpty)
		})

		// 4. Nil ProfileResult
		resNilProfile := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{"default": nil},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "req1"}, resNilProfile)
		})

		// 5. Empty TargetEndpoints
		resEmptyEndpoints := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "req1"}, resEmptyEndpoints)
		})

		// 6. Nil Endpoint in TargetEndpoints
		resNilEndpoint := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{nil}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "req1"}, resNilEndpoint)
		})

		// 7. Endpoint with nil metadata
		stub := newStubSchedulingEndpoint("nil-meta")
		stub.metadata = nil
		resNilMeta := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{stub}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "req1"}, resNilMeta)
		})

		// 8. Missing RequestID (Leak check)
		resLeak := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint("ep-leak")}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: ""}, resLeak)
		})
		require.Equal(t, int64(0), producer.requestTracker.get(fullEndpointName("ep-leak")), "should not increment counters with empty RequestID")
	})

	t.Run("ResponseBody", func(t *testing.T) {
		// 1. Nil Request or Response
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, nil, nil, nil)
		})
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, &fwksched.InferenceRequest{}, nil, nil)
		})
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, nil, &requestcontrol.Response{}, nil)
		})

		// 2. Nil SchedulingResult
		reqNoRes := &fwksched.InferenceRequest{SchedulingResult: nil}
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, reqNoRes, &requestcontrol.Response{}, nil)
		})

		// 3. Various nil components in result
		resNilProfile := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{"default": nil},
		}
		reqNilProfile := &fwksched.InferenceRequest{SchedulingResult: resNilProfile}
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, reqNilProfile, &requestcontrol.Response{EndOfStream: true}, nil)
		})

		resNilEndpoint := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{nil}},
			},
		}
		reqNilEndpoint := &fwksched.InferenceRequest{SchedulingResult: resNilEndpoint}
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, reqNilEndpoint, &requestcontrol.Response{EndOfStream: true}, nil)
		})
	})

	t.Run("Factory_NilHandle", func(t *testing.T) {
		p, err := InFlightLoadProducerFactory("test", nil, nil)
		require.Error(t, err)
		require.Nil(t, p)
	})
}

func TestInFlightLoadProducerFactory_OutputRatio(t *testing.T) {
	t.Parallel()

	newProducer := func(t *testing.T, cfg Config) (*InFlightLoadProducer, error) {
		raw, err := json.Marshal(cfg)
		require.NoError(t, err)
		ctx, cancel := context.WithCancel(context.Background())
		t.Cleanup(cancel)
		p, err := InFlightLoadProducerFactory("inflight-load-producer",
			json.NewDecoder(bytes.NewReader(raw)), testutils.NewTestHandle(ctx))
		if err != nil {
			return nil, err
		}
		return p.(*InFlightLoadProducer), nil
	}

	t.Run("default when unset", func(t *testing.T) {
		t.Parallel()
		p, err := newProducer(t, Config{AddEstimatedOutputTokens: true})
		require.NoError(t, err)
		require.Equal(t, int64(15), p.tokenEstimator.EstimateOutput(10, nil)) // 10 * 1.5
	})

	t.Run("custom ratio applied", func(t *testing.T) {
		t.Parallel()
		p, err := newProducer(t, Config{AddEstimatedOutputTokens: true, OutputRatio: ptr.To(2.0)})
		require.NoError(t, err)
		require.Equal(t, int64(20), p.tokenEstimator.EstimateOutput(10, nil)) // 10 * 2.0
	})

	t.Run("zero ratio is valid", func(t *testing.T) {
		t.Parallel()
		p, err := newProducer(t, Config{AddEstimatedOutputTokens: true, OutputRatio: ptr.To(0.0)})
		require.NoError(t, err)
		require.Equal(t, int64(0), p.tokenEstimator.EstimateOutput(10, nil))
	})

	t.Run("negative ratio rejected", func(t *testing.T) {
		t.Parallel()
		_, err := newProducer(t, Config{AddEstimatedOutputTokens: true, OutputRatio: ptr.To(-1.0)})
		require.Error(t, err)
	})
}

func TestInFlightLoadProducerFactory_MaxEstimatedOutputTokens(t *testing.T) {
	t.Parallel()

	newProducer := func(t *testing.T, cfg Config) (*InFlightLoadProducer, error) {
		raw, err := json.Marshal(cfg)
		require.NoError(t, err)
		ctx, cancel := context.WithCancel(context.Background())
		t.Cleanup(cancel)
		p, err := InFlightLoadProducerFactory("inflight-load-producer",
			json.NewDecoder(bytes.NewReader(raw)), testutils.NewTestHandle(ctx))
		if err != nil {
			return nil, err
		}
		return p.(*InFlightLoadProducer), nil
	}

	t.Run("operator cap bounds the estimate", func(t *testing.T) {
		t.Parallel()
		p, err := newProducer(t, Config{AddEstimatedOutputTokens: true, MaxEstimatedOutputTokens: ptr.To(int64(40))})
		require.NoError(t, err)
		// 100 * 1.5 = 150, capped to 40.
		require.Equal(t, int64(40), p.tokenEstimator.EstimateOutput(100, nil))
	})

	t.Run("estimate below cap is unaffected", func(t *testing.T) {
		t.Parallel()
		p, err := newProducer(t, Config{AddEstimatedOutputTokens: true, MaxEstimatedOutputTokens: ptr.To(int64(1000))})
		require.NoError(t, err)
		require.Equal(t, int64(150), p.tokenEstimator.EstimateOutput(100, nil))
	})

	t.Run("negative cap rejected", func(t *testing.T) {
		t.Parallel()
		_, err := newProducer(t, Config{AddEstimatedOutputTokens: true, MaxEstimatedOutputTokens: ptr.To(int64(-1))})
		require.Error(t, err)
	})
}
