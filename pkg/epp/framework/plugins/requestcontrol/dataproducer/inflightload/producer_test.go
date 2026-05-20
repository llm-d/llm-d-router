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

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

func newTestProducer(t testing.TB) *InFlightLoadProducer {
	params := Config{AddEstimatedOutputTokens: true}
	raw, err := json.Marshal(params)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	decoder := json.NewDecoder(bytes.NewReader(raw))
	p, err := InFlightLoadProducerFactory("inflight-load-producer", decoder, igwtestutils.NewTestHandle(ctx))
	require.NoError(t, err)
	return p.(*InFlightLoadProducer)
}

func TestInFlightLoadProducer_Produce(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)

	endpointName := "test-endpoint"
	endpointID := fullEndpointName(endpointName)

	// Mock some initial load
	producer.requestTracker.add(endpointID, 5)
	producer.tokenTracker.add(endpointID, 500)

	ctx := context.Background()
	endpoints := []fwksched.Endpoint{newStubSchedulingEndpoint(endpointName)}

	err := producer.Produce(ctx, nil, endpoints)
	require.NoError(t, err)

	// Verify AttributeMap population
	key := producer.dk.String()
	val, ok := endpoints[0].Get(key)
	require.True(t, ok)
	load := val.(*attrconcurrency.InFlightLoad)
	require.Equal(t, int64(5), load.Requests)
	require.Equal(t, int64(500), load.Tokens)
}

func TestInFlightLoadProducer_Lifecycle(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "lifecycle-endpoint"
	endpointID := fullEndpointName(endpointName)

	// 1. PreRequest (Inc)
	req := makeTokenRequest("req1", "1234567890123456") // 16 chars / 4 = 4 input + 6 output = 10 tokens
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
	req := makeTokenRequest("multi-req", "1234567890123456") // 10 tokens
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

func makeTokenRequest(requestID, prompt string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: requestID,
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
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

	// 16 chars / 4 = 4 input tokens. Output tokens are excluded.
	req := makeTokenRequest("req-no-output", "1234567890123456")
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

	req := makeTokenRequest("req-single", "1234567890123456")
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(4), producer.tokenTracker.get(endpointID))

	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true, EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

// TestInFlightLoadProducer_PrefixCacheDiscount verifies that when PrefixCacheMatchInfo
// is published on the endpoint, the matched prefix is excluded from the tracked input
// tokens, and that release subtracts the same (discounted) amount.
func TestInFlightLoadProducer_PrefixCacheDiscount(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	ctx := context.Background()
	endpointName := "prefix-cache-endpoint"
	endpointID := fullEndpointName(endpointName)

	// Prompt: 32 chars / 4 = 8 input tokens. Output = 8 * 1.5 = 12.
	// With block_size=4, total=2 blocks, matched=1 block (4 tokens cached):
	//   uncached_input = (2-1)*4 + max(0, 8-2*4) = 4
	//   total tokens = 4 + 12 = 16
	endpoint := newStubSchedulingEndpoint(endpointName)
	endpoint.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(1, 2, 4))

	req := makeTokenRequest("req-prefix", "12345678901234567890123456789012")
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}

	producer.PreRequest(ctx, req, res)
	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(16), producer.tokenTracker.get(endpointID),
		"only uncached input (4) plus output (12) should be tracked")

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

	req := makeTokenRequest("req-multi-cache", "12345678901234567890123456789012")
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

	// 16 chars / 4 = 4 input tokens, 6 output, total 10 tokens per profile.
	// Two profiles both targeting the same endpoint => 2 requests, 20 tokens.
	req := makeTokenRequest("req-shared", "1234567890123456")
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

	req := makeTokenRequest("req-no-start", "1234567890123456") // 4 input tokens
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
	req := makeTokenRequest("req-eviction", "1234567890123456") // 10 tokens
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

	req := makeTokenRequest("req-touch", "1234567890123456")
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

	req := makeTokenRequest("req-late", "1234567890123456") // 10 tokens
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

	req := makeTokenRequest("req-race", "1234567890123456") // 10 tokens
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

func TestUncachedInputTokens_Overestimate(t *testing.T) {
	// Setup:
	// inputTokens (estimated) = 5
	// PrefixCacheMatchInfo: matchBlocks=1, totalBlocks=2, blockSizeTokens=4
	//   indexedTokens = 2 * 4 = 8
	//   matchedTokens = 1 * 4 = 4

	endpoint := newStubSchedulingEndpoint("test-ep")
	endpoint.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(1, 2, 4))

	inputTokens := int64(5)

	uncached := uncachedInputTokens(endpoint, inputTokens)

	// When the prefix cache says 4 tokens are definitely uncached in the indexed portion (8-4),
	// we trust that over the smaller (approximate) estimate of 5.
	require.Equal(t, int64(4), uncached, "should trust the prefix cache's uncached count (indexed-matched) over the smaller estimate")
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
