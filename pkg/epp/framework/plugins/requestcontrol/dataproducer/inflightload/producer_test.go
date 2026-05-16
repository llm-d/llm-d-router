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
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestInFlightLoadProducer_Produce(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
	}

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
	val, ok := endpoints[0].Get(attrconcurrency.InFlightLoadKey)
	require.True(t, ok)
	load := val.(*attrconcurrency.InFlightLoad)
	require.Equal(t, int64(5), load.Requests)
	require.Equal(t, int64(500), load.Tokens)
}

func TestInFlightLoadProducer_Lifecycle(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: true,
	}
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

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: true,
	}
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

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
	}
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

	err := producer.ExtractEndpoint(ctx, eventEndpoint)
	require.NoError(t, err)

	// Verify Cleanup
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

func TestInFlightLoadProducer_ConcurrencyStress(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: true,
	}
	ctx := context.Background()
	endpointName := "stress-endpoint"
	endpointID := fullEndpointName(endpointName)

	const (
		numGoroutines = 50
		opsPerRoutine = 1000
	)

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	// Launch increments
	for range numGoroutines {
		go func() {
			defer wg.Done()
			res := makeSchedulingResult(endpointName)
			for range opsPerRoutine {
				producer.PreRequest(ctx, nil, res)
			}
		}()
	}

	// Launch decrements
	for range numGoroutines {
		go func() {
			defer wg.Done()
			res := makeSchedulingResult(endpointName)
			req := &fwksched.InferenceRequest{SchedulingResult: res}
			for range opsPerRoutine {
				producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
			}
		}()
	}

	wg.Wait()

	require.Equal(t, int64(0), producer.requestTracker.get(endpointID), "request count drift detected")
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
// IncludeOutputTokens is false, token counters are released as soon as the first chunk
// arrives (StartOfStream), while request counters are released only on EndOfStream.
func TestInFlightLoadProducer_ExcludeOutputTokens_StartOfStreamRelease(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: false,
	}
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

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: false,
	}
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

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: true,
	}
	ctx := context.Background()
	endpointName := "prefix-cache-endpoint"
	endpointID := fullEndpointName(endpointName)

	// Prompt: 32 chars / 4 = 8 input tokens. Output = 8 * 1.5 = 12.
	// With block_size=4, total=2 blocks, matched=1 block (4 tokens cached):
	//   uncached_input = (2-1)*4 + max(0, 8-2*4) = 4
	//   total tokens = 4 + 12 = 16
	endpoint := newStubSchedulingEndpoint(endpointName)
	endpoint.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(1, 2, 4))

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

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: true,
	}
	ctx := context.Background()
	podA := "pod-a-cached"
	podB := "pod-b-uncached"
	idA := fullEndpointName(podA)
	idB := fullEndpointName(podB)

	// 8 input tokens, output 12.
	epA := newStubSchedulingEndpoint(podA)
	epA.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(2, 2, 4)) // fully cached
	epB := newStubSchedulingEndpoint(podB)
	epB.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(0, 2, 4)) // none cached

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

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: true,
	}
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
// safety net for non-streaming or error paths: when includeOutputTokens=false and
// ResponseBody delivers EndOfStream without ever seeing StartOfStream, the token
// counter and request counter must both drain (tokens are normally released at
// StartOfStream, so a missing StartOfStream would otherwise leak them). Also
// asserts the addedTokens map entry is removed so accounting stays balanced.
func TestInFlightLoadProducer_ExcludeOutputTokens_EndOfStreamWithoutStart(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: false,
	}
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

	// addedTokens entry should be gone too (no leak).
	_, loaded := producer.addedTokens.Load(addedTokensKey(req.RequestID, endpointID, "default"))
	require.False(t, loaded, "addedTokens entry must be released")
}
