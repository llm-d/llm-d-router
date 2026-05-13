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
