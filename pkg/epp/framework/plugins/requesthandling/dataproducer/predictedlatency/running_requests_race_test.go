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

package predictedlatency

import (
	"context"
	"fmt"
	"sync"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// A request dispatched to an endpoint while that endpoint's last running
// request completes must still be counted as running on the endpoint.
func TestPredictedLatency_RunningRequestCountSurvivesConcurrentCompletion(t *testing.T) {
	tests := []struct {
		name        string
		newRequests int
		iterations  int
	}{
		{name: "one new request while last running request completes", newRequests: 1, iterations: 3000},
		{name: "many new requests while last running request completes", newRequests: 8, iterations: 1500},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			router := createTestRouter()
			router.latencypredictor = new(mockPredictor)
			endpoint := createTestEndpoint("race-pod", 0.5, 1, 0)
			result := createTestSchedulingResult(endpoint.GetMetadata())

			start := func(id string) *fwksched.InferenceRequest {
				req := createTestInferenceRequest(id, 100, 50)
				router.setPredictedLatencyContextForRequest(req, newPredictedLatencyContext(req))
				return req
			}
			finish := func(req *fwksched.InferenceRequest) {
				router.ResponseBody(ctx, req, &fwkrh.Response{EndOfStream: true}, endpoint.GetMetadata())
			}

			for i := 0; i < tt.iterations; i++ {
				old := start(fmt.Sprintf("old-%d", i))
				if err := router.PreRequest(ctx, old, result); err != nil {
					t.Fatalf("iteration %d: PreRequest(old): %v", i, err)
				}
				news := make([]*fwksched.InferenceRequest, tt.newRequests)
				for j := range news {
					news[j] = start(fmt.Sprintf("new-%d-%d", i, j))
				}

				gate := make(chan struct{})
				var wg sync.WaitGroup
				wg.Add(1 + len(news))
				go func() { defer wg.Done(); <-gate; finish(old) }()
				for _, req := range news {
					go func() { defer wg.Done(); <-gate; _ = router.PreRequest(ctx, req, result) }()
				}
				close(gate)
				wg.Wait()

				if got := router.getEndpointRunningRequestCount(endpoint); got != tt.newRequests {
					t.Fatalf("iteration %d: running request count = %d after old request completed, want %d in-flight new requests", i, got, tt.newRequests)
				}
				for _, req := range news {
					finish(req)
				}
				if got := router.getEndpointRunningRequestCount(endpoint); got != 0 {
					t.Fatalf("iteration %d: running request count = %d after all requests completed, want 0", i, got)
				}
			}
		})
	}
}
