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

package requestcontrol

import (
	"context"
	"testing"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

const (
	benchChunksPerOp = 10
	// maxAllocsPerResponse is the allocation ceiling for one complete
	// response (benchChunksPerOp intermediate chunks + 1 end-of-stream).
	// Bump this intentionally when a change adds justified allocations.
	maxAllocsPerResponse = 60
)

func TestHandleResponseBodyAllocs(t *testing.T) {
	plugin := newTestResponseStreaming("alloc-plugin")
	director := NewDirectorWithConfig(nil, &mockScheduler{}, nil, nil,
		NewConfig().WithResponseStreamingPlugins(plugin))

	ctx := log.IntoContext(context.Background(), logr.Discard())

	avg := testing.AllocsPerRun(100, func() {
		reqCtx := &handlers.RequestContext{
			Request: &handlers.Request{
				Headers: map[string]string{
					reqcommon.RequestIDHeaderKey: "alloc-request",
				},
			},
			Response: &handlers.Response{
				Headers: map[string]string{},
			},
			TargetPod: &fwkdl.EndpointMetadata{
				ID: types.NamespacedName{Namespace: "ns", Name: "pod"},
			},
			Usage: fwkrh.Usage{},
		}
		for chunk := 0; chunk < benchChunksPerOp; chunk++ {
			director.HandleResponseBody(ctx, reqCtx, false)
		}
		director.HandleResponseBody(ctx, reqCtx, true)

		plugin.mu.Lock()
		plugin.respsOnStreaming = plugin.respsOnStreaming[:0]
		plugin.targetPodsOnStreaming = plugin.targetPodsOnStreaming[:0]
		plugin.mu.Unlock()
	})
	if avg > maxAllocsPerResponse {
		t.Errorf("HandleResponseBody allocations regressed: got %.0f, ceiling %d", avg, maxAllocsPerResponse)
	}
}

// BenchmarkHandleResponseBody measures the per-response cost of
// Director.HandleResponseBody with one streaming plugin registered.
// Each operation simulates 10 intermediate chunks followed by one
// end-of-stream chunk.
//
// Run:
//
//	go test -run='^$' -bench=BenchmarkHandleResponseBody -benchmem -count=10 \
//	    ./pkg/epp/requestcontrol/ | tee bench.out
//	benchstat bench.out
func BenchmarkHandleResponseBody(b *testing.B) {
	plugin := newTestResponseStreaming("bench-plugin")
	director := NewDirectorWithConfig(nil, &mockScheduler{}, nil, nil,
		NewConfig().WithResponseStreamingPlugins(plugin))

	ctx := log.IntoContext(context.Background(), logr.Discard())

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reqCtx := &handlers.RequestContext{
			Request: &handlers.Request{
				Headers: map[string]string{
					reqcommon.RequestIDHeaderKey: "bench-request",
				},
			},
			Response: &handlers.Response{
				Headers: map[string]string{},
			},
			TargetPod: &fwkdl.EndpointMetadata{
				ID: types.NamespacedName{Namespace: "ns", Name: "pod"},
			},
			Usage: fwkrh.Usage{},
		}

		for chunk := 0; chunk < benchChunksPerOp; chunk++ {
			director.HandleResponseBody(ctx, reqCtx, false)
		}
		// Wait for async queue to drain before the final synchronous chunk.
		director.HandleResponseBody(ctx, reqCtx, true)

		// Reset plugin state for the next iteration.
		plugin.mu.Lock()
		plugin.respsOnStreaming = plugin.respsOnStreaming[:0]
		plugin.targetPodsOnStreaming = plugin.targetPodsOnStreaming[:0]
		plugin.mu.Unlock()
	}
}
