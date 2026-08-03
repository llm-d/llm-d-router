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

	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	crzap "sigs.k8s.io/controller-runtime/pkg/log/zap"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

const (
	benchChunksPerOp = 10
	// maxAllocsPerResponse caps allocations per response; a failure is a
	// regression to investigate, not a number to raise.
	maxAllocsPerResponse = 65
)

// newStreamingDirectorFixture builds a Director with one streaming plugin and
// a production-verbosity logger (DEBUG/TRACE disabled) so the benchmark
// measures the allocation cost production actually pays.
func newStreamingDirectorFixture(name string) (*Director, *testResponseStreaming, context.Context) {
	plugin := newTestResponseStreaming(name)
	director := NewDirectorWithConfig(nil, &mockScheduler{}, nil, nil,
		NewConfig().WithResponseStreamingPlugins(plugin))
	// Info level leaves V(DEBUG) and V(TRACE) disabled, as in production. A
	// discarding logger would hide the cost of deriving one, which is exactly
	// what this benchmark exists to track.
	logger := crzap.New(crzap.Level(uberzap.NewAtomicLevelAt(zapcore.InfoLevel)))
	ctx := log.IntoContext(context.Background(), logger)
	return director, plugin, ctx
}

func newStreamingRequestContext(requestID string) *handlers.RequestContext {
	return &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{reqcommon.RequestIDHeaderKey: requestID},
		},
		Response: &handlers.Response{
			Headers: map[string]string{},
		},
		TargetPod: &fwkdl.EndpointMetadata{
			ID: types.NamespacedName{Namespace: "ns", Name: "pod"},
		},
		Usage: fwkrh.Usage{},
	}
}

// streamOneResponse runs one full response and resets state for reuse.
func streamOneResponse(ctx context.Context, director *Director, reqCtx *handlers.RequestContext, plugin *testResponseStreaming) {
	for chunk := 0; chunk < benchChunksPerOp; chunk++ {
		director.HandleResponseBody(ctx, reqCtx, false)
	}
	director.HandleResponseBody(ctx, reqCtx, true)

	reqCtx.ResponseBodyStarted = false
	plugin.mu.Lock()
	plugin.respsOnStreaming = plugin.respsOnStreaming[:0]
	plugin.targetPodsOnStreaming = plugin.targetPodsOnStreaming[:0]
	plugin.mu.Unlock()
}

func TestHandleResponseBodyAllocs(t *testing.T) {
	director, plugin, ctx := newStreamingDirectorFixture("alloc-plugin")
	reqCtx := newStreamingRequestContext("alloc-request")

	avg := testing.AllocsPerRun(100, func() {
		streamOneResponse(ctx, director, reqCtx, plugin)
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
	director, plugin, ctx := newStreamingDirectorFixture("bench-plugin")
	reqCtx := newStreamingRequestContext("bench-request")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamOneResponse(ctx, director, reqCtx, plugin)
	}
}
