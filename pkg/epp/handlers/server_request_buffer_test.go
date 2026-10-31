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

package handlers

import (
	"bytes"
	"context"
	"io"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"weak"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// inFlightProcessServer replays reqs and then blocks in Recv until finish is
// called, which keeps Process in the response phase of the stream.
type inFlightProcessServer struct {
	mockProcessServer
	ctx       context.Context
	reqs      []*extProcPb.ProcessingRequest
	forwarded chan struct{}
	release   chan struct{}
	done      chan error
}

func newInFlightProcessServer(reqs []*extProcPb.ProcessingRequest) *inFlightProcessServer {
	return &inFlightProcessServer{
		ctx:       log.IntoContext(context.Background(), logr.Discard()),
		reqs:      reqs,
		forwarded: make(chan struct{}),
		release:   make(chan struct{}),
		done:      make(chan error, 1),
	}
}

func (m *inFlightProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(m.reqs) == 0 {
		<-m.release
		return nil, io.EOF
	}
	req := m.reqs[0]
	m.reqs = m.reqs[1:]
	return req, nil
}

func (m *inFlightProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	if resp.GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetEndOfStream() {
		close(m.forwarded)
	}
	return nil
}

func (m *inFlightProcessServer) Context() context.Context { return m.ctx }

// start runs Process and waits until the request body is forwarded to the proxy.
func (m *inFlightProcessServer) start(tb testing.TB, server *StreamingServer) {
	tb.Helper()
	go func() { m.done <- server.Process(m) }()
	select {
	case <-m.forwarded:
	case err := <-m.done:
		tb.Fatalf("Process returned before forwarding the request body: %v", err)
	}
}

// finish ends the stream and returns the result of Process.
func (m *inFlightProcessServer) finish() error {
	close(m.release)
	return <-m.done
}

// requestMessages returns request headers for path followed by body split into
// chunks of chunkSize.
func requestMessages(path string, body []byte, chunkSize int) []*extProcPb.ProcessingRequest {
	reqs := []*extProcPb.ProcessingRequest{newRequestHeaders(map[string]string{":path": path})}
	for start := 0; start < len(body); start += chunkSize {
		end := min(start+chunkSize, len(body))
		reqs = append(reqs, &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_RequestBody{
				RequestBody: &extProcPb.HttpBody{Body: body[start:end], EndOfStream: end == len(body)},
			},
		})
	}
	return reqs
}

// chatCompletionsBody returns a chat completions request of size bytes.
func chatCompletionsBody(size int) []byte {
	const prefix, suffix = `{"model":"m","messages":[{"role":"user","content":"`, `"}]}`
	return []byte(prefix + strings.Repeat("a", size-len(prefix)-len(suffix)) + suffix)
}

func TestProcessReleasesRequestBufferAfterRequestPhase(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`)
	tests := []struct {
		name      string
		telemetry bool
		wantHeld  bool
	}{
		{name: "released once the request body is forwarded"},
		{name: "held while the tool-calling snapshot references it", telemetry: true, wantHeld: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			server := NewStreamingServer(nil, &mockDirector{}, registry, 0)
			server.SetToolCallingTelemetryEnabled(tt.telemetry)
			var data weak.Pointer[byte]
			server.bufferPool.New = func() any {
				backing := make([]byte, 0, 4<<10)
				data = weak.Make(&backing[:1][0])
				return bytes.NewBuffer(backing)
			}

			srv := newInFlightProcessServer(requestMessages(reqcommon.PathChatCompletions, body, len(body)))
			srv.start(t, server)
			// sync.Pool drops unused items within two collections.
			for range 3 {
				runtime.GC()
			}
			require.Equal(t, tt.wantHeld, data.Value() != nil, "request buffer reachable while the stream waits for the response")
			require.NoError(t, srv.finish())
		})
	}
}

// retainingDirector keeps the parsed body on the request, as the Director does
// for scheduling.
type retainingDirector struct{ mockDirector }

func (d *retainingDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, body *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.SchedulingRequest = &fwksched.InferenceRequest{Body: body}
	return reqCtx, nil
}

// BenchmarkProcessInFlightRetainedHeap reports the heap each stream keeps live
// while it waits for the model server's response. Streams start one at a time,
// as requests arriving over time do; ns/op covers the request phase of all
// streams in an iteration.
func BenchmarkProcessInFlightRetainedHeap(b *testing.B) {
	const (
		streams   = 64
		chunkSize = 16 << 10 // default HTTP/2 maximum frame size
	)
	benchmarks := []struct {
		name string
		body int
		// Size of the bodies of streams that complete before the measured ones
		// start, leaving buffers of that capacity in the pool.
		warmupBody int
	}{
		{name: "body=4KiB", body: 4 << 10},
		{name: "body=64KiB", body: 64 << 10},
		{name: "body=512KiB", body: 512 << 10},
		{name: "body=4KiB/warmup=1MiB", body: 4 << 10, warmupBody: 1 << 20},
	}
	// Pooled buffers survive until the benchmark collects explicitly.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			server := NewStreamingServer(nil, &retainingDirector{}, registry, 0)
			startStreams := func(reqs []*extProcPb.ProcessingRequest) []*inFlightProcessServer {
				srvs := make([]*inFlightProcessServer, streams)
				for i := range srvs {
					srvs[i] = newInFlightProcessServer(reqs)
					srvs[i].start(b, server)
				}
				return srvs
			}
			finishStreams := func(srvs []*inFlightProcessServer) {
				for _, srv := range srvs {
					if err := srv.finish(); err != nil {
						b.Fatal(err)
					}
				}
			}
			reqs := requestMessages(reqcommon.PathChatCompletions, chatCompletionsBody(bm.body), chunkSize)
			var warmup []*extProcPb.ProcessingRequest
			if bm.warmupBody > 0 {
				warmup = requestMessages(reqcommon.PathChatCompletions, chatCompletionsBody(bm.warmupBody), chunkSize)
			}

			var retained, allocated float64
			b.ResetTimer()
			for range b.N {
				b.StopTimer()
				var base, start, opened, live runtime.MemStats
				runtime.GC()
				runtime.GC()
				runtime.ReadMemStats(&base)
				if warmup != nil {
					finishStreams(startStreams(warmup))
				}
				runtime.ReadMemStats(&start)
				b.StartTimer()
				srvs := startStreams(reqs)
				b.StopTimer()
				runtime.ReadMemStats(&opened)
				runtime.GC()
				runtime.GC()
				runtime.ReadMemStats(&live)
				retained += float64(live.HeapAlloc) - float64(base.HeapAlloc)
				allocated += float64(opened.TotalAlloc) - float64(start.TotalAlloc)
				finishStreams(srvs)
				b.StartTimer()
			}
			b.ReportMetric(retained/float64(b.N*streams), "retained-B/stream")
			b.ReportMetric(allocated/float64(b.N*streams), "alloc-B/stream")
		})
	}
}
