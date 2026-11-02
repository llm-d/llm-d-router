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
	"encoding/json"
	"io"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"
	"weak"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
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

// streamingResponse returns the response headers and a first response body chunk
// of a streamed response that has not ended.
func streamingResponse() []*extProcPb.ProcessingRequest {
	return []*extProcPb.ProcessingRequest{
		{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: ":status", RawValue: []byte("200")},
				{Key: "content-type", RawValue: []byte("text/event-stream")},
			}}},
		}},
		{Request: &extProcPb.ProcessingRequest_ResponseBody{
			ResponseBody: &extProcPb.HttpBody{Body: []byte("data: {}\n\n")},
		}},
	}
}

// responsePhaseDirector keeps the parsed body on the request, as the Director
// does for scheduling. It records weak pointers to the request content and the
// request body that response-phase plugins would receive.
type responsePhaseDirector struct {
	mockDirector
	tokenized      *fwkrequest.TokenizedRequest
	rawBody        weak.Pointer[byte]
	completions    weak.Pointer[fwkrequest.CompletionsRequest]
	prompt         weak.Pointer[byte]
	headerBody     fwkrequest.InferenceRequestBody
	bodyChunkBody  fwkrequest.InferenceRequestBody
	responseBodies int
}

func (d *responsePhaseDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, body *fwkrequest.InferenceRequestBody) (*RequestContext, error) {
	// A DataProducer sets TokenizedRequest before scheduling.
	body.TokenizedRequest = d.tokenized
	reqCtx.SchedulingRequest = &fwksched.InferenceRequest{Body: body}
	d.rawBody = weak.Make(&reqCtx.Request.RawBody[0])
	d.completions = weak.Make(body.Completions)
	prompt := body.Payload.(fwkrequest.PayloadMap)["prompt"].(json.RawMessage)
	d.prompt = weak.Make(&prompt[0])
	return reqCtx, nil
}

func (d *responsePhaseDirector) HandleResponseHeader(_ context.Context, reqCtx *RequestContext) *RequestContext {
	d.headerBody = *reqCtx.SchedulingRequest.Body
	return reqCtx
}

func (d *responsePhaseDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, _ bool) *RequestContext {
	if d.responseBodies == 0 {
		d.bodyChunkBody = *reqCtx.SchedulingRequest.Body
	}
	d.responseBodies++
	return reqCtx
}

func TestProcessReleasesRequestContentAfterForwarding(t *testing.T) {
	const prefix, suffix = `{"model":"m","stream":true,"max_tokens":16,"prompt":"`, `"}`
	body := []byte(prefix + strings.Repeat("a", 64<<10) + suffix)
	registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
	director := &responsePhaseDirector{
		tokenized: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}}},
	}
	server := NewStreamingServer(nil, director, registry, 0)

	reqs := append(requestMessages(reqcommon.PathCompletions, body, 16<<10), streamingResponse()...)
	srv := newInFlightProcessServer(reqs)
	srv.start(t, server)
	require.Eventually(t, func() bool {
		runtime.GC()
		return director.rawBody.Value() == nil && director.completions.Value() == nil && director.prompt.Value() == nil
	}, 5*time.Second, 10*time.Millisecond, "request content reachable while the stream waits for the response")
	require.NoError(t, srv.finish())

	maxTokens := int64(16)
	want := fwkrequest.InferenceRequestBody{
		TokenizedRequest: director.tokenized,
		Stream:           true,
		MaxOutputTokens:  &maxTokens,
		Model:            "m",
	}
	require.Equal(t, want, director.headerBody, "request body in ResponseHeader")
	require.Same(t, director.tokenized, director.headerBody.TokenizedRequest)
	require.Equal(t, want, director.bodyChunkBody, "request body in ResponseBody")
	require.Same(t, director.tokenized, director.bodyChunkBody.TokenizedRequest)
}

// stragglerDirector starts a goroutine that holds a copy of the request and reads
// its body, as a DataProducer does when it outlives its timeout:
// dataProducerPluginsWithTimeout passes each producer a copy of the request and
// returns without waiting for it.
type stragglerDirector struct {
	mockDirector
	responseStarted chan struct{}
	stop            chan struct{}
	done            chan struct{}
	contentKept     bool
}

func (d *stragglerDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, body *fwkrequest.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.SchedulingRequest = &fwksched.InferenceRequest{Body: body}
	producerRequest := *reqCtx.SchedulingRequest
	go func() {
		defer close(d.done)
		for {
			select {
			case <-d.stop:
				d.contentKept = producerRequest.Body.Completions != nil && producerRequest.Body.Payload != nil
				return
			default:
				// Reads that overlap the release, for the race detector.
				_ = producerRequest.Body.Completions
				runtime.Gosched()
			}
		}
	}()
	return reqCtx, nil
}

func (d *stragglerDirector) HandleResponseHeader(_ context.Context, reqCtx *RequestContext) *RequestContext {
	close(d.responseStarted)
	return reqCtx
}

func TestProcessReleaseKeepsBodyForStragglingProducer(t *testing.T) {
	body := []byte(`{"model":"m","stream":true,"prompt":"hello"}`)
	registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
	director := &stragglerDirector{
		responseStarted: make(chan struct{}),
		stop:            make(chan struct{}),
		done:            make(chan struct{}),
	}
	server := NewStreamingServer(nil, director, registry, 0)

	reqs := append(requestMessages(reqcommon.PathCompletions, body, len(body)), streamingResponse()...)
	srv := newInFlightProcessServer(reqs)
	srv.start(t, server)
	<-director.responseStarted
	close(director.stop)
	<-director.done
	require.NoError(t, srv.finish())
	require.True(t, director.contentKept, "request content cleared under a producer that still reads it")
}

// rawBodyDirector records the RawBody that the Director receives.
type rawBodyDirector struct {
	mockDirector
	rawBody []byte
}

func (d *rawBodyDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrequest.InferenceRequestBody) (*RequestContext, error) {
	d.rawBody = reqCtx.Request.RawBody
	return reqCtx, nil
}

func TestProcessRawBodyUsesBufferAbovePoolCap(t *testing.T) {
	const bufferCap = 64 << 10
	body := []byte(`{"model":"m","prompt":"` + strings.Repeat("a", 40<<10) + `"}`)
	tests := []struct {
		name          string
		poolCap       int
		wantSharesBuf bool
	}{
		{name: "buffer within the pool cap is copied", poolCap: 2 * bufferCap},
		{name: "pool without a cap copies", poolCap: 0},
		{name: "buffer above the pool cap is taken over", poolCap: bufferCap / 2, wantSharesBuf: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			director := &rawBodyDirector{}
			server := NewStreamingServer(nil, director, registry, tt.poolCap)
			var backing []byte
			server.bufferPool.New = func() any {
				backing = make([]byte, 0, bufferCap)
				return bytes.NewBuffer(backing)
			}

			srv := newInFlightProcessServer(requestMessages(reqcommon.PathCompletions, body, 16<<10))
			srv.start(t, server)
			require.Equal(t, body, director.rawBody)
			require.Equal(t, tt.wantSharesBuf, &director.rawBody[0] == &backing[:1][0],
				"RawBody shares the request buffer's memory")
			require.NoError(t, srv.finish())
		})
	}
}

// retainingDirector keeps the parsed body on the request, as the Director does
// for scheduling.
type retainingDirector struct{ mockDirector }

func (d *retainingDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, body *fwkrequest.InferenceRequestBody) (*RequestContext, error) {
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
