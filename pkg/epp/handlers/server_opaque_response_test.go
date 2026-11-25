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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/passthrough"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	grpcMethodPath   = "/grpc.testing.TestService/StreamingOutputCall"
	finalMetadataKey = "final"
	// lockstepTimeout bounds a wait for a response that must already be on its way.
	// It only turns a missing response into a failure instead of a hang.
	lockstepTimeout = 5 * time.Second
)

// lifecycleDirector keeps the parsed body for scheduling, as the Director does.
// It counts terminal response calls and writes dynamic metadata on them.
type lifecycleDirector struct {
	mockDirector
	incomingModel string
	targetModel   string

	mu            sync.Mutex
	terminalCalls int
	causes        []fwkrh.TerminationCause
	usage         fwkrequest.Usage
}

func (d *lifecycleDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, body *fwkrequest.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.SchedulingRequest = &fwksched.InferenceRequest{Body: body}
	// A scheduled endpoint arms the abnormal-termination cleanup in Process.
	reqCtx.TargetPod = &fwkdl.EndpointMetadata{}
	reqCtx.IncomingModelName = d.incomingModel
	reqCtx.TargetModelName = d.targetModel
	return reqCtx, nil
}

func (d *lifecycleDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	if !endOfStream {
		return reqCtx
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.terminalCalls++
	d.causes = append(d.causes, reqCtx.TerminationCause)
	d.usage = reqCtx.Usage
	reqCtx.Response.DynamicMetadata = &structpb.Struct{Fields: map[string]*structpb.Value{
		finalMetadataKey: structpb.NewNumberValue(float64(d.terminalCalls)),
	}}
	return reqCtx
}

func (d *lifecycleDirector) terminal() (int, []fwkrh.TerminationCause, fwkrequest.Usage) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.terminalCalls, append([]fwkrh.TerminationCause(nil), d.causes...), d.usage
}

// pacedProcessServer hands Process the messages the test feeds it and passes on each response Process sends.
type pacedProcessServer struct {
	mockProcessServer
	ctx context.Context
	in  chan *extProcPb.ProcessingRequest
	out chan *extProcPb.ProcessingResponse
}

func newPacedProcessServer(ctx context.Context) *pacedProcessServer {
	return &pacedProcessServer{
		ctx: ctx,
		in:  make(chan *extProcPb.ProcessingRequest),
		out: make(chan *extProcPb.ProcessingResponse, 64),
	}
}

func (m *pacedProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	select {
	case req, ok := <-m.in:
		if !ok {
			return nil, io.EOF
		}
		return req, nil
	case <-m.ctx.Done():
		// A gRPC server stream reports a cancelled call as a Canceled status.
		return nil, status.FromContextError(m.ctx.Err()).Err()
	}
}

func (m *pacedProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	select {
	case m.out <- resp:
		return nil
	case <-m.ctx.Done():
		return m.ctx.Err()
	}
}

func (m *pacedProcessServer) Context() context.Context { return m.ctx }

func (m *pacedProcessServer) feed(t *testing.T, req *extProcPb.ProcessingRequest) {
	t.Helper()
	select {
	case m.in <- req:
	case <-time.After(lockstepTimeout):
		t.Fatal("Process stopped reading the stream")
	}
}

// next returns the next response Process sends, failing the test when none arrives.
func (m *pacedProcessServer) next(t *testing.T, waitingFor string) *extProcPb.ProcessingResponse {
	t.Helper()
	select {
	case resp := <-m.out:
		return resp
	case <-time.After(lockstepTimeout):
		t.Fatalf("no response from the EPP while waiting for %s", waitingFor)
		return nil
	}
}

func grpcFrame(t *testing.T, payload []byte) []byte {
	t.Helper()
	frame, err := grpcframe.FramePayload(payload)
	require.NoError(t, err)
	return frame
}

func requestBodyMsg(body []byte) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_RequestBody{
		RequestBody: &extProcPb.HttpBody{Body: body, EndOfStream: true},
	}}
}

func responseHeadersMsg(contentType string, endOfStream bool, extra ...string) *extProcPb.ProcessingRequest {
	headers := []*configPb.HeaderValue{{Key: ":status", RawValue: []byte("200")}}
	if contentType != "" {
		headers = append(headers, &configPb.HeaderValue{Key: "content-type", RawValue: []byte(contentType)})
	}
	for i := 0; i+1 < len(extra); i += 2 {
		headers = append(headers, &configPb.HeaderValue{Key: extra[i], RawValue: []byte(extra[i+1])})
	}
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: headers}, EndOfStream: endOfStream},
	}}
}

func responseBodyMsg(chunk []byte, endOfStream bool) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
		ResponseBody: &extProcPb.HttpBody{Body: chunk, EndOfStream: endOfStream},
	}}
}

func responseTrailersMsg(grpcStatus string) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseTrailers{
		ResponseTrailers: &extProcPb.HttpTrailers{Trailers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
			{Key: "grpc-status", RawValue: []byte(grpcStatus)},
		}}},
	}}
}

func passthroughRegistry() *ParserRegistry {
	return NewParserRegistry([]fwkrh.Parser{passthrough.NewPassthroughParser()}, logr.Discard())
}

// runningRequests sums llm_d_epp_request_running over every series.
func runningRequests(t *testing.T) float64 {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	var total float64
	for _, family := range families {
		if family.GetName() != "llm_d_epp_request_running" {
			continue
		}
		for _, metric := range family.GetMetric() {
			total += metric.GetGauge().GetValue()
		}
	}
	return total
}

// startRouted runs Process and walks a gRPC request through routing and the response headers.
func startRouted(ctx context.Context, t *testing.T, director Director, registry *ParserRegistry, reqHeaders map[string]string, reqBody []byte, respHeaders *extProcPb.ProcessingRequest) (*pacedProcessServer, chan error) {
	t.Helper()
	srv := newPacedProcessServer(ctx)
	done := make(chan error, 1)
	go func() { done <- NewStreamingServer(nil, director, registry, 0).Process(srv) }()
	srv.feed(t, newRequestHeaders(reqHeaders))
	srv.feed(t, requestBodyMsg(reqBody))
	require.NotNil(t, srv.next(t, "the request header response").GetRequestHeaders())
	require.NotNil(t, srv.next(t, "the request body response").GetRequestBody())
	srv.feed(t, respHeaders)
	require.NotNil(t, srv.next(t, "the response header response").GetResponseHeaders())
	return srv, done
}

func streamedBody(resp *extProcPb.ProcessingResponse) *extProcPb.StreamedBodyResponse {
	return resp.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
}

// TestProcessForwardsPassthroughGRPCMessageBeforeBackendFinishes is the llm-d#3089 regression.
// The backend holds message 2 until the client has message 1, so an EPP that holds the response until the trailers never delivers message 1.
func TestProcessForwardsPassthroughGRPCMessageBeforeBackendFinishes(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()

	director := &lifecycleDirector{incomingModel: "m", targetModel: "m"}
	srv, done := startRouted(ctx, t, director, passthroughRegistry(),
		map[string]string{":path": grpcMethodPath, "content-type": "application/grpc"},
		grpcFrame(t, []byte("request")), responseHeadersMsg("application/grpc", false))

	message1, message2 := grpcFrame(t, []byte("message 1")), grpcFrame(t, []byte("message 2"))
	srv.feed(t, responseBodyMsg(message1, false))
	got := srv.next(t, "message 1 while the backend holds message 2")
	require.Equal(t, message1, streamedBody(got).GetBody())
	require.False(t, streamedBody(got).GetEndOfStream())
	require.Equal(t, 1.0, runningRequests(t), "the request is still running")

	srv.feed(t, responseBodyMsg(message2, false))
	require.Equal(t, message2, streamedBody(srv.next(t, "message 2")).GetBody())
	srv.feed(t, responseTrailersMsg("0"))
	final := srv.next(t, "the final body response")
	require.Empty(t, streamedBody(final).GetBody())
	require.NotNil(t, final.GetDynamicMetadata().GetFields()[finalMetadataKey], "final dynamic metadata precedes the trailers")
	require.NotNil(t, srv.next(t, "the trailers").GetResponseTrailers())
	close(srv.in)
	require.NoError(t, <-done)

	calls, causes, _ := director.terminal()
	require.Equal(t, 1, calls)
	require.Equal(t, []fwkrh.TerminationCause{""}, causes)
	require.Zero(t, runningRequests(t))
}

// opaqueCase is one passthrough response, fed to Process one message at a time.
type opaqueCase struct {
	name        string
	contentType string
	chunks      [][]byte
	// grpcStatus ends the response with trailers; empty ends it with an end-of-stream body chunk.
	grpcStatus string
	wantCause  fwkrh.TerminationCause
}

func TestProcessPassthroughResponseProtocols(t *testing.T) {
	frameA := grpcFrame(t, []byte("alpha"))
	frameB := grpcFrame(t, []byte("bravo-bravo"))
	frameC := grpcFrame(t, bytes.Repeat([]byte{0x00, 0xff}, 40))
	large := grpcFrame(t, bytes.Repeat([]byte("L"), 200_000))
	cases := []opaqueCase{
		{name: "unary", contentType: "application/grpc", chunks: [][]byte{frameA}, grpcStatus: "0"},
		{name: "server streaming", contentType: "application/grpc", chunks: [][]byte{frameA, frameB, frameC}, grpcStatus: "0"},
		{name: "zero messages", contentType: "application/grpc", grpcStatus: "0"},
		{name: "non-zero grpc-status", contentType: "application/grpc", chunks: [][]byte{frameA, frameB}, grpcStatus: "13", wantCause: fwkrh.TerminationCauseError},
		{name: "split inside the 5-byte prefix", contentType: "application/grpc", chunks: [][]byte{frameB[:2], frameB[2:], frameC}, grpcStatus: "0"},
		{name: "split inside the payload", contentType: "application/grpc", chunks: [][]byte{frameB[:8], frameB[8:]}, grpcStatus: "0"},
		{name: "multiple frames per chunk", contentType: "application/grpc", chunks: [][]byte{append(append(bytes.Clone(frameA), frameB...), frameC...)}, grpcStatus: "0"},
		{name: "large single message", contentType: "application/grpc+proto", chunks: [][]byte{large}, grpcStatus: "0"},
		{name: "SSE with content type", contentType: "text/event-stream", chunks: [][]byte{[]byte("data: {\"a\":1}\n\n"), []byte("data: [DONE]\n\n")}},
		{name: "SSE without content type", chunks: [][]byte{[]byte("data: {\"a\":1}\n\n"), []byte("data: [DONE]\n\n")}},
		{name: "JSON over HTTP", contentType: "application/json", chunks: [][]byte{[]byte(`{"model":"m",`), []byte(`"x":1}`)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)
			ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
			defer cancel()

			reqHeaders := map[string]string{":path": "/v1/raw"}
			if tc.grpcStatus != "" {
				reqHeaders = map[string]string{":path": grpcMethodPath, "content-type": "application/grpc"}
			}
			director := &lifecycleDirector{incomingModel: "m", targetModel: "m"}
			srv, done := startRouted(ctx, t, director, passthroughRegistry(), reqHeaders, []byte("request"), responseHeadersMsg(tc.contentType, false))

			var forwarded []byte
			var last *extProcPb.ProcessingResponse
			for i, chunk := range tc.chunks {
				eos := tc.grpcStatus == "" && i == len(tc.chunks)-1
				srv.feed(t, responseBodyMsg(chunk, eos))
				// The whole chunk goes out before the next one comes in, split at the ext_proc body limit.
				var got []byte
				for len(got) < len(chunk) {
					last = srv.next(t, "the bytes of chunk "+tc.name)
					got = append(got, streamedBody(last).GetBody()...)
				}
				require.Equal(t, chunk, got)
				require.Equal(t, eos, streamedBody(last).GetEndOfStream())
				forwarded = append(forwarded, got...)
			}
			if tc.grpcStatus != "" {
				srv.feed(t, responseTrailersMsg(tc.grpcStatus))
				last = srv.next(t, "the final body response")
				require.Empty(t, streamedBody(last).GetBody())
				require.False(t, streamedBody(last).GetEndOfStream(), "the trailers end a gRPC response")
				require.NotNil(t, srv.next(t, "the trailers").GetResponseTrailers())
			}
			require.NotNil(t, last.GetDynamicMetadata().GetFields()[finalMetadataKey], "final dynamic metadata")
			close(srv.in)
			require.NoError(t, <-done)
			require.Equal(t, string(bytes.Join(tc.chunks, nil)), string(forwarded), "bytes are forwarded unchanged")

			calls, causes, usage := director.terminal()
			require.Equal(t, 1, calls, "exactly one terminal response call")
			require.Equal(t, []fwkrh.TerminationCause{tc.wantCause}, causes)
			require.Equal(t, fwkrequest.Usage{}, usage, "an unparsed response reports no usage")
			if tc.contentType == "text/event-stream" {
				// An event stream stays on the streaming path, which records inter-token latency per chunk.
				require.Equal(t, uint64(len(tc.chunks)-1), histogramSampleCount(t, "llm_d_epp_request_streaming_itl_seconds", nil))
			}
			require.Zero(t, runningRequests(t))
		})
	}
}

// recordingPassthroughParser records the calls the handler makes to ParseResponse.
type recordingPassthroughParser struct {
	*passthrough.PassthroughParser
	mu     sync.Mutex
	bodies []string
	ends   []bool
}

func (p *recordingPassthroughParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endOfStream bool) (*fwkrh.ParsedResponse, error) {
	p.mu.Lock()
	p.bodies = append(p.bodies, string(body))
	p.ends = append(p.ends, endOfStream)
	p.mu.Unlock()
	return p.PassthroughParser.ParseResponse(ctx, body, headers, endOfStream)
}

// TestProcessPassthroughTrailersOnlyResponse covers a gRPC response sent as headers with end of stream.
// The headers end the response, so the final ParseResponse call and the completion metrics happen there.
func TestProcessPassthroughTrailersOnlyResponse(t *testing.T) {
	for _, tc := range []struct {
		name, grpcStatus string
		wantCause        fwkrh.TerminationCause
		wantOutcome      [2]string
	}{
		{name: "error status", grpcStatus: "5", wantCause: fwkrh.TerminationCauseError, wantOutcome: [2]string{outcomeFailed, errcommon.ModelServerError}},
		{name: "OK status", grpcStatus: "0", wantOutcome: [2]string{outcomeCompleted, ""}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)
			ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
			defer cancel()

			parser := &recordingPassthroughParser{PassthroughParser: passthrough.NewPassthroughParser()}
			director := &lifecycleDirector{incomingModel: "m", targetModel: "m"}
			srv, done := startRouted(ctx, t, director, NewParserRegistry([]fwkrh.Parser{parser}, logr.Discard()),
				map[string]string{":path": grpcMethodPath, "content-type": "application/grpc"},
				grpcFrame(t, []byte("request")), responseHeadersMsg("application/grpc", true, "grpc-status", tc.grpcStatus))
			close(srv.in)
			require.NoError(t, <-done)
			require.Empty(t, srv.out, "no body response for a trailers-only response")

			require.Equal(t, []string{""}, parser.bodies, "one final ParseResponse call with an empty body")
			require.Equal(t, []bool{true}, parser.ends)
			calls, causes, _ := director.terminal()
			require.Equal(t, 1, calls)
			require.Equal(t, []fwkrh.TerminationCause{tc.wantCause}, causes)
			require.Equal(t, map[[2]string]float64{tc.wantOutcome: 1}, requestOutcomes(t))
			require.Equal(t, uint64(1), histogramSampleCount(t, "llm_d_epp_response_size_bytes", nil))
			require.Equal(t, uint64(1), histogramSampleCount(t, "llm_d_epp_request_duration_seconds", nil))
			require.Zero(t, runningRequests(t))
		})
	}
}

// TestProcessPassthroughKeepsResponseMetadata checks that an opaque response's status, binary metadata and grpc-message reach the client unchanged.
// The proxy applies only the mutations the EPP returns, so an upstream header is unchanged when no mutation removes it or sets it to another value.
func TestProcessPassthroughKeepsResponseMetadata(t *testing.T) {
	binValue := []byte("AAECA/8")
	for _, tc := range []struct {
		name, status, contentType string
		reqHeaders                map[string]string
		trailers                  []*configPb.HeaderValue
	}{
		{name: "gRPC", status: "200", contentType: "application/grpc",
			reqHeaders: map[string]string{":path": grpcMethodPath, "content-type": "application/grpc"},
			trailers: []*configPb.HeaderValue{{Key: "grpc-status", RawValue: []byte("3")},
				{Key: "grpc-message", RawValue: []byte("bad%20arg")}, {Key: "x-trace-bin", RawValue: binValue}}},
		{name: "HTTP", status: "404", contentType: "application/json", reqHeaders: map[string]string{":path": "/v1/raw"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)
			ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
			defer cancel()

			srv := newPacedProcessServer(ctx)
			done := make(chan error, 1)
			go func() {
				done <- NewStreamingServer(nil, &lifecycleDirector{incomingModel: "m", targetModel: "m"}, passthroughRegistry(), 0).Process(srv)
			}()
			srv.feed(t, newRequestHeaders(tc.reqHeaders))
			srv.feed(t, requestBodyMsg([]byte("request")))
			srv.next(t, "the request header response")
			srv.next(t, "the request body response")
			upstream := map[string][]byte{":status": []byte(tc.status), "content-type": []byte(tc.contentType), "x-meta-bin": binValue}
			headers := &configPb.HeaderMap{}
			for key, value := range upstream {
				headers.Headers = append(headers.Headers, &configPb.HeaderValue{Key: key, RawValue: value})
			}
			srv.feed(t, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &extProcPb.HttpHeaders{Headers: headers},
			}})
			mutation := srv.next(t, "the response header response").GetResponseHeaders().GetResponse().GetHeaderMutation()
			for _, set := range mutation.GetSetHeaders() {
				if want, ok := upstream[set.GetHeader().GetKey()]; ok {
					require.Equal(t, string(want), string(set.GetHeader().GetRawValue()), "header %s", set.GetHeader().GetKey())
				}
			}
			for key := range upstream {
				require.NotContains(t, mutation.GetRemoveHeaders(), key)
			}

			srv.feed(t, responseBodyMsg([]byte("body"), tc.trailers == nil))
			require.Equal(t, "body", string(streamedBody(srv.next(t, "the body")).GetBody()))
			if tc.trailers != nil {
				srv.feed(t, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseTrailers{
					ResponseTrailers: &extProcPb.HttpTrailers{Trailers: &configPb.HeaderMap{Headers: tc.trailers}},
				}})
				srv.next(t, "the final body response")
				require.Nil(t, srv.next(t, "the trailers").GetResponseTrailers().GetHeaderMutation(), "trailers pass through without a mutation")
			}
			close(srv.in)
			require.NoError(t, <-done)
		})
	}
}

// TestProcessPassthroughDisconnectAfterFirstMessage checks cleanup when the client leaves mid-stream.
func TestProcessPassthroughDisconnectAfterFirstMessage(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()

	director := &lifecycleDirector{incomingModel: "m", targetModel: "m"}
	srv, done := startRouted(ctx, t, director, passthroughRegistry(),
		map[string]string{":path": grpcMethodPath, "content-type": "application/grpc"},
		grpcFrame(t, []byte("request")), responseHeadersMsg("application/grpc", false))
	message1 := grpcFrame(t, []byte("message 1"))
	srv.feed(t, responseBodyMsg(message1, false))
	require.Equal(t, message1, streamedBody(srv.next(t, "message 1")).GetBody())

	cancel()
	// Process returns nil when the cancelled receive wins its select, and the context error otherwise.
	if err := <-done; err != nil {
		require.ErrorIs(t, err, context.Canceled)
	}
	calls, causes, _ := director.terminal()
	require.Equal(t, 1, calls)
	require.Equal(t, []fwkrh.TerminationCause{fwkrh.TerminationCauseClientDisconnect}, causes)
	require.Zero(t, runningRequests(t))
}

// countingProcessServer streams a long passthrough gRPC response and discards what Process sends.
type countingProcessServer struct {
	mockProcessServer
	ctx      context.Context
	prologue []*extProcPb.ProcessingRequest
	chunk    []byte
	chunks   int
	// consume paces Send for response bodies, standing in for a slow reader.
	consume <-chan struct{}
	// onChunk runs before Process receives response chunk n, counted from 1.
	onChunk func(n int)

	pulled    int
	forwarded atomic.Int64
	bytesOut  atomic.Int64
}

func (m *countingProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(m.prologue) > 0 {
		req := m.prologue[0]
		m.prologue = m.prologue[1:]
		return req, nil
	}
	switch {
	case m.pulled < m.chunks:
		m.pulled++
		if m.onChunk != nil {
			m.onChunk(m.pulled)
		}
		return responseBodyMsg(m.chunk, false), nil
	case m.pulled == m.chunks:
		m.pulled++
		return responseTrailersMsg("0"), nil
	default:
		return nil, io.EOF
	}
}

func (m *countingProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	if body := streamedBody(resp).GetBody(); len(body) > 0 {
		if m.consume != nil {
			select {
			case <-m.consume:
			case <-m.ctx.Done():
				return m.ctx.Err()
			}
		}
		m.bytesOut.Add(int64(len(body)))
		m.forwarded.Add(1)
	}
	return nil
}

func (m *countingProcessServer) Context() context.Context { return m.ctx }

func newCountingProcessServer(ctx context.Context, t *testing.T, chunk []byte, chunks int) *countingProcessServer {
	return &countingProcessServer{
		ctx:   ctx,
		chunk: chunk,
		prologue: []*extProcPb.ProcessingRequest{
			newRequestHeaders(map[string]string{":path": grpcMethodPath, "content-type": "application/grpc"}),
			requestBodyMsg(grpcFrame(t, []byte("request"))),
			responseHeadersMsg("application/grpc", false),
		},
		chunks: chunks,
	}
}

// TestProcessPassthroughSlowReaderBoundsReadAhead checks that a slow reader stops the EPP from pulling the response.
// Process reads at most a fixed number of chunks past the one it is sending, so the backend sees backpressure.
func TestProcessPassthroughSlowReaderBoundsReadAhead(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()
	consume := make(chan struct{})
	srv := newCountingProcessServer(ctx, t, grpcFrame(t, []byte("token")), 200)
	srv.consume = consume
	// Process holds one chunk in Send, one in its receive channel, and the reader holds a third.
	const maxAhead = 3
	srv.onChunk = func(n int) {
		if ahead := int64(n) - srv.forwarded.Load(); ahead > maxAhead {
			t.Errorf("Process pulled chunk %d with %d chunks forwarded", n, srv.forwarded.Load())
			cancel()
		}
	}
	go func() {
		ticker := time.NewTicker(200 * time.Microsecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				select {
				case consume <- struct{}{}:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	err := NewStreamingServer(nil, &lifecycleDirector{incomingModel: "m", targetModel: "m"}, passthroughRegistry(), 0).Process(srv)
	if t.Failed() {
		return
	}
	require.NoError(t, err)
	require.Equal(t, int64(200), srv.forwarded.Load())
}

// TestProcessPassthroughLongStreamKeepsMemoryFlat streams far more bytes than the allowed heap growth.
func TestProcessPassthroughLongStreamKeepsMemoryFlat(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()
	const (
		payloadBytes = 60_000
		chunks       = 1100
		maxGrowth    = 16 << 20
	)
	srv := newCountingProcessServer(ctx, t, grpcFrame(t, bytes.Repeat([]byte("x"), payloadBytes)), chunks)
	var stats runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&stats)
	baseline := stats.HeapAlloc
	var peak uint64
	srv.onChunk = func(n int) {
		if n%250 != 0 {
			return
		}
		runtime.GC()
		runtime.ReadMemStats(&stats)
		if stats.HeapAlloc > baseline {
			peak = max(peak, stats.HeapAlloc-baseline)
		}
	}
	require.NoError(t, NewStreamingServer(nil, &lifecycleDirector{incomingModel: "m", targetModel: "m"}, passthroughRegistry(), 0).Process(srv))
	require.Equal(t, int64(chunks*len(srv.chunk)), srv.bytesOut.Load())
	require.Less(t, peak, uint64(maxGrowth), "live heap grew by %d bytes over a %d-byte stream", peak, chunks*len(srv.chunk))
}

// recordingParser records the bodies the handler passes to ParseResponse.
type recordingParser struct {
	*openai.OpenAIParser
	mu     sync.Mutex
	bodies []string
	ends   []bool
}

func (p *recordingParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endOfStream bool) (*fwkrh.ParsedResponse, error) {
	p.mu.Lock()
	p.bodies = append(p.bodies, string(body))
	p.ends = append(p.ends, endOfStream)
	p.mu.Unlock()
	return p.OpenAIParser.ParseResponse(ctx, body, headers, endOfStream)
}

// TestProcessNonStreamingJSONParserReceivesCompleteBody pins the complete-body contract of a parser that did not opt out of it.
// The response arrives in fragments that split the model field, so a forwarded fragment would escape the model rewrite.
func TestProcessNonStreamingJSONParserReceivesCompleteBody(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()

	parser := &recordingParser{OpenAIParser: openai.NewOpenAIParser()}
	director := &lifecycleDirector{incomingModel: "client-model", targetModel: "backend-model"}
	srv, done := startRouted(ctx, t, director, NewParserRegistry([]fwkrh.Parser{parser}, logr.Discard()),
		map[string]string{":path": "/v1/chat/completions", "content-type": "application/json"},
		[]byte(`{"model":"client-model","messages":[{"role":"user","content":"hi"}]}`),
		responseHeadersMsg("application/json", false))

	body := `{"id":"x","model":"backend-model","choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}`
	fragments := []string{body[:15], body[15:30], body[30:]}
	for i, fragment := range fragments {
		srv.feed(t, responseBodyMsg([]byte(fragment), i == len(fragments)-1))
	}
	resp := srv.next(t, "the complete body")
	require.Equal(t, strings.Replace(body, "backend-model", "client-model", 1), string(streamedBody(resp).GetBody()))
	require.True(t, streamedBody(resp).GetEndOfStream())
	require.NotNil(t, resp.GetDynamicMetadata().GetFields()[finalMetadataKey])
	close(srv.in)
	require.NoError(t, <-done)
	require.Empty(t, srv.out, "one body response for the whole body")

	require.Equal(t, []string{body}, parser.bodies)
	require.Equal(t, []bool{true}, parser.ends)
	calls, _, usage := director.terminal()
	require.Equal(t, 1, calls)
	require.Equal(t, fwkrequest.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8}, usage)
	require.Zero(t, runningRequests(t))
}

// TestProcessChatCompletionsSSEUsageAccounting pins per-chunk forwarding and usage for a recognized token stream.
func TestProcessChatCompletionsSSEUsageAccounting(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()

	director := &lifecycleDirector{incomingModel: "m", targetModel: "m"}
	srv, done := startRouted(ctx, t, director, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
		map[string]string{":path": "/v1/chat/completions", "content-type": "application/json"},
		[]byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`),
		responseHeadersMsg("text/event-stream", false))

	usageEvent := `data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":10,"total_tokens":17}}` + "\n\n"
	chunks := []string{
		`data: {"choices":[{"delta":{"content":"hi"}}]}` + "\n\n",
		usageEvent[:40],
		usageEvent[40:],
		"data: [DONE]\n\n",
	}
	for i, chunk := range chunks {
		eos := i == len(chunks)-1
		srv.feed(t, responseBodyMsg([]byte(chunk), eos))
		got := srv.next(t, "SSE chunk")
		require.Equal(t, chunk, string(streamedBody(got).GetBody()))
		require.Equal(t, eos, streamedBody(got).GetEndOfStream())
	}
	close(srv.in)
	require.NoError(t, <-done)
	calls, _, usage := director.terminal()
	require.Equal(t, 1, calls)
	require.Equal(t, fwkrequest.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17}, usage)
	require.Equal(t, uint64(len(chunks)-1), histogramSampleCount(t, "llm_d_epp_request_streaming_itl_seconds", nil))
}
