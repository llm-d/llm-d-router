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

package grpctest

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// VLLMCall records one GenerateRequest a VLLMUpstream received.
type VLLMCall struct {
	// Path is the request path, one of the vllm.Inference method names.
	Path string
	// Stream reports that the call used GenerateStream.
	Stream bool
	// Header is a copy of the request headers.
	Header http.Header
	// Request is the decoded GenerateRequest.
	Request *vllmpb.GenerateRequest
}

// VLLMUpstreamOptions configures NewVLLMUpstream.
type VLLMUpstreamOptions struct {
	// Respond returns the frames to send for a call, in order. A non-nil
	// error ends the response with that gRPC status (status.FromError), as
	// a trailers-only response when no frame was returned. When Respond is
	// nil the upstream sends DefaultVLLMResponse.
	Respond func(call VLLMCall) ([]*vllmpb.GenerateResponse, error)
}

// VLLMUpstream is a scripted fake vllm.Inference server over h2c. It
// records every GenerateRequest it receives and replies with the frames
// the Respond callback returns. Both grpc-go clients and the coordinator's
// raw HTTP/2 gateway client can talk to it.
type VLLMUpstream struct {
	server *httptest.Server

	t    testing.TB
	opts VLLMUpstreamOptions

	mu    sync.Mutex
	calls []VLLMCall
}

// NewVLLMUpstream starts a VLLMUpstream and closes it at cleanup.
func NewVLLMUpstream(t testing.TB, opts VLLMUpstreamOptions) *VLLMUpstream {
	t.Helper()
	u := &VLLMUpstream{t: t, opts: opts}
	u.server = NewH2CServer(t, http.HandlerFunc(u.serve))
	return u
}

// URL returns the http:// base URL of the upstream.
func (u *VLLMUpstream) URL() string { return u.server.URL }

// Addr returns the host:port of the upstream for grpc-go clients.
func (u *VLLMUpstream) Addr() string { return strings.TrimPrefix(u.server.URL, "http://") }

// Calls returns a copy of every recorded call in arrival order.
func (u *VLLMUpstream) Calls() []VLLMCall {
	u.mu.Lock()
	defer u.mu.Unlock()
	out := make([]VLLMCall, len(u.calls))
	copy(out, u.calls)
	return out
}

// LastCall returns the most recent call and whether one was recorded.
func (u *VLLMUpstream) LastCall() (VLLMCall, bool) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(u.calls) == 0 {
		return VLLMCall{}, false
	}
	return u.calls[len(u.calls)-1], true
}

// DefaultVLLMResponse is the single terminal frame the upstream sends when
// no Respond callback is configured.
func DefaultVLLMResponse() *vllmpb.GenerateResponse {
	return &vllmpb.GenerateResponse{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
		Outputs: &vllmpb.SequenceOutput{
			Text:      "ok",
			NumTokens: 1,
			TokenIds:  []uint32{42},
			FinishInfo: &vllmpb.FinishInfo{
				NumOutputTokens: 1,
				FinishReason:    vllmpb.FinishInfo_STOP,
				NumCachedTokens: proto.Uint32(0),
			},
		},
	}
}

func (u *VLLMUpstream) serve(w http.ResponseWriter, r *http.Request) {
	var stream bool
	switch r.URL.Path {
	case vllmpb.Inference_Generate_FullMethodName:
	case vllmpb.Inference_GenerateStream_FullMethodName:
		stream = true
	default:
		grpcframe.WriteError(w, codes.Unimplemented, "unknown method "+r.URL.Path)
		return
	}

	framed, err := grpcframe.ReadMessage(r.Body, grpcframe.MaxPayloadLen)
	if err != nil {
		grpcframe.WriteError(w, codes.InvalidArgument, err.Error())
		return
	}
	payload, err := grpcframe.ParsePayload(framed)
	if err != nil {
		grpcframe.WriteError(w, codes.InvalidArgument, err.Error())
		return
	}
	req := &vllmpb.GenerateRequest{}
	if err := proto.Unmarshal(payload, req); err != nil {
		grpcframe.WriteError(w, codes.InvalidArgument, err.Error())
		return
	}
	call := VLLMCall{Path: r.URL.Path, Stream: stream, Header: r.Header.Clone(), Request: req}
	u.mu.Lock()
	u.calls = append(u.calls, call)
	u.mu.Unlock()

	frames := []*vllmpb.GenerateResponse{DefaultVLLMResponse()}
	var respErr error
	if u.opts.Respond != nil {
		frames, respErr = u.opts.Respond(call)
	}
	if respErr != nil && len(frames) == 0 {
		st := status.Convert(respErr)
		grpcframe.WriteError(w, st.Code(), st.Message())
		return
	}

	w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	for _, resp := range frames {
		b, err := grpcframe.MarshalProto(resp)
		if err != nil {
			u.t.Errorf("grpctest: marshal GenerateResponse: %v", err)
			WriteStatusTrailer(w, codes.Internal, err.Error())
			return
		}
		if _, err := w.Write(b); err != nil {
			return
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	st := status.Convert(respErr)
	WriteStatusTrailer(w, st.Code(), st.Message())
}
