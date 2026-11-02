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
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestFrameHelpers(t *testing.T) {
	msg := &vllmpb.GenerateRequest{Model: "m", Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}}
	framed := Frame(t, msg)
	var got vllmpb.GenerateRequest
	Unmarshal(t, framed, &got)
	require.True(t, proto.Equal(msg, &got))

	raw := FramePayload(t, []byte{1, 2, 3})
	payload, err := grpcframe.ParsePayload(raw)
	require.NoError(t, err)
	require.Equal(t, []byte{1, 2, 3}, payload)

	frames := ReadFrames(t, bytes.NewReader(append(append([]byte{}, framed...), raw...)))
	require.Len(t, frames, 2)
	require.Equal(t, framed, frames[0])
	require.Equal(t, raw, frames[1])
	require.Empty(t, ReadFrames(t, bytes.NewReader(nil)))
}

func TestH2CServerAndTrailers(t *testing.T) {
	srv := NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, 2, r.ProtoMajor)
		switch r.URL.Path {
		case "/ok":
			var req vllmpb.GenerateRequest
			ReadRequest(t, r, &req)
			WriteFrames(t, w, &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{Text: req.GetText()}})
		case "/trailers-only":
			grpcframe.WriteError(w, codes.NotFound, "no such thing")
		default:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			WriteStatusTrailer(w, codes.Internal, "after body")
		}
	}))
	client := NewH2CClient()

	resp, err := client.Post(srv.URL+"/ok", grpcframe.MediaType, bytes.NewReader(Frame(t, &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_Text{Text: "echo"},
	})))
	require.NoError(t, err)
	frames := ReadFrames(t, resp.Body)
	require.NoError(t, resp.Body.Close())
	require.Len(t, frames, 1)
	var out vllmpb.GenerateResponse
	Unmarshal(t, frames[0], &out)
	require.Equal(t, "echo", out.GetOutputs().GetText())
	require.Equal(t, grpcframe.StatusOK, resp.Trailer.Get(grpcframe.HeaderStatus))

	resp, err = client.Post(srv.URL+"/trailers-only", grpcframe.MediaType, nil)
	require.NoError(t, err)
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	require.Empty(t, body)
	require.Equal(t, "5", resp.Header.Get(grpcframe.HeaderStatus))
	require.Equal(t, "no such thing", resp.Header.Get(grpcframe.HeaderMessage))

	resp, err = client.Post(srv.URL+"/status-after-body", grpcframe.MediaType, nil)
	require.NoError(t, err)
	_, err = io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	require.Equal(t, "13", resp.Trailer.Get(grpcframe.HeaderStatus))
	require.Equal(t, "after body", resp.Trailer.Get(grpcframe.HeaderMessage))
}

func TestVLLMUpstreamWithGRPCClient(t *testing.T) {
	up := NewVLLMUpstream(t, VLLMUpstreamOptions{
		Respond: func(call VLLMCall) ([]*vllmpb.GenerateResponse, error) {
			switch call.Request.GetText() {
			case "fail-trailers-only":
				return nil, status.Error(codes.ResourceExhausted, "busy")
			case "fail-after-frame":
				return []*vllmpb.GenerateResponse{{Outputs: &vllmpb.SequenceOutput{Text: "partial"}}},
					status.Error(codes.Aborted, "mid-stream")
			}
			if call.Stream {
				return []*vllmpb.GenerateResponse{
					{PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 2}, Outputs: &vllmpb.SequenceOutput{Text: "a", NumTokens: 1}},
					{Outputs: &vllmpb.SequenceOutput{Text: "b", NumTokens: 1, FinishInfo: &vllmpb.FinishInfo{NumOutputTokens: 2}}},
				}, nil
			}
			return []*vllmpb.GenerateResponse{DefaultVLLMResponse()}, nil
		},
	})
	client := vllmpb.NewInferenceClient(Dial(t, up.Addr()))
	ctx := metadata.AppendToOutgoingContext(t.Context(), "x-request-id", "rid-1")

	unary, err := client.Generate(ctx, &vllmpb.GenerateRequest{RequestId: "u1", Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}})
	require.NoError(t, err)
	require.Equal(t, "ok", unary.GetOutputs().GetText())

	stream, err := client.GenerateStream(ctx, &vllmpb.GenerateRequest{RequestId: "s1", Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}})
	require.NoError(t, err)
	var texts []string
	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		require.NoError(t, err)
		texts = append(texts, resp.GetOutputs().GetText())
	}
	require.Equal(t, []string{"a", "b"}, texts)

	_, err = client.Generate(ctx, &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "fail-trailers-only"}})
	require.Equal(t, codes.ResourceExhausted, status.Code(err))
	require.Equal(t, "busy", status.Convert(err).Message())

	stream, err = client.GenerateStream(ctx, &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "fail-after-frame"}})
	require.NoError(t, err)
	first, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, "partial", first.GetOutputs().GetText())
	_, err = stream.Recv()
	require.Equal(t, codes.Aborted, status.Code(err))

	calls := up.Calls()
	require.Len(t, calls, 4)
	require.Equal(t, vllmpb.Inference_Generate_FullMethodName, calls[0].Path)
	require.False(t, calls[0].Stream)
	require.Equal(t, "u1", calls[0].Request.GetRequestId())
	require.Equal(t, "rid-1", calls[0].Header.Get("x-request-id"))
	require.Equal(t, vllmpb.Inference_GenerateStream_FullMethodName, calls[1].Path)
	require.True(t, calls[1].Stream)
	last, ok := up.LastCall()
	require.True(t, ok)
	require.Equal(t, "fail-after-frame", last.Request.GetText())

	// Unknown methods are Unimplemented.
	_, err = vllmpb.NewControlClient(Dial(t, up.Addr())).GetServerInfo(ctx, &vllmpb.GetServerInfoRequest{})
	require.Equal(t, codes.Unimplemented, status.Code(err))
}

func TestVLLMUpstreamDefaultsAndRawClient(t *testing.T) {
	up := NewVLLMUpstream(t, VLLMUpstreamOptions{})
	client := NewH2CClient()

	resp, err := client.Post(up.URL()+vllmpb.Inference_Generate_FullMethodName, grpcframe.MediaType,
		bytes.NewReader(Frame(t, &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "raw"}})))
	require.NoError(t, err)
	frames := ReadFrames(t, resp.Body)
	require.NoError(t, resp.Body.Close())
	require.Len(t, frames, 1)
	var got vllmpb.GenerateResponse
	Unmarshal(t, frames[0], &got)
	require.True(t, proto.Equal(DefaultVLLMResponse(), &got))
	require.Equal(t, grpcframe.StatusOK, resp.Trailer.Get(grpcframe.HeaderStatus))
	_, ok := up.LastCall()
	require.True(t, ok)

	// A malformed frame gets an InvalidArgument trailers-only response.
	resp, err = client.Post(up.URL()+vllmpb.Inference_Generate_FullMethodName, grpcframe.MediaType, bytes.NewReader([]byte{0, 0, 0, 0, 9, 1}))
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	require.Equal(t, "3", resp.Header.Get(grpcframe.HeaderStatus))
}

func TestStartGRPCServer(t *testing.T) {
	addr := StartGRPCServer(t, func(s *grpc.Server) {
		vllmpb.RegisterInferenceServer(s, echoServer{})
	})
	client := vllmpb.NewInferenceClient(Dial(t, addr))
	resp, err := client.Generate(t.Context(), &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "ping"}})
	require.NoError(t, err)
	require.Equal(t, "ping", resp.GetOutputs().GetText())
}

type echoServer struct {
	vllmpb.UnimplementedInferenceServer
}

func (echoServer) Generate(_ context.Context, req *vllmpb.GenerateRequest) (*vllmpb.GenerateResponse, error) {
	return &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{Text: req.GetText()}}, nil
}
