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

package proxy

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
	"k8s.io/utils/set"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

func startCustomSidecar(t *testing.T, cfg Config, customize func(*Server)) string {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	srv := NewProxy(cfg)
	srv.allowlistValidator = &AllowlistValidator{enabled: false}
	done := make(chan error, 1)
	go func() {
		done <- srv.Start(ctx)
	}()
	select {
	case <-srv.readyCh:
	case err := <-done:
		require.NoError(t, err, "sidecar exited before ready")
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for sidecar listener")
	}
	if customize != nil {
		customize(srv)
	}
	t.Cleanup(func() {
		cancel()
		<-done
	})
	return srv.addr.String()
}

func newVLLMGRPCClient(t *testing.T, addr string) vllmpb.InferenceClient {
	t.Helper()
	return vllmpb.NewInferenceClient(grpctest.Dial(t, addr))
}

func makePrefillFinishResponse(t *testing.T, kvParams, ecParams map[string]any, cachedTokens uint32) *vllmpb.GenerateResponse {
	t.Helper()
	var kvSt, ecSt *structpb.Struct
	var err error
	if kvParams != nil {
		kvSt, err = structpb.NewStruct(kvParams)
		require.NoError(t, err)
	}
	if ecParams != nil {
		ecSt, err = structpb.NewStruct(ecParams)
		require.NoError(t, err)
	}
	return &vllmpb.GenerateResponse{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 6},
		Outputs: &vllmpb.SequenceOutput{
			Text:      "p",
			NumTokens: 1,
			TokenIds:  []uint32{99},
			FinishInfo: &vllmpb.FinishInfo{
				NumOutputTokens:  1,
				FinishReason:     vllmpb.FinishInfo_LENGTH,
				KvTransferParams: kvSt,
				EcTransferParams: ecSt,
				NumCachedTokens:  &cachedTokens,
			},
		},
	}
}

func collectStreamResponses(t *testing.T, stream grpc.ServerStreamingClient[vllmpb.GenerateResponse]) ([]*vllmpb.GenerateResponse, error) {
	t.Helper()
	var frames []*vllmpb.GenerateResponse
	for {
		msg, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return frames, nil
		}
		if err != nil {
			return frames, err
		}
		frames = append(frames, msg)
	}
}

func defaultMockVLLMStreamOrUnary(call grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
	zero := uint32(0)
	if !call.Stream {
		return []*vllmpb.GenerateResponse{grpctest.DefaultVLLMResponse()}, nil
	}
	return []*vllmpb.GenerateResponse{
		{
			PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
			Outputs:    &vllmpb.SequenceOutput{Text: "hey", NumTokens: 1, TokenIds: []uint32{10}},
		},
		{
			Outputs: &vllmpb.SequenceOutput{
				Text:      "lo",
				NumTokens: 1,
				TokenIds:  []uint32{11},
				FinishInfo: &vllmpb.FinishInfo{
					NumOutputTokens: 2,
					FinishReason:    vllmpb.FinishInfo_STOP,
					NumCachedTokens: &zero,
				},
			},
		},
	}, nil
}

type noopFlushResponseWriter struct {
	header http.Header
}

func (w *noopFlushResponseWriter) Header() http.Header { return w.header }
func (w *noopFlushResponseWriter) WriteHeader(int)     {}
func (w *noopFlushResponseWriter) Write(b []byte) (int, error) {
	return len(b), nil
}

func TestVLLMGRPCWireRoundTripAndAllocations(t *testing.T) {
	reqBytes, err := proto.Marshal(&vllmpb.GenerateRequest{
		RequestId: "r1",
		Model:     "m",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3, 4, 5}}},
		Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 64, MinNewTokens: 8},
		Kv:        &vllmpb.KVCacheParameters{CacheSalt: "s"},
		Media: []*vllmpb.MediaItem{
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Uuid:     "img-1",
				Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("raw-pixels")},
				MimeType: "image/png",
			},
		},
	})
	require.NoError(t, err)

	scanned, err := vllmwire.ScanGenerateRequest(reqBytes)
	require.NoError(t, err)

	opts := vllmwire.LegOptions{
		RequestID:        "req-override-1",
		CapSingleToken:   true,
		Media:            vllmwire.MediaMetadataOnly,
		KVTransferParams: map[string]any{"do_remote_decode": true, "do_remote_prefill": false},
	}

	wantFrame, err := scanned.Frame(opts)
	require.NoError(t, err)
	segs, err := scanned.FrameSegments(opts)
	require.NoError(t, err)
	wantSegBytes := bytes.Join(segs, nil)
	assert.Equal(t, int64(len(wantFrame)), grpcframe.SegmentsLen(segs))
	assert.Equal(t, len(wantFrame), len(wantSegBytes))

	// Verify grpcframe.NewSegmentsReader matches joined segments byte-for-byte across partial chunk sizes.
	for _, chunkSize := range []int{1, 3, 7, 64} {
		r := grpcframe.NewSegmentsReader(segs)
		var out bytes.Buffer
		buf := make([]byte, chunkSize)
		for {
			n, readErr := r.Read(buf)
			if n > 0 {
				out.Write(buf[:n])
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			require.NoError(t, readErr)
		}
		require.NoError(t, r.Close())
		assert.Equal(t, wantSegBytes, out.Bytes())
	}

	var decodedLeg, decodedSegLeg vllmpb.GenerateRequest
	grpctest.Unmarshal(t, wantFrame, &decodedLeg)
	grpctest.Unmarshal(t, wantSegBytes, &decodedSegLeg)
	assert.True(t, proto.Equal(&decodedLeg, &decodedSegLeg))
	assert.Equal(t, "req-override-1", decodedLeg.GetRequestId())
	assert.Equal(t, uint32(1), decodedLeg.GetStopping().GetMaxNewTokens())
	assert.Equal(t, uint32(0), decodedLeg.GetStopping().GetMinNewTokens())
	require.Len(t, decodedLeg.GetMedia(), 1)
	assert.Empty(t, decodedLeg.GetMedia()[0].GetRawBytes())
	assert.Equal(t, "img-1", decodedLeg.GetMedia()[0].GetUuid())

	segAllocs := testing.AllocsPerRun(100, func() {
		_, buildErr := scanned.FrameSegments(vllmwire.LegOptions{
			Media: vllmwire.MediaNone,
		})
		if buildErr != nil {
			t.Fatal(buildErr)
		}
	})
	assert.LessOrEqual(t, segAllocs, float64(2), "vectored leg segments must allocate at most 2 slices")

	// Non-terminal chunk frames must pass through with 0 allocations.
	deltaFrame := grpctest.Frame(t, &vllmpb.GenerateResponse{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 12},
		Outputs: &vllmpb.SequenceOutput{
			Text:      "hello ",
			NumTokens: 1,
			TokenIds:  []uint32{15339},
		},
	})
	wireAllocs := testing.AllocsPerRun(100, func() {
		got, rwErr := vllmwire.RewriteTerminalCachedTokens(deltaFrame, 32)
		if rwErr != nil || len(got) != len(deltaFrame) {
			t.Fatal("unexpected rewrite failure")
		}
	})
	assert.Equal(t, float64(0), wireAllocs, "non-terminal token frames must not allocate in wire rewriter")

	sink := &noopFlushResponseWriter{header: make(http.Header)}
	rw := newVLLMCachedTokensResponseWriter(sink, 32)
	rw.WriteHeader(http.StatusOK)
	// Warm internal bytes.Buffer once to steady-state stream capacity.
	_, err = rw.Write(deltaFrame)
	require.NoError(t, err)
	streamAllocs := testing.AllocsPerRun(100, func() {
		if _, wErr := rw.Write(deltaFrame); wErr != nil {
			t.Fatal(wErr)
		}
	})
	assert.Equal(t, float64(0), streamAllocs, "steady-state non-terminal frame relay must not allocate")

	zero := uint32(0)
	kvOut, err := structpb.NewStruct(map[string]any{"remote_engine_id": "e1"})
	require.NoError(t, err)
	terminalFrame := grpctest.Frame(t, &vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{
			Text:      "world",
			NumTokens: 1,
			TokenIds:  []uint32{1917},
			FinishInfo: &vllmpb.FinishInfo{
				NumOutputTokens:  2,
				FinishReason:     vllmpb.FinishInfo_STOP,
				StopReason:       &vllmpb.FinishInfo_EosTokenId{EosTokenId: 128001},
				KvTransferParams: kvOut,
				NumCachedTokens:  &zero,
			},
		},
	})
	for _, targetCached := range []uint32{48, 0} {
		rewritten, rwErr := vllmwire.RewriteTerminalCachedTokens(terminalFrame, targetCached)
		require.NoError(t, rwErr)
		var got vllmpb.GenerateResponse
		grpctest.Unmarshal(t, rewritten, &got)
		require.NotNil(t, got.GetOutputs().GetFinishInfo().NumCachedTokens)
		assert.Equal(t, targetCached, got.GetOutputs().GetFinishInfo().GetNumCachedTokens())
		assert.Equal(t, uint32(128001), got.GetOutputs().GetFinishInfo().GetEosTokenId())
		assert.Equal(t, "world", got.GetOutputs().GetText())
	}
}

func TestExtractVLLMPrefillFinishInfoFailureModes(t *testing.T) {
	t.Run("http_error", func(t *testing.T) {
		pw := &bufferedResponseWriter{statusCode: http.StatusServiceUnavailable}
		out, err := extractVLLMPrefillFinishInfo(pw, nil)
		require.Error(t, err)
		assert.True(t, out.relayError)
	})

	t.Run("grpc_trailer_error", func(t *testing.T) {
		pw := &bufferedResponseWriter{}
		grpcframe.WriteError(pw, codes.InvalidArgument, "bad prompt")
		out, err := extractVLLMPrefillFinishInfo(pw, nil)
		require.Error(t, err)
		assert.True(t, out.relayError)
	})

	t.Run("missing_grpc_status_trailer", func(t *testing.T) {
		pw := &bufferedResponseWriter{statusCode: http.StatusOK}
		pw.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		_, _ = pw.Write(grpctest.Frame(t, &vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH}},
		}))
		out, err := extractVLLMPrefillFinishInfo(pw, nil)
		require.Error(t, err)
		assert.False(t, out.relayError)
	})

	t.Run("truncated_frame", func(t *testing.T) {
		pw := &bufferedResponseWriter{statusCode: http.StatusOK}
		pw.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		pw.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
		_, _ = pw.Write([]byte{0, 0, 0, 0, 10, 1, 2})
		out, err := extractVLLMPrefillFinishInfo(pw, nil)
		require.Error(t, err)
		assert.False(t, out.relayError)
	})

	t.Run("missing_finish_info", func(t *testing.T) {
		pw := &bufferedResponseWriter{statusCode: http.StatusOK}
		pw.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		pw.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
		_, _ = pw.Write(grpctest.Frame(t, &vllmpb.GenerateResponse{
			PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 3},
		}))
		out, err := extractVLLMPrefillFinishInfo(pw, nil)
		require.Error(t, err)
		assert.False(t, out.relayError)
	})
}

func TestVLLMGRPCPassthroughWithoutPrefillOrEncoder(t *testing.T) {
	decodeBackend := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
		Respond: defaultMockVLLMStreamOrUnary,
	})
	decodeURL, err := url.Parse(decodeBackend.URL())
	require.NoError(t, err)

	sidecarAddr := startCustomSidecar(t, Config{
		Port:        "0",
		DecoderURL:  decodeURL,
		KVConnector: constants.KVConnectorNIXLV2,
	}, nil)
	client := newVLLMGRPCClient(t, sidecarAddr)

	// Decoder-only passthrough without P/D, EC, or x-llm-d-kv-cache-source
	// streams the request body unread, matching TestServer_DecoderOnlyPassthroughLeavesBodyUnread
	// (coordinator ingress strips client transfer params in coordinator deployments).
	passthroughKV, err := structpb.NewStruct(map[string]any{"custom_key": "kept_on_passthrough"})
	require.NoError(t, err)
	req := &vllmpb.GenerateRequest{
		RequestId: "passthrough-1",
		Model:     "llama",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 10, 20}}},
		Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 32, MinNewTokens: 8},
		Kv:        &vllmpb.KVCacheParameters{CacheSalt: "salt-pass", KvTransferParams: passthroughKV},
		Media: []*vllmpb.MediaItem{
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Uuid:     "pass-img-0",
				Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("passthrough-pixels")},
				MimeType: "image/jpeg",
			},
		},
	}

	unaryResp, err := client.Generate(t.Context(), req)
	require.NoError(t, err)
	assert.Equal(t, "ok", unaryResp.GetOutputs().GetText())

	stream, err := client.GenerateStream(t.Context(), req)
	require.NoError(t, err)
	frames, err := collectStreamResponses(t, stream)
	require.NoError(t, err)
	require.Len(t, frames, 2)
	assert.Equal(t, "hey", frames[0].GetOutputs().GetText())
	assert.Equal(t, "lo", frames[1].GetOutputs().GetText())

	calls := decodeBackend.Calls()
	require.Len(t, calls, 2)
	assert.False(t, calls[0].Stream)
	assert.True(t, calls[1].Stream)
	for _, c := range calls {
		assert.Equal(t, uint32(32), c.Request.GetStopping().GetMaxNewTokens())
		assert.Equal(t, uint32(8), c.Request.GetStopping().GetMinNewTokens())
		assert.Equal(t, "kept_on_passthrough", c.Request.GetKv().GetKvTransferParams().AsMap()["custom_key"])
		require.Len(t, c.Request.GetMedia(), 1)
		assert.Equal(t, []byte("passthrough-pixels"), c.Request.GetMedia()[0].GetRawBytes())
	}
}

func TestVLLMGRPCPDConnectorsUnaryAndStream(t *testing.T) {
	p2pTierPort := 29555

	// Mock Mooncake /query bootstrap HTTP server.
	mcBootstrap := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/query", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"2":{"engine_id":"mc-engine-rank-2"}}`))
	}))
	defer mcBootstrap.Close()
	mcBootstrapURL, err := url.Parse(mcBootstrap.URL)
	require.NoError(t, err)
	mcBootstrapPort, err := strconv.Atoi(mcBootstrapURL.Port())
	require.NoError(t, err)

	tests := []struct {
		name               string
		connector          string
		moriWrite          bool
		moriParallel       bool
		promptIsText       bool
		prefillCachedCount uint32
		wantCachedCount    uint32
		extraHeaders       func(prefillAddr string) map[string]string
		verifyLegs         func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, prefillAddr string)
	}{
		{
			name:               "nixlv2_read_tok_tok",
			connector:          constants.KVConnectorNIXLV2,
			prefillCachedCount: 24,
			wantCachedCount:    24,
			verifyLegs: func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, _ string) {
				require.NotEmpty(t, prefillCall.Request.GetRequestId())
				assert.NotEqual(t, "client-req-99", prefillCall.Request.GetRequestId())
				assert.Equal(t, prefillCall.Request.GetRequestId(), decodeCall.Request.GetRequestId())

				pkv := prefillCall.Request.GetKv().GetKvTransferParams().AsMap()
				assert.Equal(t, true, pkv["do_remote_decode"])
				assert.Equal(t, false, pkv["do_remote_prefill"])

				dkv := decodeCall.Request.GetKv().GetKvTransferParams().AsMap()
				assert.Equal(t, true, dkv["do_remote_prefill"])
				assert.Equal(t, "nixl-eng-1", dkv["remote_engine_id"])
			},
		},
		{
			name:               "nixlv2_read_text_text_zero_cached",
			connector:          constants.KVConnectorNIXLV2,
			promptIsText:       true,
			prefillCachedCount: 0,
			wantCachedCount:    0,
			verifyLegs: func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, _ string) {
				require.NotEmpty(t, prefillCall.Request.GetRequestId())
				assert.NotEqual(t, "client-req-99", prefillCall.Request.GetRequestId())
				assert.Equal(t, prefillCall.Request.GetRequestId(), decodeCall.Request.GetRequestId())
				assert.Equal(t, "Hello from EPP-Text-Text", prefillCall.Request.GetText())
				assert.Equal(t, "Hello from EPP-Text-Text", decodeCall.Request.GetText())
			},
		},
		{
			name:               "nixlv2_moriio_write_serial",
			connector:          constants.KVConnectorNIXLV2,
			moriWrite:          true,
			prefillCachedCount: 18,
			wantCachedCount:    18,
			verifyLegs: func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, _ string) {
				require.NotEmpty(t, prefillCall.Request.GetRequestId())
				assert.NotEqual(t, "client-req-99", prefillCall.Request.GetRequestId())
				assert.Equal(t, prefillCall.Request.GetRequestId(), decodeCall.Request.GetRequestId())

				pkv := prefillCall.Request.GetKv().GetKvTransferParams().AsMap()
				assert.Equal(t, "10.9.0.2", pkv["remote_host"])
				assert.Equal(t, float64(17001), pkv["remote_notify_port"])
				assert.NotEmpty(t, prefillCall.Header.Get(requestHeaderDataParallelRank))

				dkv := decodeCall.Request.GetKv().GetKvTransferParams().AsMap()
				assert.True(t, strings.HasPrefix(dkv[requestFieldTransferID].(string), "tx"))
				assert.Equal(t, float64(17001), dkv["remote_notify_port"])
				assert.NotEmpty(t, decodeCall.Header.Get(requestHeaderDataParallelRank))
			},
		},
		{
			name:               "nixlv2_moriio_write_parallel",
			connector:          constants.KVConnectorNIXLV2,
			moriWrite:          true,
			moriParallel:       true,
			prefillCachedCount: 12,
			wantCachedCount:    12,
			verifyLegs: func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, prefillAddr string) {
				pkv := prefillCall.Request.GetKv().GetKvTransferParams().AsMap()
				dkv := decodeCall.Request.GetKv().GetKvTransferParams().AsMap()
				require.NotEmpty(t, prefillCall.Request.GetRequestId())
				assert.NotEqual(t, "client-req-99", prefillCall.Request.GetRequestId())
				require.Equal(t, dkv["remote_request_id"], prefillCall.Request.GetRequestId())
				assert.Equal(t, prefillCall.Request.GetRequestId(), decodeCall.Request.GetRequestId())
				assert.Equal(t, pkv[requestFieldTransferID], dkv[requestFieldTransferID])
				assert.Equal(t, extractHost(prefillAddr), dkv["remote_host"])
				assert.Equal(t, true, dkv["do_remote_prefill"])
			},
		},
		{
			name:               "mooncake",
			connector:          constants.KVConnectorMooncake,
			prefillCachedCount: 10,
			wantCachedCount:    3, // concurrent P/D streams decode directly
			verifyLegs: func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, _ string) {
				pkv := prefillCall.Request.GetKv().GetKvTransferParams().AsMap()
				dkv := decodeCall.Request.GetKv().GetKvTransferParams().AsMap()
				assert.Equal(t, "2", prefillCall.Header.Get(requestHeaderDataParallelRank))
				assert.Equal(t, true, pkv["do_remote_decode"])
				assert.Equal(t, false, pkv["do_remote_prefill"])
				assert.Equal(t, pkv["transfer_id"], dkv["transfer_id"])
				assert.Equal(t, "mc-engine-rank-2", dkv["remote_engine_id"])
				assert.Contains(t, dkv["remote_bootstrap_addr"].(string), strconv.Itoa(mcBootstrapPort))
			},
		},
		{
			name:               "p2p_offloading",
			connector:          constants.KVConnectorOffloading,
			prefillCachedCount: 40,
			wantCachedCount:    40,
			extraHeaders: func(prefillAddr string) map[string]string {
				return map[string]string{
					routing.KVCachePrefillHeader: net.JoinHostPort(extractHost(prefillAddr), strconv.Itoa(p2pTierPort)),
				}
			},
			verifyLegs: func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, prefillAddr string) {
				pkv := prefillCall.Request.GetKv().GetKvTransferParams().AsMap()
				dkv := decodeCall.Request.GetKv().GetKvTransferParams().AsMap()
				rd, ok := pkv["remote_decoder"].(map[string]any)
				require.True(t, ok)
				rp, ok := dkv["remote_prefiller"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, rd["kv_request_id"], rp["kv_request_id"])
				assert.Equal(t, extractHost(prefillAddr), rp["remote_host"])
				assert.Equal(t, float64(p2pTierPort), rp["remote_port"])
			},
		},
		{
			name:               "shared_storage",
			connector:          constants.KVConnectorSharedStorage,
			prefillCachedCount: 28,
			wantCachedCount:    28,
			verifyLegs: func(t *testing.T, prefillCall, decodeCall grpctest.VLLMCall, _ string) {
				assert.Nil(t, prefillCall.Request.GetKv().GetKvTransferParams())
				assert.Nil(t, decodeCall.Request.GetKv().GetKvTransferParams())
			},
		},
	}

	for _, tc := range tests {
		for _, streamRPC := range []bool{false, true} {
			mode := "unary"
			if streamRPC {
				mode = "stream"
			}
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				prefillKVOut := map[string]any{
					"do_remote_prefill": true,
					"do_remote_decode":  false,
					"remote_engine_id":  "nixl-eng-1",
					"remote_block_ids":  []any{7, 8},
					"remote_host":       "10.9.0.1",
					"remote_port":       5555,
				}
				if tc.connector != constants.KVConnectorNIXLV2 {
					prefillKVOut = nil
				}

				prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
					Respond: func(call grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
						return []*vllmpb.GenerateResponse{
							makePrefillFinishResponse(t, prefillKVOut, nil, tc.prefillCachedCount),
						}, nil
					},
				})
				decodeCached := uint32(3)
				decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
					Respond: func(call grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
						if !call.Stream {
							return []*vllmpb.GenerateResponse{
								{
									PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 6},
									Outputs: &vllmpb.SequenceOutput{
										Text:      "decoded-unary",
										NumTokens: 2,
										TokenIds:  []uint32{501, 502},
										FinishInfo: &vllmpb.FinishInfo{
											NumOutputTokens: 2,
											FinishReason:    vllmpb.FinishInfo_STOP,
											NumCachedTokens: &decodeCached,
										},
									},
								},
							}, nil
						}
						return []*vllmpb.GenerateResponse{
							{
								PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 6},
								Outputs:    &vllmpb.SequenceOutput{Text: "tok1-", NumTokens: 1, TokenIds: []uint32{501}},
							},
							{
								Outputs: &vllmpb.SequenceOutput{Text: "tok2", NumTokens: 1, TokenIds: []uint32{502}},
							},
							{
								Outputs: &vllmpb.SequenceOutput{
									FinishInfo: &vllmpb.FinishInfo{
										NumOutputTokens: 2,
										FinishReason:    vllmpb.FinishInfo_STOP,
										NumCachedTokens: &decodeCached,
									},
								},
							},
						}, nil
					},
				})

				prefillAddr := prefillSrv.Addr()
				decodeURL, err := url.Parse(decodeSrv.URL())
				require.NoError(t, err)

				cfg := Config{
					Port:                       "0",
					DecoderURL:                 decodeURL,
					KVConnector:                tc.connector,
					MooncakeBootstrapPort:      mcBootstrapPort,
					MoRIIOWriteMode:            tc.moriWrite,
					MoRIIOParallelDispatch:     tc.moriParallel,
					MoRIIODecodePodIP:          "10.9.0.2",
					MoRIIODecodeNotifyPort:     17001,
					MoRIIODecodeHandshakePort:  17002,
					MoRIIOPrefillHandshakePort: 18002,
					MoRIIOPrefillNotifyPort:    18001,
					MoRIIOTPSize:               2,
					MoRIIODPSize:               4,
					MoRIIODPSizeLocal:          2,
					MoRIIORemoteHosts:          []string{extractHost(prefillAddr), "10.9.0.99"},
				}
				sidecarAddr := startCustomSidecar(t, cfg, nil)
				client := newVLLMGRPCClient(t, sidecarAddr)

				untrustedKV, err := structpb.NewStruct(map[string]any{"attacker_key": "drop_me"})
				require.NoError(t, err)
				req := &vllmpb.GenerateRequest{
					RequestId: "client-req-99",
					Model:     "Qwen/Qwen2.5-72B",
					Stopping: &vllmpb.StoppingCriteria{
						MaxNewTokens:       64,
						MinNewTokens:       12,
						StopStrings:        []string{"<|eot_id|>"},
						IncludeStopStrings: true,
					},
					Kv: &vllmpb.KVCacheParameters{
						BypassPrefixCache: true,
						CacheSalt:         "salt-xyz",
						KvTransferParams:  untrustedKV,
					},
					Priority: 5,
					LoraName: "adapter-v1",
				}
				if tc.promptIsText {
					req.Prompt = &vllmpb.GenerateRequest_Text{Text: "Hello from EPP-Text-Text"}
				} else {
					req.Prompt = &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 100, 200, 300}}}
					req.Media = []*vllmpb.MediaItem{
						{
							Modality: vllmpb.Modality_MODALITY_IMAGE,
							Uuid:     "pd-img-0",
							Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("pd-pixels")},
							MimeType: "image/png",
						},
					}
				}

				mdPairs := []string{routing.PrefillEndpointHeader, prefillAddr}
				if tc.extraHeaders != nil {
					for k, v := range tc.extraHeaders(prefillAddr) {
						mdPairs = append(mdPairs, k, v)
					}
				}
				callCtx := metadata.AppendToOutgoingContext(t.Context(), mdPairs...)

				if !streamRPC {
					resp, rpcErr := client.Generate(callCtx, req)
					require.NoError(t, rpcErr)
					assert.Equal(t, "decoded-unary", resp.GetOutputs().GetText())
					require.NotNil(t, resp.GetOutputs().GetFinishInfo().NumCachedTokens)
					assert.Equal(t, tc.wantCachedCount, resp.GetOutputs().GetFinishInfo().GetNumCachedTokens())
				} else {
					st, rpcErr := client.GenerateStream(callCtx, req)
					require.NoError(t, rpcErr)
					frames, recvErr := collectStreamResponses(t, st)
					require.NoError(t, recvErr)
					require.Len(t, frames, 3)
					assert.Equal(t, "tok1-", frames[0].GetOutputs().GetText())
					assert.Equal(t, "tok2", frames[1].GetOutputs().GetText())
					require.NotNil(t, frames[2].GetOutputs().GetFinishInfo().NumCachedTokens)
					assert.Equal(t, tc.wantCachedCount, frames[2].GetOutputs().GetFinishInfo().GetNumCachedTokens())
				}

				// Mooncake runs prefill asynchronously. Wait briefly for prefill call capture.
				require.Eventually(t, func() bool {
					return len(prefillSrv.Calls()) == 1
				}, 2*time.Second, 10*time.Millisecond)
				pCalls := prefillSrv.Calls()
				dCalls := decodeSrv.Calls()
				require.Len(t, pCalls, 1)
				require.Len(t, dCalls, 1)

				// Prefill always uses unary /vllm.Inference/Generate even when client called GenerateStream.
				assert.False(t, pCalls[0].Stream)
				assert.Equal(t, vllmpb.Inference_Generate_FullMethodName, pCalls[0].Path)
				assert.Equal(t, streamRPC, dCalls[0].Stream)

				pReq := pCalls[0].Request
				dReq := dCalls[0].Request

				// Prefill leg must cap max_new_tokens=1 and clear min_new_tokens=0.
				assert.Equal(t, uint32(1), pReq.GetStopping().GetMaxNewTokens())
				assert.Equal(t, uint32(0), pReq.GetStopping().GetMinNewTokens())
				assert.Equal(t, []string{"<|eot_id|>"}, pReq.GetStopping().GetStopStrings())
				assert.True(t, pReq.GetStopping().GetIncludeStopStrings())

				// Decode leg must preserve client stopping criteria verbatim.
				assert.Equal(t, uint32(64), dReq.GetStopping().GetMaxNewTokens())
				assert.Equal(t, uint32(12), dReq.GetStopping().GetMinNewTokens())

				// When no EC ran, prefill receives MediaAll while decode behind prefill receives MediaNone.
				if !tc.promptIsText {
					require.Len(t, pReq.GetMedia(), 1)
					assert.Equal(t, []byte("pd-pixels"), pReq.GetMedia()[0].GetRawBytes())
					assert.Empty(t, dReq.GetMedia())
				}

				// Both legs must preserve kv.cache_salt, kv.bypass_prefix_cache, lora_name, priority, and strip untrusted keys.
				for _, legReq := range []*vllmpb.GenerateRequest{pReq, dReq} {
					assert.Equal(t, "salt-xyz", legReq.GetKv().GetCacheSalt())
					assert.True(t, legReq.GetKv().GetBypassPrefixCache())
					assert.Equal(t, "adapter-v1", legReq.GetLoraName())
					assert.Equal(t, int32(5), legReq.GetPriority())
					if legReq.GetKv().GetKvTransferParams() != nil {
						assert.NotContains(t, legReq.GetKv().GetKvTransferParams().AsMap(), "attacker_key")
					}
					if !tc.promptIsText {
						assert.Equal(t, []uint32{1, 100, 200, 300}, legReq.GetTokenIds().GetIds())
					}
				}

				tc.verifyLegs(t, pCalls[0], dCalls[0], prefillAddr)
			})
		}
	}
}

func TestVLLMGRPCP2PSourcePullAndSSRFStripping(t *testing.T) {
	decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
	dpDecodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
	decodeURL, err := url.Parse(decodeSrv.URL())
	require.NoError(t, err)
	dpDecodeURL, err := url.Parse(dpDecodeSrv.URL())
	require.NoError(t, err)

	allowedPeerHost := "10.20.0.5"
	dpTarget := "10.20.0.9:8001"
	sidecarAddr := startCustomSidecar(t, Config{
		Port:          "0",
		DecoderURL:    decodeURL,
		KVConnector:   constants.KVConnectorNIXLV2,
		EnableP2PPull: true,
	}, func(s *Server) {
		s.forwardDataParallel = true
		s.dataParallelProxies[dpTarget] = s.createDecoderProxyHandler(dpDecodeURL, false)
		s.allowlistValidator = &AllowlistValidator{
			enabled:        true,
			allowedTargets: set.New(net.JoinHostPort(allowedPeerHost, "8000")),
			allowedHosts:   set.New(allowedPeerHost),
		}
	})
	client := newVLLMGRPCClient(t, sidecarAddr)

	media := []*vllmpb.MediaItem{
		{
			Modality: vllmpb.Modality_MODALITY_IMAGE,
			Uuid:     "src-img",
			Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("src-pixels")},
		},
	}

	// 1. Allowed remote_kv_source on decoder-only request: kept and sanitized.
	validSourceKV, err := structpb.NewStruct(map[string]any{
		"attacker_extra": "drop",
		"remote_kv_source": map[string]any{
			"kv_request_id": "cached-req-1",
			"remote_host":   allowedPeerHost,
			"remote_port":   28888,
			"extra_field":   "drop",
		},
	})
	require.NoError(t, err)

	ctxAllowed := metadata.AppendToOutgoingContext(t.Context(),
		routing.KVCacheSourceHeader, net.JoinHostPort(allowedPeerHost, "28888"),
	)
	_, err = client.Generate(ctxAllowed, &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		Kv:     &vllmpb.KVCacheParameters{CacheSalt: "s1", KvTransferParams: validSourceKV},
		Media:  media,
	})
	require.NoError(t, err)

	// 2. Denied remote_kv_source host: stripped via lazy rebuildSanitizedVLLMGRPCRequest before forwarding.
	deniedSourceKV, err := structpb.NewStruct(map[string]any{
		"remote_kv_source": map[string]any{
			"kv_request_id": "cached-req-2",
			"remote_host":   "169.254.169.254",
			"remote_port":   28888,
		},
	})
	require.NoError(t, err)

	ctxDenied := metadata.AppendToOutgoingContext(t.Context(),
		routing.KVCacheSourceHeader, "169.254.169.254:28888",
	)
	_, err = client.Generate(ctxDenied, &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		Kv:     &vllmpb.KVCacheParameters{CacheSalt: "s2", KvTransferParams: deniedSourceKV},
		Media:  media,
	})
	require.NoError(t, err)

	calls := decodeSrv.Calls()
	require.Len(t, calls, 2)

	gotAllowedKV := calls[0].Request.GetKv().GetKvTransferParams().AsMap()
	assert.NotContains(t, gotAllowedKV, "attacker_extra")
	srcMap, ok := gotAllowedKV["remote_kv_source"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "cached-req-1", srcMap["kv_request_id"])
	assert.Equal(t, allowedPeerHost, srcMap["remote_host"])
	assert.NotContains(t, srcMap, "extra_field")
	require.Len(t, calls[0].Request.GetMedia(), 1)
	assert.Equal(t, []byte("src-pixels"), calls[0].Request.GetMedia()[0].GetRawBytes())

	// Rejected source must be completely absent while MediaAll stays intact.
	assert.Equal(t, "s2", calls[1].Request.GetKv().GetCacheSalt())
	assert.Nil(t, calls[1].Request.GetKv().GetKvTransferParams())
	require.Len(t, calls[1].Request.GetMedia(), 1)
	assert.Equal(t, []byte("src-pixels"), calls[1].Request.GetMedia()[0].GetRawBytes())

	// 3. Allowed remote_kv_source + x-data-parallel-host-port: DP rank receives sanitized remote_kv_source.
	ctxDPAllowed := metadata.AppendToOutgoingContext(t.Context(),
		routing.KVCacheSourceHeader, net.JoinHostPort(allowedPeerHost, "28888"),
		routing.DataParallelEndpointHeader, dpTarget,
	)
	_, err = client.Generate(ctxDPAllowed, &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		Kv:     &vllmpb.KVCacheParameters{CacheSalt: "s3", KvTransferParams: validSourceKV},
		Media:  media,
	})
	require.NoError(t, err)

	// 4. Denied remote_kv_source + x-data-parallel-host-port: DP rank receives sanitized frame without source.
	ctxDPDenied := metadata.AppendToOutgoingContext(t.Context(),
		routing.KVCacheSourceHeader, "169.254.169.254:28888",
		routing.DataParallelEndpointHeader, dpTarget,
	)
	_, err = client.Generate(ctxDPDenied, &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		Kv:     &vllmpb.KVCacheParameters{CacheSalt: "s4", KvTransferParams: deniedSourceKV},
		Media:  media,
	})
	require.NoError(t, err)

	require.Len(t, decodeSrv.Calls(), 2, "DP-forwarded requests must not hit default decoder")
	dpCalls := dpDecodeSrv.Calls()
	require.Len(t, dpCalls, 2)

	gotDPAllowedKV := dpCalls[0].Request.GetKv().GetKvTransferParams().AsMap()
	assert.NotContains(t, gotDPAllowedKV, "attacker_extra")
	dpSrcMap, ok := gotDPAllowedKV["remote_kv_source"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "cached-req-1", dpSrcMap["kv_request_id"])
	assert.Equal(t, allowedPeerHost, dpSrcMap["remote_host"])
	assert.NotContains(t, dpSrcMap, "extra_field")
	require.Len(t, dpCalls[0].Request.GetMedia(), 1)
	assert.Equal(t, []byte("src-pixels"), dpCalls[0].Request.GetMedia()[0].GetRawBytes())

	assert.Equal(t, "s4", dpCalls[1].Request.GetKv().GetCacheSalt())
	assert.Nil(t, dpCalls[1].Request.GetKv().GetKvTransferParams())
	require.Len(t, dpCalls[1].Request.GetMedia(), 1)
	assert.Equal(t, []byte("src-pixels"), dpCalls[1].Request.GetMedia()[0].GetRawBytes())
}

func TestVLLMGRPCECDisaggregationConnectors(t *testing.T) {
	for _, tc := range []struct {
		name        string
		ecConnector string
		withPrefill bool
		useFeatures bool
	}{
		{name: "ec_nixl_epd_features", ecConnector: constants.ECConnectorNIXL, withPrefill: true, useFeatures: true},
		{name: "ec_nixl_ed_features", ecConnector: constants.ECConnectorNIXL, withPrefill: false, useFeatures: true},
		{name: "ec_mooncake_epd_raw_media", ecConnector: constants.ECConnectorMooncake, withPrefill: true, useFeatures: false},
		{name: "ec_mooncake_ed_raw_media", ecConnector: constants.ECConnectorMooncake, withPrefill: false, useFeatures: false},
		{name: "ec_shared_storage_epd", ecConnector: constants.ECExampleConnector, withPrefill: true, useFeatures: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			makeEncResponder := func(encName, fallbackPrefix string) func(grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
				return func(call grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
					ecOut := make(map[string]any, len(call.Request.GetMedia()))
					for i, item := range call.Request.GetMedia() {
						key := fmt.Sprintf("%s-%d", fallbackPrefix, i)
						if item.GetUuid() != "" {
							key = item.GetUuid()
						}
						ecOut[key] = map[string]any{
							"encoder":  encName,
							"metadata": map[string]any{"tokens": 2},
						}
					}
					return []*vllmpb.GenerateResponse{makePrefillFinishResponse(t, nil, ecOut, 0)}, nil
				}
			}

			encSrv0 := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
				Respond: makeEncResponder("enc0", "mm-hash-0"),
			})
			encSrv1 := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
				Respond: makeEncResponder("enc1", "mm-hash-1"),
			})
			prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
				Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
					return []*vllmpb.GenerateResponse{
						makePrefillFinishResponse(t, map[string]any{
							"do_remote_prefill": true,
							"remote_engine_id":  "prefill-e0",
						}, nil, 14),
					}, nil
				},
			})
			decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
				Respond: defaultMockVLLMStreamOrUnary,
			})
			decodeURL, err := url.Parse(decodeSrv.URL())
			require.NoError(t, err)

			sidecarAddr := startCustomSidecar(t, Config{
				Port:            "0",
				DecoderURL:      decodeURL,
				KVConnector:     constants.KVConnectorNIXLV2,
				ECConnector:     tc.ecConnector,
				ECMooncakePort:  29777,
				ECMooncakePodIP: "10.50.0.9",
			}, nil)
			client := newVLLMGRPCClient(t, sidecarAddr)

			var promptIDs []uint32
			var media []*vllmpb.MediaItem
			if tc.useFeatures {
				// Prompt: [BOS=1, img0=(900,900), text=42, img1=(900,900), end=99]
				promptIDs = []uint32{1, 900, 900, 42, 900, 900, 99}
				h0, h1 := "mm-hash-0", "mm-hash-1"
				media = []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     h0,
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{
								Kwargs: []byte{0x81, 0xa1, 0x78, 0x01},
								Offset: 1,
								Length: 2,
								MmHash: &h0,
							},
						},
					},
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     h1,
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{
								Kwargs: []byte{0x81, 0xa1, 0x79, 0x02},
								Offset: 4,
								Length: 2,
								MmHash: &h1,
							},
						},
					},
				}
			} else {
				// Vision start/pad/end markers around two raw images:
				// [BOS=1, vision_start=151652, image_pad=151655, vision_end=151653, text=42, vision_start=151652, image_pad=151655, vision_end=151653, end=99]
				promptIDs = []uint32{1, 151652, 151655, 151653, 42, 151652, 151655, 151653, 99}
				media = []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("img-bytes-0")},
						MimeType: "image/jpeg",
					},
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("img-bytes-1")},
						MimeType: "image/jpeg",
					},
				}
			}

			mdPairs := []string{routing.EncoderEndpointsHeader, encSrv0.Addr() + "," + encSrv1.Addr()}
			if tc.withPrefill {
				mdPairs = append(mdPairs, routing.PrefillEndpointHeader, prefillSrv.Addr())
			}
			callCtx := metadata.AppendToOutgoingContext(t.Context(), mdPairs...)

			st, err := client.GenerateStream(callCtx, &vllmpb.GenerateRequest{
				RequestId: "epd-req-1",
				Model:     "Qwen/Qwen2.5-VL",
				Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: promptIDs}},
				Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 40, MinNewTokens: 5},
				Media:     media,
			})
			require.NoError(t, err)
			frames, err := collectStreamResponses(t, st)
			require.NoError(t, err)
			require.Len(t, frames, 2)

			e0Calls := encSrv0.Calls()
			e1Calls := encSrv1.Calls()
			if tc.useFeatures {
				// Preprocessed features fan out one unary sub-request per item across encoders.
				require.Len(t, e0Calls, 1)
				require.Len(t, e1Calls, 1)
				assert.False(t, e0Calls[0].Stream)
				assert.False(t, e1Calls[0].Stream)
				assert.Equal(t, uint32(1), e0Calls[0].Request.GetStopping().GetMaxNewTokens())
				assert.Equal(t, uint32(1), e1Calls[0].Request.GetStopping().GetMaxNewTokens())
				require.Len(t, e0Calls[0].Request.GetMedia(), 1)
				require.Len(t, e1Calls[0].Request.GetMedia(), 1)
				assert.NotEmpty(t, e0Calls[0].Request.GetMedia()[0].GetFeatures().GetKwargs())
				assert.NotEmpty(t, e1Calls[0].Request.GetMedia()[0].GetFeatures().GetKwargs())
				assert.Equal(t, []uint32{1, 900, 900}, e0Calls[0].Request.GetTokenIds().GetIds())
				assert.Equal(t, uint64(1), e0Calls[0].Request.GetMedia()[0].GetFeatures().GetOffset())
				assert.Equal(t, []uint32{1, 900, 900}, e1Calls[0].Request.GetTokenIds().GetIds())
				assert.Equal(t, uint64(1), e1Calls[0].Request.GetMedia()[0].GetFeatures().GetOffset())
			} else {
				// Raw media sends one unary full-prompt request so start/end markers pair cleanly.
				require.Len(t, e0Calls, 1)
				assert.Empty(t, e1Calls)
				assert.False(t, e0Calls[0].Stream)
				assert.Equal(t, uint32(1), e0Calls[0].Request.GetStopping().GetMaxNewTokens())
				assert.Equal(t, promptIDs, e0Calls[0].Request.GetTokenIds().GetIds())
				require.Len(t, e0Calls[0].Request.GetMedia(), 2)
				assert.Equal(t, []byte("img-bytes-0"), e0Calls[0].Request.GetMedia()[0].GetRawBytes())
				assert.Equal(t, []byte("img-bytes-1"), e0Calls[0].Request.GetMedia()[1].GetRawBytes())
			}

			dCalls := decodeSrv.Calls()
			require.Len(t, dCalls, 1)
			assert.True(t, dCalls[0].Stream)

			if tc.withPrefill {
				pCalls := prefillSrv.Calls()
				require.Len(t, pCalls, 1)
				assert.False(t, pCalls[0].Stream)
				if tc.ecConnector != constants.ECExampleConnector {
					require.NotNil(t, pCalls[0].Request.GetKv().GetEcTransferParams())
					assert.NotEmpty(t, pCalls[0].Request.GetKv().GetEcTransferParams().AsMap())
				}
				// Prefill after encode receives MediaMetadataOnly (no pixels or kwargs).
				require.Len(t, pCalls[0].Request.GetMedia(), 2)
				for i, m := range pCalls[0].Request.GetMedia() {
					assert.Empty(t, m.GetRawBytes(), "prefill media[%d] must strip raw_bytes after encode", i)
					assert.Empty(t, m.GetFeatures().GetKwargs(), "prefill media[%d] must strip features.kwargs after encode", i)
					assert.NotEmpty(t, m.GetUuid())
				}
				// Decode behind a prefiller receives MediaNone and no ec_transfer_params.
				assert.Empty(t, dCalls[0].Request.GetMedia())
				assert.Nil(t, dCalls[0].Request.GetKv().GetEcTransferParams())
				assert.Equal(t, uint32(14), frames[1].GetOutputs().GetFinishInfo().GetNumCachedTokens())
			} else {
				// Without a prefiller, decode is the EC consumer: it receives MediaMetadataOnly and ec_transfer_params.
				require.Len(t, dCalls[0].Request.GetMedia(), 2)
				for i, m := range dCalls[0].Request.GetMedia() {
					assert.Empty(t, m.GetRawBytes(), "decode media[%d] must strip raw_bytes after encode", i)
					assert.Empty(t, m.GetFeatures().GetKwargs(), "decode media[%d] must strip features.kwargs after encode", i)
				}
				if tc.ecConnector != constants.ECExampleConnector {
					require.NotNil(t, dCalls[0].Request.GetKv().GetEcTransferParams())
					ecMap := dCalls[0].Request.GetKv().GetEcTransferParams().AsMap()
					assert.NotEmpty(t, ecMap)
					if tc.ecConnector == constants.ECConnectorMooncake {
						assert.Len(t, ecMap[ecMooncakeFieldItems], 2)
						for i := range 2 {
							assert.NotEmpty(t, dCalls[0].Request.GetMedia()[i].GetUuid())
							assert.Equal(t, e0Calls[0].Request.GetMedia()[i].GetUuid(), dCalls[0].Request.GetMedia()[i].GetUuid())
						}
					}
				}
			}
		})
	}
}

func TestVLLMGRPCPrefillRetryOnlyWhenUncommitted(t *testing.T) {
	t.Run("trailers_only_unavailable_retries", func(t *testing.T) {
		var attempts atomic.Int32
		prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
			Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
				if attempts.Add(1) == 1 {
					return nil, status.Error(codes.Unavailable, "accept queue full")
				}
				return []*vllmpb.GenerateResponse{
					makePrefillFinishResponse(t, map[string]any{
						"do_remote_prefill": true,
						"remote_engine_id":  "nixl-retry-ok",
					}, nil, 9),
				}, nil
			},
		})
		decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
		decodeURL, err := url.Parse(decodeSrv.URL())
		require.NoError(t, err)

		sidecarAddr := startCustomSidecar(t, Config{
			Port:                "0",
			DecoderURL:          decodeURL,
			KVConnector:         constants.KVConnectorNIXLV2,
			PrefillMaxRetries:   2,
			PrefillRetryBackoff: time.Millisecond,
		}, nil)
		client := newVLLMGRPCClient(t, sidecarAddr)

		callCtx := metadata.AppendToOutgoingContext(t.Context(), routing.PrefillEndpointHeader, prefillSrv.Addr())
		resp, err := client.Generate(callCtx, &vllmpb.GenerateRequest{
			Model:  "m",
			Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
		})
		require.NoError(t, err)
		assert.Equal(t, uint32(9), resp.GetOutputs().GetFinishInfo().GetNumCachedTokens())
		assert.Len(t, prefillSrv.Calls(), 2)
		assert.Len(t, decodeSrv.Calls(), 1)
	})

	t.Run("partial_frame_before_unavailable_does_not_retry", func(t *testing.T) {
		prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
			Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
				return []*vllmpb.GenerateResponse{
					{PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4}},
				}, status.Error(codes.Unavailable, "engine lost mid-prefill")
			},
		})
		decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
		decodeURL, err := url.Parse(decodeSrv.URL())
		require.NoError(t, err)

		sidecarAddr := startCustomSidecar(t, Config{
			Port:                "0",
			DecoderURL:          decodeURL,
			KVConnector:         constants.KVConnectorNIXLV2,
			PrefillMaxRetries:   3,
			PrefillRetryBackoff: time.Millisecond,
		}, nil)
		client := newVLLMGRPCClient(t, sidecarAddr)

		callCtx := metadata.AppendToOutgoingContext(t.Context(), routing.PrefillEndpointHeader, prefillSrv.Addr())
		_, err = client.Generate(callCtx, &vllmpb.GenerateRequest{
			Model:  "m",
			Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
		})
		require.Error(t, err)
		assert.Equal(t, codes.Unavailable, status.Code(err))
		assert.Contains(t, status.Convert(err).Message(), "engine lost mid-prefill")
		assert.Len(t, prefillSrv.Calls(), 1, "prefill must not retry after streaming any body frame")
		assert.Empty(t, decodeSrv.Calls())
	})
}

func TestVLLMGRPCPrefillErrorsNeverDispatchDecode(t *testing.T) {
	for _, tc := range []struct {
		name        string
		prefillH2C  http.HandlerFunc
		wantCode    codes.Code
		wantMsgPart string
	}{
		{
			name: "trailers_only_invalid_argument",
			prefillH2C: func(w http.ResponseWriter, _ *http.Request) {
				grpcframe.WriteError(w, codes.InvalidArgument, "prompt exceeds max model len")
			},
			wantCode:    codes.InvalidArgument,
			wantMsgPart: "prompt exceeds max model len",
		},
		{
			name: "mid_stream_trailer_error",
			prefillH2C: func(w http.ResponseWriter, _ *http.Request) {
				frame := grpctest.Frame(t, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
				})
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(frame)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				grpctest.WriteStatusTrailer(w, codes.Unavailable, "kv engine overloaded")
			},
			wantCode:    codes.Unavailable,
			wantMsgPart: "kv engine overloaded",
		},
		{
			name: "truncated_prefill_frame",
			prefillH2C: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.WriteHeader(http.StatusOK)
				// Declare 32-byte payload, send only 3 bytes, and still emit grpc-status=0.
				_, _ = w.Write([]byte{0, 0, 0, 0, 32, 1, 2, 3})
				grpctest.WriteStatusTrailer(w, codes.OK, "")
			},
			wantCode: codes.Unavailable,
		},
		{
			name: "stream_ended_without_finish_info",
			prefillH2C: func(w http.ResponseWriter, _ *http.Request) {
				grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
				})
			},
			wantCode: codes.Unavailable,
		},
	} {
		for _, streamRPC := range []bool{false, true} {
			mode := "unary"
			if streamRPC {
				mode = "stream"
			}
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				prefillH2C := grpctest.NewH2CServer(t, tc.prefillH2C)
				prefillURL, err := url.Parse(prefillH2C.URL)
				require.NoError(t, err)

				decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
					Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
						return nil, status.Error(codes.Internal, "decode must not be called")
					},
				})
				decodeURL, err := url.Parse(decodeSrv.URL())
				require.NoError(t, err)

				sidecarAddr := startCustomSidecar(t, Config{
					Port:        "0",
					DecoderURL:  decodeURL,
					KVConnector: constants.KVConnectorNIXLV2,
				}, nil)
				client := newVLLMGRPCClient(t, sidecarAddr)

				callCtx := metadata.AppendToOutgoingContext(t.Context(),
					routing.PrefillEndpointHeader, prefillURL.Host,
				)
				req := &vllmpb.GenerateRequest{
					Model:  "m",
					Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
				}

				var rpcErr error
				if !streamRPC {
					_, rpcErr = client.Generate(callCtx, req)
				} else {
					st, startErr := client.GenerateStream(callCtx, req)
					if startErr != nil {
						rpcErr = startErr
					} else {
						_, rpcErr = collectStreamResponses(t, st)
					}
				}
				require.Error(t, rpcErr)
				assert.Equal(t, tc.wantCode, status.Code(rpcErr))
				if tc.wantMsgPart != "" {
					assert.Contains(t, status.Convert(rpcErr).Message(), tc.wantMsgPart)
				}
				assert.Empty(t, decodeSrv.Calls(), "decode must never be dispatched after prefill failure")
			})
		}
	}
}

func TestVLLMGRPCMalformedClientFramesRejectedBeforeUpstream(t *testing.T) {
	var prefillCalled, decodeCalled atomic.Bool
	prefillH2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		prefillCalled.Store(true)
		grpcframe.WriteError(w, codes.OK, "")
	}))
	decodeH2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		decodeCalled.Store(true)
		grpcframe.WriteError(w, codes.OK, "")
	}))
	prefillURL, err := url.Parse(prefillH2C.URL)
	require.NoError(t, err)
	decodeURL, err := url.Parse(decodeH2C.URL)
	require.NoError(t, err)

	sidecarBaseURL := startTestSidecar(t, Config{
		Port:        "0",
		DecoderURL:  decodeURL,
		KVConnector: constants.KVConnectorNIXLV2,
	})
	h2cClient := grpctest.NewH2CClient()

	for _, tc := range []struct {
		name     string
		encoding string
		body     []byte
		wantCode codes.Code
	}{
		{name: "truncated_header", body: []byte{0, 0, 0}, wantCode: codes.InvalidArgument},
		{name: "compressed_flag_set", body: []byte{1, 0, 0, 0, 0}, wantCode: codes.Unimplemented},
		{name: "grpc_encoding_gzip", encoding: "gzip", body: []byte{0, 0, 0, 0, 0}, wantCode: codes.Unimplemented},
		{name: "truncated_payload", body: []byte{0, 0, 0, 0, 10, 1, 2}, wantCode: codes.InvalidArgument},
		{name: "corrupt_protobuf", body: []byte{0, 0, 0, 0, 2, 0xff, 0xff}, wantCode: codes.InvalidArgument},
		{name: "frame_exceeds_max_payload", body: []byte{0, 0xff, 0xff, 0xff, 0xff}, wantCode: codes.ResourceExhausted},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req, err := http.NewRequestWithContext(
				t.Context(),
				http.MethodPost,
				sidecarBaseURL+vllmpb.Inference_Generate_FullMethodName,
				bytes.NewReader(tc.body),
			)
			require.NoError(t, err)
			req.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			if tc.encoding != "" {
				req.Header.Set(grpcframe.HeaderEncoding, tc.encoding)
			}
			req.Header.Set(routing.PrefillEndpointHeader, prefillURL.Host)

			resp, err := h2cClient.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			_, _ = io.ReadAll(resp.Body)

			codeStr, ok := grpcframe.Status(resp.Header)
			if !ok {
				codeStr, ok = grpcframe.Status(resp.Trailer)
			}
			require.True(t, ok)
			assert.Equal(t, strconv.Itoa(int(tc.wantCode)), codeStr)
			assert.False(t, prefillCalled.Load())
			assert.False(t, decodeCalled.Load())
		})
	}
}

type errorDetailsInferenceServer struct {
	vllmpb.UnimplementedInferenceServer
	err error
}

func (s *errorDetailsInferenceServer) Generate(context.Context, *vllmpb.GenerateRequest) (*vllmpb.GenerateResponse, error) {
	return nil, s.err
}

func (s *errorDetailsInferenceServer) GenerateStream(*vllmpb.GenerateRequest, grpc.ServerStreamingServer[vllmpb.GenerateResponse]) error {
	return s.err
}

func TestVLLMGRPCDecodeRelayErrors(t *testing.T) {
	t.Run("trailers_only_error_with_details", func(t *testing.T) {
		stWithDetails, err := status.New(codes.ResourceExhausted, "decoder kv cache full").WithDetails(
			structpb.NewStringValue("kv_pool_saturated"),
		)
		require.NoError(t, err)

		decodeAddr := grpctest.StartGRPCServer(t, func(gs *grpc.Server) {
			vllmpb.RegisterInferenceServer(gs, &errorDetailsInferenceServer{err: stWithDetails.Err()})
		})
		decodeURL, err := url.Parse("http://" + decodeAddr)
		require.NoError(t, err)

		prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
			Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
				return []*vllmpb.GenerateResponse{
					makePrefillFinishResponse(t, map[string]any{
						"do_remote_prefill": true,
						"remote_engine_id":  "prefill-ok",
					}, nil, 16),
				}, nil
			},
		})

		sidecarAddr := startCustomSidecar(t, Config{
			Port:        "0",
			DecoderURL:  decodeURL,
			KVConnector: constants.KVConnectorNIXLV2,
		}, nil)
		client := newVLLMGRPCClient(t, sidecarAddr)

		for _, withPrefill := range []bool{false, true} {
			callCtx := t.Context()
			if withPrefill {
				callCtx = metadata.AppendToOutgoingContext(callCtx, routing.PrefillEndpointHeader, prefillSrv.Addr())
			}
			_, rpcErr := client.Generate(callCtx, &vllmpb.GenerateRequest{
				Model:  "m",
				Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
			})
			require.Error(t, rpcErr)
			gotSt := status.Convert(rpcErr)
			assert.Equal(t, codes.ResourceExhausted, gotSt.Code())
			assert.Equal(t, "decoder kv cache full", gotSt.Message())
			details := gotSt.Details()
			require.Len(t, details, 1)
			val, ok := details[0].(*structpb.Value)
			require.True(t, ok)
			assert.Equal(t, "kv_pool_saturated", val.GetStringValue())
		}
	})

	t.Run("mid_stream_trailer_error", func(t *testing.T) {
		prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
			Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
				return []*vllmpb.GenerateResponse{
					makePrefillFinishResponse(t, map[string]any{
						"do_remote_prefill": true,
						"remote_engine_id":  "prefill-ok",
					}, nil, 11),
				}, nil
			},
		})
		decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
			Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
				return []*vllmpb.GenerateResponse{
					{
						PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
						Outputs:    &vllmpb.SequenceOutput{Text: "partial-", NumTokens: 1, TokenIds: []uint32{7}},
					},
				}, status.Error(codes.Internal, "decode GPU fault")
			},
		})
		decodeURL, err := url.Parse(decodeSrv.URL())
		require.NoError(t, err)

		sidecarAddr := startCustomSidecar(t, Config{
			Port:        "0",
			DecoderURL:  decodeURL,
			KVConnector: constants.KVConnectorNIXLV2,
		}, nil)
		client := newVLLMGRPCClient(t, sidecarAddr)

		callCtx := metadata.AppendToOutgoingContext(t.Context(), routing.PrefillEndpointHeader, prefillSrv.Addr())
		st, err := client.GenerateStream(callCtx, &vllmpb.GenerateRequest{
			Model:  "m",
			Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		})
		require.NoError(t, err)
		frames, recvErr := collectStreamResponses(t, st)
		require.Error(t, recvErr)
		require.Len(t, frames, 1)
		assert.Equal(t, "partial-", frames[0].GetOutputs().GetText())
		assert.Equal(t, codes.Internal, status.Code(recvErr))
		assert.Contains(t, status.Convert(recvErr).Message(), "decode GPU fault")
	})

	t.Run("decoder_connection_refused", func(t *testing.T) {
		var lc net.ListenConfig
		closedLn, err := lc.Listen(t.Context(), "tcp", "127.0.0.1:0")
		require.NoError(t, err)
		closedAddr := closedLn.Addr().String()
		require.NoError(t, closedLn.Close())

		decodeURL, err := url.Parse("http://" + closedAddr)
		require.NoError(t, err)
		sidecarAddr := startCustomSidecar(t, Config{
			Port:        "0",
			DecoderURL:  decodeURL,
			KVConnector: constants.KVConnectorNIXLV2,
		}, nil)
		client := newVLLMGRPCClient(t, sidecarAddr)

		_, rpcErr := client.Generate(t.Context(), &vllmpb.GenerateRequest{
			Model:  "m",
			Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		})
		require.Error(t, rpcErr)
		assert.Equal(t, codes.Unavailable, status.Code(rpcErr))
	})
}

func TestVLLMCachedTokensResponseWriter(t *testing.T) {
	t.Run("split_writes_across_frame_boundary", func(t *testing.T) {
		zero := uint32(0)
		frame1 := grpctest.Frame(t, &vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{Text: "a", NumTokens: 1, TokenIds: []uint32{1}},
		})
		frame2 := grpctest.Frame(t, &vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{
				Text:      "b",
				NumTokens: 1,
				TokenIds:  []uint32{2},
				FinishInfo: &vllmpb.FinishInfo{
					NumOutputTokens: 2,
					FinishReason:    vllmpb.FinishInfo_STOP,
					NumCachedTokens: &zero,
				},
			},
		})
		stream := append(append([]byte{}, frame1...), frame2...)

		rec := httptest.NewRecorder()
		rw := newVLLMCachedTokensResponseWriter(rec, 37)
		for i := 0; i < len(stream); i += 3 {
			end := min(i+3, len(stream))
			_, err := rw.Write(stream[i:end])
			require.NoError(t, err)
		}
		req := httptest.NewRequest(http.MethodPost, vllmpb.Inference_GenerateStream_FullMethodName, nil)
		rw.finish(req)

		frames := grpctest.ReadFrames(t, rec.Body)
		require.Len(t, frames, 2)
		var gotTerminal vllmpb.GenerateResponse
		grpctest.Unmarshal(t, frames[1], &gotTerminal)
		assert.Equal(t, uint32(37), gotTerminal.GetOutputs().GetFinishInfo().GetNumCachedTokens())
	})

	t.Run("trailing_partial_frame_marks_unavailable", func(t *testing.T) {
		rec := httptest.NewRecorder()
		rw := newVLLMCachedTokensResponseWriter(rec, 10)
		rec.Header().Set(http.TrailerPrefix+"Grpc-Status", grpcframe.StatusOK)
		rw.WriteHeader(http.StatusOK)
		_, err := rw.Write([]byte{0, 0, 0, 0, 20, 1, 2})
		require.NoError(t, err)

		req := httptest.NewRequest(http.MethodPost, vllmpb.Inference_Generate_FullMethodName, nil)
		req.Header.Set(reqcommon.RequestIDHeaderKey, "req-trunc-1")
		rw.finish(req)

		code, ok := grpcframe.Status(rec.Header())
		require.True(t, ok)
		assert.Equal(t, strconv.Itoa(int(codes.Unavailable)), code)
	})

	t.Run("compressed_flag_falls_back_to_passthrough", func(t *testing.T) {
		rec := httptest.NewRecorder()
		rw := newVLLMCachedTokensResponseWriter(rec, 10)
		compressed := []byte{1, 0, 0, 0, 2, 0xaa, 0xbb}
		_, err := rw.Write(compressed)
		require.NoError(t, err)
		assert.True(t, rw.passthrough)
		assert.Equal(t, compressed, rec.Body.Bytes())
	})
}

func TestVLLMGRPCDataParallelDecodeRoutingWithCachedTokens(t *testing.T) {
	prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
		Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
			return []*vllmpb.GenerateResponse{
				makePrefillFinishResponse(t, map[string]any{
					"do_remote_prefill": true,
					"remote_engine_id":  "dp-prefill-1",
				}, nil, 19),
			}, nil
		},
	})
	defaultDecodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
	dpRankDecodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
		Respond: defaultMockVLLMStreamOrUnary,
	})

	defaultDecodeURL, err := url.Parse(defaultDecodeSrv.URL())
	require.NoError(t, err)
	dpRankDecodeURL, err := url.Parse(dpRankDecodeSrv.URL())
	require.NoError(t, err)

	dpTarget := "10.0.0.2:8001"
	sidecarAddr := startCustomSidecar(t, Config{
		Port:        "0",
		DecoderURL:  defaultDecodeURL,
		KVConnector: constants.KVConnectorNIXLV2,
	}, func(s *Server) {
		s.forwardDataParallel = true
		s.dataParallelProxies[dpTarget] = s.createDecoderProxyHandler(dpRankDecodeURL, false)
	})
	client := newVLLMGRPCClient(t, sidecarAddr)

	callCtx := metadata.AppendToOutgoingContext(t.Context(),
		routing.PrefillEndpointHeader, prefillSrv.Addr(),
		routing.DataParallelEndpointHeader, dpTarget,
	)
	req := &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 10, 20}}},
		Media: []*vllmpb.MediaItem{
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Uuid:     "dp-img",
				Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("dp-pixels")},
			},
		},
	}

	unaryResp, err := client.Generate(callCtx, req)
	require.NoError(t, err)
	assert.Equal(t, uint32(19), unaryResp.GetOutputs().GetFinishInfo().GetNumCachedTokens())

	st, err := client.GenerateStream(callCtx, req)
	require.NoError(t, err)
	frames, err := collectStreamResponses(t, st)
	require.NoError(t, err)
	require.Len(t, frames, 2)
	assert.Equal(t, uint32(19), frames[1].GetOutputs().GetFinishInfo().GetNumCachedTokens())

	assert.Empty(t, defaultDecodeSrv.Calls(), "default decoder must not be called when DP header selects rank")
	dpCalls := dpRankDecodeSrv.Calls()
	require.Len(t, dpCalls, 2)
	for _, c := range dpCalls {
		assert.Empty(t, c.Request.GetMedia())
		assert.Equal(t, "dp-prefill-1", c.Request.GetKv().GetKvTransferParams().AsMap()["remote_engine_id"])
	}
}

func TestVLLMGRPCECEncoderFailureCancelsPeersAndAbortsPD(t *testing.T) {
	peerStarted := make(chan struct{})
	peerCanceled := make(chan struct{})

	enc0H2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		<-peerStarted
		grpcframe.WriteError(w, codes.ResourceExhausted, "encoder GPU OOM")
	}))
	enc1H2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(peerStarted)
		<-r.Context().Done()
		close(peerCanceled)
	}))
	enc0URL, err := url.Parse(enc0H2C.URL)
	require.NoError(t, err)
	enc1URL, err := url.Parse(enc1H2C.URL)
	require.NoError(t, err)

	prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
	decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
	decodeURL, err := url.Parse(decodeSrv.URL())
	require.NoError(t, err)

	sidecarAddr := startCustomSidecar(t, Config{
		Port:        "0",
		DecoderURL:  decodeURL,
		KVConnector: constants.KVConnectorNIXLV2,
		ECConnector: constants.ECConnectorNIXL,
	}, nil)
	client := newVLLMGRPCClient(t, sidecarAddr)

	hash0, hash1 := "img-fail-0", "img-peer-1"
	req := &vllmpb.GenerateRequest{
		RequestId: "ec-cancel-req",
		Model:     "Qwen2-VL",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 151655, 151655, 42, 151655, 151655}}},
		Media: []*vllmpb.MediaItem{
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Uuid:     hash0,
				Source: &vllmpb.MediaItem_Features{
					Features: &vllmpb.PreprocessedMediaFeatures{Offset: 1, Length: 2},
				},
			},
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Uuid:     hash1,
				Source: &vllmpb.MediaItem_Features{
					Features: &vllmpb.PreprocessedMediaFeatures{Offset: 4, Length: 2},
				},
			},
		},
	}

	callCtx := metadata.AppendToOutgoingContext(t.Context(),
		routing.EncoderEndpointsHeader, enc0URL.Host+","+enc1URL.Host,
		routing.PrefillEndpointHeader, prefillSrv.Addr(),
	)
	_, rpcErr := client.Generate(callCtx, req)
	require.Error(t, rpcErr)
	assert.Equal(t, codes.Unavailable, status.Code(rpcErr))

	select {
	case <-peerCanceled:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for peer encoder cancellation")
	}
	assert.Empty(t, prefillSrv.Calls(), "prefill must not run when encoder stage fails")
	assert.Empty(t, decodeSrv.Calls(), "decode must not run when encoder stage fails")
}

func TestVLLMGRPCMoRIIOParallelPrefillErrorAbortsDecode(t *testing.T) {
	decodeStarted := make(chan struct{})
	decodeCanceled := make(chan struct{})

	prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
		Respond: func(_ grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
			<-decodeStarted
			return nil, status.Error(codes.InvalidArgument, "prefill rejected MoRI-IO request")
		},
	})
	decodeH2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(decodeStarted)
		<-r.Context().Done()
		close(decodeCanceled)
	}))
	decodeURL, err := url.Parse(decodeH2C.URL)
	require.NoError(t, err)

	sidecarAddr := startCustomSidecar(t, Config{
		Port:                       "0",
		DecoderURL:                 decodeURL,
		KVConnector:                constants.KVConnectorNIXLV2,
		MoRIIOWriteMode:            true,
		MoRIIOParallelDispatch:     true,
		MoRIIODecodePodIP:          "10.9.0.2",
		MoRIIODecodeNotifyPort:     17001,
		MoRIIODecodeHandshakePort:  17002,
		MoRIIOPrefillHandshakePort: 18002,
		MoRIIOPrefillNotifyPort:    18001,
		MoRIIOTPSize:               2,
		MoRIIODPSize:               2,
		MoRIIODPSizeLocal:          2,
	}, nil)
	client := newVLLMGRPCClient(t, sidecarAddr)

	callCtx := metadata.AppendToOutgoingContext(t.Context(), routing.PrefillEndpointHeader, prefillSrv.Addr())
	_, rpcErr := client.Generate(callCtx, &vllmpb.GenerateRequest{
		RequestId: "mori-parallel-fail",
		Model:     "llama",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
		Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 16},
	})
	require.Error(t, rpcErr)
	assert.Equal(t, codes.InvalidArgument, status.Code(rpcErr))
	assert.Contains(t, status.Convert(rpcErr).Message(), "prefill rejected MoRI-IO request")

	select {
	case <-decodeCanceled:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for parallel decode cancellation")
	}
}

func TestVLLMGRPCMooncakeStreamPrefillMidStreamErrorDropsPartialFrames(t *testing.T) {
	decodeStarted := make(chan struct{})
	decodeCanceled := make(chan struct{})

	prefillSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
		Respond: func(call grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
			assert.False(t, call.Stream, "Mooncake prefill must dispatch unary Generate")
			<-decodeStarted
			return []*vllmpb.GenerateResponse{
				{PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4}},
			}, status.Error(codes.Internal, "mooncake prefill engine crashed")
		},
	})
	decodeH2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(decodeStarted)
		<-r.Context().Done()
		close(decodeCanceled)
	}))
	decodeURL, err := url.Parse(decodeH2C.URL)
	require.NoError(t, err)

	bootstrapSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"0":{"engine_id":"mc-engine-0"}}`))
	}))
	t.Cleanup(bootstrapSrv.Close)
	bootstrapURL, err := url.Parse(bootstrapSrv.URL)
	require.NoError(t, err)
	_, bootstrapPortStr, err := net.SplitHostPort(bootstrapURL.Host)
	require.NoError(t, err)
	bootstrapPort, err := strconv.Atoi(bootstrapPortStr)
	require.NoError(t, err)

	sidecarAddr := startCustomSidecar(t, Config{
		Port:                  "0",
		DecoderURL:            decodeURL,
		KVConnector:           constants.KVConnectorMooncake,
		MooncakeBootstrapPort: bootstrapPort,
	}, nil)
	client := newVLLMGRPCClient(t, sidecarAddr)

	callCtx := metadata.AppendToOutgoingContext(t.Context(), routing.PrefillEndpointHeader, prefillSrv.Addr())
	st, err := client.GenerateStream(callCtx, &vllmpb.GenerateRequest{
		RequestId: "mc-stream-fail",
		Model:     "llama",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
	})
	if err == nil {
		var frames []*vllmpb.GenerateResponse
		frames, err = collectStreamResponses(t, st)
		assert.Empty(t, frames, "partial prefill Start frame must not leak to client")
	}
	require.Error(t, err)
	assert.Equal(t, codes.Internal, status.Code(err))
	assert.Contains(t, status.Convert(err).Message(), "mooncake prefill engine crashed")

	select {
	case <-decodeCanceled:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for mooncake decode cancellation")
	}
}

func TestVLLMGRPCClientCancellationPropagatesToWorkers(t *testing.T) {
	t.Run("cancel_during_prefill", func(t *testing.T) {
		prefillStarted := make(chan struct{})
		prefillCanceled := make(chan struct{})

		prefillH2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			close(prefillStarted)
			<-r.Context().Done()
			close(prefillCanceled)
		}))
		prefillURL, err := url.Parse(prefillH2C.URL)
		require.NoError(t, err)

		decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
		decodeURL, err := url.Parse(decodeSrv.URL())
		require.NoError(t, err)

		sidecarAddr := startCustomSidecar(t, Config{
			Port:        "0",
			DecoderURL:  decodeURL,
			KVConnector: constants.KVConnectorNIXLV2,
		}, nil)
		client := newVLLMGRPCClient(t, sidecarAddr)

		callCtx, cancel := context.WithCancel(metadata.AppendToOutgoingContext(t.Context(),
			routing.PrefillEndpointHeader, prefillURL.Host,
		))
		defer cancel()

		errCh := make(chan error, 1)
		go func() {
			_, rpcErr := client.Generate(callCtx, &vllmpb.GenerateRequest{
				Model:  "m",
				Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
			})
			errCh <- rpcErr
		}()

		select {
		case <-prefillStarted:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for prefill to start")
		}
		cancel()

		select {
		case <-prefillCanceled:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for prefill cancellation")
		}
		require.Error(t, <-errCh)
		assert.Empty(t, decodeSrv.Calls())
	})

	t.Run("cancel_during_decode_stream", func(t *testing.T) {
		decodeCanceled := make(chan struct{})
		decodeH2C := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(grpctest.Frame(t, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{Text: "first", NumTokens: 1, TokenIds: []uint32{1}},
			}))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
			close(decodeCanceled)
		}))
		decodeURL, err := url.Parse(decodeH2C.URL)
		require.NoError(t, err)

		sidecarAddr := startCustomSidecar(t, Config{
			Port:        "0",
			DecoderURL:  decodeURL,
			KVConnector: constants.KVConnectorNIXLV2,
		}, nil)
		client := newVLLMGRPCClient(t, sidecarAddr)

		callCtx, cancel := context.WithCancel(t.Context())
		defer cancel()

		st, err := client.GenerateStream(callCtx, &vllmpb.GenerateRequest{
			Model:  "m",
			Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		})
		require.NoError(t, err)
		first, err := st.Recv()
		require.NoError(t, err)
		assert.Equal(t, "first", first.GetOutputs().GetText())

		cancel()
		select {
		case <-decodeCanceled:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for decode stream cancellation")
		}
	})
}

func TestVLLMGRPCEncodedPathRedirectOverH2C(t *testing.T) {
	decodeSrv := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{})
	decodeURL, err := url.Parse(decodeSrv.URL())
	require.NoError(t, err)

	sidecarBaseURL := startTestSidecar(t, Config{
		Port:        "0",
		DecoderURL:  decodeURL,
		KVConnector: constants.KVConnectorNIXLV2,
	})
	h2cClient := grpctest.NewH2CClient()
	h2cClient.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}

	for _, tc := range []struct {
		rawPath      string
		wantLocation string
	}{
		{
			rawPath:      "/vllm.Inference%2FGenerate",
			wantLocation: vllmpb.Inference_Generate_FullMethodName,
		},
		{
			rawPath:      "/vllm.Inference%2FGenerateStream?trace=1",
			wantLocation: vllmpb.Inference_GenerateStream_FullMethodName + "?trace=1",
		},
	} {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, sidecarBaseURL+tc.rawPath, http.NoBody)
		require.NoError(t, err)
		req.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)

		resp, err := h2cClient.Do(req)
		require.NoError(t, err)
		_ = resp.Body.Close()

		assert.Equal(t, http.StatusTemporaryRedirect, resp.StatusCode)
		assert.Equal(t, tc.wantLocation, resp.Header.Get("Location"))
	}
	assert.Empty(t, decodeSrv.Calls())
}
