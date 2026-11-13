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

package steps

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func scanTestVLLMRequest(t *testing.T, pb *vllmpb.GenerateRequest) vllmwire.Request {
	t.Helper()
	payload, err := proto.Marshal(pb)
	require.NoError(t, err)
	req, err := vllmwire.ScanGenerateRequest(payload)
	require.NoError(t, err)
	return req
}

func TestVLLMGRPC_RejectsSGLangConnector(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	gwClient := gateway.New(config.GatewayConfig{Address: "http://127.0.0.1:1"})
	prefill, err := NewPrefillStep(gwClient, map[string]any{ParamKVConnector: kv.SGLang})
	require.NoError(t, err)
	decode, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.SGLang})
	require.NoError(t, err)

	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-1",
		KVTransferParams: map[string]any{},
		ResponseWriter:   httptest.NewRecorder(),
		GRPC: &pipeline.VLLMGRPCRequest{
			Req: scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"},
			}),
		},
	}

	require.ErrorContains(t, prefill.Execute(ctx, reqCtx), "does not support kv-sglang")
	require.ErrorContains(t, decode.Execute(ctx, reqCtx), "does not support kv-sglang")
}

func TestVLLMGRPC_JSONStepsSkipVLLMRequests(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var renderCalls atomic.Int32
	renderSrv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		renderCalls.Add(1)
	}))
	t.Cleanup(renderSrv.Close)

	renderStep, err := NewRenderStep(nil, map[string]any{"address": renderSrv.URL})
	require.NoError(t, err)
	replaceStep, err := NewReplaceMediaURLsStep(nil, nil)
	require.NoError(t, err)

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-skip",
		OriginalPath: vllmpb.Inference_Generate_FullMethodName,
		GRPC: &pipeline.VLLMGRPCRequest{
			Req: scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
			}),
		},
	}

	require.NoError(t, replaceStep.Execute(ctx, reqCtx))
	require.NoError(t, renderStep.Execute(ctx, reqCtx))
	require.Zero(t, renderCalls.Load())
}

func TestVLLMGRPC_EncodeSkipsTextOnlyAndForwardsECOnDirectEncodeDecode(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var (
		mu          sync.Mutex
		encodeCalls int
		encodeReq   *vllmpb.GenerateRequest
		decodeReq   *vllmpb.GenerateRequest
	)

	gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var genReq vllmpb.GenerateRequest
		grpctest.ReadRequest(t, r, &genReq)

		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhaseEncode:
			mu.Lock()
			encodeCalls++
			encodeReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
			mu.Unlock()
			ecStruct, err := structpb.NewStruct(map[string]any{
				"hash-img-1": map[string]any{"remote_host": "10.9.0.1", "remote_port": 5555},
			})
			require.NoError(t, err)
			grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{
					FinishInfo: &vllmpb.FinishInfo{
						FinishReason:     vllmpb.FinishInfo_LENGTH,
						NumOutputTokens:  1,
						EcTransferParams: ecStruct,
					},
				},
			})
		case gateway.PhaseDecode:
			mu.Lock()
			decodeReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
			mu.Unlock()
			grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 12},
				Outputs: &vllmpb.SequenceOutput{
					TokenIds: []uint32{42},
					FinishInfo: &vllmpb.FinishInfo{
						FinishReason:    vllmpb.FinishInfo_STOP,
						NumOutputTokens: 1,
						NumCachedTokens: proto.Uint32(4),
					},
				},
			})
		default:
			t.Errorf("unexpected phase %q", r.Header.Get(reqcommon.EPPProfileHeaderKey))
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	encodeStep, err := NewEncodeStep(gwClient, map[string]any{ParamECConnector: ec.NIXL})
	require.NoError(t, err)
	decodeStep, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL, ParamECConnector: ec.NIXL})
	require.NoError(t, err)

	// 1. Text-only request skips EncodeStep with 0 fanout.
	textCtx := &pipeline.RequestContext{
		RequestID:        "req-text",
		KVTransferParams: map[string]any{},
		ResponseWriter:   httptest.NewRecorder(),
		GRPC: &pipeline.VLLMGRPCRequest{
			Req: scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "text only"},
			}),
		},
	}
	require.NoError(t, encodeStep.Execute(ctx, textCtx))
	require.Zero(t, textCtx.EncodeFanout)
	mu.Lock()
	require.Zero(t, encodeCalls)
	mu.Unlock()

	// 2. Multimodal request in an encode -> decode pipeline (no prefill)
	// caps encode to 1 token, sends full media to encode, strips raw media
	// payloads to metadata-only on decode, and forwards ec_transfer_params.
	mmCtx := &pipeline.RequestContext{
		RequestID:        "req-mm",
		KVTransferParams: map[string]any{},
		ResponseWriter:   httptest.NewRecorder(),
		GRPC: &pipeline.VLLMGRPCRequest{
			Req: scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
				Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 32000, 2}}},
				Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 50, MinNewTokens: 10},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     "hash-img-1",
						Source:   &vllmpb.MediaItem_DataUri{DataUri: "data:image/png;base64,AAAA"},
					},
				},
			}),
		},
	}
	require.NoError(t, encodeStep.Execute(ctx, mmCtx))
	require.Equal(t, 1, mmCtx.EncodeFanout)
	require.NoError(t, decodeStep.Execute(ctx, mmCtx))

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, 1, encodeCalls)
	require.Equal(t, uint32(1), encodeReq.GetStopping().GetMaxNewTokens())
	require.Zero(t, encodeReq.GetStopping().GetMinNewTokens())
	require.Len(t, encodeReq.GetMedia(), 1)
	require.Equal(t, "data:image/png;base64,AAAA", encodeReq.GetMedia()[0].GetDataUri())

	require.Equal(t, uint32(50), decodeReq.GetStopping().GetMaxNewTokens())
	require.Equal(t, uint32(10), decodeReq.GetStopping().GetMinNewTokens())
	require.Len(t, decodeReq.GetMedia(), 1)
	require.Equal(t, "hash-img-1", decodeReq.GetMedia()[0].GetUuid())
	require.Equal(t, vllmpb.Modality_MODALITY_IMAGE, decodeReq.GetMedia()[0].GetModality())
	require.Empty(t, decodeReq.GetMedia()[0].GetDataUri(), "decode after encode must strip raw media source")
	require.Nil(t, decodeReq.GetKv().GetKvTransferParams())
	require.Equal(t, map[string]any{
		"hash-img-1": map[string]any{"remote_host": "10.9.0.1", "remote_port": float64(5555)},
	}, decodeReq.GetKv().GetEcTransferParams().AsMap())
}

func TestVLLMGRPC_MediaStrippingAcrossAllPipelineTopologies(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	makeMediaReq := func() vllmwire.Request {
		return scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
			Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 32000, 32000, 2}}},
			Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 16},
			Media: []*vllmpb.MediaItem{
				{
					Modality: vllmpb.Modality_MODALITY_IMAGE,
					Uuid:     "uuid-img-42",
					Source: &vllmpb.MediaItem_Features{
						Features: &vllmpb.PreprocessedMediaFeatures{
							Identifier: "mm-hash-42",
							Offset:     1,
							Length:     2,
							MmHash:     proto.String("mm-hash-42"),
							Kwargs:     []byte("ipc-tensor-payload"),
						},
					},
				},
				{
					Modality: vllmpb.Modality_MODALITY_IMAGE,
					Uuid:     "uuid-raw-43",
					Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("raw-pixel-payload")},
				},
			},
		})
	}

	t.Run("encode -> prefill -> decode strips media bytes after encode and all media on decode", func(t *testing.T) {
		var (
			mu         sync.Mutex
			encodeReq  *vllmpb.GenerateRequest
			prefillReq *vllmpb.GenerateRequest
			decodeReq  *vllmpb.GenerateRequest
		)
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var genReq vllmpb.GenerateRequest
			grpctest.ReadRequest(t, r, &genReq)
			switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
			case gateway.PhaseEncode:
				mu.Lock()
				encodeReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				ecOut, err := structpb.NewStruct(map[string]any{
					"mm-hash-42": map[string]any{"remote_host": "10.0.0.1", "remote_port": 5001},
				})
				require.NoError(t, err)
				grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1, EcTransferParams: ecOut},
					},
				})
			case gateway.PhasePrefill:
				mu.Lock()
				prefillReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				kvOut, err := structpb.NewStruct(map[string]any{
					"do_remote_prefill": true,
					"remote_host":       "10.0.0.2",
					"remote_port":       5002,
				})
				require.NoError(t, err)
				grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1, KvTransferParams: kvOut},
					},
				})
			case gateway.PhaseDecode:
				mu.Lock()
				decodeReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
					Outputs: &vllmpb.SequenceOutput{
						TokenIds:   []uint32{9},
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_STOP, NumOutputTokens: 1},
					},
				})
			}
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		encStep, err := NewEncodeStep(gwClient, map[string]any{ParamECConnector: ec.NIXL})
		require.NoError(t, err)
		preStep, err := NewPrefillStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL, ParamECConnector: ec.NIXL})
		require.NoError(t, err)
		decStep, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL, ParamECConnector: ec.NIXL})
		require.NoError(t, err)

		reqCtx := &pipeline.RequestContext{
			RequestID:        "req-epd",
			KVTransferParams: map[string]any{},
			ResponseWriter:   httptest.NewRecorder(),
			GRPC:             &pipeline.VLLMGRPCRequest{Req: makeMediaReq()},
		}
		require.NoError(t, encStep.Execute(ctx, reqCtx))
		require.NoError(t, preStep.Execute(ctx, reqCtx))
		require.NoError(t, decStep.Execute(ctx, reqCtx))

		mu.Lock()
		defer mu.Unlock()
		// Encode receives MediaAll.
		require.Len(t, encodeReq.GetMedia(), 2)
		require.Equal(t, []byte("ipc-tensor-payload"), encodeReq.GetMedia()[0].GetFeatures().GetKwargs())
		require.Equal(t, []byte("raw-pixel-payload"), encodeReq.GetMedia()[1].GetRawBytes())

		// Prefill after encode receives MediaMetadataOnly.
		require.Len(t, prefillReq.GetMedia(), 2)
		pm := prefillReq.GetMedia()[0]
		require.Equal(t, vllmpb.Modality_MODALITY_IMAGE, pm.GetModality())
		require.Equal(t, "uuid-img-42", pm.GetUuid())
		require.Equal(t, "mm-hash-42", pm.GetFeatures().GetIdentifier())
		require.Equal(t, uint64(1), pm.GetFeatures().GetOffset())
		require.Equal(t, uint64(2), pm.GetFeatures().GetLength())
		require.Equal(t, "mm-hash-42", pm.GetFeatures().GetMmHash())
		require.Empty(t, pm.GetFeatures().GetKwargs())
		pmRaw := prefillReq.GetMedia()[1]
		require.Equal(t, "uuid-raw-43", pmRaw.GetUuid())
		require.Empty(t, pmRaw.GetRawBytes())
		require.Empty(t, pmRaw.GetDataUri())
		require.Empty(t, pmRaw.GetUrl())

		// Decode after prefill receives MediaNone.
		require.Empty(t, decodeReq.GetMedia())
	})

	t.Run("prefill -> decode without encode sends full media to prefill and none to decode", func(t *testing.T) {
		var (
			mu         sync.Mutex
			prefillReq *vllmpb.GenerateRequest
			decodeReq  *vllmpb.GenerateRequest
		)
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var genReq vllmpb.GenerateRequest
			grpctest.ReadRequest(t, r, &genReq)
			switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
			case gateway.PhasePrefill:
				mu.Lock()
				prefillReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1},
					},
				})
			case gateway.PhaseDecode:
				mu.Lock()
				decodeReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
					Outputs: &vllmpb.SequenceOutput{
						TokenIds:   []uint32{9},
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_STOP, NumOutputTokens: 1},
					},
				})
			}
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		preStep, err := NewPrefillStep(gwClient, map[string]any{ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)
		decStep, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)

		reqCtx := &pipeline.RequestContext{
			RequestID:        "req-pd-mm",
			KVTransferParams: map[string]any{},
			ResponseWriter:   httptest.NewRecorder(),
			GRPC:             &pipeline.VLLMGRPCRequest{Req: makeMediaReq()},
		}
		require.NoError(t, preStep.Execute(ctx, reqCtx))
		require.NoError(t, decStep.Execute(ctx, reqCtx))

		mu.Lock()
		defer mu.Unlock()
		require.Len(t, prefillReq.GetMedia(), 2)
		require.Equal(t, []byte("ipc-tensor-payload"), prefillReq.GetMedia()[0].GetFeatures().GetKwargs())
		require.Equal(t, []byte("raw-pixel-payload"), prefillReq.GetMedia()[1].GetRawBytes())
		require.Empty(t, decodeReq.GetMedia())
	})

	t.Run("conditional-decode probe strips all media and decode-only preserves full media", func(t *testing.T) {
		var (
			mu        sync.Mutex
			probeReq  *vllmpb.GenerateRequest
			decodeReq *vllmpb.GenerateRequest
		)
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var genReq vllmpb.GenerateRequest
			grpctest.ReadRequest(t, r, &genReq)
			if r.Header.Get("Prefer") == "if-available" {
				mu.Lock()
				probeReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				grpcframe.WriteError(w, codes.FailedPrecondition, "cache miss")
				return
			}
			mu.Lock()
			decodeReq = proto.Clone(&genReq).(*vllmpb.GenerateRequest)
			mu.Unlock()
			grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{
					TokenIds:   []uint32{9},
					FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_STOP, NumOutputTokens: 1},
				},
			})
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		condStep, err := NewConditionalDecodeStep(gwClient, nil)
		require.NoError(t, err)
		decStep, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)

		reqCtx := &pipeline.RequestContext{
			RequestID:        "req-cond-mm",
			KVTransferParams: map[string]any{},
			ResponseWriter:   httptest.NewRecorder(),
			GRPC:             &pipeline.VLLMGRPCRequest{Req: makeMediaReq()},
		}
		require.NoError(t, condStep.Execute(ctx, reqCtx))
		require.NoError(t, decStep.Execute(ctx, reqCtx))

		mu.Lock()
		defer mu.Unlock()
		require.Empty(t, probeReq.GetMedia(), "conditional-decode probe must send MediaNone")
		require.Len(t, decodeReq.GetMedia(), 2, "decode-only passthrough must preserve MediaAll")
		require.Equal(t, []byte("raw-pixel-payload"), decodeReq.GetMedia()[1].GetRawBytes())
	})
}

func TestVLLMGRPC_PrefillMissingNIXLParamsKeepsDecodeCachedTokens(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Prefill returns num_cached_tokens=20, but NO kv_transfer_params under kv-nixl.
		grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{
				FinishInfo: &vllmpb.FinishInfo{
					FinishReason:    vllmpb.FinishInfo_LENGTH,
					NumOutputTokens: 1,
					NumCachedTokens: proto.Uint32(20),
				},
			},
		})
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	prefillStep, err := NewPrefillStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL})
	require.NoError(t, err)

	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-no-nixl",
		KVTransferParams: map[string]any{},
		ResponseWriter:   httptest.NewRecorder(),
		GRPC: &pipeline.VLLMGRPCRequest{
			Req: scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
			}),
		},
	}

	require.NoError(t, prefillStep.Execute(ctx, reqCtx))
	require.Nil(t, reqCtx.PrefillCachedTokens, "missing kv_transfer_params under kv-nixl means decode recomputes prompt")
}

func TestVLLMGRPC_DecodeClassifiesTransportError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	closedSrv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closedURL := closedSrv.URL
	closedSrv.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: closedURL})
	decodeStep, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL})
	require.NoError(t, err)

	rec := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-transport-err",
		KVTransferParams: map[string]any{},
		ResponseWriter:   rec,
		GRPC: &pipeline.VLLMGRPCRequest{
			Req: scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
			}),
		},
	}

	err = decodeStep.Execute(ctx, reqCtx)
	require.Error(t, err)
	require.Equal(t, coordmetrics.ErrorCodeUpstreamTransport, coordmetrics.ClassifyErrorCode(err, pipeline.ClassifyOpts))
	require.Equal(t, "14", rec.Header().Get(grpcframe.HeaderStatus)) // codes.Unavailable == 14
}

func TestVLLMGRPC_FrameSegmentsWithTransferParamsAllocationBudget(t *testing.T) {
	ids := make([]uint32, 8192)
	for i := range ids {
		ids[i] = uint32(i%32000 + 1)
	}
	req := scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
		RequestId: "alloc-budget-req",
		Model:     "meta-llama/Llama-3.1-70B",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: ids}},
		Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 256, MinNewTokens: 8},
		Kv:        &vllmpb.KVCacheParameters{CacheSalt: "tenant-salt"},
	})

	nixlKV := map[string]any{
		"do_remote_prefill": true,
		"do_remote_decode":  false,
		"remote_engine_id":  "eng-prefill-1",
		"remote_host":       "10.1.2.3",
		"remote_port":       json.Number("5557"),
		"remote_block_ids":  []any{json.Number("11"), json.Number("12"), json.Number("13"), json.Number("14")},
	}

	prefillAllocs := testing.AllocsPerRun(100, func() {
		segs, err := req.FrameSegments(vllmwire.LegOptions{
			CapSingleToken:   true,
			Media:            vllmwire.MediaAll,
			KVTransferParams: nixlKV,
		})
		if err != nil || len(segs) == 0 {
			t.Fatalf("FrameSegments prefill: %v", err)
		}
	})
	decodeAllocs := testing.AllocsPerRun(100, func() {
		segs, err := req.FrameSegments(vllmwire.LegOptions{
			Media:            vllmwire.MediaNone,
			KVTransferParams: nixlKV,
		})
		if err != nil || len(segs) == 0 {
			t.Fatalf("FrameSegments decode: %v", err)
		}
	})
	require.LessOrEqual(t, prefillAllocs, 4.0, "prefill FrameSegments with NIXL KV must not regress to structpb allocations")
	require.LessOrEqual(t, decodeAllocs, 4.0, "decode FrameSegments with NIXL KV must not regress to structpb allocations")
}
