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

package server_test

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"io"
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/encoding/gzip"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/protoadapt"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline/builder"
	"github.com/llm-d/llm-d-router/pkg/coordinator/server"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps/asyncbroker"
)

func dialVLLMClient(t *testing.T, addr string) vllmpb.InferenceClient {
	t.Helper()
	return vllmpb.NewInferenceClient(grpctest.Dial(t, addr))
}

func newVLLMPDPipeline(t *testing.T, gw *gateway.Client, kvName string) *pipeline.Pipeline {
	t.Helper()
	prefill, err := steps.NewPrefillStep(gw, map[string]any{steps.ParamKVConnector: kvName})
	require.NoError(t, err)
	decode, err := steps.NewDecodeStep(gw, map[string]any{steps.ParamKVConnector: kvName})
	require.NoError(t, err)
	p, err := pipeline.NewWithForwardResponseHeaders(
		[]pipeline.Step{prefill, decode},
		[]string{reqcommon.DisaggregatedRevisionHeaderKey},
	)
	require.NoError(t, err)
	return p
}

func probeCount(t *testing.T, reg *prometheus.Registry, result string) float64 {
	t.Helper()
	families, err := reg.Gather()
	require.NoError(t, err)
	total := 0.0
	for _, family := range families {
		if family.GetName() != "llm_d_coordinator_conditional_decode_probes_total" {
			continue
		}
		for _, m := range family.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == "result" && l.GetValue() == result {
					total += m.GetCounter().GetValue()
				}
			}
		}
	}
	return total
}

func unmarshalGatewayVLLMReq(t *testing.T, r *http.Request) *vllmpb.GenerateRequest {
	t.Helper()
	var genReq vllmpb.GenerateRequest
	grpctest.ReadRequest(t, r, &genReq)
	return &genReq
}

func writeGatewayVLLMFrames(t *testing.T, w http.ResponseWriter, frames ...*vllmpb.GenerateResponse) {
	t.Helper()
	msgs := make([]proto.Message, len(frames))
	for i, f := range frames {
		msgs[i] = f
	}
	grpctest.WriteFrames(t, w, msgs...)
}

func TestVLLMGRPC_PrefillDecodeAllKVConnectors(t *testing.T) {
	for _, kvName := range []string{kv.NIXL, kv.SharedStorage, kv.Offloading} {
		t.Run(kvName, func(t *testing.T) {
			var (
				mu             sync.Mutex
				stageOrder     []string
				prefillDone    atomic.Bool
				prefillPath    string
				decodePath     string
				prefillReq     *vllmpb.GenerateRequest
				decodeReq      *vllmpb.GenerateRequest
				prefillHeaders http.Header
				decodeHeaders  http.Header
			)

			// Build a non-canonical intermediate stream frame (2-byte varint 0x81,0x00
			// for SequenceOutput.num_tokens = 1) so any re-marshal in the coordinator
			// would normalize it to 0x01 and fail the byte-exact check below.
			var seqBody []byte
			seqBody = protowire.AppendTag(seqBody, 3, protowire.VarintType) // num_tokens
			seqBody = append(seqBody, 0x81, 0x00)                           // non-canonical 1
			seqBody = protowire.AppendTag(seqBody, 4, protowire.VarintType) // token_ids
			seqBody = protowire.AppendVarint(seqBody, 101)
			var respPayload []byte
			respPayload = protowire.AppendTag(respPayload, 2, protowire.BytesType) // outputs
			respPayload = protowire.AppendBytes(respPayload, seqBody)
			nonCanonicalIntermediateFrame := grpctest.FramePayload(t, respPayload)

			gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, 2, r.ProtoMajor, "gateway must receive HTTP/2")
				genReq := unmarshalGatewayVLLMReq(t, r)
				phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)

				switch phase {
				case gateway.PhasePrefill:
					mu.Lock()
					stageOrder = append(stageOrder, "prefill")
					prefillPath = r.URL.Path
					prefillReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
					prefillHeaders = r.Header.Clone()
					mu.Unlock()

					w.Header().Set(reqcommon.PeerTopologyHeaderKey, "host=node-p,zone=us-central1-a")
					w.Header().Set(reqcommon.DisaggregatedRevisionHeaderKey, "rev-vllm-42")
					if kvName == kv.Offloading {
						w.Header().Set(routing.KVCachePrefillHeader, "10.4.5.6:29500")
					}

					var kvOut *structpb.Struct
					if kvName == kv.NIXL {
						var sErr error
						kvOut, sErr = structpb.NewStruct(map[string]any{
							"do_remote_prefill": true,
							"remote_engine_id":  "eng-prefill-1",
							"remote_host":       "10.1.2.3",
							"remote_port":       5557,
							"remote_block_ids":  []any{11, 12},
						})
						require.NoError(t, sErr)
					}
					prefillDone.Store(true)
					writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
						PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 64},
						Outputs: &vllmpb.SequenceOutput{
							NumTokens: 1,
							TokenIds:  []uint32{99},
							FinishInfo: &vllmpb.FinishInfo{
								FinishReason:     vllmpb.FinishInfo_LENGTH,
								NumOutputTokens:  1,
								NumCachedTokens:  proto.Uint32(36),
								KvTransferParams: kvOut,
							},
						},
					})

				case gateway.PhaseDecode:
					require.True(t, prefillDone.Load(), "vLLM P/D must execute sequentially: prefill before decode")
					mu.Lock()
					stageOrder = append(stageOrder, "decode")
					decodePath = r.URL.Path
					decodeReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
					decodeHeaders = r.Header.Clone()
					mu.Unlock()

					w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
					w.Header().Set("x-worker-pod", "decode-pod-7")
					w.Header().Set(reqcommon.PeerTopologyHeaderKey, "internal-must-be-stripped")
					w.WriteHeader(http.StatusOK)

					terminalFrame := grpctest.Frame(t, &vllmpb.GenerateResponse{
						PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 64},
						Outputs: &vllmpb.SequenceOutput{
							NumTokens: 1,
							TokenIds:  []uint32{102},
							Text:      "world",
							FinishInfo: &vllmpb.FinishInfo{
								FinishReason:    vllmpb.FinishInfo_STOP,
								StopReason:      &vllmpb.FinishInfo_StopTokenId{StopTokenId: 2},
								NumOutputTokens: 2,
								// Decode worker reports 64 handoff blocks; coordinator must
								// overwrite this with prefill's 36 prompt-cache hits.
								NumCachedTokens: proto.Uint32(64),
							},
						},
					})

					if r.URL.Path == vllmpb.Inference_GenerateStream_FullMethodName {
						_, _ = w.Write(nonCanonicalIntermediateFrame)
						_ = http.NewResponseController(w).Flush()
					}
					_, _ = w.Write(terminalFrame)
					grpctest.WriteStatusTrailer(w, codes.OK, "")

				default:
					t.Errorf("unexpected phase %q", phase)
				}
			}))

			gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
			addr := startCoordinatorH2C(t, config.ServerConfig{}, newVLLMPDPipeline(t, gwClient, kvName), gwClient)
			client := dialVLLMClient(t, addr)

			attackerKV, err := structpb.NewStruct(map[string]any{"remote_host": "evil.internal"})
			require.NoError(t, err)
			attackerEC, err := structpb.NewStruct(map[string]any{"evil_hash": "evil"})
			require.NoError(t, err)

			clientReq := &vllmpb.GenerateRequest{
				RequestId: "client-req-1",
				Model:     "meta-llama/Llama-3.1-8B",
				Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 10, 20, 30}}},
				Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 64, MinNewTokens: 8},
				Response:  &vllmpb.ResponseOptions{OutputTokenIds: true, OutputText: proto.Bool(true)},
				Kv: &vllmpb.KVCacheParameters{
					CacheSalt:         "tenant-salt",
					BypassPrefixCache: true,
					KvTransferParams:  attackerKV,
					EcTransferParams:  attackerEC,
				},
			}

			mdCtx := metadata.NewOutgoingContext(testCtx(t), metadata.Pairs(
				reqcommon.RequestIDHeaderKey, "trace-req-99",
				"x-data-parallel-rank", "3",
				reqcommon.EPPProfileHeaderKey, "attacker-injected-profile",
			))

			// 1. Unary /vllm.Inference/Generate
			var respHeader metadata.MD
			unaryResp, err := client.Generate(mdCtx, clientReq, grpc.Header(&respHeader))
			require.NoError(t, err)
			require.Equal(t, uint32(36), unaryResp.GetOutputs().GetFinishInfo().GetNumCachedTokens())
			require.Equal(t, []uint32{102}, unaryResp.GetOutputs().GetTokenIds())
			require.Equal(t, []string{"decode-pod-7"}, respHeader.Get("x-worker-pod"))
			require.Empty(t, respHeader.Get(reqcommon.PeerTopologyHeaderKey), "internal topology header must be stripped")

			mu.Lock()
			require.Equal(t, []string{"prefill", "decode"}, stageOrder)
			require.Equal(t, vllmpb.Inference_Generate_FullMethodName, prefillPath)
			require.Equal(t, vllmpb.Inference_Generate_FullMethodName, decodePath)

			// Prefill overrides stopping (max_new_tokens=1, min_new_tokens=0), preserves
			// cache_salt & bypass_prefix_cache, and strips client transfer params.
			require.Equal(t, uint32(1), prefillReq.GetStopping().GetMaxNewTokens())
			require.Zero(t, prefillReq.GetStopping().GetMinNewTokens())
			require.Equal(t, "tenant-salt", prefillReq.GetKv().GetCacheSalt())
			require.True(t, prefillReq.GetKv().GetBypassPrefixCache())
			require.Nil(t, prefillReq.GetKv().GetEcTransferParams())

			// Decode preserves client stopping (max=64, min=8) and KV cache flags.
			require.Equal(t, uint32(64), decodeReq.GetStopping().GetMaxNewTokens())
			require.Equal(t, uint32(8), decodeReq.GetStopping().GetMinNewTokens())
			require.Equal(t, "tenant-salt", decodeReq.GetKv().GetCacheSalt())
			require.True(t, decodeReq.GetKv().GetBypassPrefixCache())
			require.Nil(t, decodeReq.GetKv().GetEcTransferParams())

			// Headers propagated cleanly across stages while stripping attacker profile.
			require.Equal(t, gateway.PhasePrefill, prefillHeaders.Get(reqcommon.EPPProfileHeaderKey))
			require.Equal(t, "trace-req-99", prefillHeaders.Get(reqcommon.RequestIDHeaderKey))
			require.Equal(t, "3", prefillHeaders.Get("X-Data-Parallel-Rank"))
			require.Equal(t, "trace-req-99", decodeHeaders.Get(reqcommon.RequestIDHeaderKey))
			require.Equal(t, "3", decodeHeaders.Get("X-Data-Parallel-Rank"))
			require.Equal(t, "host=node-p,zone=us-central1-a", decodeHeaders.Get(reqcommon.PeerTopologyHeaderKey))
			require.Equal(t, "rev-vllm-42", decodeHeaders.Get(reqcommon.DisaggregatedRevisionHeaderKey))

			switch kvName {
			case kv.NIXL:
				require.Equal(t, true, prefillReq.GetKv().GetKvTransferParams().AsMap()["do_remote_decode"])
				require.Equal(t, "10.1.2.3", decodeReq.GetKv().GetKvTransferParams().AsMap()["remote_host"])
			case kv.SharedStorage:
				require.Equal(t, map[string]any{
					"do_remote_decode":  true,
					"do_remote_prefill": false,
				}, prefillReq.GetKv().GetKvTransferParams().AsMap())
				require.Equal(t, map[string]any{
					"do_remote_decode":  false,
					"do_remote_prefill": true,
				}, decodeReq.GetKv().GetKvTransferParams().AsMap())
			case kv.Offloading:
				remoteDec, _ := prefillReq.GetKv().GetKvTransferParams().AsMap()["remote_decoder"].(map[string]any)
				kvReqID, _ := remoteDec["kv_request_id"].(string)
				require.NotEmpty(t, kvReqID)
				decodeKV := decodeReq.GetKv().GetKvTransferParams().AsMap()
				require.Equal(t, map[string]any{
					"kv_request_id": kvReqID,
					"remote_host":   "10.4.5.6",
					"remote_port":   float64(29500),
				}, decodeKV["remote_prefiller"])
			}
			stageOrder = nil
			prefillDone.Store(false)
			mu.Unlock()

			// 2. Streaming /vllm.Inference/GenerateStream — verify raw wire bytes on
			// intermediate frame are forwarded verbatim without re-marshaling.
			rawBody := grpctest.Frame(t, clientReq)
			h2Client := gateway.New(config.GatewayConfig{Address: "http://" + addr})
			streamHTTPResp, err := h2Client.StreamGRPC(
				testCtx(t),
				vllmpb.Inference_GenerateStream_FullMethodName,
				map[string]string{reqcommon.RequestIDHeaderKey: "stream-req-1"},
				rawBody,
			)
			require.NoError(t, err)
			defer func() { _ = streamHTTPResp.Body.Close() }()

			frames := grpctest.ReadFrames(t, streamHTTPResp.Body)
			require.Len(t, frames, 2)
			require.Equal(t, nonCanonicalIntermediateFrame, frames[0], "intermediate token frame must be forwarded verbatim without re-marshal")

			var termResp vllmpb.GenerateResponse
			grpctest.Unmarshal(t, frames[1], &termResp)
			require.Equal(t, uint32(36), termResp.GetOutputs().GetFinishInfo().GetNumCachedTokens())

			statusStr, ok := grpcframe.Status(streamHTTPResp.Trailer)
			require.True(t, ok)
			require.Equal(t, grpcframe.StatusOK, statusStr)
		})
	}

	t.Run("kv-offloading missing prefill p2p header returns Internal and records prefill step error", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		var decodeCalled atomic.Bool
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
			case gateway.PhasePrefill:
				// Intentionally omit x-llm-d-kv-cache-prefill-p2p-host-port.
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1},
					},
				})
			case gateway.PhaseDecode:
				decodeCalled.Store(true)
			}
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newVLLMPDPipeline(t, gwClient, kv.Offloading), gwClient)
		client := dialVLLMClient(t, addr)

		_, err := client.Generate(testCtx(t), &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: "missing offloading header"},
		})
		require.Equal(t, codes.Internal, status.Code(err))
		require.False(t, decodeCalled.Load())
		require.Equal(t, float64(1), stepErrors(t, reg, steps.PrefillStepName))
		require.Zero(t, stepErrors(t, reg, steps.DecodeStepName))
	})
}

func TestVLLMGRPC_ConditionalDecodeHitAndMiss(t *testing.T) {
	newCondPipeline := func(t *testing.T, gw *gateway.Client) *pipeline.Pipeline {
		t.Helper()
		cond, err := steps.NewConditionalDecodeStep(gw, nil)
		require.NoError(t, err)
		prefill, err := steps.NewPrefillStep(gw, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)
		decode, err := steps.NewDecodeStep(gw, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)
		return pipeline.New([]pipeline.Step{cond, prefill, decode})
	}

	t.Run("cache hit short-circuits prefill and decode and strips media on probe", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		var (
			prefillCalls, regularDecodeCalls atomic.Int32
			mu                               sync.Mutex
			probeReq                         *vllmpb.GenerateRequest
		)

		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			genReq := unmarshalGatewayVLLMReq(t, r)
			phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
			if phase == gateway.PhaseDecode && r.Header.Get("Prefer") == "if-available" {
				mu.Lock()
				probeReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{
						Text:     "cached-hit",
						TokenIds: []uint32{77},
						FinishInfo: &vllmpb.FinishInfo{
							FinishReason:    vllmpb.FinishInfo_STOP,
							NumOutputTokens: 1,
							NumCachedTokens: proto.Uint32(16),
						},
					},
				})
				return
			}
			if phase == gateway.PhasePrefill {
				prefillCalls.Add(1)
			} else {
				regularDecodeCalls.Add(1)
			}
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newCondPipeline(t, gwClient), gwClient)
		client := dialVLLMClient(t, addr)

		resp, err := client.Generate(testCtx(t), &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: "hit"},
			Media: []*vllmpb.MediaItem{
				{
					Modality: vllmpb.Modality_MODALITY_IMAGE,
					Uuid:     "probe-img-1",
					Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte("pixel-bytes")},
				},
			},
		})
		require.NoError(t, err)
		require.Equal(t, "cached-hit", resp.GetOutputs().GetText())
		require.Zero(t, prefillCalls.Load())
		require.Zero(t, regularDecodeCalls.Load())
		require.Equal(t, float64(1), probeCount(t, reg, coordmetrics.ProbeResultServed))
		mu.Lock()
		require.Empty(t, probeReq.GetMedia(), "conditional-decode probe must strip media")
		mu.Unlock()
	})

	for _, missMode := range []string{"grpc_trailers_only", "grpc_trailing_header", "http_412"} {
		t.Run("cache miss via "+missMode+" falls through to prefill and decode", func(t *testing.T) {
			reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			var prefillCalls, decodeCalls atomic.Int32

			gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
				if phase == gateway.PhaseDecode && r.Header.Get("Prefer") == "if-available" {
					switch missMode {
					case "grpc_trailers_only":
						grpcframe.WriteError(w, codes.FailedPrecondition, "prefix cache below threshold")
					case "grpc_trailing_header":
						w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
						w.WriteHeader(http.StatusOK)
						_ = http.NewResponseController(w).Flush()
						grpctest.WriteStatusTrailer(w, codes.FailedPrecondition, "")
					case "http_412":
						w.WriteHeader(http.StatusPreconditionFailed)
					}
					return
				}
				switch phase {
				case gateway.PhasePrefill:
					prefillCalls.Add(1)
					writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
						Outputs: &vllmpb.SequenceOutput{
							FinishInfo: &vllmpb.FinishInfo{
								FinishReason:    vllmpb.FinishInfo_LENGTH,
								NumOutputTokens: 1,
								NumCachedTokens: proto.Uint32(12),
							},
						},
					})
				case gateway.PhaseDecode:
					decodeCalls.Add(1)
					writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
						Outputs: &vllmpb.SequenceOutput{
							Text:     "from-pd",
							TokenIds: []uint32{88},
							FinishInfo: &vllmpb.FinishInfo{
								FinishReason:    vllmpb.FinishInfo_STOP,
								NumOutputTokens: 1,
								NumCachedTokens: proto.Uint32(64),
							},
						},
					})
				}
			}))

			gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
			addr := startCoordinatorH2C(t, config.ServerConfig{}, newCondPipeline(t, gwClient), gwClient)
			client := dialVLLMClient(t, addr)

			stream, err := client.GenerateStream(testCtx(t), &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "miss"},
			})
			require.NoError(t, err)
			msg, err := stream.Recv()
			require.NoError(t, err)
			require.Equal(t, "from-pd", msg.GetOutputs().GetText())
			require.Equal(t, uint32(12), msg.GetOutputs().GetFinishInfo().GetNumCachedTokens())
			_, err = stream.Recv()
			require.ErrorIs(t, err, io.EOF)

			require.Equal(t, int32(1), prefillCalls.Load())
			require.Equal(t, int32(1), decodeCalls.Load())
			require.Equal(t, float64(1), probeCount(t, reg, coordmetrics.ProbeResultDeferred))
		})
	}
}

func TestVLLMGRPC_EncodePrefillDecodeMultimodalAndSkipsJSONSteps(t *testing.T) {
	mr := miniredis.RunT(t)

	for _, ecName := range []string{ec.NIXL, ec.SharedStorage} {
		t.Run(ecName, func(t *testing.T) {
			var (
				mu         sync.Mutex
				stages     []string
				encodeReq  *vllmpb.GenerateRequest
				prefillReq *vllmpb.GenerateRequest
				decodeReq  *vllmpb.GenerateRequest
			)

			gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				genReq := unmarshalGatewayVLLMReq(t, r)
				phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)

				mu.Lock()
				stages = append(stages, phase)
				mu.Unlock()

				switch phase {
				case gateway.PhaseEncode:
					mu.Lock()
					encodeReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
					mu.Unlock()
					require.Equal(t, uint32(1), genReq.GetStopping().GetMaxNewTokens())
					require.Zero(t, genReq.GetStopping().GetMinNewTokens())
					var ecOut *structpb.Struct
					if ecName == ec.NIXL {
						var sErr error
						ecOut, sErr = structpb.NewStruct(map[string]any{
							"img-hash-1": map[string]any{"remote_host": "10.8.0.1", "remote_port": 6001},
						})
						require.NoError(t, sErr)
					}
					writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
						Outputs: &vllmpb.SequenceOutput{
							FinishInfo: &vllmpb.FinishInfo{
								FinishReason:     vllmpb.FinishInfo_LENGTH,
								NumOutputTokens:  1,
								EcTransferParams: ecOut,
							},
						},
					})

				case gateway.PhasePrefill:
					mu.Lock()
					prefillReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
					mu.Unlock()
					kvOut, sErr := structpb.NewStruct(map[string]any{
						"do_remote_prefill": true,
						"remote_host":       "10.8.0.2",
						"remote_port":       6002,
					})
					require.NoError(t, sErr)
					writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
						Outputs: &vllmpb.SequenceOutput{
							FinishInfo: &vllmpb.FinishInfo{
								FinishReason:     vllmpb.FinishInfo_LENGTH,
								NumOutputTokens:  1,
								NumCachedTokens:  proto.Uint32(8),
								KvTransferParams: kvOut,
							},
						},
					})

				case gateway.PhaseDecode:
					mu.Lock()
					decodeReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
					mu.Unlock()
					writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
						Outputs: &vllmpb.SequenceOutput{
							Text:     "mm-answer",
							TokenIds: []uint32{500},
							FinishInfo: &vllmpb.FinishInfo{
								FinishReason:    vllmpb.FinishInfo_STOP,
								NumOutputTokens: 1,
								NumCachedTokens: proto.Uint32(32),
							},
						},
					})
				}
			}))

			gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
			cfg := &config.Config{
				Pipeline: config.PipelineConfig{
					WorkerProtocol:  config.WorkerProtocolGRPC,
					KVConnector:     kv.NIXL,
					ECConnector:     ecName,
					UseOpenAIFormat: false,
					Steps: []config.StepConfig{
						{
							Type: asyncbroker.StepName,
							Params: map[string]any{
								"redis_url":   "redis://" + mr.Addr(),
								"wakeup_mode": "poll",
							},
						},
						{Type: steps.ReplaceMediaURLsStepName},
						{Type: steps.RenderStepName, Params: map[string]any{"address": "http://127.0.0.1:1"}},
						{Type: steps.EncodeStepName},
						{Type: steps.PrefillStepName},
						{Type: steps.DecodeStepName},
					},
				},
			}
			p, err := builder.Build(cfg, gwClient)
			require.NoError(t, err)

			addr := startCoordinatorH2C(t, config.ServerConfig{}, p, gwClient, server.WithWorkerProtocol(config.WorkerProtocolGRPC))
			client := dialVLLMClient(t, addr)

			// Even with x-async-mode: enqueue, async-broker leaves native gRPC untouched.
			mdCtx := metadata.NewOutgoingContext(testCtx(t), metadata.Pairs("x-async-mode", "enqueue"))
			resp, err := client.Generate(mdCtx, &vllmpb.GenerateRequest{
				Model:    "qwen-vl",
				Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 151655, 2}}},
				Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 32, MinNewTokens: 5},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     "img-hash-1",
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{
								Identifier: "ipc",
								Offset:     1,
								Length:     1,
								MmHash:     proto.String("img-hash-1"),
								Kwargs:     []byte("raw-tensor"),
							},
						},
					},
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     "img-uri-2",
						Source:   &vllmpb.MediaItem_DataUri{DataUri: "data:image/png;base64,AAAA"},
					},
				},
			})
			require.NoError(t, err)
			require.Equal(t, "mm-answer", resp.GetOutputs().GetText())
			require.Equal(t, uint32(8), resp.GetOutputs().GetFinishInfo().GetNumCachedTokens())

			mu.Lock()
			defer mu.Unlock()
			require.Equal(t, []string{gateway.PhaseEncode, gateway.PhasePrefill, gateway.PhaseDecode}, stages)

			// F1/F7: Encode receives MediaAll; prefill after encode receives MediaMetadataOnly; decode receives MediaNone.
			require.Len(t, encodeReq.GetMedia(), 2)
			require.Equal(t, []byte("raw-tensor"), encodeReq.GetMedia()[0].GetFeatures().GetKwargs())
			require.Equal(t, "data:image/png;base64,AAAA", encodeReq.GetMedia()[1].GetDataUri())

			require.Len(t, prefillReq.GetMedia(), 2)
			require.Equal(t, "img-hash-1", prefillReq.GetMedia()[0].GetUuid())
			require.Equal(t, vllmpb.Modality_MODALITY_IMAGE, prefillReq.GetMedia()[0].GetModality())
			require.Equal(t, "ipc", prefillReq.GetMedia()[0].GetFeatures().GetIdentifier())
			require.Equal(t, uint64(1), prefillReq.GetMedia()[0].GetFeatures().GetOffset())
			require.Equal(t, uint64(1), prefillReq.GetMedia()[0].GetFeatures().GetLength())
			require.Equal(t, "img-hash-1", prefillReq.GetMedia()[0].GetFeatures().GetMmHash())
			require.Empty(t, prefillReq.GetMedia()[0].GetFeatures().GetKwargs())
			require.Equal(t, "img-uri-2", prefillReq.GetMedia()[1].GetUuid())
			require.Empty(t, prefillReq.GetMedia()[1].GetDataUri())

			require.Empty(t, decodeReq.GetMedia(), "decode after prefill must receive MediaNone")

			if ecName == ec.NIXL {
				require.Equal(t, map[string]any{
					"img-hash-1": map[string]any{"remote_host": "10.8.0.1", "remote_port": float64(6001)},
				}, prefillReq.GetKv().GetEcTransferParams().AsMap())
			} else {
				require.Nil(t, prefillReq.GetKv().GetEcTransferParams())
			}
			require.NotNil(t, decodeReq.GetKv().GetKvTransferParams())
			require.Nil(t, decodeReq.GetKv().GetEcTransferParams(), "decode after prefill must not attach ec_transfer_params")
		})
	}
}

type stubVLLMInferenceServer struct {
	vllmpb.UnimplementedInferenceServer
	mu      sync.Mutex
	lastReq *vllmpb.GenerateRequest
}

func (s *stubVLLMInferenceServer) Generate(_ context.Context, req *vllmpb.GenerateRequest) (*vllmpb.GenerateResponse, error) {
	s.mu.Lock()
	s.lastReq = proto.Clone(req).(*vllmpb.GenerateRequest)
	s.mu.Unlock()
	return &vllmpb.GenerateResponse{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
		Outputs: &vllmpb.SequenceOutput{
			Text:      "unary:" + req.GetText(),
			NumTokens: 1,
			TokenIds:  []uint32{7},
			FinishInfo: &vllmpb.FinishInfo{
				FinishReason:    vllmpb.FinishInfo_STOP,
				NumOutputTokens: 1,
				NumCachedTokens: proto.Uint32(2),
			},
		},
	}, nil
}

func (s *stubVLLMInferenceServer) GenerateStream(req *vllmpb.GenerateRequest, stream grpc.ServerStreamingServer[vllmpb.GenerateResponse]) error {
	s.mu.Lock()
	s.lastReq = proto.Clone(req).(*vllmpb.GenerateRequest)
	s.mu.Unlock()
	if err := stream.Send(&vllmpb.GenerateResponse{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
		Outputs:    &vllmpb.SequenceOutput{Text: "tok1:", NumTokens: 1, TokenIds: []uint32{7}},
	}); err != nil {
		return err
	}
	return stream.Send(&vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{
			Text:      req.GetText(),
			NumTokens: 1,
			TokenIds:  []uint32{8},
			FinishInfo: &vllmpb.FinishInfo{
				FinishReason:    vllmpb.FinishInfo_STOP,
				NumOutputTokens: 2,
				NumCachedTokens: proto.Uint32(2),
			},
		},
	})
}

type stubVLLMControlServer struct {
	vllmpb.UnimplementedControlServer
}

func (stubVLLMControlServer) GetServerInfo(context.Context, *vllmpb.GetServerInfoRequest) (*vllmpb.ServerInfo, error) {
	return &vllmpb.ServerInfo{EngineVersion: "vllm-grpc"}, nil
}

func TestVLLMGRPC_DecodeOnlyTLSAndControlPassthrough(t *testing.T) {
	stubInf := &stubVLLMInferenceServer{}
	backendAddr := grpctest.StartGRPCServer(t, func(s *grpc.Server) {
		vllmpb.RegisterInferenceServer(s, stubInf)
		vllmpb.RegisterControlServer(s, stubVLLMControlServer{})
	})

	gwClient := gateway.New(config.GatewayConfig{Address: "http://" + backendAddr})
	decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.NIXL})
	require.NoError(t, err)
	p := pipeline.New([]pipeline.Step{decodeStep})

	srv, err := server.New(config.ServerConfig{SecureServing: true}, p, gwClient)
	require.NoError(t, err)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = srv.Serve(ctx, ln)
	}()
	t.Cleanup(func() {
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		_ = srv.Shutdown(shutdownCtx)
		<-done
	})

	tlsCreds := credentials.NewTLS(&tls.Config{InsecureSkipVerify: true}) //#nosec G402 -- self-signed test cert
	conn, err := grpc.NewClient(ln.Addr().String(), grpc.WithTransportCredentials(tlsCreds))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })

	infClient := vllmpb.NewInferenceClient(conn)
	evilKV, err := structpb.NewStruct(map[string]any{"remote_host": "evil"})
	require.NoError(t, err)

	unaryResp, err := infClient.Generate(testCtx(t), &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_Text{Text: "tls-check"},
		Kv:     &vllmpb.KVCacheParameters{CacheSalt: "s1", KvTransferParams: evilKV},
	})
	require.NoError(t, err)
	require.Equal(t, "unary:tls-check", unaryResp.GetOutputs().GetText())

	stubInf.mu.Lock()
	require.Nil(t, stubInf.lastReq.GetKv().GetKvTransferParams(), "decode-only must strip client kv_transfer_params and not inject new ones")
	require.Equal(t, "s1", stubInf.lastReq.GetKv().GetCacheSalt())
	stubInf.mu.Unlock()

	stream, err := infClient.GenerateStream(testCtx(t), &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_Text{Text: "stream-tls"},
	})
	require.NoError(t, err)
	m1, err := stream.Recv()
	require.NoError(t, err)
	m2, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, "tok1:stream-tls", m1.GetOutputs().GetText()+m2.GetOutputs().GetText())
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)

	// Auxiliary vllm.Control RPC passes through to the backend via passthrough.
	ctrlResp, err := vllmpb.NewControlClient(conn).GetServerInfo(testCtx(t), &vllmpb.GetServerInfoRequest{})
	require.NoError(t, err)
	require.Equal(t, "vllm-grpc", ctrlResp.GetEngineVersion())
}

func TestVLLMGRPC_ErrorPropagationTruncationAndCancellation(t *testing.T) {
	t.Run("ingress rejects compressed oversized and malformed requests", func(t *testing.T) {
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			t.Error("invalid ingress requests must not reach gateway")
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(
			t,
			config.ServerConfig{MaxRequestBodySize: 1}, // 1 MB
			newVLLMPDPipeline(t, gwClient, kv.SharedStorage),
			gwClient,
		)
		client := dialVLLMClient(t, addr)

		// Compressed gRPC request -> Unimplemented
		_, err := client.Generate(testCtx(t), &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: "gz"},
		}, grpc.UseCompressor(gzip.Name))
		require.Equal(t, codes.Unimplemented, status.Code(err))

		// Oversized (>1 MB) request -> ResourceExhausted
		_, err = client.Generate(testCtx(t), &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: strings.Repeat("x", 1024*1024+128)},
		})
		require.Equal(t, codes.ResourceExhausted, status.Code(err))

		// Malformed protobuf payload -> InvalidArgument
		malformedFrame := grpctest.FramePayload(t, []byte{0xff, 0xff, 0xff})
		rawClient := gateway.New(config.GatewayConfig{Address: "http://" + addr})
		resp, err := rawClient.StreamGRPC(testCtx(t), vllmpb.Inference_Generate_FullMethodName, nil, malformedFrame)
		require.NoError(t, err)
		_ = resp.Body.Close()
		stCode, _ := grpcframe.Status(resp.Header)
		require.Equal(t, strconv.Itoa(int(codes.InvalidArgument)), stCode)
	})

	t.Run("trailers-only error with grpc-status-details-bin preserved", func(t *testing.T) {
		stWithDetails, err := status.New(codes.InvalidArgument, "unknown lora adapter").
			WithDetails(protoadapt.MessageV1Of(&vllmpb.TokenIds{Ids: []uint32{404}}))
		require.NoError(t, err)
		detailsBytes, err := proto.Marshal(stWithDetails.Proto())
		require.NoError(t, err)
		encodedDetails := base64.RawStdEncoding.EncodeToString(detailsBytes)

		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(grpcframe.HeaderStatus, strconv.Itoa(int(codes.InvalidArgument)))
			w.Header().Set(grpcframe.HeaderMessage, grpcframe.EncodeStatusMessage("unknown lora adapter"))
			w.Header().Set(grpcframe.HeaderStatusDetailsBin, encodedDetails)
			w.WriteHeader(http.StatusOK)
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newVLLMPDPipeline(t, gwClient, kv.SharedStorage), gwClient)
		client := dialVLLMClient(t, addr)

		_, callErr := client.Generate(testCtx(t), &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: "bad lora"},
		})
		gotSt, ok := status.FromError(callErr)
		require.True(t, ok)
		require.Equal(t, codes.InvalidArgument, gotSt.Code())
		require.Equal(t, "unknown lora adapter", gotSt.Message())
		rawDetails := gotSt.Proto().GetDetails()
		require.Len(t, rawDetails, 1)
		var gotDetail vllmpb.TokenIds
		require.NoError(t, rawDetails[0].UnmarshalTo(&gotDetail))
		require.Equal(t, []uint32{404}, gotDetail.GetIds())
	})

	t.Run("mid-stream gRPC error trailer with grpc-status-details-bin preserved", func(t *testing.T) {
		stWithDetails, err := status.New(codes.ResourceExhausted, "kv blocks exhausted mid-stream").
			WithDetails(protoadapt.MessageV1Of(&vllmpb.TokenIds{Ids: []uint32{503, 504}}))
		require.NoError(t, err)
		detailsBytes, err := proto.Marshal(stWithDetails.Proto())
		require.NoError(t, err)
		encodedDetails := base64.RawStdEncoding.EncodeToString(detailsBytes)

		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get(reqcommon.EPPProfileHeaderKey) == gateway.PhasePrefill {
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1},
					},
				})
				return
			}
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			first := grpctest.Frame(t, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{Text: "part-1", NumTokens: 1},
			})
			_, _ = w.Write(first)
			_ = http.NewResponseController(w).Flush()
			grpctest.WriteStatusTrailer(w, codes.ResourceExhausted, "kv blocks exhausted mid-stream")
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatusDetailsBin, encodedDetails)
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newVLLMPDPipeline(t, gwClient, kv.SharedStorage), gwClient)
		client := dialVLLMClient(t, addr)

		stream, err := client.GenerateStream(testCtx(t), &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: "mid-stream err"},
		})
		require.NoError(t, err)
		firstMsg, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, "part-1", firstMsg.GetOutputs().GetText())

		_, recvErr := stream.Recv()
		gotSt, ok := status.FromError(recvErr)
		require.True(t, ok)
		require.Equal(t, codes.ResourceExhausted, gotSt.Code())
		require.Equal(t, "kv blocks exhausted mid-stream", gotSt.Message())
		rawDetails := gotSt.Proto().GetDetails()
		require.Len(t, rawDetails, 1)
		var gotDetail vllmpb.TokenIds
		require.NoError(t, rawDetails[0].UnmarshalTo(&gotDetail))
		require.Equal(t, []uint32{503, 504}, gotDetail.GetIds())
	})

	t.Run("truncated prefill and decode frames return Unavailable and missing grpc-status returns Internal", func(t *testing.T) {
		var mode atomic.Int32
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			cur := mode.Load()
			if r.Header.Get(reqcommon.EPPProfileHeaderKey) == gateway.PhasePrefill {
				if cur == 0 {
					// Declare 20-byte frame, write only 3 bytes -> io.ErrUnexpectedEOF.
					w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte{0, 0, 0, 0, 20, 1, 2, 3})
					return
				}
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1},
					},
				})
				return
			}
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			if cur == 1 {
				// Truncated decode frame.
				_, _ = w.Write([]byte{0, 0, 0, 0, 25, 1, 2})
				return
			}
			// Mode 2: valid frame but no grpc-status trailer.
			valid := grpctest.Frame(t, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{Text: "ok"},
			})
			_, _ = w.Write(valid)
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newVLLMPDPipeline(t, gwClient, kv.SharedStorage), gwClient)
		client := dialVLLMClient(t, addr)

		mode.Store(0)
		_, err := client.Generate(testCtx(t), &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "trunc-prefill"}})
		require.Equal(t, codes.Unavailable, status.Code(err))

		mode.Store(1)
		stream, err := client.GenerateStream(testCtx(t), &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "trunc-decode"}})
		require.NoError(t, err)
		_, err = stream.Recv()
		require.Equal(t, codes.Unavailable, status.Code(err))

		mode.Store(2)
		stream, err = client.GenerateStream(testCtx(t), &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "no-status"}})
		require.NoError(t, err)
		_, err = stream.Recv()
		require.NoError(t, err)
		_, err = stream.Recv()
		require.Equal(t, codes.Internal, status.Code(err))
	})

	t.Run("client disconnect cancels upstream decode stream", func(t *testing.T) {
		decodeCancelled := make(chan struct{})
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get(reqcommon.EPPProfileHeaderKey) == gateway.PhasePrefill {
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1},
					},
				})
				return
			}
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			first := grpctest.Frame(t, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{Text: "first-tok", NumTokens: 1},
			})
			_, _ = w.Write(first)
			_ = http.NewResponseController(w).Flush()
			<-r.Context().Done()
			close(decodeCancelled)
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newVLLMPDPipeline(t, gwClient, kv.SharedStorage), gwClient)
		client := dialVLLMClient(t, addr)

		callCtx, cancelCall := context.WithCancel(testCtx(t))
		stream, err := client.GenerateStream(callCtx, &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: "cancel-stream"},
		})
		require.NoError(t, err)
		_, err = stream.Recv()
		require.NoError(t, err)
		cancelCall()

		select {
		case <-decodeCancelled:
		case <-testCtx(t).Done():
			t.Fatal("timed out waiting for upstream decode cancellation")
		}
	})
}

func TestVLLMGRPC_PromptTokenParityMismatch(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)

	gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhasePrefill:
			writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 64},
				Outputs: &vllmpb.SequenceOutput{
					FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1},
				},
			})
		case gateway.PhaseDecode:
			writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 72}, // mismatch vs 64
				Outputs: &vllmpb.SequenceOutput{
					Text:     "mismatch-out",
					TokenIds: []uint32{5},
					FinishInfo: &vllmpb.FinishInfo{
						FinishReason:    vllmpb.FinishInfo_STOP,
						NumOutputTokens: 1,
					},
				},
			})
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	prefill, err := steps.NewPrefillStep(gwClient, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
	require.NoError(t, err)
	decode, err := steps.NewDecodeStep(gwClient, map[string]any{
		steps.ParamKVConnector:       kv.SharedStorage,
		steps.ParamPromptTokenParity: "error",
	})
	require.NoError(t, err)
	p := pipeline.New([]pipeline.Step{prefill, decode})
	addr := startCoordinatorH2C(t, config.ServerConfig{}, p, gwClient, server.WithWorkerProtocol(config.WorkerProtocolGRPC))
	client := dialVLLMClient(t, addr)

	// 1. Native /vllm.Inference/Generate with TokenIds -> ids_forwarded=true.
	_, err = client.Generate(testCtx(t), &vllmpb.GenerateRequest{
		Model:  "llama",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
	})
	require.NoError(t, err)
	require.Equal(t, float64(1), promptMismatchCount(t, reg, reqcommon.APITypeVLLMGenerate.String(), true))
	require.Equal(t, float64(1), stepErrors(t, reg, steps.DecodeStepName))

	// 2. HTTP /inference/v1/generate with text prompt -> ids_forwarded=false.
	httpReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"llama","prompt":"hello"}`))
	require.NoError(t, err)
	resp, err := (&http.Client{}).Do(httpReq)
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Equal(t, float64(1), promptMismatchCount(t, reg, reqcommon.APITypeVLLMGenerate.String(), false))
	require.Equal(t, float64(2), stepErrors(t, reg, steps.DecodeStepName))
}

func TestVLLMHTTPGenerate_WorkerProtocolGRPCAndHTTPDefault(t *testing.T) {
	t.Run("worker_protocol grpc maps HTTP /inference/v1/generate unary and SSE onto vllm.Inference", func(t *testing.T) {
		var (
			mu         sync.Mutex
			encodeReq  *vllmpb.GenerateRequest
			prefillReq *vllmpb.GenerateRequest
			decodeReq  *vllmpb.GenerateRequest
		)

		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			genReq := unmarshalGatewayVLLMReq(t, r)
			if genReq.GetModel() == "reject-upstream" {
				grpcframe.WriteError(w, codes.InvalidArgument, "unsupported model")
				return
			}
			phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
			if phase == gateway.PhaseDecode && r.Header.Get("Prefer") == "if-available" {
				grpcframe.WriteError(w, codes.FailedPrecondition, "cache miss")
				return
			}
			switch phase {
			case gateway.PhaseEncode:
				mu.Lock()
				encodeReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				ecOut, err := structpb.NewStruct(map[string]any{
					"mm-hash-1": map[string]any{"remote_host": "10.7.0.1", "remote_port": 7001},
				})
				require.NoError(t, err)
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{
							FinishReason:     vllmpb.FinishInfo_LENGTH,
							NumOutputTokens:  1,
							EcTransferParams: ecOut,
						},
					},
				})
			case gateway.PhasePrefill:
				mu.Lock()
				prefillReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				kvOut, err := structpb.NewStruct(map[string]any{
					"do_remote_prefill": true,
					"remote_host":       "10.7.0.2",
					"remote_port":       7002,
				})
				require.NoError(t, err)
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 5},
					Outputs: &vllmpb.SequenceOutput{
						FinishInfo: &vllmpb.FinishInfo{
							FinishReason:     vllmpb.FinishInfo_LENGTH,
							NumOutputTokens:  1,
							NumCachedTokens:  proto.Uint32(3),
							KvTransferParams: kvOut,
						},
					},
				})
			case gateway.PhaseDecode:
				mu.Lock()
				decodeReq = proto.Clone(genReq).(*vllmpb.GenerateRequest)
				mu.Unlock()
				if r.URL.Path == vllmpb.Inference_GenerateStream_FullMethodName {
					writeGatewayVLLMFrames(t, w,
						&vllmpb.GenerateResponse{
							PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 5},
							Outputs:    &vllmpb.SequenceOutput{Index: 0, NumTokens: 1, TokenIds: []uint32{201}},
						},
						&vllmpb.GenerateResponse{
							Outputs: &vllmpb.SequenceOutput{
								Index:     0,
								NumTokens: 1,
								TokenIds:  []uint32{202},
								FinishInfo: &vllmpb.FinishInfo{
									FinishReason:    vllmpb.FinishInfo_STOP,
									StopReason:      &vllmpb.FinishInfo_StopTokenId{StopTokenId: 202},
									NumOutputTokens: 2,
									NumCachedTokens: proto.Uint32(5),
								},
							},
						},
					)
					return
				}
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 5},
					Outputs: &vllmpb.SequenceOutput{
						Index:     0,
						NumTokens: 2,
						TokenIds:  []uint32{201, 202},
						FinishInfo: &vllmpb.FinishInfo{
							FinishReason:    vllmpb.FinishInfo_STOP,
							StopReason:      &vllmpb.FinishInfo_StopTokenId{StopTokenId: 202},
							NumOutputTokens: 2,
							NumCachedTokens: proto.Uint32(5),
						},
					},
				})
			}
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		cfg := &config.Config{
			Pipeline: config.PipelineConfig{
				WorkerProtocol:  config.WorkerProtocolGRPC,
				KVConnector:     kv.NIXL,
				ECConnector:     ec.NIXL,
				UseOpenAIFormat: false,
				Steps: []config.StepConfig{
					{Type: steps.RenderStepName, Params: map[string]any{"address": "http://127.0.0.1:1"}},
					{Type: steps.ConditionalDecodeStepName},
					{Type: steps.EncodeStepName},
					{Type: steps.PrefillStepName},
					{Type: steps.DecodeStepName},
				},
			},
		}
		p, err := builder.Build(cfg, gwClient)
		require.NoError(t, err)
		addr := startCoordinatorH2C(t, config.ServerConfig{}, p, gwClient, server.WithWorkerProtocol(config.WorkerProtocolGRPC))
		httpClient := &http.Client{}

		// 1. Unary JSON POST /inference/v1/generate with features + vllm_xargs sanitization.
		rawKwargs := base64.StdEncoding.EncodeToString([]byte("ipc-tensor-bytes"))
		unaryPayload := map[string]any{
			"request_id":   "http-gen-1",
			"model":        "qwen-vl",
			"lora_request": map[string]any{"lora_name": "adapter-v1"},
			"token_ids":    []int{1, 32000, 32000, 15, 2},
			"sampling_params": map[string]any{
				"max_tokens":     16,
				"min_tokens":     4,
				"temperature":    0.5,
				"stop_token_ids": []int{202},
				"vllm_xargs": map[string]any{
					"cache_salt":         "http-salt",
					"kv_transfer_params": map[string]any{"remote_host": "evil"},
				},
			},
			"features": map[string]any{
				"mm_hashes":       map[string]any{"image": []string{"mm-hash-1"}},
				"mm_placeholders": map[string]any{"image": []map[string]any{{"offset": 1, "length": 2}}},
				"kwargs_data":     map[string]any{"image": []string{rawKwargs}},
			},
			"kv_transfer_params": map[string]any{"remote_host": "evil-top"},
		}
		bodyBytes, err := json.Marshal(unaryPayload)
		require.NoError(t, err)

		httpReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, bytes.NewReader(bodyBytes))
		require.NoError(t, err)
		httpReq.Header.Set(reqcommon.HeaderContentType, reqcommon.ContentTypeJSON)
		httpResp, err := httpClient.Do(httpReq)
		require.NoError(t, err)
		defer func() { _ = httpResp.Body.Close() }()
		require.Equal(t, http.StatusOK, httpResp.StatusCode)

		var gotJSON map[string]any
		require.NoError(t, json.NewDecoder(httpResp.Body).Decode(&gotJSON))
		require.Equal(t, "http-gen-1", gotJSON["request_id"])
		require.Equal(t, "qwen-vl", gotJSON["model"])
		choices := gotJSON["choices"].([]any)
		require.Len(t, choices, 1)
		choice0 := choices[0].(map[string]any)
		require.Equal(t, []any{float64(201), float64(202)}, choice0["token_ids"])
		require.Equal(t, "stop", choice0["finish_reason"])
		require.Equal(t, float64(202), choice0["stop_reason"])

		usageMap := gotJSON["usage"].(map[string]any)
		require.Equal(t, float64(5), usageMap["prompt_tokens"])
		require.Equal(t, float64(2), usageMap["completion_tokens"])
		require.Equal(t, float64(3), usageMap["prompt_tokens_details"].(map[string]any)["cached_tokens"])

		mu.Lock()
		require.True(t, decodeReq.GetResponse().GetOutputTokenIds())
		require.Equal(t, "adapter-v1", decodeReq.GetLoraName())
		require.Equal(t, "http-salt", prefillReq.GetKv().GetCacheSalt())
		require.NotEqual(t, "evil", prefillReq.GetKv().GetKvTransferParams().AsMap()["remote_host"])
		// F1/F7: Encode receives full kwargs; prefill after encode strips kwargs and keeps metadata; decode receives MediaNone.
		require.Len(t, encodeReq.GetMedia(), 1)
		require.Equal(t, []byte("ipc-tensor-bytes"), encodeReq.GetMedia()[0].GetFeatures().GetKwargs())
		require.Len(t, prefillReq.GetMedia(), 1)
		require.Equal(t, "ipc", prefillReq.GetMedia()[0].GetFeatures().GetIdentifier())
		require.Equal(t, uint64(1), prefillReq.GetMedia()[0].GetFeatures().GetOffset())
		require.Equal(t, uint64(2), prefillReq.GetMedia()[0].GetFeatures().GetLength())
		require.Equal(t, "mm-hash-1", prefillReq.GetMedia()[0].GetFeatures().GetMmHash())
		require.Empty(t, prefillReq.GetMedia()[0].GetFeatures().GetKwargs())
		require.Empty(t, decodeReq.GetMedia(), "decode after prefill must strip all media")
		mu.Unlock()

		// 2. Streaming SSE POST /inference/v1/generate with include_usage=true.
		streamPayload, err := json.Marshal(map[string]any{
			"request_id":     "http-stream-1",
			"model":          "qwen-vl",
			"token_ids":      []int{1, 2, 3, 4, 5},
			"stream":         true,
			"stream_options": map[string]any{"include_usage": true},
		})
		require.NoError(t, err)
		sReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, bytes.NewReader(streamPayload))
		require.NoError(t, err)
		sResp, err := httpClient.Do(sReq)
		require.NoError(t, err)
		defer func() { _ = sResp.Body.Close() }()
		require.Equal(t, http.StatusOK, sResp.StatusCode)
		require.Contains(t, sResp.Header.Get(reqcommon.HeaderContentType), "text/event-stream")
		sseRaw, err := io.ReadAll(sResp.Body)
		require.NoError(t, err)
		sseStr := string(sseRaw)
		require.Contains(t, sseStr, `"token_ids":[201]`)
		require.Contains(t, sseStr, `"token_ids":[202]`)
		require.Contains(t, sseStr, `"cached_tokens":3`)
		require.True(t, strings.HasSuffix(sseStr, "data: [DONE]\n\n"))

		// 3. Invalid client body (missing prompt/token_ids) -> HTTP 400.
		badReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"m"}`))
		require.NoError(t, err)
		badResp, err := httpClient.Do(badReq)
		require.NoError(t, err)
		_ = badResp.Body.Close()
		require.Equal(t, http.StatusBadRequest, badResp.StatusCode)

		// 4. Upstream gRPC InvalidArgument -> HTTP 400.
		upErrReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"reject-upstream","token_ids":[1,2]}`))
		require.NoError(t, err)
		upErrResp, err := httpClient.Do(upErrReq)
		require.NoError(t, err)
		_ = upErrResp.Body.Close()
		require.Equal(t, http.StatusBadRequest, upErrResp.StatusCode)
	})

	t.Run("default worker_protocol http preserves HTTP /inference/v1/generate", func(t *testing.T) {
		var sawHTTPPath atomic.Bool
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == reqcommon.PathVLLMGenerate {
				sawHTTPPath.Store(true)
			}
			w.Header().Set(reqcommon.HeaderContentType, reqcommon.ContentTypeJSON)
			_, _ = w.Write([]byte(`{"request_id":"r1","choices":[{"index":0,"token_ids":[9],"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		cfg := &config.Config{
			Pipeline: config.PipelineConfig{
				UseOpenAIFormat: false,
				Steps: []config.StepConfig{
					{Type: steps.RenderStepName},
					{Type: steps.DecodeStepName},
				},
			},
		}
		p, err := builder.Build(cfg, gwClient)
		require.NoError(t, err)
		addr := startCoordinatorH2C(t, config.ServerConfig{}, p, gwClient)

		httpReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"m","token_ids":[1,2]}`))
		require.NoError(t, err)
		resp, err := (&http.Client{}).Do(httpReq)
		require.NoError(t, err)
		_ = resp.Body.Close()
		require.Equal(t, http.StatusOK, resp.StatusCode)
		require.True(t, sawHTTPPath.Load(), "default worker_protocol http must forward HTTP /inference/v1/generate")
	})
}

func TestVLLMHTTPGenerate_SamplingParityLogprobsAndUnknownKeyRejection(t *testing.T) {
	upstream := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
		Respond: func(call grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
			switch call.Request.GetModel() {
			case "rate-limited":
				return nil, status.Error(codes.ResourceExhausted, "rate limited")
			case "not-found":
				return nil, status.Error(codes.NotFound, "model missing")
			case "deadline":
				return nil, status.Error(codes.DeadlineExceeded, "timed out")
			case "unavailable":
				return nil, status.Error(codes.Unavailable, "worker down")
			}
			return []*vllmpb.GenerateResponse{{
				PromptInfo: &vllmpb.PromptInfo{
					NumPromptTokens: 2,
					TokenIds:        []uint32{11, 22},
					Logprobs:        []float32{float32(math.NaN()), -0.25},
					Ranks:           []uint32{0, 1},
					CandidateTokens: []*vllmpb.CandidateTokenInfo{
						{},
						{
							Tokens: []*vllmpb.CandidateTokenInfo_TokenInfo{
								{Id: 22, Logprob: -0.25, Rank: 1},
							},
						},
					},
				},
				Outputs: &vllmpb.SequenceOutput{
					Index:     0,
					Text:      `{"answer":42}`,
					NumTokens: 1,
					TokenIds:  []uint32{99},
					Logprobs:  []float32{-0.125},
					Ranks:     []uint32{1},
					CandidateTokens: []*vllmpb.CandidateTokenInfo{
						{
							Tokens: []*vllmpb.CandidateTokenInfo_TokenInfo{
								{Id: 99, Logprob: -0.125, Rank: 1},
								{Id: 100, Logprob: -2.5, Rank: 2},
							},
						},
					},
					FinishInfo: &vllmpb.FinishInfo{
						FinishReason:    vllmpb.FinishInfo_STOP,
						NumOutputTokens: 1,
						NumCachedTokens: proto.Uint32(1),
					},
				},
			}}, nil
		},
	})

	gwClient := gateway.New(config.GatewayConfig{Address: upstream.URL()})
	decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
	require.NoError(t, err)
	addr := startCoordinatorH2C(
		t,
		config.ServerConfig{},
		pipeline.New([]pipeline.Step{decodeStep}),
		gwClient,
		server.WithWorkerProtocol(config.WorkerProtocolGRPC),
	)
	httpClient := &http.Client{}

	postJSON := func(body string) (int, map[string]any) {
		t.Helper()
		req, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(body))
		require.NoError(t, err)
		req.Header.Set(reqcommon.HeaderContentType, reqcommon.ContentTypeJSON)
		resp, err := httpClient.Do(req)
		require.NoError(t, err)
		defer func() { _ = resp.Body.Close() }()
		var parsed map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&parsed)
		return resp.StatusCode, parsed
	}

	// 1. Full sampling + structured_outputs + logprobs + return_token_ids parity.
	fullBody, err := json.Marshal(map[string]any{
		"model":            "llama",
		"token_ids":        []int{11, 22},
		"return_token_ids": true,
		"sampling_params": map[string]any{
			"max_tokens":                 32,
			"truncate_prompt_tokens":     64,
			"thinking_token_budget":      128,
			"logit_bias":                 map[string]any{"10": 1.5, "20": -2.0},
			"allowed_token_ids":          []int{99, 100},
			"bad_words_token_ids":        [][]int{{666, 667}},
			"detokenize":                 true,
			"logprobs":                   2,
			"prompt_logprobs":            true,
			"prompt_logprob_token_ids":   []int{22},
			"structured_outputs":         map[string]any{"json": map[string]any{"type": "object"}},
			"presence_penalty":           0.25,
			"frequency_penalty":          0.5,
			"repetition_penalty":         1.1,
			"skip_special_tokens":        false,
			"include_stop_str_in_output": true,
			"ignore_eos":                 true,
		},
	})
	require.NoError(t, err)

	code, got := postJSON(string(fullBody))
	require.Equal(t, http.StatusOK, code)

	call, ok := upstream.LastCall()
	require.True(t, ok)
	pbReq := call.Request
	require.Equal(t, uint32(64), pbReq.GetTruncatePromptTokens())
	require.Equal(t, int64(128), pbReq.GetStopping().GetThinkingTokenBudget())
	require.True(t, pbReq.GetStopping().GetIncludeStopStrings())
	require.True(t, pbReq.GetStopping().GetIgnoreEos())
	require.InDelta(t, 1.5, pbReq.GetDecoding().GetLogitBias()[10], 1e-5)
	require.InDelta(t, -2.0, pbReq.GetDecoding().GetLogitBias()[20], 1e-5)
	require.Equal(t, []uint32{99, 100}, pbReq.GetDecoding().GetAllowedTokenIds())
	require.Len(t, pbReq.GetDecoding().GetBadWordsTokenIds(), 1)
	require.Equal(t, []uint32{666, 667}, pbReq.GetDecoding().GetBadWordsTokenIds()[0].GetIds())
	require.Contains(t, pbReq.GetDecoding().GetJson(), `"type":"object"`)
	require.True(t, pbReq.GetResponse().GetPromptTokenIds())
	require.True(t, pbReq.GetResponse().GetOutputText())
	require.True(t, pbReq.GetResponse().GetOutputLogprobs())
	require.Equal(t, uint32(2), pbReq.GetResponse().GetOutputCandidates().GetTopN())
	require.True(t, pbReq.GetResponse().GetPromptLogprobs())
	require.Equal(t, []uint32{22}, pbReq.GetResponse().GetPromptCandidates().GetTokenIds().GetIds())

	// Verify response echoed prompt_token_ids, text, output logprobs, and prompt_logprobs (with null first NaN logprob).
	require.Equal(t, []any{float64(11), float64(22)}, got["prompt_token_ids"])
	choice0 := got["choices"].([]any)[0].(map[string]any)
	require.Equal(t, `{"answer":42}`, choice0["text"])
	outLP := choice0["logprobs"].(map[string]any)["content"].([]any)
	require.Len(t, outLP, 1)
	require.Equal(t, float64(99), outLP[0].(map[string]any)["token_id"])
	require.Len(t, outLP[0].(map[string]any)["top_logprobs"].([]any), 2)

	promptLP := got["prompt_logprobs"].(map[string]any)["content"].([]any)
	require.Len(t, promptLP, 2)
	require.Nil(t, promptLP[0].(map[string]any)["logprob"], "NaN first prompt logprob must serialize as null")
	require.InDelta(t, -0.25, promptLP[1].(map[string]any)["logprob"].(float64), 1e-5)

	// 2. Verify all remaining structured output variants map cleanly.
	structVariants := []struct {
		name  string
		sp    map[string]any
		check func(*testing.T, *vllmpb.DecodingParameters)
	}{
		{
			name: "guided_regex",
			sp:   map[string]any{"guided_regex": "^[a-z]+$"},
			check: func(t *testing.T, d *vllmpb.DecodingParameters) {
				require.Equal(t, "^[a-z]+$", d.GetRegex())
			},
		},
		{
			name: "guided_choice",
			sp:   map[string]any{"guided_choice": []string{"yes", "no"}},
			check: func(t *testing.T, d *vllmpb.DecodingParameters) {
				require.Equal(t, []string{"yes", "no"}, d.GetChoice().GetChoices())
			},
		},
		{
			name: "guided_grammar",
			sp:   map[string]any{"guided_grammar": "root ::= 'ok'"},
			check: func(t *testing.T, d *vllmpb.DecodingParameters) {
				require.Equal(t, "root ::= 'ok'", d.GetGrammar())
			},
		},
		{
			name: "structured_outputs json_object",
			sp:   map[string]any{"structured_outputs": map[string]any{"json_object": true}},
			check: func(t *testing.T, d *vllmpb.DecodingParameters) {
				require.True(t, d.GetJsonObject())
			},
		},
		{
			name: "structural_tag",
			sp:   map[string]any{"structural_tag": "<tool>"},
			check: func(t *testing.T, d *vllmpb.DecodingParameters) {
				require.Equal(t, "<tool>", d.GetStructuralTag())
			},
		},
		{
			name: "bool logprobs + top_logprobs",
			sp:   map[string]any{"logprobs": true, "top_logprobs": 5},
			check: func(t *testing.T, _ *vllmpb.DecodingParameters) {
				last, _ := upstream.LastCall()
				require.True(t, last.Request.GetResponse().GetOutputLogprobs())
				require.Equal(t, uint32(5), last.Request.GetResponse().GetOutputCandidates().GetTopN())
			},
		},
		{
			name: "bad_words token id sequences",
			sp:   map[string]any{"bad_words": [][]int{{777, 778}}},
			check: func(t *testing.T, d *vllmpb.DecodingParameters) {
				require.Len(t, d.GetBadWordsTokenIds(), 1)
				require.Equal(t, []uint32{777, 778}, d.GetBadWordsTokenIds()[0].GetIds())
			},
		},
	}
	for _, tc := range structVariants {
		t.Run(tc.name, func(t *testing.T) {
			b, mErr := json.Marshal(map[string]any{
				"model":           "llama",
				"token_ids":       []int{1},
				"sampling_params": tc.sp,
			})
			require.NoError(t, mErr)
			st, _ := postJSON(string(b))
			require.Equal(t, http.StatusOK, st)
			last, ok := upstream.LastCall()
			require.True(t, ok)
			tc.check(t, last.Request.GetDecoding())
		})
	}

	// 3. Reject unknown keys, string bad_words, conflicting structured outputs, and malformed JSON with HTTP 400.
	badInputs := []struct {
		name    string
		body    string
		errText string
	}{
		{
			name:    "malformed JSON",
			body:    `{"model":`,
			errText: "",
		},
		{
			name:    "unknown top-level key",
			body:    `{"model":"m","token_ids":[1],"unknown_top_key":true}`,
			errText: "unknown_top_key",
		},
		{
			name:    "unknown sampling_params key",
			body:    `{"model":"m","token_ids":[1],"sampling_params":{"unknown_sampling_flag":1}}`,
			errText: "unknown_sampling_flag",
		},
		{
			name:    "unsupported guided_decoding_backend",
			body:    `{"model":"m","token_ids":[1],"sampling_params":{"guided_decoding_backend":"outlines"}}`,
			errText: "guided_decoding_backend",
		},
		{
			name:    "string bad_words without tokenizer",
			body:    `{"model":"m","token_ids":[1],"sampling_params":{"bad_words":["secret"]}}`,
			errText: "bad_words",
		},
		{
			name:    "conflicting structured_outputs",
			body:    `{"model":"m","token_ids":[1],"sampling_params":{"structured_outputs":{"json":"{}","regex":".*"}}}`,
			errText: "at most one structured output",
		},
		{
			name:    "unsupported continuous_usage_stats",
			body:    `{"model":"m","token_ids":[1],"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}`,
			errText: "continuous_usage_stats",
		},
		{
			name:    "unknown vllm_xargs key",
			body:    `{"model":"m","token_ids":[1],"sampling_params":{"vllm_xargs":{"mystery_xarg":1}}}`,
			errText: "mystery_xarg",
		},
		{
			name:    "unknown features key",
			body:    `{"model":"m","token_ids":[1],"features":{"audio_raw":[]}}`,
			errText: "audio_raw",
		},
	}
	for _, tc := range badInputs {
		t.Run("400 "+tc.name, func(t *testing.T) {
			st, errJSON := postJSON(tc.body)
			require.Equal(t, http.StatusBadRequest, st)
			if tc.errText != "" {
				raw, _ := json.Marshal(errJSON)
				require.Contains(t, string(raw), tc.errText)
			}
		})
	}

	// 4. Upstream gRPC status -> HTTP status mapping (4xx preserved, 5xx -> 502).
	statusCases := []struct {
		model    string
		wantHTTP int
	}{
		{model: "rate-limited", wantHTTP: http.StatusTooManyRequests},
		{model: "not-found", wantHTTP: http.StatusNotFound},
		{model: "deadline", wantHTTP: http.StatusBadGateway},
		{model: "unavailable", wantHTTP: http.StatusBadGateway},
	}
	for _, tc := range statusCases {
		st, _ := postJSON(`{"model":"` + tc.model + `","token_ids":[1]}`)
		require.Equal(t, tc.wantHTTP, st, "model=%s", tc.model)
	}
}

func TestVLLMHTTPGenerate_ConditionalDecodeHitMidStreamSSEErrorAndCancel(t *testing.T) {
	t.Run("conditional-decode cache hit serves HTTP unary and SSE directly", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		var prefillCalls, decodeCalls atomic.Int32

		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = unmarshalGatewayVLLMReq(t, r)
			if r.Header.Get(reqcommon.EPPProfileHeaderKey) == gateway.PhaseDecode && r.Header.Get("Prefer") == "if-available" {
				writeGatewayVLLMFrames(t, w, &vllmpb.GenerateResponse{
					PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 3},
					Outputs: &vllmpb.SequenceOutput{
						TokenIds: []uint32{55},
						FinishInfo: &vllmpb.FinishInfo{
							FinishReason:    vllmpb.FinishInfo_STOP,
							NumOutputTokens: 1,
							NumCachedTokens: proto.Uint32(3),
						},
					},
				})
				return
			}
			if r.Header.Get(reqcommon.EPPProfileHeaderKey) == gateway.PhasePrefill {
				prefillCalls.Add(1)
			} else {
				decodeCalls.Add(1)
			}
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		cond, err := steps.NewConditionalDecodeStep(gwClient, nil)
		require.NoError(t, err)
		prefill, err := steps.NewPrefillStep(gwClient, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)
		decode, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)
		addr := startCoordinatorH2C(
			t,
			config.ServerConfig{},
			pipeline.New([]pipeline.Step{cond, prefill, decode}),
			gwClient,
			server.WithWorkerProtocol(config.WorkerProtocolGRPC),
		)

		// Unary hit
		uReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"m","token_ids":[1,2,3]}`))
		require.NoError(t, err)
		uResp, err := (&http.Client{}).Do(uReq)
		require.NoError(t, err)
		uBody, _ := io.ReadAll(uResp.Body)
		_ = uResp.Body.Close()
		require.Equal(t, http.StatusOK, uResp.StatusCode)
		require.Contains(t, string(uBody), `"token_ids":[55]`)

		// SSE hit
		sReq, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"m","token_ids":[1,2,3],"stream":true}`))
		require.NoError(t, err)
		sResp, err := (&http.Client{}).Do(sReq)
		require.NoError(t, err)
		sBody, _ := io.ReadAll(sResp.Body)
		_ = sResp.Body.Close()
		require.Equal(t, http.StatusOK, sResp.StatusCode)
		require.Contains(t, string(sBody), `"token_ids":[55]`)
		require.True(t, strings.HasSuffix(string(sBody), "data: [DONE]\n\n"))

		require.Zero(t, prefillCalls.Load())
		require.Zero(t, decodeCalls.Load())
		require.Equal(t, float64(2), probeCount(t, reg, coordmetrics.ProbeResultServed))
	})

	t.Run("mid-stream upstream error after SSE commit omits DONE sentinel", func(t *testing.T) {
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = unmarshalGatewayVLLMReq(t, r)
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			first := grpctest.Frame(t, &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 2},
				Outputs:    &vllmpb.SequenceOutput{NumTokens: 1, TokenIds: []uint32{77}},
			})
			_, _ = w.Write(first)
			_ = http.NewResponseController(w).Flush()
			grpctest.WriteStatusTrailer(w, codes.Internal, "engine crashed mid-stream")
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		decode, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)
		addr := startCoordinatorH2C(
			t,
			config.ServerConfig{},
			pipeline.New([]pipeline.Step{decode}),
			gwClient,
			server.WithWorkerProtocol(config.WorkerProtocolGRPC),
		)

		req, err := http.NewRequestWithContext(testCtx(t), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"m","token_ids":[1,2],"stream":true}`))
		require.NoError(t, err)
		resp, err := (&http.Client{}).Do(req)
		require.NoError(t, err)
		raw, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		require.Equal(t, http.StatusOK, resp.StatusCode)
		require.Contains(t, string(raw), `"token_ids":[77]`)
		require.NotContains(t, string(raw), "data: [DONE]", "mid-stream error must not emit DONE sentinel")
	})

	t.Run("client cancel mid-stream during SSE cancels upstream decode stream", func(t *testing.T) {
		decodeCancelled := make(chan struct{})
		gw := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = unmarshalGatewayVLLMReq(t, r)
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			first := grpctest.Frame(t, &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 2},
				Outputs:    &vllmpb.SequenceOutput{NumTokens: 1, TokenIds: []uint32{88}},
			})
			_, _ = w.Write(first)
			_ = http.NewResponseController(w).Flush()
			<-r.Context().Done()
			close(decodeCancelled)
		}))

		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		decode, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.SharedStorage})
		require.NoError(t, err)
		addr := startCoordinatorH2C(
			t,
			config.ServerConfig{},
			pipeline.New([]pipeline.Step{decode}),
			gwClient,
			server.WithWorkerProtocol(config.WorkerProtocolGRPC),
		)

		callCtx, cancelCall := context.WithCancel(testCtx(t))
		req, err := http.NewRequestWithContext(callCtx, http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(`{"model":"m","token_ids":[1,2],"stream":true}`))
		require.NoError(t, err)
		resp, err := (&http.Client{}).Do(req)
		require.NoError(t, err)
		defer func() { _ = resp.Body.Close() }()

		line, err := bufio.NewReader(resp.Body).ReadString('\n')
		require.NoError(t, err)
		require.Contains(t, line, `"token_ids":[88]`)
		cancelCall()

		select {
		case <-decodeCancelled:
		case <-testCtx(t).Done():
			t.Fatal("timed out waiting for upstream decode cancellation on SSE cancel")
		}
	})
}
