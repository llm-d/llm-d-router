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

package p2psource

import (
	"encoding/json"
	"math"
	"net"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

const testKVCRControlPort = 17771

func kvcrProducer() *Producer {
	return New("test", Config{MinCachedTokenDelta: 1, Transport: TransportKVCR, KVCRControlPort: testKVCRControlPort})
}

// generateFrame returns a framed vllm.Inference GenerateRequest.
func generateFrame(t *testing.T) []byte {
	t.Helper()
	frame, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
	})
	require.NoError(t, err)
	return frame
}

// grpcRequest builds a request on path whose body is frame as the
// vllm.Inference parser leaves it, with best stashed when non-nil.
func grpcRequest(p *Producer, id, path string, frame []byte, best *bestMatchPeer) *scheduling.InferenceRequest {
	req := &scheduling.InferenceRequest{
		RequestID: id,
		Headers:   map[string]string{":path": path},
		Body:      &fwkrequest.InferenceRequestBody{Payload: fwkrequest.PayloadBytes(frame), RawBody: frame},
	}
	if best != nil {
		req.PutAttribute(p.attrKey(), best)
	}
	return req
}

// kvcrPeer builds a stashed best match at 10.0.0.2:8080 on rank 0 holding 64
// cached tokens.
func kvcrPeer() *bestMatchPeer {
	return &bestMatchPeer{
		hostPort:     net.JoinHostPort("10.0.0.2", "8080"),
		address:      "10.0.0.2",
		p2pPort:      testKVCRControlPort,
		cachedTokens: 64,
	}
}

// forwardedRequest decodes the GenerateRequest the request body now carries.
func forwardedRequest(t *testing.T, req *scheduling.InferenceRequest) *vllmpb.GenerateRequest {
	t.Helper()
	frame, ok := req.Body.Payload.(fwkrequest.PayloadBytes)
	require.True(t, ok, "body is %T", req.Body.Payload)
	payload, err := grpcframe.ParsePayload(frame)
	require.NoError(t, err)
	var gen vllmpb.GenerateRequest
	require.NoError(t, proto.Unmarshal(payload, &gen))
	return &gen
}

func decisionOutcome(t *testing.T, p *Producer, req *scheduling.InferenceRequest) string {
	t.Helper()
	d, ok := scheduling.ReadRequestAttribute[attrprefix.P2PSourceDecision](req, p.decisionDataKey)
	require.True(t, ok)
	return d.Outcome
}

// assertFetchHint checks that the forwarded request carries exactly one
// kv.fetch action naming endpoint, that its block hash survives vLLM's gRPC
// Struct conversion as an integer, and that no P2P source header was written.
func assertFetchHint(t *testing.T, req *scheduling.InferenceRequest, endpoint string) {
	t.Helper()
	assert.True(t, req.Body.Mutated)
	gen := forwardedRequest(t, req)
	assert.Equal(t, "hello", gen.GetText(), "the rest of the request is kept")

	hints := gen.GetKvHints()
	require.NotNil(t, hints)
	assert.Equal(t, "0.1", hints.GetProtocolVersion())
	assert.NotEmpty(t, hints.GetMessageId())
	require.Len(t, hints.GetActions(), 1)
	action := hints.GetActions()[0]
	assert.Equal(t, "kv.fetch", action.GetActionType())
	assert.Equal(t, "1.0", action.GetActionVersion())
	assert.NotEmpty(t, action.GetActionId())

	payload := action.GetPayload().AsMap()
	assert.Equal(t, endpoint, payload["source_control_endpoint"])
	hashes, ok := payload["block_hashes"].([]any)
	require.True(t, ok, "block_hashes is %T", payload["block_hashes"])
	require.Len(t, hashes, 1)
	h, ok := hashes[0].(float64)
	require.True(t, ok)
	assert.Equal(t, math.Trunc(h), h, "block hash must be a whole number")
	assert.GreaterOrEqual(t, h, 0.0)
	assert.Less(t, h, float64(1<<53), "vLLM's gRPC server converts only numbers below 2^53 to ints")

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

func TestPluginFactory_Transport(t *testing.T) {
	tests := []struct {
		name      string
		config    string
		wantErr   bool
		transport string
		port      int
	}{
		{name: "default", config: `{}`, transport: TransportOffloading},
		{name: "offloading", config: `{"transport": "offloading"}`, transport: TransportOffloading},
		{name: "kvcr", config: `{"transport": "kvcr", "kvcrControlPort": 17771}`, transport: TransportKVCR, port: 17771},
		{name: "kvcr without port", config: `{"transport": "kvcr"}`, wantErr: true},
		{name: "kvcr port out of range", config: `{"transport": "kvcr", "kvcrControlPort": 70000}`, wantErr: true},
		{name: "unknown transport", config: `{"transport": "nixl"}`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := PluginFactory("test", json.NewDecoder(strings.NewReader(tt.config)), nil)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			producer := p.(*Producer)
			assert.Equal(t, tt.transport, producer.transport)
			if tt.port != 0 {
				assert.Equal(t, tt.port, producer.kvcrControlPort)
			}
		})
	}
}

func TestPreRequest_KVCR_WritesFetchHint(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := kvcrProducer()
	computing := endpoint(p, "pod-a", "10.0.0.1", 0)
	req := grpcRequest(p, "req-kvcr", vllmpb.Inference_GenerateStream_FullMethodName, generateFrame(t), kvcrPeer())

	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(computing)))

	assertFetchHint(t, req, "tcp://10.0.0.2:17771")
	assert.Equal(t, "pulled", decisionOutcome(t, p, req))
}

// A written hint replaces every kv_hints the client sent.
func TestPreRequest_KVCR_ReplacesClientHints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := kvcrProducer()
	clientHints := &vllmpb.KVHintsEnvelope{
		ProtocolVersion: kvHintProtocolVersion,
		MessageId:       "client",
		Actions:         []*vllmpb.KVHintAction{{ActionId: "client-0", ActionType: kvFetchActionType}},
	}
	frame, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
		Model:   "m",
		Prompt:  &vllmpb.GenerateRequest_Text{Text: "hello"},
		KvHints: clientHints,
	})
	require.NoError(t, err)
	req := grpcRequest(p, "req-client", vllmpb.Inference_GenerateStream_FullMethodName, frame, kvcrPeer())

	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0))))

	assertFetchHint(t, req, "tcp://10.0.0.2:17771")
	assert.NotEqual(t, "client", forwardedRequest(t, req).GetKvHints().GetMessageId())
}

// The unary Generate path carries hints the same way, including every path
// form the vllminference parser accepts.
func TestPreRequest_KVCR_GeneratePaths(t *testing.T) {
	for _, path := range []string{
		vllmpb.Inference_Generate_FullMethodName,
		strings.TrimPrefix(vllmpb.Inference_Generate_FullMethodName, "/"),
		vllmpb.Inference_Generate_FullMethodName + "/",
		" " + vllmpb.Inference_GenerateStream_FullMethodName + "/ ",
	} {
		t.Run(path, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			p := kvcrProducer()
			req := grpcRequest(p, "req-unary", path, generateFrame(t), kvcrPeer())
			require.NoError(t, p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0))))
			assertFetchHint(t, req, "tcp://10.0.0.2:17771")
		})
	}
}

// The control port is kvcrControlPort plus the source's data parallel rank.
// A prefill target still gets its P2P tier header: the offloading P/D
// connector builds the decode request from it.
func TestPreRequest_KVCR_ControlPortAddsRankIndex(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := kvcrProducer()
	source := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: k8stypes.NamespacedName{Name: "pod-src"}, Address: "10.0.0.3", Port: "8081", RankIndex: 1,
	}, nil, nil)
	source.Put(p.prefixMatchDataKey, attrprefix.NewPrefixCacheMatchInfo(4, 4, testBlockSize).WithCachedBlockCount(4))
	prefill := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: k8stypes.NamespacedName{Name: "pod-p"}, Address: "10.0.0.2", Port: "8082", RankIndex: 2,
	}, nil, nil)
	prefill.Put(p.prefixMatchDataKey, attrprefix.NewPrefixCacheMatchInfo(0, 4, testBlockSize).WithCachedBlockCount(0))

	req := grpcRequest(p, "req-rank", vllmpb.Inference_GenerateStream_FullMethodName, generateFrame(t), nil)
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{source, prefill}))
	require.NoError(t, p.PreRequest(ctx, req, prefillOnly(prefill)))

	assertFetchHint(t, req, "tcp://10.0.0.3:17772")
	assert.Equal(t, net.JoinHostPort("10.0.0.2", "7779"), req.Headers[routing.KVCachePrefillHeader])
}

// A request gets the reusable prefix floor in Produce exactly when PreRequest
// can give it a hint. A decode leg receives the prompt's KV from a prefill pod
// (remote_prefiller on the offloading connector, do_remote_prefill on the
// coordinator's NIXL and shared-storage connectors), a request that bypasses
// the prefix cache does not load fetched blocks, and only vllm.Inference
// requests can carry the hint.
func TestKVCR_FloorAndHintSkipTheSameRequests(t *testing.T) {
	stream := vllmpb.Inference_GenerateStream_FullMethodName
	for _, tc := range []struct {
		name    string
		path    string
		kv      map[string]any
		bypass  bool
		outcome string
	}{
		{name: "generate", path: stream, outcome: "pulled"},
		{name: "remote_prefiller", path: stream, kv: reqcommon.RemotePrefillerParams("kv-1", "10.0.0.9", 7777), outcome: "decode-request"},
		{name: "do_remote_prefill", path: stream, kv: map[string]any{reqcommon.FieldDoRemotePrefill: true, reqcommon.FieldDoRemoteDecode: false}, outcome: "decode-request"},
		{name: "do_remote_prefill false", path: stream, kv: map[string]any{reqcommon.FieldDoRemotePrefill: false}, outcome: "pulled"},
		{name: "bypass_prefix_cache", path: stream, bypass: true, outcome: "cache-bypassed"},
		{name: "openai", path: "/v1/completions", outcome: "unsupported-request"},
		{name: "sglang", path: apiv1.SglangService_Generate_FullMethodName, outcome: "unsupported-request"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			p := kvcrProducer()
			kv := &vllmpb.KVCacheParameters{BypassPrefixCache: tc.bypass}
			if tc.kv != nil {
				params, err := structpb.NewStruct(tc.kv)
				require.NoError(t, err)
				kv.KvTransferParams = params
			}
			frame, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
				Model:  "m",
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
				Kv:     kv,
			})
			require.NoError(t, err)
			req := grpcRequest(p, "req-"+tc.name, tc.path, frame, nil)

			source := tierEndpointWithLoad(p, "pod-src", "10.0.0.2", 4, 0)
			require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{source}))
			_, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
			require.True(t, ok, "the source is stashed either way")
			_, floor := readReusablePrefixTokens(req, "test")
			assert.Equal(t, tc.outcome == "pulled", floor, "floor published")

			require.NoError(t, p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0))))
			assert.Equal(t, tc.outcome, decisionOutcome(t, p, req))
			if tc.outcome != "pulled" {
				assert.False(t, req.Body.Mutated)
				assert.Equal(t, fwkrequest.PayloadBytes(frame), req.Body.Payload)
			}
		})
	}
}

func TestPreRequest_KVCR_BelowFloor_NoHint(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 100, Transport: TransportKVCR, KVCRControlPort: testKVCRControlPort})
	frame := generateFrame(t)
	req := grpcRequest(p, "req-floor", vllmpb.Inference_GenerateStream_FullMethodName, frame, kvcrPeer())

	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0))))

	assert.False(t, req.Body.Mutated)
	assert.Equal(t, fwkrequest.PayloadBytes(frame), req.Body.Payload)
	assert.Equal(t, "below-floor", decisionOutcome(t, p, req))
}

// Only vllm.Inference requests can carry a hint: an OpenAI JSON body or
// another gRPC API's frame is left as received.
func TestPreRequest_KVCR_UnsupportedRequest_NoHint(t *testing.T) {
	t.Run("openai json", func(t *testing.T) {
		ctx := utils.NewTestContext(t)
		p := kvcrProducer()
		req := preRequest(p, "req-http", kvcrPeer())
		req.Headers[":path"] = "/v1/completions"

		require.NoError(t, p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0))))

		assertNoSource(t, req)
		assert.Equal(t, "unsupported-request", decisionOutcome(t, p, req))
	})
	t.Run("sglang grpc", func(t *testing.T) {
		ctx := utils.NewTestContext(t)
		p := kvcrProducer()
		frame := generateFrame(t)
		req := grpcRequest(p, "req-sglang", apiv1.SglangService_Generate_FullMethodName, frame, kvcrPeer())

		require.NoError(t, p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0))))

		assert.False(t, req.Body.Mutated)
		assert.Equal(t, fwkrequest.PayloadBytes(frame), req.Body.Payload)
		assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
		assert.Equal(t, "unsupported-request", decisionOutcome(t, p, req))
	})
}
