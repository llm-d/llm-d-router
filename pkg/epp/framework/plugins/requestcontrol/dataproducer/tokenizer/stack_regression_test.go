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

package tokenizer

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/sglanghttp"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc"
	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmhttp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func TestSuppliedTokensReachScheduling(t *testing.T) {
	msg, err := proto.Marshal(&pb.GenerateRequest{
		Input: &pb.GenerateRequest_Tokenized{Tokenized: &pb.TokenizedInput{InputIds: []uint32{1, 2, 3}}},
		MmInputs: &pb.MultimodalInputs{
			MmHashes:       []string{"image-hash"},
			MmPlaceholders: []*pb.PlaceholderRange{{Offset: 1, Length: 2}},
		},
	})
	require.NoError(t, err)
	framed := make([]byte, 5+len(msg))
	binary.BigEndian.PutUint32(framed[1:5], uint32(len(msg)))
	copy(framed[5:], msg)
	features := []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "image-hash", Offset: 1, Length: 2}}
	for _, tc := range []struct {
		name, path, salt string
		parser           fwkrh.Parser
		body             []byte
		features         []fwkrh.MultiModalFeature
	}{
		{
			name: "sglang", path: "/generate", salt: "tenant-a", parser: sglanghttp.NewSGLangHTTPParser(),
			body: []byte(`{"input_ids":[1,2,3],"extra_key":"tenant-a","sampling_params":{"max_new_tokens":1}}`),
		},
		{
			name: "vllm HTTP", path: "/inference/v1/generate", salt: "tenant-a", parser: vllmhttp.NewVllmHTTPParser(),
			body:     []byte(`{"model":"m","token_ids":[1,2,3],"cache_salt":"tenant-a","features":{"mm_hashes":{"image":["image-hash"]},"mm_placeholders":{"image":[{"offset":1,"length":2}]}},"sampling_params":{"max_tokens":1}}`),
			features: features,
		},
		{
			name: "vllm gRPC", path: "/vllm.grpc.engine.VllmEngine/Generate", parser: vllmgrpc.NewVllmGRPCParser(),
			body: framed, features: features,
		},
	} {
		for _, backend := range []string{backendVLLM, backendEstimate} {
			t.Run(tc.name+"/"+backend, func(t *testing.T) {
				parsed, err := tc.parser.ParseRequest(context.Background(), tc.body, map[string]string{":path": tc.path})
				require.NoError(t, err)
				before, err := json.Marshal(parsed.Body.Payload)
				require.NoError(t, err)
				existing := parsed.Body.TokenizedRequest
				p := newTestPlugin(&mockTokenizer{})
				if backend == backendEstimate {
					p.backend = estimateBackend{}
					p.backendName = backendEstimate
				}
				req := &scheduling.InferenceRequest{Body: parsed.Body}
				require.NoError(t, p.Produce(context.Background(), req, nil))
				require.Equal(t, &fwkrh.TokenizedRequest{
					Prompts:   []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}, MultiModalFeatures: tc.features}},
					CacheSalt: tc.salt,
				}, req.Body.TokenizedRequest)
				if existing != nil {
					require.Same(t, existing, req.Body.TokenizedRequest)
				}
				after, err := json.Marshal(req.Body.Payload)
				require.NoError(t, err)
				require.Equal(t, before, after)
				require.False(t, req.Body.Mutated)
			})
		}
	}
}

func TestCompletionTokenBatchesUseRenderedTokens(t *testing.T) {
	const raw = ` {"model":"m","prompt":[[1,2,3],[4,5,6]],"truncate_prompt_tokens":2,"cache_salt":"tenant-a"} `
	parsed, err := openai.NewOpenAIParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/completions"})
	require.NoError(t, err)
	require.Nil(t, parsed.Body.TokenizedRequest)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/completions/render", r.URL.Path)
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Equal(t, raw, string(body))
		_, _ = io.WriteString(w, `[{"token_ids":[2,3]},{"token_ids":[5,6]}]`)
	}))
	defer srv.Close()
	req := &scheduling.InferenceRequest{Body: parsed.Body}
	require.NoError(t, newTestPlugin(newHTTPRenderer(t, srv)).Produce(context.Background(), req, nil))
	require.Equal(t, &fwkrh.TokenizedRequest{
		Prompts:   []fwkrh.PromptTokens{{TokenIDs: []uint32{2, 3}}, {TokenIDs: []uint32{5, 6}}},
		CacheSalt: "tenant-a",
	}, req.Body.TokenizedRequest)
	require.Equal(t, fwkrh.RawPayload(raw), req.Body.WirePayload())
}

func TestGRPCTextProducesTokens(t *testing.T) {
	msg, err := proto.Marshal(&pb.GenerateRequest{Input: &pb.GenerateRequest_Text{Text: "Hello world"}})
	require.NoError(t, err)
	framed := make([]byte, 5+len(msg))
	binary.BigEndian.PutUint32(framed[1:5], uint32(len(msg)))
	copy(framed[5:], msg)
	parsed, err := vllmgrpc.NewVllmGRPCParser().ParseRequest(context.Background(), framed, map[string]string{":path": "/vllm.grpc.engine.VllmEngine/Generate"})
	require.NoError(t, err)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Equal(t, "/v1/completions/render", r.URL.Path)
		require.JSONEq(t, `{"model":"grpc-model","prompt":"Hello world"}`, string(body))
		_, _ = io.WriteString(w, `[{"token_ids":[1,2,3]}]`)
	}))
	defer srv.Close()
	req := &scheduling.InferenceRequest{Body: parsed.Body}
	renderer := newHTTPRenderer(t, srv)
	p := newTestPlugin(renderer)
	p.backend = renderBackend{tk: renderer, modelName: "grpc-model"}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedRequest)
	require.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
	require.Equal(t, "Hello world", req.Body.Payload.(fwkrh.PayloadProto).Message.(*pb.GenerateRequest).GetText())
}

func TestDirectRenderKeepsModel(t *testing.T) {
	for _, tc := range []struct {
		path   string
		parser fwkrh.Parser
	}{
		{"/v1/chat/completions/render", openai.NewOpenAIParser()},
		{"/v1/completions/render", openai.NewOpenAIParser()},
		{"/v1/messages/render", anthropic.NewAnthropicParser()},
	} {
		t.Run(tc.path, func(t *testing.T) {
			raw := []byte(` {"model":"adapter","messages":[{"role":"user","content":"hi"}],"prompt":"hi","system":"keep","tools":[{"z":9007199254740993,"a":1.00}]} `)
			parsed, err := tc.parser.ParseRequest(context.Background(), raw, map[string]string{":path": tc.path})
			require.NoError(t, err)
			require.Equal(t, "adapter", parsed.Body.Model)
			require.True(t, parsed.Body.RenderRequest)
			require.True(t, parsed.SkipResponseProcessing)
			require.Equal(t, fwkrh.RawPayload(raw), parsed.Body.WirePayload())
			parsed.Body.Payload, err = tc.parser.(fwkrh.ModelNameRewriter).RewriteModelName(parsed.Body.Payload.(fwkrh.MarshalablePayload), "resolved")
			require.NoError(t, err)
			parsed.Body.Mutated = true
			req := &scheduling.InferenceRequest{Body: parsed.Body}
			require.NoError(t, newTestPlugin(&mockTokenizer{}).Produce(context.Background(), req, nil))
			require.Nil(t, req.Body.TokenizedRequest)
			forwarded, err := parsed.Body.WirePayload().(fwkrh.Marshaler).Marshal()
			require.NoError(t, err)
			assertNativeFieldsUnchanged(t, raw, forwarded, "model")
			require.Contains(t, string(forwarded), `"model":"resolved"`)
		})
	}
}
