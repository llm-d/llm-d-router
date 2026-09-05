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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc"
	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

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
