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

package requestcontrol

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/protobuf/proto"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/sglanghttp"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vertexai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmhttp"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const eppPromptIDsSanitizedMetric = "llm_d_epp_prompt_ids_sanitized_total"

type sanitizeInspectProducer struct {
	called   bool
	seenKeys map[string]bool
}

func (p *sanitizeInspectProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-sanitize-inspect", Name: "inspect"}
}

func (p *sanitizeInspectProducer) Produces() map[fwkplugin.DataKey]any { return nil }
func (p *sanitizeInspectProducer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{}
}

func (p *sanitizeInspectProducer) Produce(_ context.Context, req *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	p.called = true
	p.seenKeys = make(map[string]bool)
	if m, ok := req.Body.Payload.(fwkrequest.PayloadMap); ok {
		for k := range m {
			p.seenKeys[k] = true
		}
	}
	return nil
}

func newSanitizeScheduleResult() *fwksched.SchedulingResult {
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		Address: "192.168.1.100",
		Port:    "8000",
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
	}, nil, fwkdl.NewAttributes())
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func TestDirector_SanitizesExternalTextRoutes(t *testing.T) {
	metrics.Register()

	cases := []struct {
		name      string
		path      string
		wantRoute string
		parser    fwkrh.Parser
		baseField string
	}{
		{
			name:      "chat completions",
			path:      reqcommon.PathChatCompletions,
			wantRoute: reqcommon.APITypeChatCompletions.String(),
			parser:    openai.NewOpenAIParser(),
			baseField: `"messages":[{"role":"user","content":"hi"}]`,
		},
		{
			name:      "string completions",
			path:      reqcommon.PathCompletions,
			wantRoute: reqcommon.APITypeCompletions.String(),
			parser:    openai.NewOpenAIParser(),
			baseField: `"prompt":"hello"`,
		},
		{
			name:      "string-array completions",
			path:      reqcommon.PathCompletions,
			wantRoute: reqcommon.APITypeCompletions.String(),
			parser:    openai.NewOpenAIParser(),
			baseField: `"prompt":["hello","world"]`,
		},
		{
			name:      "messages",
			path:      reqcommon.PathMessages,
			wantRoute: reqcommon.APITypeMessages.String(),
			parser:    anthropic.NewAnthropicParser(),
			baseField: `"max_tokens":16,"messages":[{"role":"user","content":"hi"}]`,
		},
		{
			name:      "responses",
			path:      reqcommon.PathResponses,
			wantRoute: reqcommon.APITypeResponses.String(),
			parser:    openai.NewOpenAIParser(),
			baseField: `"input":"hi"`,
		},
	}

	for _, tc := range cases {
		raw := []byte(`{"model":"llama-3-8b",` + tc.baseField +
			`,"prompt_token_ids":[1,2,3]` +
			`,"prompt_render_fingerprint":"fp-1"` +
			`,"render_fingerprint":"fp-2"` +
			`,"kv_transfer_params":{"prompt_token_ids":[4,5],"do_remote_prefill":true}` +
			`,"ec_transfer_params":{"image":[]}` +
			`,"input_ids":[6,7]}`)

		t.Run(tc.name+"/default strips before producers and preserves PreRequest KV injection", func(t *testing.T) {
			metrics.Reset()
			producer := &sanitizeInspectProducer{}
			kvInjector := &mockPreRequestPlugin{
				name: "kv-injector",
				modifyFn: func(r *fwksched.InferenceRequest) {
					r.Body.MutatePayloadMap(func(m fwkrequest.PayloadMap) {
						m[reqcommon.FieldKVTransferParams] = map[string]any{"remote_kv_source": "10.0.0.1:8000"}
					})
				},
			}
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{producer, kvInjector})

			dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
			dir.requestControlPlugins = *NewConfig().
				WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
				WithDataProducerPlugins(producer).
				WithPreRequestPlugins(kvInjector)

			headers := map[string]string{":path": tc.path}
			parsed, err := tc.parser.ParseRequest(ctx, raw, headers)
			require.NoError(t, err)
			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{ID: "req-1", Headers: headers, RawBody: raw},
				Parser:  tc.parser,
			}

			_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
			require.NoError(t, err)
			require.True(t, producer.called, "producer must run on stripped request")
			for _, field := range reqcommon.ExternalTextRouteSanitizedFields {
				require.False(t, producer.seenKeys[field], "producer must not observe client field %s", field)
			}

			var out map[string]any
			require.NoError(t, json.Unmarshal(reqCtx.Request.RawBody, &out))
			for _, field := range []string{
				reqcommon.FieldPromptTokenIDs,
				reqcommon.FieldPromptRenderFingerprint,
				reqcommon.FieldRenderFingerprint,
				reqcommon.FieldECTransferParams,
				reqcommon.FieldInputIDs,
			} {
				require.NotContains(t, out, field, "repackaged body must omit client field %s", field)
			}
			require.Equal(t, map[string]any{"remote_kv_source": "10.0.0.1:8000"}, out[reqcommon.FieldKVTransferParams],
				"PreRequest plugin KV transfer metadata must survive client stripping")

			wantMetric := `
# HELP llm_d_epp_prompt_ids_sanitized_total [ALPHA] Total number of external text-route requests where client-supplied prompt token, fingerprint, KV transfer, or EC transfer fields were stripped or rejected.
# TYPE llm_d_epp_prompt_ids_sanitized_total counter
llm_d_epp_prompt_ids_sanitized_total{action="stripped",component="epp",route="` + tc.wantRoute + `"} 1
`
			require.NoError(t, promtestutil.GatherAndCompare(ctrlmetrics.Registry, strings.NewReader(wantMetric), eppPromptIDsSanitizedMetric))
		})

		t.Run(tc.name+"/rejectClientPromptIDs returns 400 before producers", func(t *testing.T) {
			metrics.Reset()
			producer := &sanitizeInspectProducer{}
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{producer})

			dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
			dir.requestControlPlugins = *NewConfig().
				WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
				WithDataProducerPlugins(producer).
				WithRejectClientPromptIDs(true)

			headers := map[string]string{":path": tc.path}
			parsed, err := tc.parser.ParseRequest(ctx, raw, headers)
			require.NoError(t, err)
			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{ID: "req-2", Headers: headers, RawBody: raw},
				Parser:  tc.parser,
			}

			_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
			var reqErr errcommon.Error
			require.ErrorAs(t, err, &reqErr)
			require.Equal(t, errcommon.BadRequest, reqErr.Code)
			require.False(t, producer.called, "rejected request must not reach data producers")

			wantMetric := `
# HELP llm_d_epp_prompt_ids_sanitized_total [ALPHA] Total number of external text-route requests where client-supplied prompt token, fingerprint, KV transfer, or EC transfer fields were stripped or rejected.
# TYPE llm_d_epp_prompt_ids_sanitized_total counter
llm_d_epp_prompt_ids_sanitized_total{action="rejected",component="epp",route="` + tc.wantRoute + `"} 1
`
			require.NoError(t, promtestutil.GatherAndCompare(ctrlmetrics.Registry, strings.NewReader(wantMetric), eppPromptIDsSanitizedMetric))
		})

		t.Run(tc.name+"/trustCallerFields preserves fields for internal callers", func(t *testing.T) {
			metrics.Reset()
			producer := &sanitizeInspectProducer{}
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{producer})

			dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
			dir.requestControlPlugins = *NewConfig().
				WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
				WithDataProducerPlugins(producer).
				WithTrustCallerFields(true)

			headers := map[string]string{":path": tc.path}
			parsed, err := tc.parser.ParseRequest(ctx, raw, headers)
			require.NoError(t, err)
			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{ID: "req-3", Headers: headers, RawBody: raw},
				Parser:  tc.parser,
			}

			_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
			require.NoError(t, err)
			require.True(t, producer.called)
			for _, field := range reqcommon.ExternalTextRouteSanitizedFields {
				require.True(t, producer.seenKeys[field], "trusted caller field %s must be preserved", field)
			}
			require.False(t, parsed.Body.Mutated, "trusted caller request should not force re-marshaling")
			require.NoError(t, promtestutil.GatherAndCompare(ctrlmetrics.Registry, strings.NewReader(""), eppPromptIDsSanitizedMetric))
		})
	}
}

func TestDirector_RejectClientPromptIDsAlsoRejectsKVTransferParams(t *testing.T) {
	metrics.Register()
	metrics.Reset()

	dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
	dir.requestControlPlugins = *NewConfig().
		WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
		WithRejectClientPromptIDs(true)

	parser := openai.NewOpenAIParser()
	raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"kv_transfer_params":{"do_remote_prefill":true}}`)
	headers := map[string]string{":path": reqcommon.PathChatCompletions}
	parsed, err := parser.ParseRequest(ctx, raw, headers)
	require.NoError(t, err)
	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{ID: "req-kv", Headers: headers, RawBody: raw},
		Parser:  parser,
	}

	_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
	var reqErr errcommon.Error
	require.ErrorAs(t, err, &reqErr)
	require.Equal(t, errcommon.BadRequest, reqErr.Code)
	require.Contains(t, reqErr.Msg, reqcommon.FieldKVTransferParams)
}

// handleSanitizeRequest parses raw with parser and runs it through a
// single-pod Director that uses cfg.
func handleSanitizeRequest(t *testing.T, cfg *Config, parser fwkrh.Parser, path string, raw []byte) (*handlers.RequestContext, *fwkrequest.InferenceRequestBody, error) {
	t.Helper()
	metrics.Reset()
	dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
	dir.requestControlPlugins = *cfg.WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil))

	headers := map[string]string{":path": path}
	parsed, err := parser.ParseRequest(ctx, raw, headers)
	require.NoError(t, err)
	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{ID: "req-sanitize", Headers: headers, RawBody: raw},
		Parser:  parser,
	}
	_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
	return reqCtx, parsed.Body, err
}

func forwardedBody(t *testing.T, reqCtx *handlers.RequestContext) map[string]any {
	t.Helper()
	var out map[string]any
	require.NoError(t, json.Unmarshal(reqCtx.Request.RawBody, &out))
	return out
}

func requireNoSanitizeMetric(t *testing.T) {
	t.Helper()
	require.NoError(t, promtestutil.GatherAndCompare(ctrlmetrics.Registry, strings.NewReader(""), eppPromptIDsSanitizedMetric))
}

// A dedicated token route keeps the client token IDs, because sending them is
// the purpose of the route. It drops kv_transfer_params, which only the
// Coordinator, the EPP plugins, and the sidecar may write.
func TestDirector_TokenRoutesKeepTokenIDsAndDropKVTransferParams(t *testing.T) {
	metrics.Register()

	const clientKV = `,"kv_transfer_params":{"do_remote_prefill":true}`
	cases := []struct {
		name      string
		path      string
		parser    fwkrh.Parser
		body      string // without the closing brace, so a subtest can append clientKV
		keptField string
	}{
		{
			name:      "vLLM HTTP generate",
			path:      reqcommon.PathVLLMGenerate,
			parser:    vllmhttp.NewVllmHTTPParser(),
			body:      `{"model":"llama-3-8b","token_ids":[1,2,3],"prompt_token_ids":[1,2,3]`,
			keptField: reqcommon.FieldPromptTokenIDs,
		},
		{
			name:      "SGLang HTTP generate",
			path:      reqcommon.PathSGLangGenerate,
			parser:    sglanghttp.NewSGLangHTTPParser(),
			body:      `{"model":"llama-3-8b","input_ids":[10,20,30]`,
			keptField: reqcommon.FieldInputIDs,
		},
		{
			name:      "token-array completions",
			path:      reqcommon.PathCompletions,
			parser:    openai.NewOpenAIParser(),
			body:      `{"model":"llama-3-8b","prompt":[1,2,3],"prompt_token_ids":[1,2,3]`,
			keptField: reqcommon.FieldPromptTokenIDs,
		},
		{
			name:      "batch token-array completions",
			path:      reqcommon.PathCompletions,
			parser:    openai.NewOpenAIParser(),
			body:      `{"model":"llama-3-8b","prompt":[[1,2],[3,4]],"prompt_token_ids":[1,2]`,
			keptField: reqcommon.FieldPromptTokenIDs,
		},
	}

	for _, tc := range cases {
		for _, strict := range []bool{false, true} {
			mode := "strip mode"
			if strict {
				mode = "reject mode"
			}
			t.Run(tc.name+"/"+mode+" drops client kv_transfer_params and keeps token IDs", func(t *testing.T) {
				cfg := NewConfig().WithRejectClientPromptIDs(strict)
				reqCtx, body, err := handleSanitizeRequest(t, cfg, tc.parser, tc.path, []byte(tc.body+clientKV+"}"))

				require.NoError(t, err, "a token route drops the field and never returns 400")
				require.True(t, body.Mutated)
				out := forwardedBody(t, reqCtx)
				require.NotContains(t, out, reqcommon.FieldKVTransferParams)
				require.Contains(t, out, tc.keptField)
				requireNoSanitizeMetric(t)
			})
		}

		t.Run(tc.name+"/request without kv_transfer_params forwards the original bytes", func(t *testing.T) {
			raw := []byte(tc.body + "}")
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig(), tc.parser, tc.path, raw)

			require.NoError(t, err)
			require.False(t, body.Mutated)
			require.Equal(t, raw, reqCtx.Request.RawBody)
			requireNoSanitizeMetric(t)
		})

		t.Run(tc.name+"/trustCallerFields keeps kv_transfer_params", func(t *testing.T) {
			raw := []byte(tc.body + clientKV + "}")
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig().WithTrustCallerFields(true), tc.parser, tc.path, raw)

			require.NoError(t, err)
			require.False(t, body.Mutated)
			require.Equal(t, raw, reqCtx.Request.RawBody)
		})
	}
}

// A render request goes to the render endpoint and not to a model server, so
// the director does not sanitize it.
func TestDirector_RenderRequestIsNotSanitized(t *testing.T) {
	metrics.Register()

	raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"prompt_token_ids":[1,2,3],"kv_transfer_params":{"do_remote_prefill":true}}`)
	for _, strict := range []bool{false, true} {
		cfg := NewConfig().WithRejectClientPromptIDs(strict)
		reqCtx, body, err := handleSanitizeRequest(t, cfg, openai.NewOpenAIParser(), "/v1/chat/completions/render", raw)

		require.NoError(t, err)
		require.False(t, body.Mutated)
		require.Equal(t, raw, reqCtx.Request.RawBody)
		requireNoSanitizeMetric(t)
	}
}

// A JSON parser decodes an escaped key and keeps the last of two equal keys.
// The model server does the same, so the director must match the decoded key
// and must not forward either copy.
func TestDirector_StripsEscapedAndDuplicateKeys(t *testing.T) {
	metrics.Register()

	cases := map[string]string{
		"escaped key":   `"kv_transfer_\u0070arams":{"do_remote_prefill":true}`,
		"duplicate key": `"kv_transfer_params":{"do_remote_prefill":true},"kv_transfer_params":{"do_remote_decode":true}`,
	}
	for name, field := range cases {
		t.Run(name, func(t *testing.T) {
			raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],` + field + `}`)
			reqCtx, _, err := handleSanitizeRequest(t, NewConfig(), openai.NewOpenAIParser(), reqcommon.PathChatCompletions, raw)

			require.NoError(t, err)
			require.NotContains(t, string(reqCtx.Request.RawBody), "do_remote")
			require.NotContains(t, forwardedBody(t, reqCtx), reqcommon.FieldKVTransferParams)
		})
	}
}

// The Vertex AI parser keeps a text request as a proto that wraps the JSON
// body. The director cannot strip a field from it, so it rejects the request
// under both settings and does not forward the field.
func TestDirector_VertexAITextRequestWithForbiddenFieldIsRejected(t *testing.T) {
	metrics.Register()

	const service = "/google.cloud.aiplatform.v1beta1.PredictionService/"
	cases := []struct {
		name      string
		path      string
		wantRoute string
		json      string // without the closing brace, so a subtest can append a field
		wrap      func(json []byte) proto.Message
	}{
		{
			name:      "ChatCompletions",
			path:      service + "ChatCompletions",
			wantRoute: reqcommon.APITypeChatCompletions.String(),
			json:      `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]`,
			wrap: func(json []byte) proto.Message {
				return &aiplatformpb.ChatCompletionsRequest{HttpBody: &httpbody.HttpBody{Data: json}}
			},
		},
		{
			name:      "RawPredict",
			path:      service + "RawPredict",
			wantRoute: reqcommon.APITypeResponses.String(),
			json:      `{"model":"llama-3-8b","input":"hi"`,
			wrap: func(json []byte) proto.Message {
				return &aiplatformpb.RawPredictRequest{HttpBody: &httpbody.HttpBody{Data: json}}
			},
		},
	}
	frame := func(t *testing.T, msg proto.Message) []byte {
		t.Helper()
		framed, err := grpcframe.MarshalProto(msg)
		require.NoError(t, err)
		return framed
	}

	for _, tc := range cases {
		forbidden := map[string]string{
			"prompt_token_ids":           `,"prompt_token_ids":[1,2,3]`,
			"kv_transfer_params":         `,"kv_transfer_params":{"do_remote_prefill":true}`,
			"escaped kv_transfer_params": `,"kv_transfer_\u0070arams":{"do_remote_prefill":true}`,
			"nested kv_transfer_params":  `,"vllm_xargs":{"kv_transfer_params":{"do_remote_prefill":true}}`,
		}
		for fieldName, field := range forbidden {
			for _, strict := range []bool{false, true} {
				mode := "strip mode"
				if strict {
					mode = "reject mode"
				}
				t.Run(tc.name+"/"+fieldName+"/"+mode+" returns 400", func(t *testing.T) {
					raw := frame(t, tc.wrap([]byte(tc.json+field+"}")))
					cfg := NewConfig().WithRejectClientPromptIDs(strict)
					_, _, err := handleSanitizeRequest(t, cfg, vertexai.NewVertexAIParser(), tc.path, raw)

					var reqErr errcommon.Error
					require.ErrorAs(t, err, &reqErr)
					require.Equal(t, errcommon.BadRequest, reqErr.Code)
					wantMetric := `
# HELP llm_d_epp_prompt_ids_sanitized_total [ALPHA] Total number of external text-route requests where client-supplied prompt token, fingerprint, KV transfer, or EC transfer fields were stripped or rejected.
# TYPE llm_d_epp_prompt_ids_sanitized_total counter
llm_d_epp_prompt_ids_sanitized_total{action="rejected",component="epp",route="` + tc.wantRoute + `"} 1
`
					require.NoError(t, promtestutil.GatherAndCompare(ctrlmetrics.Registry, strings.NewReader(wantMetric), eppPromptIDsSanitizedMetric))
				})
			}
		}

		t.Run(tc.name+"/request without a forbidden field forwards the original bytes", func(t *testing.T) {
			raw := frame(t, tc.wrap([]byte(tc.json+"}")))
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig().WithRejectClientPromptIDs(true), vertexai.NewVertexAIParser(), tc.path, raw)

			require.NoError(t, err)
			require.False(t, body.Mutated)
			require.Equal(t, raw, reqCtx.Request.RawBody)
			requireNoSanitizeMetric(t)
		})

		t.Run(tc.name+"/trustCallerFields forwards the original bytes", func(t *testing.T) {
			raw := frame(t, tc.wrap([]byte(tc.json+`,"kv_transfer_params":{"do_remote_prefill":true}}`)))
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig().WithTrustCallerFields(true), vertexai.NewVertexAIParser(), tc.path, raw)

			require.NoError(t, err)
			require.False(t, body.Mutated)
			require.Equal(t, raw, reqCtx.Request.RawBody)
			requireNoSanitizeMetric(t)
		})
	}
}

func TestDirector_CleanExternalTextRequestLeavesRawBodyUnmutated(t *testing.T) {
	metrics.Register()
	metrics.Reset()

	dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
	dir.requestControlPlugins = *NewConfig().
		WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
		WithRejectClientPromptIDs(true)

	parser := openai.NewOpenAIParser()
	raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hello"}]}`)
	headers := map[string]string{":path": reqcommon.PathChatCompletions}
	parsed, err := parser.ParseRequest(ctx, raw, headers)
	require.NoError(t, err)
	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{ID: "req-clean", Headers: headers, RawBody: raw},
		Parser:  parser,
	}

	_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
	require.NoError(t, err)
	require.False(t, parsed.Body.Mutated)
	require.Equal(t, raw, reqCtx.Request.RawBody)
	require.NoError(t, promtestutil.GatherAndCompare(ctrlmetrics.Registry, strings.NewReader(""), eppPromptIDsSanitizedMetric))
}

// A vLLM worker also reads kv_transfer_params and ec_transfer_params from
// vllm_xargs and from sampling_params, so the director drops those copies with
// the top-level fields.
func TestDirector_DropsNestedTransferParams(t *testing.T) {
	metrics.Register()

	const clientKV = `"kv_transfer_params":{"do_remote_prefill":true},"ec_transfer_params":{"image":[]}`
	tests := []struct {
		name       string
		path       string
		parser     fwkrh.Parser
		body       string
		parent     string
		wantParent string
		keptField  string
	}{
		{
			name:       "vllm_generate_extra_args",
			path:       reqcommon.PathVLLMGenerate,
			parser:     vllmhttp.NewVllmHTTPParser(),
			body:       `{"model":"llama-3-8b","token_ids":[1,2,3],"sampling_params":{"max_tokens":4,"extra_args":{` + clientKV + `,"k":"v"}}}`,
			parent:     reqcommon.FieldSamplingParams,
			wantParent: `{"max_tokens":4,"extra_args":{"k":"v"}}`,
			keptField:  "token_ids",
		},
		{
			name:       "vllm_generate_vllm_xargs",
			path:       reqcommon.PathVLLMGenerate,
			parser:     vllmhttp.NewVllmHTTPParser(),
			body:       `{"model":"llama-3-8b","token_ids":[1,2,3],"sampling_params":{"max_tokens":4,"vllm_xargs":{` + clientKV + `,"k":"v"}}}`,
			parent:     reqcommon.FieldSamplingParams,
			wantParent: `{"max_tokens":4,"vllm_xargs":{"k":"v"}}`,
			keptField:  "token_ids",
		},
		{
			name:       "token_array_completions_vllm_xargs",
			path:       reqcommon.PathCompletions,
			parser:     openai.NewOpenAIParser(),
			body:       `{"model":"llama-3-8b","prompt":[1,2,3],"vllm_xargs":{` + clientKV + `,"k":"v"}}`,
			parent:     reqcommon.FieldVLLMXArgs,
			wantParent: `{"k":"v"}`,
			keptField:  reqcommon.FieldPrompt,
		},
		{
			name:       "chat_completions_vllm_xargs",
			path:       reqcommon.PathChatCompletions,
			parser:     openai.NewOpenAIParser(),
			body:       `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"vllm_xargs":{` + clientKV + `,"k":"v"}}`,
			parent:     reqcommon.FieldVLLMXArgs,
			wantParent: `{"k":"v"}`,
			keptField:  "messages",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig(), tc.parser, tc.path, []byte(tc.body))

			require.NoError(t, err)
			require.True(t, body.Mutated)
			out := forwardedBody(t, reqCtx)
			require.Contains(t, out, tc.keptField)
			parent, err := json.Marshal(out[tc.parent])
			require.NoError(t, err)
			require.JSONEq(t, tc.wantParent, string(parent))
		})

		t.Run(tc.name+"/trust_caller_fields", func(t *testing.T) {
			raw := []byte(tc.body)
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig().WithTrustCallerFields(true), tc.parser, tc.path, raw)

			require.NoError(t, err)
			require.False(t, body.Mutated)
			require.Equal(t, raw, reqCtx.Request.RawBody)
		})
	}
}

func TestDirector_RejectClientPromptIDsRejectsNestedKVTransferParams(t *testing.T) {
	metrics.Register()

	raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"vllm_xargs":{"kv_transfer_params":{"do_remote_prefill":true}}}`)
	_, _, err := handleSanitizeRequest(t, NewConfig().WithRejectClientPromptIDs(true), openai.NewOpenAIParser(), reqcommon.PathChatCompletions, raw)

	var reqErr errcommon.Error
	require.ErrorAs(t, err, &reqErr)
	require.Equal(t, errcommon.BadRequest, reqErr.Code)
	require.Contains(t, reqErr.Msg, "vllm_xargs.kv_transfer_params")
}

func TestDirector_TokenRoutesDropECTransferParams(t *testing.T) {
	metrics.Register()

	const clientEC = `,"ec_transfer_params":{"image":[]}`
	tests := []struct {
		name      string
		path      string
		parser    fwkrh.Parser
		body      string // without the closing brace, so the test can append clientEC
		keptField string
	}{
		{
			name:      "vllm_generate",
			path:      reqcommon.PathVLLMGenerate,
			parser:    vllmhttp.NewVllmHTTPParser(),
			body:      `{"model":"llama-3-8b","token_ids":[1,2,3]`,
			keptField: "token_ids",
		},
		{
			name:      "token_array_completions",
			path:      reqcommon.PathCompletions,
			parser:    openai.NewOpenAIParser(),
			body:      `{"model":"llama-3-8b","prompt":[1,2,3]`,
			keptField: reqcommon.FieldPrompt,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig(), tc.parser, tc.path, []byte(tc.body+clientEC+"}"))

			require.NoError(t, err)
			require.True(t, body.Mutated)
			out := forwardedBody(t, reqCtx)
			require.NotContains(t, out, reqcommon.FieldECTransferParams)
			require.Contains(t, out, tc.keptField)
			requireNoSanitizeMetric(t)
		})

		t.Run(tc.name+"/trust_caller_fields", func(t *testing.T) {
			raw := []byte(tc.body + clientEC + "}")
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig().WithTrustCallerFields(true), tc.parser, tc.path, raw)

			require.NoError(t, err)
			require.False(t, body.Mutated)
			require.Equal(t, raw, reqCtx.Request.RawBody)
		})
	}
}

func TestDirector_RejectClientPromptIDsRejectsECTransferParams(t *testing.T) {
	metrics.Register()

	tests := []struct {
		name      string
		field     string
		wantField string
	}{
		{name: "top_level", field: `"ec_transfer_params":{"image":[]}`, wantField: `"ec_transfer_params"`},
		{name: "vllm_xargs", field: `"vllm_xargs":{"ec_transfer_params":{"image":[]}}`, wantField: "vllm_xargs.ec_transfer_params"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],` + tc.field + `}`)
			_, _, err := handleSanitizeRequest(t, NewConfig().WithRejectClientPromptIDs(true), openai.NewOpenAIParser(), reqcommon.PathChatCompletions, raw)

			var reqErr errcommon.Error
			require.ErrorAs(t, err, &reqErr)
			require.Equal(t, errcommon.BadRequest, reqErr.Code)
			require.Contains(t, reqErr.Msg, tc.wantField)
		})
	}
}

func TestDirector_CompletionsDuplicateAndCaseVariantPromptCannotBypassSanitization(t *testing.T) {
	metrics.Register()

	cases := []struct {
		name string
		raw  string
	}{
		{
			name: "token array prompt overwritten by string prompt",
			raw:  `{"model":"llama-3-8b","prompt":[1,2,3],"prompt":"hello","prompt_token_ids":[9,9,9]}`,
		},
		{
			name: "token array prompt overwritten by null with prompt_embeds",
			raw:  `{"model":"llama-3-8b","prompt":[1,2,3],"prompt":null,"prompt_embeds":"AAAA","prompt_token_ids":[9,9,9]}`,
		},
		{
			name: "string prompt followed by uppercase PROMPT token array",
			raw:  `{"model":"llama-3-8b","prompt":"hello","PROMPT":[1,2,3],"prompt_token_ids":[9,9,9]}`,
		},
		{
			name: "prompt_embeds with uppercase PROMPT token array",
			raw:  `{"model":"llama-3-8b","PROMPT":[1,2,3],"prompt_embeds":"AAAA","prompt_token_ids":[9,9,9]}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name+"/strip", func(t *testing.T) {
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig(), openai.NewOpenAIParser(), reqcommon.PathCompletions, []byte(tc.raw))

			require.NoError(t, err)
			require.Empty(t, body.Completions.Prompt.TokenIDs)
			require.True(t, body.Mutated)
			out := forwardedBody(t, reqCtx)
			require.NotContains(t, out, reqcommon.FieldPromptTokenIDs)
		})

		t.Run(tc.name+"/reject", func(t *testing.T) {
			_, _, err := handleSanitizeRequest(t, NewConfig().WithRejectClientPromptIDs(true), openai.NewOpenAIParser(), reqcommon.PathCompletions, []byte(tc.raw))

			var reqErr errcommon.Error
			require.ErrorAs(t, err, &reqErr)
			require.Equal(t, errcommon.BadRequest, reqErr.Code)
			require.Contains(t, reqErr.Msg, reqcommon.FieldPromptTokenIDs)
		})
	}
}

func TestDirector_OtherPayloadMapRoutesDropTransferParams(t *testing.T) {
	metrics.Register()

	const clientTransfer = `,"kv_transfer_params":{"remote_prefiller":"10.0.0.1:8000"},"ec_transfer_params":{"image":[]},"vllm_xargs":{"kv_transfer_params":{"do_remote_prefill":true},"keep":1}`
	routes := []struct {
		name   string
		path   string
		parser fwkrh.Parser
		body   string // without closing brace
	}{
		{
			name:   "audio_speech",
			path:   reqcommon.PathAudioSpeech,
			parser: openai.NewOpenAIParser(),
			body:   `{"model":"tts-1","input":"hello","voice":"alloy"`,
		},
		{
			name:   "embeddings",
			path:   "/v1/embeddings",
			parser: openai.NewOpenAIParser(),
			body:   `{"model":"embed-1","input":"hello"`,
		},
		{
			name:   "audio_generate",
			path:   "/v1/audio/generate",
			parser: openai.NewOpenAIParser(),
			body:   `{"model":"audio-1","input":"rain"`,
		},
		{
			name:   "images_generations",
			path:   "/v1/images/generations",
			parser: openai.NewOpenAIParser(),
			body:   `{"model":"img-1","prompt":"sunset"`,
		},
		{
			name:   "conversations",
			path:   "/v1/conversations",
			parser: openai.NewOpenAIParser(),
			body:   `{"model":"llama-3-8b","items":[{"type":"message","role":"user","content":"hi"}]`,
		},
		{
			name:   "messages_count_tokens",
			path:   "/v1/messages/count_tokens",
			parser: anthropic.NewAnthropicParser(),
			body:   `{"model":"claude-3","messages":[{"role":"user","content":"hi"}]`,
		},
	}

	for _, tc := range routes {
		t.Run(tc.name+"/drops_transfer_params", func(t *testing.T) {
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig(), tc.parser, tc.path, []byte(tc.body+clientTransfer+"}"))

			require.NoError(t, err)
			require.True(t, body.Mutated)
			out := forwardedBody(t, reqCtx)
			require.NotContains(t, out, reqcommon.FieldKVTransferParams)
			require.NotContains(t, out, reqcommon.FieldECTransferParams)
			xargs, ok := out[reqcommon.FieldVLLMXArgs].(map[string]any)
			require.True(t, ok)
			require.NotContains(t, xargs, reqcommon.FieldKVTransferParams)
			require.Equal(t, float64(1), xargs["keep"])
			requireNoSanitizeMetric(t)
		})

		t.Run(tc.name+"/clean_request_unchanged", func(t *testing.T) {
			raw := []byte(tc.body + "}")
			reqCtx, body, err := handleSanitizeRequest(t, NewConfig(), tc.parser, tc.path, raw)

			require.NoError(t, err)
			require.False(t, body.Mutated)
			require.Equal(t, raw, reqCtx.Request.RawBody)
		})
	}
}

func TestDirector_UnsanitizedBodyHeaderOverridesTrustCallerFields(t *testing.T) {
	metrics.Register()

	runWithHeaders := func(t *testing.T, parser fwkrh.Parser, path string, extraHeaders map[string]string, raw []byte) (*handlers.RequestContext, *fwkrequest.InferenceRequestBody, error) {
		t.Helper()
		metrics.Reset()
		dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
		dir.requestControlPlugins = *NewConfig().
			WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
			WithTrustCallerFields(true)

		headers := map[string]string{":path": path}
		for k, v := range extraHeaders {
			headers[k] = v
		}
		parsed, err := parser.ParseRequest(ctx, raw, headers)
		require.NoError(t, err)
		reqCtx := &handlers.RequestContext{
			Request: &handlers.Request{ID: "req-passthrough", Headers: headers, RawBody: raw},
			Parser:  parser,
		}
		_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
		return reqCtx, parsed.Body, err
	}

	passthroughHeaders := map[string]string{reqcommon.UnsanitizedBodyHeaderKey: "true"}

	t.Run("text route strips client fields when marked unsanitized", func(t *testing.T) {
		raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"prompt_token_ids":[1,2,3],"kv_transfer_params":{"do_remote_prefill":true}}`)
		reqCtx, body, err := runWithHeaders(t, openai.NewOpenAIParser(), reqcommon.PathChatCompletions, passthroughHeaders, raw)

		require.NoError(t, err)
		require.True(t, body.Mutated)
		out := forwardedBody(t, reqCtx)
		require.NotContains(t, out, reqcommon.FieldPromptTokenIDs)
		require.NotContains(t, out, reqcommon.FieldKVTransferParams)
	})

	t.Run("false header value still trusts caller", func(t *testing.T) {
		raw := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"kv_transfer_params":{"do_remote_prefill":true}}`)
		reqCtx, body, err := runWithHeaders(t, openai.NewOpenAIParser(), reqcommon.PathChatCompletions, map[string]string{reqcommon.UnsanitizedBodyHeaderKey: "false"}, raw)

		require.NoError(t, err)
		require.False(t, body.Mutated)
		require.Equal(t, raw, reqCtx.Request.RawBody)
	})

	t.Run("non-render payload routes drop transfer params when marked unsanitized", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			path   string
			parser fwkrh.Parser
			raw    string
		}{
			{
				name:   "sglang_generate",
				path:   reqcommon.PathSGLangGenerate,
				parser: sglanghttp.NewSGLangHTTPParser(),
				raw:    `{"input_ids":[1,2,3],"kv_transfer_params":{"do_remote_prefill":true}}`,
			},
			{
				name:   "embeddings",
				path:   "/v1/embeddings",
				parser: openai.NewOpenAIParser(),
				raw:    `{"model":"embed-1","input":"hello","kv_transfer_params":{"remote_prefiller":"10.0.0.1:8000"}}`,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				reqCtx, body, err := runWithHeaders(t, tc.parser, tc.path, passthroughHeaders, []byte(tc.raw))

				require.NoError(t, err)
				require.True(t, body.Mutated)
				out := forwardedBody(t, reqCtx)
				require.NotContains(t, out, reqcommon.FieldKVTransferParams)
			})
		}
	})

	t.Run("vertex ai gRPC text route rejects forbidden field when marked unsanitized", func(t *testing.T) {
		payload := []byte(`{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"prompt_token_ids":[1,2,3]}`)
		msg := &aiplatformpb.ChatCompletionsRequest{
			Endpoint: "projects/p/locations/l/endpoints/e",
			HttpBody: &httpbody.HttpBody{ContentType: reqcommon.ContentTypeJSON, Data: payload},
		}
		framed, err := grpcframe.MarshalProto(msg)
		require.NoError(t, err)

		_, _, err = runWithHeaders(t, vertexai.NewVertexAIParser(), "/google.cloud.aiplatform.v1beta1.PredictionService/ChatCompletions", passthroughHeaders, framed)

		var reqErr errcommon.Error
		require.ErrorAs(t, err, &reqErr)
		require.Equal(t, errcommon.BadRequest, reqErr.Code)
		require.Contains(t, reqErr.Msg, reqcommon.FieldPromptTokenIDs)
	})
}
