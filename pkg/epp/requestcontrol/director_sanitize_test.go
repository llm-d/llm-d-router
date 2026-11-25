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

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
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
				reqcommon.FieldInputIDs,
			} {
				require.NotContains(t, out, field, "repackaged body must omit client field %s", field)
			}
			require.Equal(t, map[string]any{"remote_kv_source": "10.0.0.1:8000"}, out[reqcommon.FieldKVTransferParams],
				"PreRequest plugin KV transfer metadata must survive client stripping")

			wantMetric := `
# HELP llm_d_epp_prompt_ids_sanitized_total [ALPHA] Total number of external text-route requests where client-supplied prompt token, fingerprint, or KV transfer fields were stripped or rejected.
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
# HELP llm_d_epp_prompt_ids_sanitized_total [ALPHA] Total number of external text-route requests where client-supplied prompt token, fingerprint, or KV transfer fields were stripped or rejected.
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

func TestDirector_PreservesDedicatedTokenRoutesAndRenderRequests(t *testing.T) {
	metrics.Register()

	cases := []struct {
		name           string
		path           string
		parser         fwkrh.Parser
		body           string
		preservedField string
	}{
		{
			name:           "vLLM HTTP generate",
			path:           reqcommon.PathVLLMGenerate,
			parser:         vllmhttp.NewVllmHTTPParser(),
			body:           `{"model":"llama-3-8b","token_ids":[1,2,3],"prompt_token_ids":[1,2,3]}`,
			preservedField: reqcommon.FieldPromptTokenIDs,
		},
		{
			name:           "SGLang HTTP generate",
			path:           reqcommon.PathSGLangGenerate,
			parser:         sglanghttp.NewSGLangHTTPParser(),
			body:           `{"model":"llama-3-8b","input_ids":[10,20,30]}`,
			preservedField: reqcommon.FieldInputIDs,
		},
		{
			name:           "token-array completions",
			path:           reqcommon.PathCompletions,
			parser:         openai.NewOpenAIParser(),
			body:           `{"model":"llama-3-8b","prompt":[1,2,3],"prompt_token_ids":[1,2,3]}`,
			preservedField: reqcommon.FieldPromptTokenIDs,
		},
		{
			name:           "batch token-array completions",
			path:           reqcommon.PathCompletions,
			parser:         openai.NewOpenAIParser(),
			body:           `{"model":"llama-3-8b","prompt":[[1,2],[3,4]],"prompt_token_ids":[1,2]}`,
			preservedField: reqcommon.FieldPromptTokenIDs,
		},
		{
			name:           "chat completions render",
			path:           "/v1/chat/completions/render",
			parser:         openai.NewOpenAIParser(),
			body:           `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}],"prompt_token_ids":[1,2,3]}`,
			preservedField: reqcommon.FieldPromptTokenIDs,
		},
	}

	for _, tc := range cases {
		for _, strict := range []bool{false, true} {
			mode := "strip"
			if strict {
				mode = "reject"
			}
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				metrics.Reset()
				dir, ctx := newSinglePodDirector(t, newSanitizeScheduleResult())
				dir.requestControlPlugins = *NewConfig().
					WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
					WithRejectClientPromptIDs(strict)

				headers := map[string]string{":path": tc.path}
				raw := []byte(tc.body)
				parsed, err := tc.parser.ParseRequest(ctx, raw, headers)
				require.NoError(t, err)
				reqCtx := &handlers.RequestContext{
					Request: &handlers.Request{ID: "req-tok", Headers: headers, RawBody: raw},
					Parser:  tc.parser,
				}

				_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
				require.NoError(t, err)
				require.False(t, parsed.Body.Mutated)
				var out map[string]any
				require.NoError(t, json.Unmarshal(reqCtx.Request.RawBody, &out))
				require.Contains(t, out, tc.preservedField)
				require.NoError(t, promtestutil.GatherAndCompare(ctrlmetrics.Registry, strings.NewReader(""), eppPromptIDsSanitizedMetric))
			})
		}
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
