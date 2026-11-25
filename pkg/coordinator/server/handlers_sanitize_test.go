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

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const promptIDsSanitizedMetric = "llm_d_coordinator_prompt_ids_sanitized_total"

func TestHandleInference_SanitizesExternalTextRoutes(t *testing.T) {
	cases := []struct {
		name      string
		path      string
		wantRoute string
		baseField string
	}{
		{name: "chat completions", path: reqcommon.PathChatCompletions, wantRoute: coordmetrics.RouteChatCompletions, baseField: `"messages":[{"role":"user","content":"hi"}]`},
		{name: "string completions", path: reqcommon.PathCompletions, wantRoute: coordmetrics.RouteCompletions, baseField: `"prompt":"hello"`},
		{name: "string-array completions", path: reqcommon.PathCompletions, wantRoute: coordmetrics.RouteCompletions, baseField: `"prompt":["hello","world"]`},
		{name: "messages", path: reqcommon.PathMessages, wantRoute: coordmetrics.RouteMessages, baseField: `"messages":[{"role":"user","content":"hi"}]`},
		{name: "responses", path: reqcommon.PathResponses, wantRoute: coordmetrics.RouteResponses, baseField: `"input":"hi"`},
	}

	for _, tc := range cases {
		t.Run(tc.name+"/strips all forbidden fields from Body and OriginalBody", func(t *testing.T) {
			reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			step := &capturingStep{}
			srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, stubGatewayURL))
			require.NoError(t, err)

			raw := `{"model":"m",` + tc.baseField +
				`,"prompt_token_ids":[1,2,3]` +
				`,"prompt_render_fingerprint":"secret-fp-1"` +
				`,"render_fingerprint":"secret-fp-2"` +
				`,"kv_transfer_params":{"prompt_token_ids":[4,5],"do_remote_prefill":true}` +
				`,"input_ids":[6,7]}`
			rec := httptest.NewRecorder()
			lines := handlerLogLines(t, func() {
				srv.handleInference(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(raw)))
			})

			require.Equal(t, http.StatusOK, rec.Code)
			require.NotNil(t, step.captured)

			var origMap map[string]any
			require.NoError(t, json.Unmarshal(step.captured.OriginalBody, &origMap))
			for _, field := range reqcommon.ExternalTextRouteSanitizedFields {
				require.NotContains(t, step.captured.Body, field, "Body must not retain %s", field)
				require.NotContains(t, origMap, field, "OriginalBody must not retain %s", field)
			}
			require.Equal(t, "m", step.captured.Body["model"])
			require.Equal(t, "m", origMap["model"])

			for _, line := range lines {
				require.NotContains(t, line, "secret-fp-1", "client field values must not reach logs")
				require.NotContains(t, line, "secret-fp-2", "client field values must not reach logs")
			}

			got := promtestutil.ToFloat64(mustCounter(t, reg, promptIDsSanitizedMetric, map[string]string{
				"component": metricsutil.SanitizeComponentCoordinator,
				"route":     tc.wantRoute,
				"action":    metricsutil.SanitizeActionStripped,
			}))
			require.InDelta(t, 1.0, got, 1e-9)
		})

		t.Run(tc.name+"/rejects with 400 when reject_client_prompt_ids is true", func(t *testing.T) {
			reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			step := &capturingStep{}
			srv, err := New(config.ServerConfig{RejectClientPromptIDs: true}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, stubGatewayURL))
			require.NoError(t, err)

			raw := `{"model":"m",` + tc.baseField + `,"prompt_token_ids":[1,2,3]}`
			rec := httptest.NewRecorder()
			srv.handleInference(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(raw)))

			require.Equal(t, http.StatusBadRequest, rec.Code)
			require.Nil(t, step.captured, "rejected request must not reach pipeline")
			require.Contains(t, rec.Body.String(), reqcommon.FieldPromptTokenIDs)

			got := promtestutil.ToFloat64(mustCounter(t, reg, promptIDsSanitizedMetric, map[string]string{
				"component": metricsutil.SanitizeComponentCoordinator,
				"route":     tc.wantRoute,
				"action":    metricsutil.SanitizeActionRejected,
			}))
			require.InDelta(t, 1.0, got, 1e-9)
		})
	}
}

func TestHandleInference_RejectClientPromptIDsAlsoRejectsKVTransferParams(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	step := &capturingStep{}
	srv, err := New(config.ServerConfig{RejectClientPromptIDs: true}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, stubGatewayURL))
	require.NoError(t, err)

	raw := `{"model":"m","messages":[{"role":"user","content":"hi"}],"kv_transfer_params":{"do_remote_prefill":true}}`
	rec := httptest.NewRecorder()
	srv.handleInference(rec, httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(raw)))

	require.Equal(t, http.StatusBadRequest, rec.Code)
	require.Nil(t, step.captured)
	got := promtestutil.ToFloat64(mustCounter(t, reg, promptIDsSanitizedMetric, map[string]string{
		"component": metricsutil.SanitizeComponentCoordinator,
		"route":     coordmetrics.RouteChatCompletions,
		"action":    metricsutil.SanitizeActionRejected,
	}))
	require.InDelta(t, 1.0, got, 1e-9)
}

// A JSON parser decodes an escaped key and keeps the last of two equal keys.
// The model server does the same, so the handler must match the decoded key and
// must not keep either copy in Body or OriginalBody.
func TestHandleInference_StripsEscapedAndDuplicateKeys(t *testing.T) {
	cases := map[string]string{
		"escaped key":   `"kv_transfer_\u0070arams":{"do_remote_prefill":true}`,
		"duplicate key": `"kv_transfer_params":{"do_remote_prefill":true},"kv_transfer_params":{"do_remote_decode":true}`,
	}
	for name, field := range cases {
		t.Run(name, func(t *testing.T) {
			metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			step := &capturingStep{}
			srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, stubGatewayURL))
			require.NoError(t, err)

			raw := `{"model":"m","messages":[{"role":"user","content":"hi"}],` + field + `}`
			rec := httptest.NewRecorder()
			srv.handleInference(rec, httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(raw)))

			require.Equal(t, http.StatusOK, rec.Code)
			require.NotNil(t, step.captured)
			require.NotContains(t, step.captured.Body, reqcommon.FieldKVTransferParams)
			require.NotContains(t, string(step.captured.OriginalBody), "do_remote")
		})
	}
}

func TestHandleInference_PreservesClientTokensOnDedicatedTokenRoutes(t *testing.T) {
	cases := []struct {
		name           string
		path           string
		body           string
		preservedField string
	}{
		{
			name:           "vLLM HTTP generate preserves token_ids and prompt_token_ids",
			path:           reqcommon.PathVLLMGenerate,
			body:           `{"model":"m","token_ids":[1,2,3],"prompt_token_ids":[1,2,3],"kv_transfer_params":{"do_remote_prefill":true}}`,
			preservedField: "prompt_token_ids",
		},
		{
			name:           "SGLang HTTP generate preserves input_ids",
			path:           reqcommon.PathSGLangGenerate,
			body:           `{"model":"m","input_ids":[10,20,30],"kv_transfer_params":{"do_remote_prefill":true}}`,
			preservedField: "input_ids",
		},
		{
			name:           "token-array completions preserves prompt token array",
			path:           reqcommon.PathCompletions,
			body:           `{"model":"m","prompt":[1,2,3],"kv_transfer_params":{"do_remote_prefill":true}}`,
			preservedField: "prompt",
		},
		{
			name:           "batch token-array completions preserves prompt token arrays",
			path:           reqcommon.PathCompletions,
			body:           `{"model":"m","prompt":[[1,2],[3,4]]}`,
			preservedField: "prompt",
		},
	}

	for _, tc := range cases {
		for _, strict := range []bool{false, true} {
			mode := "strip"
			if strict {
				mode = "reject"
			}
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
				step := &capturingStep{}
				srv, err := New(config.ServerConfig{RejectClientPromptIDs: strict}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, stubGatewayURL))
				require.NoError(t, err)

				rec := httptest.NewRecorder()
				srv.handleInference(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body)))

				require.Equal(t, http.StatusOK, rec.Code)
				require.NotNil(t, step.captured)
				require.Contains(t, step.captured.Body, tc.preservedField)
				require.NotContains(t, step.captured.Body, reqcommon.FieldKVTransferParams, "kv_transfer_params is still dropped on token routes")
				require.Zero(t, seriesCount(t, reg, promptIDsSanitizedMetric))
			})
		}
	}
}
