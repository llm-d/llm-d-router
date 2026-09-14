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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func TestECPipelineTokenLimits(t *testing.T) {
	tests := []struct {
		name        string
		apiType     APIType
		path        string
		body        string
		tokenFields []string
	}{
		{
			name:        "chat",
			apiType:     APITypeChatCompletions,
			path:        ChatCompletionsPath,
			body:        `{"model":"m","messages":[{"role":"user","content":"hello"}],"max_tokens":80,"max_completion_tokens":90,"min_tokens":5}`,
			tokenFields: []string{"max_tokens", "max_completion_tokens", "min_tokens"},
		},
		{
			name:        "responses",
			apiType:     APITypeResponses,
			path:        ResponsesPath,
			body:        `{"model":"m","input":"hello","max_output_tokens":800}`,
			tokenFields: []string{"max_output_tokens"},
		},
		{
			name:        "responses without limit",
			apiType:     APITypeResponses,
			path:        ResponsesPath,
			body:        `{"model":"m","input":"hello"}`,
			tokenFields: []string{"max_output_tokens"},
		},
		{
			name:        "generate",
			apiType:     APITypeGenerate,
			path:        GeneratePath,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":{"max_tokens":800,"min_tokens":5,"temperature":0.7}}`,
			tokenFields: []string{"max_tokens", "min_tokens"},
		},
		{
			name:        "generate without limits",
			apiType:     APITypeGenerate,
			path:        GeneratePath,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":{"temperature":0.7}}`,
			tokenFields: []string{"max_tokens", "min_tokens"},
		},
		{
			name:        "generate without sampling params",
			apiType:     APITypeGenerate,
			path:        GeneratePath,
			body:        `{"model":"m","token_ids":[1,2]}`,
			tokenFields: []string{"max_tokens", "min_tokens"},
		},
	}

	for _, connector := range []string{ECExampleConnector, ECConnectorNIXL} {
		t.Run(connector, func(t *testing.T) {
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					prefillBodies := make(chan map[string]any, 1)
					prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						assert.Equal(t, tt.path, r.URL.Path)
						var body map[string]any
						assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
						prefillBodies <- body
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(`{"kv_transfer_params":{}}`))
					}))
					defer prefill.Close()

					decodeURL, err := url.Parse("http://decoder:8000")
					require.NoError(t, err)
					srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: KVConnectorNIXLV2, ECConnector: connector})
					srv.logger = log.Log
					srv.allowlistValidator = &AllowlistValidator{}
					var decodeBody map[string]any
					srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						assert.Equal(t, tt.path, r.URL.Path)
						assert.NoError(t, json.NewDecoder(r.Body).Decode(&decodeBody))
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(`{}`))
					})

					// Text-only inputs exercise the EC handoff without requiring multimodal API support.
					req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
					req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
					req.Header.Set(routing.EncoderEndpointsHeader, "encoder:8000")
					recorder := httptest.NewRecorder()
					srv.disaggregatedPrefillHandler(tt.apiType)(recorder, req)
					require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
					require.Len(t, prefillBodies, 1)
					prefillBody := <-prefillBodies
					require.NotNil(t, decodeBody)

					var original, wantPrefill map[string]any
					require.NoError(t, json.Unmarshal([]byte(tt.body), &original))
					require.NoError(t, json.Unmarshal([]byte(tt.body), &wantPrefill))
					limits := wantPrefill
					if tt.apiType == APITypeGenerate {
						limits, _ = wantPrefill["sampling_params"].(map[string]any)
						if limits == nil {
							limits = make(map[string]any)
							wantPrefill["sampling_params"] = limits
						}
					}
					for _, field := range tt.tokenFields {
						limits[field] = float64(1)
					}
					wantPrefill["stream"] = false
					wantPrefill["cache_hit_threshold"] = float64(0)
					delete(prefillBody, "kv_transfer_params")
					assert.Equal(t, wantPrefill, prefillBody)

					delete(decodeBody, "kv_transfer_params")
					delete(decodeBody, "cache_hit_threshold")
					assert.Equal(t, original, decodeBody)
				})
			}
		})
	}
}
