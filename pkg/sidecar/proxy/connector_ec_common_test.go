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

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func TestECPipelineTokenLimits(t *testing.T) {
	tests := []struct {
		name    string
		apiType reqcommon.APIType
		path    string
		body    string
		// Output cap fields the prefill leg must set to 1.
		tokenFields []string
	}{
		{
			name:        "chat",
			apiType:     reqcommon.APITypeChatCompletions,
			path:        reqcommon.PathChatCompletions,
			body:        `{"model":"m","messages":[{"role":"user","content":"hello"}],"max_tokens":80,"max_completion_tokens":90,"min_tokens":5}`,
			tokenFields: []string{reqcommon.FieldMaxTokens, reqcommon.FieldMaxCompletionTokens},
		},
		{
			name:        "responses",
			apiType:     reqcommon.APITypeResponses,
			path:        reqcommon.PathResponses,
			body:        `{"model":"m","input":"hello","max_output_tokens":800}`,
			tokenFields: []string{reqcommon.FieldMaxOutputTokens},
		},
		{
			name:        "responses without limit",
			apiType:     reqcommon.APITypeResponses,
			path:        reqcommon.PathResponses,
			body:        `{"model":"m","input":"hello"}`,
			tokenFields: []string{reqcommon.FieldMaxOutputTokens},
		},
		{
			name:        "generate",
			apiType:     reqcommon.APITypeGenerate,
			path:        reqcommon.PathGenerate,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":{"max_tokens":800,"min_tokens":5,"temperature":0.7}}`,
			tokenFields: []string{reqcommon.FieldMaxTokens},
		},
		{
			name:        "generate without limits",
			apiType:     reqcommon.APITypeGenerate,
			path:        reqcommon.PathGenerate,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":{"temperature":0.7}}`,
			tokenFields: []string{reqcommon.FieldMaxTokens},
		},
		{
			name:        "generate without sampling params",
			apiType:     reqcommon.APITypeGenerate,
			path:        reqcommon.PathGenerate,
			body:        `{"model":"m","token_ids":[1,2]}`,
			tokenFields: []string{reqcommon.FieldMaxTokens},
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
					if tt.apiType == reqcommon.APITypeGenerate {
						limits, _ = wantPrefill[reqcommon.FieldSamplingParams].(map[string]any)
						if limits == nil {
							limits = make(map[string]any)
							wantPrefill[reqcommon.FieldSamplingParams] = limits
						}
					}
					for _, field := range tt.tokenFields {
						limits[field] = float64(1)
					}
					// The prefill leg drops min_tokens; see reqcommon.CapSingleToken.
					delete(limits, reqcommon.FieldMinTokens)
					wantPrefill[reqcommon.FieldStream] = false
					wantPrefill[reqcommon.FieldCacheHitThreshold] = float64(0)
					delete(prefillBody, reqcommon.FieldKVTransferParams)
					assert.Equal(t, wantPrefill, prefillBody)

					delete(decodeBody, reqcommon.FieldKVTransferParams)
					delete(decodeBody, reqcommon.FieldCacheHitThreshold)
					assert.Equal(t, original, decodeBody)
				})
			}
		})
	}
}

func TestBuildEncoderRequest(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "text",
						"text": "What's in this image?",
					},
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		"max_tokens": 100,
		"stream":     true,
	}

	mmItem := map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "https://example.com/image.jpg",
		},
	}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem)

	// Verify encoder request modifications
	assert.Equal(t, 1, encoderRequest["max_tokens"])
	assert.Equal(t, false, encoderRequest["stream"])
	_, hasStreamOptions := encoderRequest["stream_options"]
	assert.False(t, hasStreamOptions)

	// Verify messages contain only the MM item
	messages, ok := encoderRequest["messages"].([]map[string]any)
	assert.True(t, ok)
	assert.Equal(t, 1, len(messages))

	content, ok := messages[0]["content"].([]map[string]any)
	assert.True(t, ok)
	assert.Equal(t, 1, len(content))
	assert.Equal(t, "image_url", content[0]["type"])
}

// TestBuildEncoderRequest_MaxCompletionTokens is a regression test: a shallow
// copy previously left the client's max_completion_tokens value untouched
// alongside the newly-capped max_tokens=1, so a reasoning-model client's
// large max_completion_tokens would survive uncapped into the encoder request.
func TestBuildEncoderRequest_MaxCompletionTokens(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		"max_tokens":            50,
		"max_completion_tokens": 100,
	}

	mmItem := map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "https://example.com/image.jpg",
		},
	}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem)

	assert.Equal(t, 1, encoderRequest["max_tokens"])
	assert.Equal(t, 1, encoderRequest["max_completion_tokens"])
}

// TestBuildEncoderRequest_MinTokens is a regression test for stripping a
// client-supplied min_tokens from the encoder leg; reqcommon.CapSingleToken
// documents why.
func TestBuildEncoderRequest_MinTokens(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		"max_tokens": 50,
		"min_tokens": 5,
	}

	mmItem := map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "https://example.com/image.jpg",
		},
	}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem)

	assert.Equal(t, 1, encoderRequest["max_tokens"])
	assert.NotContains(t, encoderRequest, "min_tokens")
}
