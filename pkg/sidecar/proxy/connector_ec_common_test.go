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
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync/atomic"
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
		// Output cap fields the prefill request must set to 1.
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
			apiType:     reqcommon.APITypeVLLMGenerate,
			path:        reqcommon.PathVLLMGenerate,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":{"max_tokens":800,"min_tokens":5,"temperature":0.7}}`,
			tokenFields: []string{reqcommon.FieldMaxTokens},
		},
		{
			name:        "generate without limits",
			apiType:     reqcommon.APITypeVLLMGenerate,
			path:        reqcommon.PathVLLMGenerate,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":{"temperature":0.7}}`,
			tokenFields: []string{reqcommon.FieldMaxTokens},
		},
		{
			name:        "generate without sampling params",
			apiType:     reqcommon.APITypeVLLMGenerate,
			path:        reqcommon.PathVLLMGenerate,
			body:        `{"model":"m","token_ids":[1,2]}`,
			tokenFields: []string{reqcommon.FieldMaxTokens},
		},
		{
			name:        "generate with null sampling params",
			apiType:     reqcommon.APITypeVLLMGenerate,
			path:        reqcommon.PathVLLMGenerate,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":null}`,
			tokenFields: []string{reqcommon.FieldMaxTokens},
		},
		{
			name:        "generate with non-object sampling params",
			apiType:     reqcommon.APITypeVLLMGenerate,
			path:        reqcommon.PathVLLMGenerate,
			body:        `{"model":"m","token_ids":[1,2],"sampling_params":"not-an-object"}`,
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

					var wantPrefill, wantDecode map[string]any
					require.NoError(t, json.Unmarshal([]byte(tt.body), &wantPrefill))
					require.NoError(t, json.Unmarshal([]byte(tt.body), &wantDecode))
					limits := wantPrefill
					if tt.apiType == reqcommon.APITypeVLLMGenerate {
						limits, _ = wantPrefill[reqcommon.FieldSamplingParams].(map[string]any)
						if limits == nil {
							limits = make(map[string]any)
							wantPrefill[reqcommon.FieldSamplingParams] = limits
						}
					}
					for _, field := range tt.tokenFields {
						limits[field] = float64(1)
					}
					// The prefill request drops min_tokens; see reqcommon.CapSingleToken.
					delete(limits, reqcommon.FieldMinTokens)
					wantPrefill[reqcommon.FieldStream] = false
					wantPrefill[reqcommon.FieldCacheHitThreshold] = float64(0)
					delete(prefillBody, reqcommon.FieldKVTransferParams)
					assert.Equal(t, wantPrefill, prefillBody)

					delete(decodeBody, reqcommon.FieldKVTransferParams)
					delete(decodeBody, reqcommon.FieldCacheHitThreshold)
					assert.Equal(t, wantDecode, decodeBody)
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

	encoderRequest := buildEncoderRequest(originalRequest, mmItem, reqcommon.APITypeChatCompletions)

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

	encoderRequest := buildEncoderRequest(originalRequest, mmItem, reqcommon.APITypeChatCompletions)

	assert.Equal(t, 1, encoderRequest["max_tokens"])
	assert.Equal(t, 1, encoderRequest["max_completion_tokens"])
}

// TestBuildEncoderRequest_ResponsesInputImage locks in that a Responses
// input_image item is forwarded to the encoder unmodified, under a native
// Responses-shaped request (input, not messages), rather than reshaped into
// chat completions' image_url nesting: vLLM's chat-completions engine
// ignores a detail hint nested there, so only a native Responses request
// carries it through to the encoder.
func TestBuildEncoderRequest_ResponsesInputImage(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"input": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":      "input_image",
						"image_url": "https://example.com/image.jpg",
						"detail":    "high",
					},
				},
			},
		},
	}

	mmItem := map[string]any{
		"type":      "input_image",
		"image_url": "https://example.com/image.jpg",
		"detail":    "high",
	}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem, reqcommon.APITypeResponses)

	input, ok := encoderRequest["input"].([]map[string]any)
	require.True(t, ok)
	require.Len(t, input, 1)

	content, ok := input[0]["content"].([]map[string]any)
	require.True(t, ok)
	require.Len(t, content, 1)

	assert.Equal(t, "input_image", content[0]["type"])
	assert.Equal(t, "https://example.com/image.jpg", content[0]["image_url"])
	assert.Equal(t, "high", content[0]["detail"])
}

// TestBuildEncoderRequest_OnlyModelAndInput locks in that buildEncoderRequest
// builds the encoder request from scratch rather than copying the client's
// request, even when the encoder is addressed with the client's own API
// (Responses here): copying the client's own input or max_output_tokens
// would leak every other multimodal item or an uncapped output limit, and
// stateful fields (previous_response_id, conversation, store, background)
// and tools/tool_choice/instructions have no place on a per-item encoder
// request. store is the one stateful field that does appear, forced to
// false regardless of the client's own value, so the priming request
// leaves no stored response object on the encoder pod.
func TestBuildEncoderRequest_OnlyModelAndInput(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"input": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_image", "image_url": "https://example.com/img1.jpg"},
					map[string]any{"type": "input_image", "image_url": "https://example.com/img2.jpg"},
				},
			},
		},
		"previous_response_id": "resp-123",
		"conversation":         "conv-123",
		"store":                true,
		"background":           true,
		"max_output_tokens":    500,
		"instructions":         "be nice",
		"tools":                []any{map[string]any{"type": "function", "name": "f", "parameters": map[string]any{}}},
		"tool_choice":          map[string]any{"type": "function", "name": "f"},
	}

	mmItem := map[string]any{"type": "input_image", "image_url": "https://example.com/img1.jpg"}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem, reqcommon.APITypeResponses)

	assert.Equal(t, "test-model", encoderRequest["model"])
	assert.Equal(t, false, encoderRequest["store"])
	assert.ElementsMatch(t, []string{"model", "input", "store", "max_output_tokens", "stream"}, slices.Collect(maps.Keys(encoderRequest)))
}

// TestBuildEncoderRequest_NoModel locks in that an absent client model stays
// absent on the encoder request rather than becoming an explicit JSON null,
// which vLLM's request validation would reject differently than a missing
// field.
func TestBuildEncoderRequest_NoModel(t *testing.T) {
	originalRequest := map[string]any{
		"input": []any{
			map[string]any{
				"role":    "user",
				"content": []any{map[string]any{"type": "input_image", "image_url": "https://example.com/img.jpg"}},
			},
		},
	}

	mmItem := map[string]any{"type": "input_image", "image_url": "https://example.com/img.jpg"}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem, reqcommon.APITypeResponses)

	_, hasModel := encoderRequest["model"]
	assert.False(t, hasModel)
}

// TestBuildEncoderRequest_MinTokens is a regression test for stripping a
// client-supplied min_tokens from the encoder request; reqcommon.CapSingleToken
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

	encoderRequest := buildEncoderRequest(originalRequest, mmItem, reqcommon.APITypeChatCompletions)

	assert.Equal(t, 1, encoderRequest["max_tokens"])
	assert.NotContains(t, encoderRequest, "min_tokens")
}

// TestECPipelineResponsesImage is an end-to-end test asserting that a
// /v1/responses request carrying an input_image part, routed through the
// EC connector, reaches the encoder.
func TestECPipelineResponsesImage(t *testing.T) {
	for _, connector := range []string{ECExampleConnector, ECConnectorNIXL} {
		t.Run(connector, func(t *testing.T) {
			var encoderCalls atomic.Int32
			encoder := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				encoderCalls.Add(1)
				var body map[string]any
				assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))

				// The encoder is addressed with the client's own API, so a
				// Responses request's input_image part arrives unmodified
				// under input, not reshaped into chat completions' messages.
				assert.Equal(t, reqcommon.PathResponses, r.URL.Path)
				input, ok := body["input"].([]any)
				require.True(t, ok, "encoder request must carry input")
				require.Len(t, input, 1)
				msg, ok := input[0].(map[string]any)
				require.True(t, ok)
				content, ok := msg["content"].([]any)
				require.True(t, ok)
				require.Len(t, content, 1)
				part, ok := content[0].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, "input_image", part["type"])
				assert.Equal(t, "https://example.com/image.jpg", part["image_url"])

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}],"ec_transfer_params":{"hash-0":{"peer_host":"10.0.0.1"}}}`))
			}))
			defer encoder.Close()

			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"kv_transfer_params":{}}`))
			}))
			defer prefill.Close()

			decodeURL, err := url.Parse("http://decoder:8000")
			require.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: KVConnectorNIXLV2, ECConnector: connector})
			srv.logger = log.Log
			srv.allowlistValidator = &AllowlistValidator{}
			var decodeCalls atomic.Int32
			srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				decodeCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{}`))
			})

			body := `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"what is this?"},{"type":"input_image","image_url":"https://example.com/image.jpg"}]}]}`
			req := httptest.NewRequest(http.MethodPost, reqcommon.PathResponses, strings.NewReader(body))
			req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
			req.Header.Set(routing.EncoderEndpointsHeader, strings.TrimPrefix(encoder.URL, "http://"))
			recorder := httptest.NewRecorder()
			srv.disaggregatedPrefillHandler(reqcommon.APITypeResponses)(recorder, req)

			require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
			assert.Equal(t, int32(1), encoderCalls.Load(), "the encoder must be called for a Responses request carrying an image")
			assert.Equal(t, int32(1), decodeCalls.Load(), "the pipeline must still reach the decoder after the encoder/prefill stages")
		})
	}
}
