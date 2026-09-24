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
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// TestRequestInput pins requestInput's own contract: extractMMItems returns an
// empty slice on the error path too, so a caller-level test cannot tell a
// malformed input from an absent one.
func TestRequestInput(t *testing.T) {
	tests := []struct {
		name      string
		input     any
		absent    bool
		wantItems int
		wantErr   bool
	}{
		{
			name:   "absent input",
			absent: true,
		},
		{
			name:      "input array",
			input:     json.RawMessage(`[{"role":"user"},{"role":"user"}]`),
			wantItems: 2,
		},
		{
			name:  "bare string is a single text turn",
			input: json.RawMessage(`"hello"`),
		},
		{
			name:  "explicit null is treated as absent",
			input: json.RawMessage(`null`),
		},
		{
			name:    "object",
			input:   json.RawMessage(`{"role":"user"}`),
			wantErr: true,
		},
		{
			name:    "number",
			input:   json.RawMessage(`42`),
			wantErr: true,
		},
		{
			name:    "bool",
			input:   json.RawMessage(`true`),
			wantErr: true,
		},
		// requestMessages accepts an already-decoded slice; requestInput
		// refuses one, because nothing writes a decoded slice under input.
		{
			name:    "decoded slice of raw messages",
			input:   []json.RawMessage{json.RawMessage(`{"role":"user"}`)},
			wantErr: true,
		},
		{
			name:    "fully decoded array",
			input:   []any{map[string]any{"role": "user"}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := map[string]any{}
			if !tt.absent {
				request[reqcommon.FieldInput] = tt.input
			}

			items, err := requestInput(request)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, items)
				return
			}
			assert.NoError(t, err)
			assert.Len(t, items, tt.wantItems)
		})
	}
}

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
			body:        `{"model":"m","input":"hello","max_output_tokens":800,"store":true}`,
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
					if tt.apiType == reqcommon.APITypeResponses {
						// The synthetic prefill leg pins store so it leaves no
						// stored response object behind. The responses case sends
						// store: true, so the wantDecode compare below pins that
						// the decode leg forwards the client's own value.
						wantPrefill[reqcommon.FieldStore] = false
					}
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

			var prefillECParams any
			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				prefillECParams = body[requestFieldECTransferParams]
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

			// Merging the encoder's params into the prefill body is what the
			// ec-nixl connector exists to do: without this the pipeline can
			// reach every stage in order and still prime nothing the prefiller
			// can look up. ec-example primes by status alone and merges nothing.
			if connector == ECConnectorNIXL {
				assert.Equal(t,
					map[string]any{"hash-0": map[string]any{"peer_host": "10.0.0.1"}},
					prefillECParams,
					"the encoder's ec_transfer_params must reach the prefiller")
			} else {
				assert.Nil(t, prefillECParams, "ec-example must not synthesize ec_transfer_params")
			}
		})
	}
}

// TestHandleEC_EncoderErrorStatus pins what a failed fanout answers the client
// and that it dispatches nothing. The encoder's status is not relayed: the
// client addressed the gateway, and which of the encoders behind it refused the
// priming leg is not a distinction the client can act on.
func TestHandleEC_EncoderErrorStatus(t *testing.T) {
	handlers := map[string]func(*Server, http.ResponseWriter, *http.Request, string, []string, reqcommon.APIType){
		"ec-nixl":           (*Server).handleECNIXL,
		"ec-shared-storage": (*Server).handleECSharedStorage,
	}
	tests := []struct {
		name        string
		encoderCode int
		wantClient  int
	}{
		{"encoder rejects the body", http.StatusBadRequest, http.StatusBadGateway},
		{"encoder finds the body unprocessable", http.StatusUnprocessableEntity, http.StatusBadGateway},
		{"encoder does not serve the route", http.StatusNotFound, http.StatusBadGateway},
		{"encoder is at capacity", http.StatusTooManyRequests, http.StatusBadGateway},
		{"encoder is unavailable", http.StatusServiceUnavailable, http.StatusBadGateway},
		{"encoder fails internally", http.StatusInternalServerError, http.StatusBadGateway},
	}

	for name, handle := range handlers {
		for _, tt := range tests {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				encoder := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tt.encoderCode)
					_, _ = w.Write([]byte(`{"error":"nope"}`))
				}))
				defer encoder.Close()

				encoderURL, err := url.Parse(encoder.URL)
				require.NoError(t, err)
				srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
				srv.logger = log.Log

				var dispatched bool
				srv.handlePDConnector = func(http.ResponseWriter, *http.Request, string, string, reqcommon.APIType) {
					dispatched = true
				}

				body, err := json.Marshal(userMessageRequest(imageURLItem("https://example.com/img.jpg")))
				require.NoError(t, err)
				req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(body))
				rw := httptest.NewRecorder()

				handle(srv, rw, req, "fake-prefiller:8000", []string{encoderURL.Host}, reqcommon.APITypeChatCompletions)

				assert.False(t, dispatched, "a failed fanout must not reach the P/D connector")
				assert.Equal(t, tt.wantClient, rw.Code, "body=%s", rw.Body.String())
			})
		}
	}
}
