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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// TestHandleEC_Multimedia asserts that video_url, audio_url, and input_audio
// items flow through both EC connectors the same way image_url items do.
// mmTypes in connector_ec_common.go treats video_url / audio_url uniformly
// with image_url (URL-based, dedup-eligible), while input_audio is inline and
// never deduplicates. This table exercises those paths against handleECNIXL
// (threads encoder ec_transfer_params into the prefill body) and
// handleECSharedStorage (primer only — encoder responses are discarded).
//
// Inline audio never deduplicates (see fanoutEncoderPrimerDeduplication note),
// so two input_audio blocks always produce two encoder calls.
func TestHandleEC_Multimedia(t *testing.T) {
	tests := []struct {
		name         string
		handler      func(*Server, http.ResponseWriter, *http.Request, string, []string)
		items        []map[string]any
		wantECParams bool
		wantECLen    int
		wantEncCalls int32
	}{
		{
			name:    "nixl video",
			handler: (*Server).handleECNIXL,
			items: []map[string]any{
				videoURLItem("https://example.com/v1.mp4"),
				videoURLItem("https://example.com/v2.mp4"),
			},
			wantECParams: true,
			wantECLen:    2,
			wantEncCalls: 2,
		},
		{
			name:    "nixl audio_url",
			handler: (*Server).handleECNIXL,
			items: []map[string]any{
				audioURLItem("https://example.com/a1.mp3"),
				audioURLItem("https://example.com/a2.mp3"),
			},
			wantECParams: true,
			wantECLen:    2,
			wantEncCalls: 2,
		},
		{
			name:    "nixl input_audio",
			handler: (*Server).handleECNIXL,
			items: []map[string]any{
				inlineAudioItem("aaa"),
				inlineAudioItem("bbb"),
			},
			wantECParams: true,
			wantECLen:    2,
			wantEncCalls: 2,
		},
		{
			name:    "shared_storage video",
			handler: (*Server).handleECSharedStorage,
			items: []map[string]any{
				videoURLItem("https://example.com/v1.mp4"),
				videoURLItem("https://example.com/v2.mp4"),
			},
			wantECParams: false,
			wantEncCalls: 2,
		},
		{
			name:    "shared_storage audio_url",
			handler: (*Server).handleECSharedStorage,
			items: []map[string]any{
				audioURLItem("https://example.com/a1.mp3"),
				audioURLItem("https://example.com/a2.mp3"),
			},
			wantECParams: false,
			wantEncCalls: 2,
		},
		{
			name:    "shared_storage input_audio",
			handler: (*Server).handleECSharedStorage,
			items: []map[string]any{
				inlineAudioItem("aaa"),
				inlineAudioItem("bbb"),
			},
			wantECParams: false,
			wantEncCalls: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				seq          atomic.Int32
				encoderCalls atomic.Int32
			)
			encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				encoderCalls.Add(1)
				i := seq.Add(1) - 1
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Always include ec_transfer_params. handleECSharedStorage discards
				// encoder responses, so returning it here is a harmless superset that
				// keeps the mock uniform across rows.
				_, _ = fmt.Fprintf(w, `{
					"choices": [{"message": {"content": ""}}],
					"ec_transfer_params": {"hash-%d": {"peer_host": "10.0.0.%d", "peer_port": 5500}}
				}`, i, i)
			}))
			defer encoderBackend.Close()

			encoderURL, err := url.Parse(encoderBackend.URL)
			assert.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
			srv.logger = log.Log

			var capturedBody []byte
			srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ string, _ APIType) {
				buf, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				capturedBody = buf
			}

			reqBody, _ := json.Marshal(userMessageRequest(tt.items...))
			httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
			rw := httptest.NewRecorder()

			tt.handler(srv, rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

			assert.Equal(t, tt.wantEncCalls, encoderCalls.Load(), "unexpected encoder call count")
			if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
				return
			}
			var parsed map[string]any
			assert.NoError(t, json.Unmarshal(capturedBody, &parsed))

			ec, hasEC := parsed[requestFieldECTransferParams].(map[string]any)
			if tt.wantECParams {
				assert.True(t, hasEC, "prefill body should carry ec_transfer_params")
				assert.Len(t, ec, tt.wantECLen, "one entry per distinct multimodal item")
				for k, v := range ec {
					entry, ok := v.(map[string]any)
					assert.Truef(t, ok, "ec[%q] should be an object", k)
					assert.Containsf(t, entry, "peer_host", "ec[%q] should carry transfer metadata", k)
				}
			} else {
				_, present := parsed[requestFieldECTransferParams]
				assert.False(t, present, "shared_storage primer must not add ec_transfer_params to the prefill body")
			}
		})
	}
}
