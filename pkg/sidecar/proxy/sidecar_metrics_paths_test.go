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
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// sharedStorageDecoder answers the decode-first attempt (cache_hit_threshold
// above 0) and the post-prefill decode (cache_hit_threshold 0) separately. A
// nil handler marks a decode that must not be dispatched.
func sharedStorageDecoder(t *testing.T, attempt, final http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&body)) {
			return
		}
		h := final
		if threshold, _ := body[reqcommon.FieldCacheHitThreshold].(float64); threshold > 0 {
			h = attempt
		}
		if h == nil {
			t.Error("unexpected decode dispatch")
			return
		}
		h.ServeHTTP(w, r)
	})
}

// sseHandler streams one chat event with the given finish reason.
func sseHandler(finishReason string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":%q}]}\n\ndata: [DONE]\n\n", finishReason)
	})
}

// abortingHandler sends a 200 and then breaks the stream the way the reverse
// proxy does when the client or upstream goes away mid-response.
var abortingHandler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	panic(http.ErrAbortHandler)
})

func TestHandleSharedStorageMetrics(t *testing.T) {
	stop := statusHandler(http.StatusOK, `{"choices":[{"finish_reason":"stop"}]}`)
	belowThreshold := statusHandler(http.StatusOK,
		fmt.Sprintf(`{"choices":[{"finish_reason":%q}]}`, finishReasonCacheThreshold))

	tests := []struct {
		name          string
		threshold     bool
		stream        bool
		prefillStatus int
		attempt       http.Handler
		final         http.Handler
		wantCode      int
		wantBody      string
		wantPanic     bool
		wantPrefill   bool
		wantDelta     stageMetrics
	}{
		{
			name:          "prefill then decode success records durations only",
			prefillStatus: http.StatusOK,
			final:         stop,
			wantCode:      http.StatusOK,
			wantPrefill:   true,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1},
		},
		{
			name:          "prefill 500 records prefill error and skips decode",
			prefillStatus: http.StatusInternalServerError,
			wantCode:      http.StatusInternalServerError,
			wantPrefill:   true,
			wantDelta:     stageMetrics{prefillCount: 1, prefillErrors: 1},
		},
		{
			name:          "decode 500 after prefill records decode error",
			prefillStatus: http.StatusOK,
			final:         statusHandler(http.StatusInternalServerError, `{"error":"boom"}`),
			wantCode:      http.StatusInternalServerError,
			wantPrefill:   true,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
		{
			name:          "decode abort after prefill records decode error and propagates",
			prefillStatus: http.StatusOK,
			final:         abortingHandler,
			wantPanic:     true,
			wantPrefill:   true,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
		{
			name:          "decode-first success records one decode sample and no prefill",
			threshold:     true,
			prefillStatus: http.StatusOK,
			attempt:       stop,
			wantCode:      http.StatusOK,
			wantDelta:     stageMetrics{decodeCount: 1},
		},
		{
			name:          "decode-first 503 records decode error",
			threshold:     true,
			prefillStatus: http.StatusOK,
			attempt:       statusHandler(http.StatusServiceUnavailable, `{"error":"busy"}`),
			wantCode:      http.StatusServiceUnavailable,
			wantDelta:     stageMetrics{decodeCount: 1, decodeErrors: 1},
		},
		{
			name:          "decode-first below threshold falls back without a decode error",
			threshold:     true,
			prefillStatus: http.StatusOK,
			attempt:       belowThreshold,
			final:         stop,
			wantCode:      http.StatusOK,
			wantPrefill:   true,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1},
		},
		{
			name:          "decode-first abort records decode error without prefill and propagates",
			threshold:     true,
			prefillStatus: http.StatusOK,
			attempt:       abortingHandler,
			wantPanic:     true,
			wantDelta:     stageMetrics{decodeCount: 1, decodeErrors: 1},
		},
		// A streamed attempt is only inspected on the streaming path while the
		// wrapped writer keeps http.Flusher; the buffered path rejects SSE with a 500.
		{
			name:          "streamed decode-first success records one decode sample and no prefill",
			threshold:     true,
			stream:        true,
			prefillStatus: http.StatusOK,
			attempt:       sseHandler("stop"),
			wantCode:      http.StatusOK,
			wantBody:      "data: [DONE]",
			wantDelta:     stageMetrics{decodeCount: 1},
		},
		{
			name:          "streamed decode-first below threshold falls back without a decode error",
			threshold:     true,
			stream:        true,
			prefillStatus: http.StatusOK,
			attempt:       sseHandler(finishReasonCacheThreshold),
			final:         stop,
			wantCode:      http.StatusOK,
			wantPrefill:   true,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var prefilled atomic.Bool
			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				prefilled.Store(true)
				statusHandler(tt.prefillStatus, `{}`).ServeHTTP(w, r)
			}))
			defer prefill.Close()
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "0", DecoderURL: prefillURL, KVConnector: constants.KVConnectorSharedStorage})
			s.logger = log.Log
			s.decoderProxy = sharedStorageDecoder(t, tt.attempt, tt.final)

			body := textChatBody()
			if tt.threshold {
				body[reqcommon.FieldCacheHitThreshold] = 0.8
			}
			if tt.stream {
				body[reqcommon.FieldStream] = true
			}

			before := snapshotStageMetrics(t)
			rw := httptest.NewRecorder()
			var recovered any
			func() {
				defer func() { recovered = recover() }()
				s.handleSharedStorage(rw, chatRequest(t, body), prefillURL.Host, reqcommon.APITypeChatCompletions)
			}()

			if tt.wantPanic {
				assert.Equal(t, http.ErrAbortHandler, recovered, "abort panic must propagate")
			} else {
				assert.Nil(t, recovered)
				assert.Equal(t, tt.wantCode, rw.Code)
			}
			if tt.wantBody != "" {
				assert.Contains(t, rw.Body.String(), tt.wantBody)
			}
			assert.Equal(t, tt.wantPrefill, prefilled.Load())
			assert.Equal(t, tt.wantDelta, snapshotStageMetrics(t).delta(before))
		})
	}
}

// Chunked decode in a disaggregated request runs under the NIXLv2 decode
// stage, which records one decode sample for the whole chunk loop.
func TestHandleNIXLV2ChunkedDecodeMetrics(t *testing.T) {
	lengthChunk := `{"choices":[{"message":{"content":"ab"},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`
	stopChunk := `{"choices":[{"message":{"content":"c"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}`

	tests := []struct {
		name       string
		stream     bool
		secondCode int
		wantCode   int
		wantDelta  stageMetrics
	}{
		{
			name:       "all chunks succeed records one decode sample",
			secondCode: http.StatusOK,
			wantCode:   http.StatusOK,
			wantDelta:  stageMetrics{prefillCount: 1, decodeCount: 1},
		},
		{
			name:       "failing chunk records one decode error",
			secondCode: http.StatusInternalServerError,
			wantCode:   http.StatusInternalServerError,
			wantDelta:  stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
		{
			name:       "streaming all chunks succeed records one decode sample",
			stream:     true,
			secondCode: http.StatusOK,
			wantCode:   http.StatusOK,
			wantDelta:  stageMetrics{prefillCount: 1, decodeCount: 1},
		},
		{
			name:       "streaming failing chunk records one decode error",
			stream:     true,
			secondCode: http.StatusInternalServerError,
			wantCode:   http.StatusOK,
			wantDelta:  stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefill := httptest.NewServer(statusHandler(http.StatusOK, `{"kv_transfer_params":{}}`))
			defer prefill.Close()
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "0", DecoderURL: prefillURL, DecodeChunkSize: 2})
			s.logger = log.Log
			var chunks atomic.Int32
			s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if chunks.Add(1) == 1 {
					statusHandler(http.StatusOK, lengthChunk).ServeHTTP(w, r)
					return
				}
				statusHandler(tt.secondCode, stopChunk).ServeHTTP(w, r)
			})

			body := textChatBody()
			body[reqcommon.FieldMaxTokens] = 10
			if tt.stream {
				body[reqcommon.FieldStream] = true
			}

			before := snapshotStageMetrics(t)
			rw := httptest.NewRecorder()
			s.handleNIXLV2(rw, chatRequest(t, body), prefillURL.Host, "", reqcommon.APITypeChatCompletions)

			assert.Equal(t, tt.wantCode, rw.Code)
			assert.Equal(t, int32(2), chunks.Load(), "request must take the chunked decode path")
			assert.Equal(t, tt.wantDelta, snapshotStageMetrics(t).delta(before))
		})
	}
}
