/*
Copyright 2025 The llm-d Authors.

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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKVTransferParamsCapture_NonStreaming(t *testing.T) {
	tests := []struct {
		name           string
		responseBody   string
		expectedParams map[string]any
	}{
		{
			name: "extracts kv_transfer_params from non-streaming response",
			responseBody: `{
				"id": "chatcmpl-123",
				"usage": {"prompt_tokens": 100, "completion_tokens": 10},
				"kv_transfer_params": {
					"remote_block_ids": [[1, 2, 3]],
					"remote_engine_id": "engine-abc",
					"remote_host": "10.0.1.42",
					"remote_port": 5678
				}
			}`,
			expectedParams: map[string]any{
				"remote_block_ids": []any{[]any{float64(1), float64(2), float64(3)}},
				"remote_engine_id": "engine-abc",
				"remote_host":      "10.0.1.42",
				"remote_port":      float64(5678),
			},
		},
		{
			name: "returns nil when kv_transfer_params absent",
			responseBody: `{
				"id": "chatcmpl-123",
				"usage": {"prompt_tokens": 100, "completion_tokens": 10}
			}`,
			expectedParams: nil,
		},
		{
			name:           "returns nil for empty response",
			responseBody:   "",
			expectedParams: nil,
		},
		{
			name:           "returns nil for invalid JSON",
			responseBody:   `{invalid json}`,
			expectedParams: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			writer, finalize := newKVTransferParamsCaptureWriter(recorder)

			n, err := writer.Write([]byte(tt.responseBody))
			require.NoError(t, err)
			assert.Equal(t, len(tt.responseBody), n)

			captured := finalize()
			if tt.expectedParams == nil {
				assert.Nil(t, captured)
			} else {
				assert.Equal(t, tt.expectedParams, captured)
			}

			assert.Equal(t, tt.responseBody, recorder.Body.String(), "response body should be unchanged")
		})
	}
}

func TestKVTransferParamsCapture_Streaming(t *testing.T) {
	tests := []struct {
		name           string
		chunks         []string
		expectedParams map[string]any
	}{
		{
			name: "extracts kv_transfer_params from final SSE chunk",
			chunks: []string{
				"data: {\"id\":\"chatcmpl-123\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
				"data: {\"id\":\"chatcmpl-123\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
				"data: {\"id\":\"chatcmpl-123\",\"usage\":{\"prompt_tokens\":100},\"kv_transfer_params\":{\"remote_block_ids\":[[1,2]],\"remote_engine_id\":\"engine-xyz\"}}\n\n",
				"data: [DONE]\n\n",
			},
			expectedParams: map[string]any{
				"remote_block_ids": []any{[]any{float64(1), float64(2)}},
				"remote_engine_id": "engine-xyz",
			},
		},
		{
			name: "updates to last-seen kv_transfer_params when multiple chunks have it",
			chunks: []string{
				"data: {\"kv_transfer_params\":{\"remote_engine_id\":\"first\"}}\n\n",
				"data: {\"kv_transfer_params\":{\"remote_engine_id\":\"second\"}}\n\n",
			},
			expectedParams: map[string]any{
				"remote_engine_id": "second",
			},
		},
		{
			name: "returns nil when no chunks contain kv_transfer_params",
			chunks: []string{
				"data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"test\"}}]}\n\n",
				"data: [DONE]\n\n",
			},
			expectedParams: nil,
		},
		{
			name: "handles chunks split mid-line",
			chunks: []string{
				"data: {\"kv_transfer_params\":{\"remote_",
				"engine_id\":\"split-line\"}}\n\n",
			},
			expectedParams: map[string]any{
				"remote_engine_id": "split-line",
			},
		},
		{
			name:           "returns nil for empty stream",
			chunks:         []string{},
			expectedParams: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			recorder.Header().Set("Content-Type", "text/event-stream")
			writer, finalize := newKVTransferParamsCaptureWriter(recorder)

			var expectedBody bytes.Buffer
			for _, chunk := range tt.chunks {
				n, err := writer.Write([]byte(chunk))
				require.NoError(t, err)
				assert.Equal(t, len(chunk), n)
				expectedBody.WriteString(chunk)
			}

			captured := finalize()
			if tt.expectedParams == nil {
				assert.Nil(t, captured)
			} else {
				assert.Equal(t, tt.expectedParams, captured)
			}

			assert.Equal(t, expectedBody.String(), recorder.Body.String(), "response body should be unchanged")
		})
	}
}

func TestKVTransferParamsCapture_DetectsSSEMode(t *testing.T) {
	t.Run("detects SSE from Content-Type header", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		recorder.Header().Set("Content-Type", "text/event-stream")
		writer, finalize := newKVTransferParamsCaptureWriter(recorder)

		_, err := writer.Write([]byte("data: {\"kv_transfer_params\":{\"remote_engine_id\":\"test\"}}\n\n"))
		require.NoError(t, err)
		captured := finalize()

		assert.NotNil(t, captured)
		assert.Equal(t, "test", captured["remote_engine_id"])
	})

	t.Run("detects SSE from data: prefix", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		writer, finalize := newKVTransferParamsCaptureWriter(recorder)

		_, err := writer.Write([]byte("data: {\"kv_transfer_params\":{\"remote_engine_id\":\"test\"}}\n\n"))
		require.NoError(t, err)
		captured := finalize()

		assert.NotNil(t, captured)
		assert.Equal(t, "test", captured["remote_engine_id"])
	})

	t.Run("treats response as non-streaming without SSE indicators", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		writer, finalize := newKVTransferParamsCaptureWriter(recorder)

		_, err := writer.Write([]byte(`{"kv_transfer_params":{"remote_engine_id":"test"}}`))
		require.NoError(t, err)
		captured := finalize()

		assert.NotNil(t, captured)
		assert.Equal(t, "test", captured["remote_engine_id"])
	})
}

func TestKVTransferParamsCapture_PreservesHTTPInterfaces(t *testing.T) {
	t.Run("preserves http.Flusher interface", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		writer, _ := newKVTransferParamsCaptureWriter(recorder)

		flusher, ok := writer.(http.Flusher)
		require.True(t, ok, "should preserve http.Flusher interface")
		flusher.Flush()
	})

	t.Run("preserves http.ResponseWriter interface", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		writer, _ := newKVTransferParamsCaptureWriter(recorder)

		writer.WriteHeader(http.StatusOK)
		writer.Header().Set("X-Test", "value")
		_, err := writer.Write([]byte("test"))
		require.NoError(t, err)

		assert.Equal(t, http.StatusOK, recorder.Code)
		assert.Equal(t, "value", recorder.Header().Get("X-Test"))
		assert.Equal(t, "test", recorder.Body.String())
	})
}
