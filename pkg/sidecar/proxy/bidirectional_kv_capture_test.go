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
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

const captureKVFrame = `{"id":"c","choices":[],"kv_transfer_params":{"do_remote_prefill":false,"do_remote_decode":true,` +
	`"remote_block_ids":[[7,8]],"remote_engine_id":"e","remote_request_id":"r","remote_host":"h","remote_port":4032,"remote_num_tokens":96}}`

func sseFrames(frames ...string) string {
	var b strings.Builder
	for _, f := range frames {
		b.WriteString("data: " + f + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func captureSSE(t *testing.T, stream string, chunk int) (*httptest.ResponseRecorder, *decodeCapture) {
	t.Helper()
	rec := httptest.NewRecorder()
	rec.Header().Set("Content-Type", "text/event-stream")
	w, c := newDecodeCapture(rec)
	flusher, ok := w.(http.Flusher)
	require.True(t, ok, "the capture must keep the underlying writer's http.Flusher")
	for i := 0; i < len(stream); i += chunk {
		end := min(i+chunk, len(stream))
		_, err := w.Write([]byte(stream[i:end]))
		require.NoError(t, err)
		flusher.Flush()
	}
	return rec, c
}

func TestDecodeCaptureStreamedContentAndToolCalls(t *testing.T) {
	stream := sseFrames(
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":""}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"other","arguments":"{}"}}]}}]}`,
		captureKVFrame,
	)

	// Seven bytes per write splits frames and lines at arbitrary points.
	rec, c := captureSSE(t, stream, 7)
	require.Equal(t, stream, rec.Body.String(), "the stream is forwarded unchanged")
	require.True(t, rec.Flushed)

	params, reply, ok := c.result()
	require.True(t, ok)
	require.Equal(t, "Hello", reply["content"])
	require.Equal(t, []any{
		map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{"q":1}`}},
		map[string]any{"id": "call_2", "type": "function", "function": map[string]any{"name": "other", "arguments": `{}`}},
	}, reply["tool_calls"])
	require.Equal(t, "e", params["remote_engine_id"])
	require.Equal(t, json.Number("4032"), params["remote_port"], "numbers replay as written")
}

func TestDecodeCaptureStreamForwardsBeforeTheStreamEnds(t *testing.T) {
	rec := httptest.NewRecorder()
	rec.Header().Set("Content-Type", "text/event-stream")
	w, _ := newDecodeCapture(rec)

	first := "data: " + `{"choices":[{"index":0,"delta":{"content":"a"}}]}` + "\n\n"
	_, err := w.Write([]byte(first))
	require.NoError(t, err)
	require.Equal(t, first, rec.Body.String(), "the first chunk reaches the client before the stream ends")
}

func TestDecodeCaptureNonStreamingResponse(t *testing.T) {
	body := `{"choices":[{"index":0,"message":{"role":"assistant","content":"Hi","tool_calls":[` +
		`{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}],` +
		`"kv_transfer_params":{"do_remote_decode":true,"remote_block_ids":[[1]],"remote_engine_id":"e","remote_request_id":"r","remote_host":"h","remote_port":1}}`
	params, reply, ok := captureJSON(t, []byte(body)).result()
	require.True(t, ok)
	require.Equal(t, "Hi", reply["content"])
	require.Len(t, reply["tool_calls"], 1)
	require.Equal(t, "r", params["remote_request_id"])
}

func TestDecodeCaptureRejectsResponsesItCannotReduceToOneMessage(t *testing.T) {
	t.Run("a second choice", func(t *testing.T) {
		stream := sseFrames(
			`{"choices":[{"index":0,"delta":{"content":"a"}},{"index":1,"delta":{"content":"b"}}]}`,
			captureKVFrame,
		)
		_, c := captureSSE(t, stream, 64)
		_, _, ok := c.result()
		require.False(t, ok)
	})
	t.Run("a data frame that is not JSON", func(t *testing.T) {
		_, c := captureSSE(t, sseFrames(`{"choices":[{"index":0,"delta":{"content":"a"}}]}`, `{not json`, captureKVFrame), 64)
		_, _, ok := c.result()
		require.False(t, ok)
	})
	t.Run("no kv_transfer_params", func(t *testing.T) {
		_, c := captureSSE(t, sseFrames(`{"choices":[{"index":0,"delta":{"content":"a"}}]}`), 64)
		_, _, ok := c.result()
		require.False(t, ok)
	})
	t.Run("a non-streaming body that is not JSON", func(t *testing.T) {
		w, c := newDecodeCapture(httptest.NewRecorder())
		_, err := w.Write([]byte("upstream exploded"))
		require.NoError(t, err)
		_, _, ok := c.result()
		require.False(t, ok)
	})
}

func TestDecodeCaptureIgnoresSSEComments(t *testing.T) {
	stream := ": keep-alive\n\n" + sseFrames(`{"choices":[{"index":0,"delta":{"content":"a"}}]}`, captureKVFrame)
	_, c := captureSSE(t, stream, 5)
	_, reply, ok := c.result()
	require.True(t, ok)
	require.Equal(t, "a", reply["content"])
}

func TestDecodeCaptureRefusesReasoningOutput(t *testing.T) {
	t.Run("streamed", func(t *testing.T) {
		stream := sseFrames(
			`{"choices":[{"index":0,"delta":{"reasoning_content":"thinking"}}]}`,
			`{"choices":[{"index":0,"delta":{"content":"answer"}}]}`,
			captureKVFrame,
		)
		_, c := captureSSE(t, stream, 64)
		_, _, ok := c.result()
		require.False(t, ok, "a template may render reasoning differently on the next turn")
	})
	t.Run("complete message", func(t *testing.T) {
		body := `{"choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning":"thinking"}}],` +
			`"kv_transfer_params":{"do_remote_decode":true,"remote_block_ids":[[1]],"remote_engine_id":"e","remote_request_id":"r","remote_host":"h","remote_port":1}}`
		_, _, ok := captureJSON(t, []byte(body)).result()
		require.False(t, ok)
	})
	t.Run("empty reasoning fields are ignored", func(t *testing.T) {
		stream := sseFrames(
			`{"choices":[{"index":0,"delta":{"role":"assistant","content":"a","reasoning_content":null,"reasoning":""}}]}`,
			captureKVFrame,
		)
		_, c := captureSSE(t, stream, 64)
		_, _, ok := c.result()
		require.True(t, ok)
	})
}

func TestDecodeCaptureStopsReadingOversizedResponses(t *testing.T) {
	rec := httptest.NewRecorder()
	rec.Header().Set("Content-Type", "text/event-stream")
	w, c := newDecodeCapture(rec)
	frame := "data: " + `{"choices":[{"index":0,"delta":{"content":"` + strings.Repeat("x", 1<<20) + `"}}]}` + "\n\n"
	for written := 0; written <= maxCapturedResponseBytes; written += len(frame) {
		_, err := w.Write([]byte(frame))
		require.NoError(t, err)
	}
	_, err := w.Write([]byte(sseFrames(captureKVFrame)))
	require.NoError(t, err)

	_, _, ok := c.result()
	require.False(t, ok, "a response over the limit is never cached")
	require.Greater(t, rec.Body.Len(), maxCapturedResponseBytes, "the client still receives all of it")
}
