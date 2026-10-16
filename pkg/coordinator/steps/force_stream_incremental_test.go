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

package steps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// chatIncrementalReqCtx builds a non-streaming chat request with a single choice
// (the shape the incremental path serves), optionally with a token limit.
func chatIncrementalReqCtx(w http.ResponseWriter) *pipeline.RequestContext {
	return &pipeline.RequestContext{
		RequestID:        "req-inc",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: w,
	}
}

// TestCanStreamIncrementally checks the shape and choice-count gate for the
// incremental path: chat and text with a single choice qualify, while n>1
// (interleaved choices) and the generate and responses shapes take the buffered
// path.
func TestCanStreamIncrementally(t *testing.T) {
	require.True(t, canStreamIncrementally(sseShapeChat,
		&pipeline.RequestContext{OriginalPath: reqcommon.PathChatCompletions, Body: map[string]any{}}))
	require.True(t, canStreamIncrementally(sseShapeText,
		&pipeline.RequestContext{OriginalPath: reqcommon.PathCompletions, Body: map[string]any{}}))
	// n>1 interleaves choices, so it takes the buffered path.
	require.False(t, canStreamIncrementally(sseShapeChat,
		&pipeline.RequestContext{OriginalPath: reqcommon.PathChatCompletions, Body: map[string]any{"n": 2}}))
	// generate and responses are not a flat content string.
	require.False(t, canStreamIncrementally(sseShapeGenerate,
		&pipeline.RequestContext{OriginalPath: reqcommon.PathVLLMGenerate, Body: map[string]any{}}))
	require.False(t, canStreamIncrementally(sseShapeResponses,
		&pipeline.RequestContext{OriginalPath: reqcommon.PathResponses, Body: map[string]any{}}))
}

func TestForceStreamLossy(t *testing.T) {
	// chat and text requests that may carry tool/function calls or logprobs are lossy.
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"tools": []any{map[string]any{"type": "function"}}}))
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"functions": []any{map[string]any{"name": "f"}}}))
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"logprobs": true}))
	require.True(t, forceStreamLossy(sseShapeText, map[string]any{"logprobs": 3}))
	// On the legacy Completions API logprobs is a count; its presence (even 0)
	// requests the sampled token's logprob, which the reassembler drops, so a
	// numeric value of any size is lossy. JSON decodes the count to float64.
	require.True(t, forceStreamLossy(sseShapeText, map[string]any{"logprobs": 0}))
	require.True(t, forceStreamLossy(sseShapeText, map[string]any{"logprobs": float64(0)}))

	// vLLM chat reasoning (delta.reasoning_content) is not folded, so a chat
	// request that activates thinking via chat_template_kwargs.enable_thinking or
	// reasoning_effort is lossy and takes the pass-through.
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": true}}))
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"reasoning_effort": "high"}))

	// Absent, empty, or false forms are not lossy.
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"tools": []any{}}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"logprobs": false}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"reasoning_effort": ""}))
	// Reasoning is a chat concept; the text shape has no reasoning_content to drop.
	require.False(t, forceStreamLossy(sseShapeText, map[string]any{"reasoning_effort": "high"}))

	// The responses shape emits the upstream object verbatim and generate has no
	// such fields, so neither is ever lossy even when the request declares them.
	require.False(t, forceStreamLossy(sseShapeResponses, map[string]any{"tools": []any{map[string]any{"type": "function"}}}))
	require.False(t, forceStreamLossy(sseShapeGenerate, map[string]any{"logprobs": true}))
}

// TestForceStreamIncremental_EqualsBuffered is the core correctness check: the
// reply the incremental path writes must parse-equal what the buffered
// reassembler produces from the same frames, including escaped content and usage.
func TestForceStreamIncremental_EqualsBuffered(t *testing.T) {
	frames := []string{
		`{"id":"c-1","model":"llama-3","created":100,"choices":[{"index":0,"delta":{"role":"assistant","content":"He"}}]}`,
		`{"id":"c-1","choices":[{"index":0,"delta":{"content":"llo <world> \"q\""}}]}`,
		`{"id":"c-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c-1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, _ := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)))

	var got map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &got), "incremental reply must be valid JSON")

	// Compare both in the form the client receives (JSON-serialized), since the
	// buffered path also marshals its result before sending; a direct map compare
	// would differ only by numeric type (int vs JSON float64).
	want := newSSEReassembler(sseShapeChat, 0)
	for _, f := range frames {
		want.add(frame(t, f))
	}
	wantBytes, err := json.Marshal(want.result())
	require.NoError(t, err)
	var wantParsed map[string]any
	require.NoError(t, json.Unmarshal(wantBytes, &wantParsed))
	require.Equal(t, wantParsed, got, "incremental output must match the buffered reassembly")
}

// TestForceStreamIncremental_ChunkedNoContentLength checks the wire framing end
// to end through a real server: the flushed reply carries no Content-Length and
// is chunked, so the client receives it as one JSON body with no length known
// up front.
func TestForceStreamIncremental_ChunkedNoContentLength(t *testing.T) {
	frames := []string{
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
		`{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	upstream := sseServer(t, frames, nil)
	defer upstream.Close()
	step, _ := newForceStreamStep(t, upstream.URL, "1GiB")

	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = step.Execute(r.Context(), chatIncrementalReqCtx(w))
	}))
	defer front.Close()

	resp, err := http.Post(front.URL+reqcommon.PathChatCompletions, "application/json", strings.NewReader("{}"))
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, "application/json", resp.Header.Get("Content-Type"))
	require.Empty(t, resp.Header.Get("Content-Length"))
	require.Equal(t, []string{"chunked"}, resp.TransferEncoding)

	body, _ := io.ReadAll(resp.Body)
	var got map[string]any
	require.NoError(t, json.Unmarshal(body, &got))
	msg := got["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	require.Equal(t, "hi", msg["content"])
}

// TestForceStreamIncremental_NoTokenLimit_StillForced documents that the
// incremental path drops the token-limit requirement: a chat single-choice
// request with no max_tokens is force-streamed, holding only a rolling buffer and
// reserving no budget.
func TestForceStreamIncremental_NoTokenLimit_StillForced(t *testing.T) {
	frames := []string{
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
		`{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	var upstreamBody map[string]any
	server := sseServer(t, frames, &upstreamBody)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)))

	require.Equal(t, true, upstreamBody["stream"], "a chat single-choice request is force-streamed even without a token limit")
	require.Equal(t, "application/json", recorder.Result().Header.Get("Content-Type"))
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "the incremental path reserves no budget")
}

// TestForceStreamIncremental_UnreadableStream_FallsBack verifies that an
// unreadable stream before the first content byte falls back to the pass-through
// rather than failing: nothing reached the client, so the request is re-sent
// non-forced and the pod serves it. The outcome is counted fallback_unreadable.
func TestForceStreamIncremental_UnreadableStream_FallsBack(t *testing.T) {
	passthroughReply := `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			fmt.Fprint(w, "data: not-json\n\n")
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()

	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)),
		"a pre-commit unreadable stream must fall back, not error")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client gets the pass-through reply, not a 5xx")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}

// TestForceStreamIncremental_TruncatesAfterCommit verifies the accepted trade:
// once the first content byte is flushed the response is committed, so a later
// upstream fault truncates the connection rather than returning a clean error.
func TestForceStreamIncremental_TruncatesAfterCommit(t *testing.T) {
	frames := []string{
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"partial"}}]}`,
		`not-json`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()

	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)),
		"a fault after commit is reported as handled, not an error")
	require.Equal(t, http.StatusOK, recorder.Result().StatusCode)
	require.Contains(t, recorder.Body.String(), "partial", "the committed partial body reaches the client")
	require.False(t, json.Valid(recorder.Body.Bytes()), "the truncated body is not complete JSON")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9,
		"a reply truncated after commit is still counted forced, matching the buffered path")
}

// TestForceStreamIncremental_TruncatedBeforeContent_FallsBack covers a forced
// stream that ends with a clean EOF before any content byte and before [DONE] or a
// finish_reason: nothing is committed, so it falls back to the pass-through rather
// than finishing a reply that was cut short. Counted fallback_unreadable.
func TestForceStreamIncremental_TruncatedBeforeContent_FallsBack(t *testing.T) {
	passthroughReply := `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			// A role-only frame carries no content, so nothing commits; the stream
			// then closes with no [DONE] and no finish_reason.
			fmt.Fprint(w, "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n")
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()

	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)),
		"a truncation before the first content byte must fall back, not error")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client gets the pass-through reply, not a 5xx")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}

// TestForceStreamIncremental_TruncatedAfterContent_Truncates covers a forced
// stream cut after content is flushed but before [DONE] or a finish_reason: the
// reply is already committed, so the connection is truncated with no clean suffix
// rather than finished as if complete. Counted forced, matching the buffered path.
func TestForceStreamIncremental_TruncatedAfterContent_Truncates(t *testing.T) {
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		// Content is flushed, then the stream closes with no [DONE] and no
		// finish_reason.
		fmt.Fprint(w, "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"}}]}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()

	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)),
		"a truncation after commit is reported as handled, not an error")
	require.Equal(t, http.StatusOK, recorder.Result().StatusCode)
	require.Contains(t, recorder.Body.String(), "partial", "the committed partial body reaches the client")
	require.False(t, json.Valid(recorder.Body.Bytes()), "the truncated body is not complete JSON")
	require.Equal(t, []any{true}, streams, "no pass-through: the reply was already committed")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}

// TestForceStreamIncremental_SeveredConnection_FallsBack covers the read-error
// branch, distinct from a JSON parse error: the forced upstream sends a role-only
// frame (no content, so nothing commits) then severs the connection (a chunked body
// with no terminating chunk and no [DONE]), so the scanner returns a read-upstream
// error before the first content byte. The request falls back to the pass-through.
func TestForceStreamIncremental_SeveredConnection_FallsBack(t *testing.T) {
	passthroughReply := `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			hj, ok := w.(http.Hijacker)
			require.True(t, ok, "the test server must support hijacking")
			conn, buf, err := hj.Hijack()
			require.NoError(t, err)
			defer conn.Close()
			// A role-only frame commits nothing; close with no terminating chunk so
			// the client's chunked read fails before any content byte is flushed.
			_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
			f := "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"
			fmt.Fprintf(buf, "%x\r\n%s\r\n", len(f), f)
			_ = buf.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()

	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)),
		"a severed connection before the first content byte must fall back, not error")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client gets the pass-through reply, not a 5xx")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}
