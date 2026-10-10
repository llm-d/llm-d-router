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
	"errors"
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

	// Absent, empty, or false forms are not lossy.
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"tools": []any{}}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"logprobs": false}))

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

// TestForceStreamIncremental_CleanErrorBeforeCommit verifies that a fault before
// the first content byte surfaces as a plain error (so the server can answer a
// clean 5xx) and writes nothing to the client.
func TestForceStreamIncremental_CleanErrorBeforeCommit(t *testing.T) {
	server := sseServer(t, []string{`not-json`}, nil)
	defer server.Close()

	step, _ := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()

	err := step.Execute(context.Background(), chatIncrementalReqCtx(recorder))
	require.Error(t, err, "a fault before any write must surface for a clean 5xx")
	var streamed *pipeline.UpstreamStreamedError
	require.False(t, errors.As(err, &streamed), "nothing was written, so it is a plain error")
	require.Equal(t, 0, recorder.Body.Len(), "nothing may be written before the first content frame")
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

	step, _ := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()

	require.NoError(t, step.Execute(context.Background(), chatIncrementalReqCtx(recorder)),
		"a fault after commit is reported as handled, not an error")
	require.Equal(t, http.StatusOK, recorder.Result().StatusCode)
	require.Contains(t, recorder.Body.String(), "partial", "the committed partial body reaches the client")
	require.False(t, json.Valid(recorder.Body.Bytes()), "the truncated body is not complete JSON")
}
