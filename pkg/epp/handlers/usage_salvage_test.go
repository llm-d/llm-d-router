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

package handlers

import (
	"context"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// A chat-completions usage line over maxSSELineBytes must still record its usage
// instead of silently zeroing the request metrics.
func TestOversizedChatUsageLineSalvaged(t *testing.T) {
	t.Parallel()

	pad := strings.Repeat("x", maxSSELineBytes+1024)
	stream := `data: {"id":"c","choices":[{"delta":{"content":"` + pad + `"}}],"usage":{"prompt_tokens":2901,"completion_tokens":42,"total_tokens":2943}}` + "\n" +
		"data: [DONE]"

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		Response:          &Response{Headers: map[string]string{"content-type": "text/event-stream"}},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}
	for i := 0; i < len(stream); i += 65536 {
		end := min(i+65536, len(stream))
		server.HandleResponseBody(context.Background(), reqCtx, []byte(stream[i:end]), end == len(stream))
	}

	assert.Equal(t, 2901, reqCtx.Usage.PromptTokens, "usage must be salvaged from the dropped line")
	assert.Equal(t, 42, reqCtx.Usage.CompletionTokens)
	assert.Equal(t, 2943, reqCtx.Usage.TotalTokens)
}

// A response.completed event far above the cap repeats instructions and output before
// the usage object; the dropped line must still yield the Responses-API usage fields.
func TestOversizedResponsesCompletedSalvaged(t *testing.T) {
	t.Parallel()

	event := "event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"r","instructions":"` + strings.Repeat("x", maxSSELineBytes*2) +
		`","usage":{"input_tokens":38617,"output_tokens":80,"total_tokens":38697,"input_tokens_details":{"cached_tokens":775}}}}` + "\n\n" +
		"data: [DONE]"

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		Response:          &Response{Headers: map[string]string{"content-type": "text/event-stream"}},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}
	for i := 0; i < len(event); i += 16384 {
		end := min(i+16384, len(event))
		server.HandleResponseBody(context.Background(), reqCtx, []byte(event[i:end]), end == len(event))
	}

	assert.Equal(t, 38617, reqCtx.Usage.PromptTokens, "input_tokens must be salvaged from the dropped response.completed")
	assert.Equal(t, 80, reqCtx.Usage.CompletionTokens)
	assert.Equal(t, 38697, reqCtx.Usage.TotalTokens)
	require.NotNil(t, reqCtx.Usage.PromptTokenDetails)
	assert.Equal(t, 775, reqCtx.Usage.PromptTokenDetails.CachedTokens)
}

// A usage key appearing inside a JSON string (e.g. echoed instructions) must not match;
// only the real usage object later in the dropped line counts.
func TestSalvageIgnoresUsageInsideStrings(t *testing.T) {
	t.Parallel()

	stream := `{` +
		`"instructions":"say ` + `\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":1}` + ` now",` +
		`"output":[],"usage":{"prompt_tokens":77,"completion_tokens":8,"total_tokens":85}}`

	reqCtx := &RequestContext{}
	salvageDroppedSSELine(context.Background(), reqCtx, []byte(stream), true)

	require.NotNil(t, reqCtx.salvagedUsage, "the real usage object must win over the one embedded in a string")
	assert.Equal(t, 77, reqCtx.salvagedUsage.PromptTokens)
	assert.Equal(t, 8, reqCtx.salvagedUsage.CompletionTokens)
	assert.Equal(t, 85, reqCtx.salvagedUsage.TotalTokens)
}

// Direct scanner coverage: chat, Responses and Anthropic field namings.
func TestScanDroppedSSEUsage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		line string
		want fwkrequest.Usage
	}{
		{
			name: "chat completions",
			line: `data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}`,
			want: fwkrequest.Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "responses api input and output tokens",
			line: `data: {"type":"response.completed","response":{"usage":{"input_tokens":31,"output_tokens":7,"total_tokens":38}}}`,
			want: fwkrequest.Usage{PromptTokens: 31, CompletionTokens: 7, TotalTokens: 38},
		},
		{
			name: "anthropic message delta",
			line: `data: {"type":"message_delta","delta":{"text":"x"},"usage":{"input_tokens":30,"output_tokens":15}}`,
			want: fwkrequest.Usage{PromptTokens: 30, CompletionTokens: 15, TotalTokens: 45},
		},
		{
			name: "no usage object",
			line: `data: {"choices":[{"delta":{"text":"hi"}}]}`,
			want: fwkrequest.Usage{},
		},
		{
			name: "no usage object, meta_info style (not covered)",
			line: `data: {"text":"hi","meta_info":{"prompt_tokens":3,"completion_tokens":4}}`,
			want: fwkrequest.Usage{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := scanDroppedSSEUsage([]byte(tc.line))
			if tc.want == (fwkrequest.Usage{}) {
				assert.Nil(t, got)
				return
			}
			require.NotNil(t, got)
			assert.Equal(t, tc.want.PromptTokens, got.PromptTokens)
			assert.Equal(t, tc.want.CompletionTokens, got.CompletionTokens)
			assert.Equal(t, tc.want.TotalTokens, got.TotalTokens)
		})
	}
}

// A "usage" key whose value is a string must not desynchronize quote tracking,
// or the token object later in the same dropped line is missed.
func TestSalvageStringValuedUsageKey(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		line string
		want fwkrequest.Usage
	}{
		{
			name: "string none before real usage",
			line: `{"metadata":{"usage":"none"},"response":{"usage":{"input_tokens":31,"output_tokens":7,"total_tokens":38}}}`,
			want: fwkrequest.Usage{PromptTokens: 31, CompletionTokens: 7, TotalTokens: 38},
		},
		{
			name: "empty string before real usage",
			line: `{"usage":"","usage":{"input_tokens":8,"output_tokens":9,"total_tokens":17}}`,
			want: fwkrequest.Usage{PromptTokens: 8, CompletionTokens: 9, TotalTokens: 17},
		},
		{
			name: "string value is the word usage",
			line: `{"usage":"usage","usage":{"input_tokens":4,"output_tokens":5,"total_tokens":9}}`,
			want: fwkrequest.Usage{PromptTokens: 4, CompletionTokens: 5, TotalTokens: 9},
		},
		{
			name: "whitespace around colon and string value",
			line: `{"metadata":{"usage" : "none"},"response":{"usage":{"input_tokens":31,"output_tokens":7,"total_tokens":38}}}`,
			want: fwkrequest.Usage{PromptTokens: 31, CompletionTokens: 7, TotalTokens: 38},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := scanDroppedSSEUsage([]byte(tc.line))
			require.NotNil(t, got)
			assert.Equal(t, tc.want.PromptTokens, got.PromptTokens)
			assert.Equal(t, tc.want.CompletionTokens, got.CompletionTokens)
			assert.Equal(t, tc.want.TotalTokens, got.TotalTokens)
		})
	}
}

// Anthropic cache_read_input_tokens is part of the prompt total and is also the
// cached-token metric. Nested cached_tokens still wins when both are present.
func TestSalvageAnthropicCacheReadTokens(t *testing.T) {
	t.Parallel()

	t.Run("root cache read and cache creation", func(t *testing.T) {
		t.Parallel()

		got := scanDroppedSSEUsage([]byte(`data: {"type":"message_delta","usage":{"input_tokens":30,"cache_creation_input_tokens":4,"cache_read_input_tokens":12,"output_tokens":15}}`))
		require.NotNil(t, got)
		assert.Equal(t, 46, got.PromptTokens)
		assert.Equal(t, 15, got.CompletionTokens)
		assert.Equal(t, 61, got.TotalTokens)
		require.NotNil(t, got.PromptTokenDetails)
		assert.Equal(t, 12, got.PromptTokenDetails.CachedTokens)
	})

	t.Run("nested cached_tokens is kept", func(t *testing.T) {
		t.Parallel()

		got := scanDroppedSSEUsage([]byte(`{"usage":{"input_tokens":30,"cache_read_input_tokens":12,"output_tokens":1,"total_tokens":43,"input_tokens_details":{"cached_tokens":7}}}`))
		require.NotNil(t, got)
		assert.Equal(t, 42, got.PromptTokens)
		assert.Equal(t, 1, got.CompletionTokens)
		assert.Equal(t, 43, got.TotalTokens)
		require.NotNil(t, got.PromptTokenDetails)
		assert.Equal(t, 7, got.PromptTokenDetails.CachedTokens)
	})
}

// A "usage" value larger than the collection cap is not a token block. Skipping
// it must stay in phase so the real usage object later in the line is recovered,
// including when the capped string itself contains usage-shaped text.
func TestSalvageSkipsOversizedUsageObject(t *testing.T) {
	t.Parallel()

	want := fwkrequest.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}
	real := `,"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}`

	t.Run("sibling usage after a capped object", func(t *testing.T) {
		t.Parallel()

		blob := strings.Repeat("x", maxSalvageObjectBytes+32)
		got := scanDroppedSSEUsage([]byte(`{"debug":{"usage":{"blob":"` + blob + `"}}` + real))
		require.NotNil(t, got)
		assert.Equal(t, want.PromptTokens, got.PromptTokens)
		assert.Equal(t, want.CompletionTokens, got.CompletionTokens)
		assert.Equal(t, want.TotalTokens, got.TotalTokens)
	})

	t.Run("usage text inside the capped string", func(t *testing.T) {
		t.Parallel()

		blob := strings.Repeat("x", maxSalvageObjectBytes) + `\"usage\":{\"input_tokens\":999,\"output_tokens\":1}`
		got := scanDroppedSSEUsage([]byte(`{"debug":{"usage":{"blob":"` + blob + `"}}` + real))
		require.NotNil(t, got)
		assert.Equal(t, want.PromptTokens, got.PromptTokens)
		assert.Equal(t, want.CompletionTokens, got.CompletionTokens)
		assert.Equal(t, want.TotalTokens, got.TotalTokens)
	})
}

// A chunk that both finishes an oversized line and carries the next event must
// salvage only the dropped line. The following event stays in the bytes returned
// for parsing, and its usage must not replace the dropped line's token counts.
func TestOverflowChunkKeepsFollowingEventOutOfSalvage(t *testing.T) {
	t.Parallel()

	prefix := `{"pad":"`
	first := prefix + strings.Repeat("x", maxSSELineBytes-len(prefix))
	require.Len(t, first, maxSSELineBytes)
	require.NotContains(t, first, "\n")

	second := strings.Repeat("y", 8) + `","usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}}` + "\n" +
		`data: {"usage":{"output_tokens":7}}` + "\n"

	reqCtx := &RequestContext{}
	ctx := context.Background()
	require.Nil(t, completeSSELines(ctx, reqCtx, []byte(first), false))
	assert.Nil(t, reqCtx.salvagedUsage, "the dropped line has not ended yet, so nothing is salvaged")

	kept := completeSSELines(ctx, reqCtx, []byte(second), true)
	require.NotNil(t, reqCtx.salvagedUsage)
	assert.Equal(t, 100, reqCtx.salvagedUsage.PromptTokens)
	assert.Equal(t, 50, reqCtx.salvagedUsage.CompletionTokens)
	assert.Equal(t, 150, reqCtx.salvagedUsage.TotalTokens)
	assert.Contains(t, string(kept), `data: {"usage":{"output_tokens":7}}`)
	assert.NotContains(t, string(kept), `"prompt_tokens":100`)
}

// A dropped line that never reaches its newline before the stream ends must
// still have its usage salvaged at end-of-stream, not lost with the request.
func TestSalvageStreamEndsBeforeNewline(t *testing.T) {
	t.Parallel()

	stream := `data: {"choices":[{"delta":{"content":"` + strings.Repeat("x", maxSSELineBytes) +
		`"}}],"usage":{"prompt_tokens":42,"completion_tokens":7,"total_tokens":49}}`

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		Response:          &Response{Headers: map[string]string{"content-type": "text/event-stream"}},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}
	for i := 0; i < len(stream); i += 65536 {
		end := min(i+65536, len(stream))
		server.HandleResponseBody(context.Background(), reqCtx, []byte(stream[i:end]), end == len(stream))
	}

	assert.Equal(t, 42, reqCtx.Usage.PromptTokens, "usage must be salvaged even though the dropped line never ends with a newline")
	assert.Equal(t, 7, reqCtx.Usage.CompletionTokens)
	assert.Equal(t, 49, reqCtx.Usage.TotalTokens)
}

// A chunk that ends an oversized data line and leaves a tail over the cap must
// salvage the two usages separately. The tail is fed first, so running
// dropOversizedLines before the tail switch keeps each salvager to one line
// instead of flushing the data line into the still-open tail scan.
func TestOversizedTailAndDataLineSalvagedSeparately(t *testing.T) {
	t.Parallel()

	dataLine := `data: {"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}}` + "\n"
	pad := strings.Repeat("x", maxSSELineBytes)
	first := pad + dataLine + `data: {"usage":{"prompt_tokens":200,"completion_tokens":70,"total_tokens":270}}` + pad

	reqCtx := &RequestContext{}
	ctx := context.Background()
	require.Nil(t, completeSSELines(ctx, reqCtx, []byte(first), false))
	// The data line's usage is salvaged by dropOversizedLines before the tail
	// scan opens; nothing is merged into Usage until HandleResponseBody runs.
	require.NotNil(t, reqCtx.salvagedUsage)
	assert.Equal(t, 100, reqCtx.salvagedUsage.PromptTokens)

	// The tail's usage finishes on the next chunk, which ends the stream.
	completeSSELines(ctx, reqCtx, nil, true)
	require.NotNil(t, reqCtx.salvagedUsage)
	assert.Equal(t, 200, reqCtx.salvagedUsage.PromptTokens, "the tail's usage must not be clobbered by the data line's")
	assert.Equal(t, 70, reqCtx.salvagedUsage.CompletionTokens)
	assert.Equal(t, 270, reqCtx.salvagedUsage.TotalTokens)
}

// A chunk holding a normal usage line followed by a dropped oversized line must
// keep the dropped line's usage: it is the last usage event in the chunk, and the
// parser merges the normal line first, so the salvaged usage must merge after it.
func TestSalvagedUsageWinsOverEarlierParsedLine(t *testing.T) {
	t.Parallel()

	small := `data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n"
	big := `data: {"choices":[{"delta":{"content":"` + strings.Repeat("x", maxSSELineBytes) +
		`"}}],"usage":{"prompt_tokens":999,"completion_tokens":999,"total_tokens":1998}}` + "\n"

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		Response:          &Response{Headers: map[string]string{"content-type": "text/event-stream"}},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}
	server.HandleResponseBody(context.Background(), reqCtx, []byte(small+big), true)

	assert.Equal(t, 999, reqCtx.Usage.PromptTokens, "the dropped oversized line is the last usage event and must win")
	assert.Equal(t, 999, reqCtx.Usage.CompletionTokens)
	assert.Equal(t, 1998, reqCtx.Usage.TotalTokens)
}
