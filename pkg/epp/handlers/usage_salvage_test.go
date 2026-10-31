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

	assert.Equal(t, 77, reqCtx.Usage.PromptTokens, "the real usage object must win over the one embedded in a string")
	assert.Equal(t, 8, reqCtx.Usage.CompletionTokens)
	assert.Equal(t, 85, reqCtx.Usage.TotalTokens)
}

// Direct scanner coverage: chat, Responses and Anthropic field namings.
func TestScanDroppedSSEUsage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		line string
		want fwkrh.Usage
	}{
		{
			name: "chat completions",
			line: `data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}`,
			want: fwkrh.Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "responses api input and output tokens",
			line: `data: {"type":"response.completed","response":{"usage":{"input_tokens":31,"output_tokens":7,"total_tokens":38}}}`,
			want: fwkrh.Usage{PromptTokens: 31, CompletionTokens: 7, TotalTokens: 38},
		},
		{
			name: "anthropic message delta",
			line: `data: {"type":"message_delta","delta":{"text":"x"},"usage":{"input_tokens":30,"output_tokens":15}}`,
			want: fwkrh.Usage{PromptTokens: 30, CompletionTokens: 15, TotalTokens: 45},
		},
		{
			name: "no usage object",
			line: `data: {"choices":[{"delta":{"text":"hi"}}]}`,
			want: fwkrh.Usage{},
		},
		{
			name: "no usage object, meta_info style (not covered)",
			line: `data: {"text":"hi","meta_info":{"prompt_tokens":3,"completion_tokens":4}}`,
			want: fwkrh.Usage{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := scanDroppedSSEUsage([]byte(tc.line))
			if tc.want == (fwkrh.Usage{}) {
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
