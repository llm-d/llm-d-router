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
	"testing"

	"github.com/go-logr/logr"
	"github.com/google/go-cmp/cmp"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/sglanghttp"
)

// Envoy forwards the upstream body as it is read from the socket, so one SSE
// event can straddle two ext_proc body chunks. The usage event is the last of
// the stream; when it is split, the handler must still record the usage.
func TestHandleResponseBody_StreamedUsageSurvivesChunkSplit(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	tests := []struct {
		name   string
		path   string
		parser fwkrh.Parser
		event  string
		want   fwkrh.Usage
	}{
		{
			name:   "openai chat completions",
			path:   "/v1/chat/completions",
			parser: openai.NewOpenAIParser(),
			event:  "data: {\"id\":\"c\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":10,\"total_tokens\":17}}\n\ndata: [DONE]\n\n",
			want:   fwkrh.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17},
		},
		{
			name:   "anthropic messages",
			path:   "/v1/messages",
			parser: anthropic.NewAnthropicParser(),
			event:  "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":7,\"output_tokens\":10}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
			want:   fwkrh.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17},
		},
		{
			name:   "sglang generate",
			path:   "/generate",
			parser: sglanghttp.NewSGLangHTTPParser(),
			event:  "data: {\"text\":\"!\",\"meta_info\":{\"finish_reason\":{\"type\":\"stop\"},\"prompt_tokens\":7,\"completion_tokens\":10}}\n\ndata: [DONE]\n\n",
			want:   fwkrh.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Cut the whole stream at every interior offset so the split can
			// fall anywhere inside the usage event.
			for cut := 1; cut < len(tt.event); cut++ {
				server := &StreamingServer{
					parserRegistry: NewParserRegistry([]fwkrh.Parser{tt.parser}, logr.Discard()),
					director:       &mockDirector{},
				}
				reqCtx := &RequestContext{
					Request:                        &Request{Headers: map[string]string{":path": tt.path}},
					Response:                       &Response{Headers: map[string]string{"content-type": "text/event-stream; charset=utf-8"}},
					SchedulingRequest:              &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
					modelServerStreaming:           true,
					toolCallingResponseEventStream: true,
				}
				server.HandleResponseBody(ctx, reqCtx, []byte(tt.event[:cut]), false)
				server.HandleResponseBody(ctx, reqCtx, []byte(tt.event[cut:]), true)
				if diff := cmp.Diff(tt.want, reqCtx.Usage); diff != "" {
					t.Errorf("cut at %d: usage (-want +got):\n%s", cut, diff)
				}
			}
		})
	}
}
