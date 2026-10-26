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

package openai

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// vLLM with stream_options {"include_usage": true, "continuous_usage_stats": true}
// reports cumulative usage on every chunk. When Envoy delivers the last two
// events in one body chunk, the usage read for the request is the final one.
func TestParseResponse_StreamingUsageTakesLastEventInChunk(t *testing.T) {
	chunk := []byte("data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9,\"total_tokens\":16}}\n\n" +
		"data: {\"id\":\"c\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":10,\"total_tokens\":17}}\n\n" +
		"data: [DONE]\n\n")

	got, err := NewOpenAIParser().ParseResponse(context.Background(), chunk, map[string]string{reqcommon.HeaderContentType: request.MediaTypeEventStream}, true)
	if err != nil {
		t.Fatal(err)
	}
	want := &fwkrh.ParsedResponse{
		Usage:          &fwkrh.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17},
		StreamedEvents: 2,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ParseResponse() mismatch (-want +got):\n%s", diff)
	}
}
