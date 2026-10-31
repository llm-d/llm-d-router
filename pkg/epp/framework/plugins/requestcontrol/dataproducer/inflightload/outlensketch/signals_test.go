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

package outlensketch

import (
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func chatBody() *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{
		Model:           "test-model",
		ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: []fwkrh.Message{msg("user", "hi")}},
		Payload:         fwkrh.PayloadMap{},
	}
}

func TestSignalSigStableAndDistinct(t *testing.T) {
	a := signalSig(chatBody())
	if a != signalSig(chatBody()) {
		t.Error("signalSig must be stable for identical inputs")
	}

	withTools := chatBody()
	withTools.ChatCompletions.Tools = []any{map[string]any{"type": "function"}}
	if signalSig(withTools) == a {
		t.Error("adding tools must change the signature")
	}

	withReasoning := chatBody()
	withReasoning.Payload = fwkrh.PayloadMap{"reasoning_effort": "high"}
	if signalSig(withReasoning) == a {
		t.Error("reasoning_effort=high must change the signature")
	}

	withCode := chatBody()
	withCode.ChatCompletions.Messages = []fwkrh.Message{msg("user", "```go\nx:=1\n```")}
	if signalSig(withCode) == a {
		t.Error("code in the prompt must change the signature")
	}
}

func TestReasoningBucket(t *testing.T) {
	high := chatBody()
	high.Payload = fwkrh.PayloadMap{"reasoning_effort": "high"}
	if got := reasoningBucket(high, payloadMap(high)); got != 2 {
		t.Errorf("reasoning high bucket = %d, want 2", got)
	}

	thinkingOff := chatBody()
	thinkingOff.ChatCompletions.ChatTemplateKWArgs = map[string]any{"enable_thinking": false}
	if got := reasoningBucket(thinkingOff, payloadMap(thinkingOff)); got != 0 {
		t.Errorf("enable_thinking=false bucket = %d, want 0", got)
	}

	none := chatBody()
	if got := reasoningBucket(none, payloadMap(none)); got != 1 {
		t.Errorf("no reasoning signal bucket = %d, want 1 (unknown)", got)
	}
}

func TestMaxOutputBucket(t *testing.T) {
	n := func(v int64) *int64 { return &v }
	cases := []struct {
		cap  *int64
		want int
	}{{nil, 0}, {n(4000), 1}, {n(2000), 1}, {n(100), 2}}
	for _, c := range cases {
		if got := maxOutputBucket(c.cap); got != c.want {
			t.Errorf("maxOutputBucket(%v) = %d, want %d", c.cap, got, c.want)
		}
	}
}

func TestNumMessagesBucket(t *testing.T) {
	mk := func(n int) *fwkrh.InferenceRequestBody {
		b := chatBody()
		b.ChatCompletions.Messages = make([]fwkrh.Message, n)
		return b
	}
	cases := []struct{ n, want int }{{1, 0}, {3, 1}, {8, 2}, {12, 3}}
	for _, c := range cases {
		if got := numMessagesBucket(mk(c.n)); got != c.want {
			t.Errorf("numMessagesBucket(%d) = %d, want %d", c.n, got, c.want)
		}
	}
}

func TestToolChoiceAndResponseFormatKind(t *testing.T) {
	if got := toolChoiceKind("required"); got != "required" {
		t.Errorf("toolChoiceKind(string) = %q, want required", got)
	}
	if got := toolChoiceKind(map[string]any{"type": "function"}); got != "named" {
		t.Errorf("toolChoiceKind(object) = %q, want named", got)
	}
	if got := responseFormatKind(map[string]any{"type": "json_schema"}); got != "json_schema" {
		t.Errorf("responseFormatKind(object) = %q, want json_schema", got)
	}
	if got := responseFormatKind(nil); got != "" {
		t.Errorf("responseFormatKind(nil) = %q, want empty", got)
	}
}
