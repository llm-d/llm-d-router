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

package parsers

import (
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func TestNormalizeStopReason(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		want   fwkrh.StopReason
	}{
		{name: "tool_calls", reason: "tool_calls", want: fwkrh.StopReasonToolCall},
		{name: "tool_use", reason: "tool_use", want: fwkrh.StopReasonToolCall},
		{name: "function_call", reason: "function_call", want: fwkrh.StopReasonToolCall},
		{name: "tool_calls_hyphen", reason: "tool-calls", want: fwkrh.StopReasonToolCall},
		{name: "stop", reason: "stop", want: fwkrh.StopReasonTurnEnded},
		{name: "end", reason: "end", want: fwkrh.StopReasonTurnEnded},
		{name: "length", reason: "length", want: fwkrh.StopReasonTurnEnded},
		{name: "end_turn", reason: "end_turn", want: fwkrh.StopReasonTurnEnded},
		{name: "stop_sequence", reason: "stop_sequence", want: fwkrh.StopReasonTurnEnded},
		{name: "max_tokens", reason: "max_tokens", want: fwkrh.StopReasonTurnEnded},
		{name: "pause_turn", reason: "pause_turn", want: fwkrh.StopReasonTurnEnded},
		{name: "context_window", reason: "model_context_window_exceeded", want: fwkrh.StopReasonTurnEnded},
		{name: "completed", reason: "completed", want: fwkrh.StopReasonTurnEnded},
		{name: "incomplete", reason: "incomplete", want: fwkrh.StopReasonTurnEnded},
		{name: "abort", reason: "abort", want: fwkrh.StopReasonUnknown},
		{name: "error", reason: "error", want: fwkrh.StopReasonUnknown},
		{name: "repetition", reason: "repetition", want: fwkrh.StopReasonUnknown},
		{name: "content_filter", reason: "content_filter", want: fwkrh.StopReasonUnknown},
		{name: "empty", reason: "", want: fwkrh.StopReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NormalizeStopReason(tt.reason); got != tt.want {
				t.Fatalf("NormalizeStopReason(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}

func TestStopReasonOrUnknown(t *testing.T) {
	if got := fwkrh.StopReason("").OrUnknown(); got != fwkrh.StopReasonUnknown {
		t.Fatalf("empty.OrUnknown() = %q, want %q", got, fwkrh.StopReasonUnknown)
	}
	if got := fwkrh.StopReasonToolCall.OrUnknown(); got != fwkrh.StopReasonToolCall {
		t.Fatalf("tool_call.OrUnknown() = %q, want %q", got, fwkrh.StopReasonToolCall)
	}
}
