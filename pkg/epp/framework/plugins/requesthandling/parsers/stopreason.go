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
	"strings"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// NormalizeStopReason maps native stop values from the supported providers into
// the shared vocabulary. An unrecognized value is unknown.
func NormalizeStopReason(reason string) fwkrh.StopReason {
	switch strings.ToLower(strings.TrimSpace(reason)) {
	case "tool_call", "tool_calls", "tool-calls", "tool_use", "function_call":
		return fwkrh.StopReasonToolCall
	case "stop", "end", "length", "end_turn", "stop_sequence", "max_tokens",
		"pause_turn", "model_context_window_exceeded", "completed", "incomplete":
		return fwkrh.StopReasonTurnEnded
	default:
		return fwkrh.StopReasonUnknown
	}
}
