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

package request

const (
	FieldMaxTokens           = "max_tokens"
	FieldMaxCompletionTokens = "max_completion_tokens"
	FieldStream              = "stream"
	FieldStreamOptions       = "stream_options"
)

// PrimeSingleTokenRequest mutates target in place into a synthetic,
// non-streaming, single-output-token chat-completions request derived from
// original (which may be the same map as target). max_tokens is always
// capped to 1; max_completion_tokens is only added when original already
// carries it, and leaves it untached otherwise.
func PrimeSingleTokenRequest(target, original map[string]any) {
	target[FieldMaxTokens] = 1
	if _, ok := original[FieldMaxCompletionTokens]; ok {
		target[FieldMaxCompletionTokens] = 1
	}

	target[FieldStream] = false
	delete(target, FieldStreamOptions)
}
