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

package proxy

import (
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// requestFieldPromptTokenIDs is the kv_transfer_params key vLLM's chat path reads to skip
// templating and tokenization (vLLM #48145).
const requestFieldPromptTokenIDs = "prompt_token_ids"

// clientPromptTokenIDs returns the client-supplied kv_transfer_params.prompt_token_ids, or nil.
func clientPromptTokenIDs(body map[string]any) any {
	params, ok := body[reqcommon.FieldKVTransferParams].(map[string]any)
	if !ok {
		return nil
	}
	return params[requestFieldPromptTokenIDs]
}

// setPromptTokenIDs adds ids to params unless params already carries its own.
func setPromptTokenIDs(params map[string]any, ids any) {
	if ids == nil || params == nil {
		return
	}
	if _, exists := params[requestFieldPromptTokenIDs]; !exists {
		params[requestFieldPromptTokenIDs] = ids
	}
}
