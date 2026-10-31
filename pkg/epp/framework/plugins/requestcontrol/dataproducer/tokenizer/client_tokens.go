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

package tokenizer

import (
	"encoding/json"
	"math"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

// clientPromptTokenIDs returns the prompt token IDs a request already carries,
// so the backend does not render the prompt again: a token-array completions
// prompt, or kv_transfer_params.prompt_token_ids, which vLLM's chat path
// consumes in place of templating and tokenizing the messages. Returns nil
// when the request carries none.
func clientPromptTokenIDs(body *fwkrh.InferenceRequestBody) [][]uint32 {
	if body.Completions != nil {
		// The completions path reads the prompt; it does not read
		// kv_transfer_params.prompt_token_ids, so a text prompt must be rendered.
		if len(body.Completions.Prompt.TokenIDs) > 0 {
			return body.Completions.Prompt.TokenIDs
		}
		return nil
	}
	payload, ok := body.Payload.(fwkrh.PayloadMap)
	if !ok {
		return nil
	}
	value, ok := payload[reqcommon.FieldKVTransferParams]
	if !ok || value == nil {
		return nil
	}
	var params map[string]any
	switch v := value.(type) {
	case map[string]any:
		params = v
	case json.RawMessage:
		if err := parserutil.Unmarshal(v, &params); err != nil {
			return nil
		}
	default:
		return nil
	}
	if ids := parseTokenIDList(params[reqcommon.FieldPromptTokenIDs]); ids != nil {
		return [][]uint32{ids}
	}
	return nil
}

// parseTokenIDList converts a prompt_token_ids value to token IDs. The value is
// raw JSON when the envelope kept kv_transfer_params opaque, a decoded list
// whose numbers are json.Number or float64, or a []uint32 written by a
// mutation. Returns nil for an absent, empty, or malformed value, so
// tokenization proceeds normally.
func parseTokenIDList(value any) []uint32 {
	switch v := value.(type) {
	case []uint32:
		if len(v) > 0 {
			return v
		}
	case json.RawMessage:
		if ids, ok := fwkrh.ParseCanonicalTokenIDs(v); ok {
			return ids
		}
	case []any:
		ids := make([]uint32, 0, len(v))
		for _, item := range v {
			id, ok := tokenID(item)
			if !ok {
				return nil
			}
			ids = append(ids, id)
		}
		if len(ids) > 0 {
			return ids
		}
	}
	return nil
}

// tokenID converts one decoded JSON number to a token ID.
func tokenID(v any) (uint32, bool) {
	var f float64
	switch n := v.(type) {
	case json.Number:
		parsed, err := n.Float64()
		if err != nil {
			return 0, false
		}
		f = parsed
	case float64:
		f = n
	default:
		return 0, false
	}
	if f < 0 || f != math.Trunc(f) || f > math.MaxUint32 {
		return 0, false
	}
	return uint32(f), true
}
