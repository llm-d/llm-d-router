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
	"bytes"
	"encoding/json"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// clientPromptTokenIDs returns the client-supplied kv_transfer_params.prompt_token_ids, or nil.
// The IDs stay as json.RawMessage when kv_transfer_params is still serialized: this is the only
// reader of that field in the sidecar, and keeping it raw avoids materializing a long token array
// (one boxed value per token) only for the connectors to serialize it right back.
// Returns nil unless the server's ForwardPromptTokenIDs flag is set: vLLM serves
// the IDs instead of rendering messages, so an untrusted caller must not be
// able to inject them.
func (s *Server) clientPromptTokenIDs(body map[string]any) any {
	if !s.config.ForwardPromptTokenIDs {
		return nil
	}
	return rawClientPromptTokenIDs(body)
}

func rawClientPromptTokenIDs(body map[string]any) any {
	switch kv := body[reqcommon.FieldKVTransferParams].(type) {
	case map[string]any:
		return kv[reqcommon.FieldPromptTokenIDs]
	case json.RawMessage:
		var params map[string]json.RawMessage
		if json.Unmarshal(kv, &params) != nil {
			return nil
		}
		ids, ok := params[reqcommon.FieldPromptTokenIDs]
		if !ok || bytes.Equal(bytes.TrimSpace(ids), []byte("null")) {
			return nil
		}
		return ids
	default:
		return nil
	}
}

// setPromptTokenIDs adds ids to params unless params already carries its own.
func setPromptTokenIDs(params map[string]any, ids any) {
	if ids == nil || params == nil {
		return
	}
	if _, exists := params[reqcommon.FieldPromptTokenIDs]; !exists {
		params[reqcommon.FieldPromptTokenIDs] = ids
	}
}
