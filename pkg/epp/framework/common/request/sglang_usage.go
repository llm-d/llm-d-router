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

import (
	"bytes"
	"encoding/json"
)

// SGLangMetaInfoKey is the field carrying usage in an SGLang /generate response.
const SGLangMetaInfoKey = "meta_info"

// SGLangMetaInfo holds the usage fields of an SGLang /generate response.
type SGLangMetaInfo struct {
	FinishReason     json.RawMessage `json:"finish_reason"`
	PromptTokens     *int            `json:"prompt_tokens"`
	CompletionTokens *int            `json:"completion_tokens"`
	CachedTokens     *int            `json:"cached_tokens"`
}

// Finished reports whether an SSE event ended generation.
func (m SGLangMetaInfo) Finished() bool {
	reason := bytes.TrimSpace(m.FinishReason)
	return len(reason) != 0 && !bytes.Equal(reason, []byte("null"))
}

// TokenUsage returns the reported usage, or nil when both token counts are absent.
func (m SGLangMetaInfo) TokenUsage() *Usage {
	if m.PromptTokens == nil && m.CompletionTokens == nil {
		return nil
	}
	u := &Usage{}
	if m.PromptTokens != nil {
		u.PromptTokens = *m.PromptTokens
	}
	if m.CompletionTokens != nil {
		u.CompletionTokens = *m.CompletionTokens
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	if m.CachedTokens != nil {
		u.PromptTokenDetails = &PromptTokenDetails{CachedTokens: *m.CachedTokens}
	}
	return u
}
