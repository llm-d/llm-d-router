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

package disaggregatedsetrollout

import (
	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// affinityDraw returns a deterministic draw in [0, 1) for the request, and
// false when affinity is off or the request has nothing to key on.
//
// pickWeightedRevision lays the revisions out over [0, 1) in a fixed order, so
// with two revisions a shift in shares only moves the keys in the slice of
// [0, 1) between the old and new boundary, and always toward the revision that
// gained share. Every other key keeps its revision and its KV cache.
func (c *Screener) affinityDraw(request *fwksched.InferenceRequest) (float64, bool) {
	if c.config.RevisionGating.affinityMode() != AffinityModePrefix || request == nil {
		return 0, false
	}
	digest := xxhash.New()
	if header := c.config.RevisionGating.Affinity.SessionHeader; header != "" {
		if session := request.Headers[header]; session != "" {
			_, _ = digest.WriteString(session)
			return drawFromHash(digest.Sum64()), true
		}
	}
	if !writeConversationPrefix(digest, request.Body) {
		return 0, false
	}
	return drawFromHash(digest.Sum64()), true
}

// fieldSeparator ends each hashed field so adjacent fields cannot run together.
var fieldSeparator = []byte{0}

// drawFromHash maps a 64-bit hash onto [0, 1) using its top 53 bits, the
// precision of a float64 mantissa.
func drawFromHash(sum uint64) float64 {
	return float64(sum>>11) / (1 << 53)
}

// writeConversationPrefix hashes the conversation up to and including its
// first user message. Every later turn of a session repeats that prefix, so
// the turns share one key, while sessions that share only a system prompt
// still spread across revisions. It returns false for bodies it cannot key.
func writeConversationPrefix(digest *xxhash.Digest, body *fwkrh.InferenceRequestBody) bool {
	if body == nil {
		return false
	}
	switch {
	case body.ChatCompletions != nil:
		chat := body.ChatCompletions
		writeField(digest, chat.CacheSalt)
		wrote := false
		for _, message := range chat.Messages {
			writeField(digest, message.Role)
			writeField(digest, message.Content.PlainText())
			wrote = true
			if message.Role == "user" {
				break
			}
		}
		return wrote
	case body.Messages != nil:
		messages := body.Messages
		writeField(digest, messages.CacheSalt)
		writeAnthropicContent(digest, messages.System)
		wrote := false
		for _, message := range messages.Messages {
			writeField(digest, message.Role)
			writeAnthropicContent(digest, message.Content)
			wrote = true
			if message.Role == "user" {
				break
			}
		}
		return wrote
	default:
		// Completions carry no turn boundary, so a fixed-length prefix would
		// collapse every session behind a long shared preamble onto one key.
		return false
	}
}

// writeField writes s followed by fieldSeparator.
func writeField(digest *xxhash.Digest, s string) {
	_, _ = digest.WriteString(s)
	_, _ = digest.Write(fieldSeparator)
}

func writeAnthropicContent(digest *xxhash.Digest, content fwkrh.AnthropicContent) {
	_, _ = digest.WriteString(content.Raw)
	for _, block := range content.Structured {
		if block.Type == "text" {
			_, _ = digest.WriteString(block.Text)
		}
	}
	_, _ = digest.Write(fieldSeparator)
}
