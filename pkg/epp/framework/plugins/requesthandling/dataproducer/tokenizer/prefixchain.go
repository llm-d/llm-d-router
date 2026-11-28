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
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"sort"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

// chainNode is a SHA-256 digest committing to the render salt and the first i
// messages of a chat-completions payload. Agentic turn N+1 repeats turn N's
// conversation with appended messages, so its node over the first i messages
// equals turn N's node i: a cache keyed by node recognizes the shared prefix
// without re-rendering it.
type chainNode [sha256.Size]byte

// chainFieldSeparator ends each hashed field so adjacent fields cannot run
// together, mirroring the conversation-prefix writer's separator.
var chainFieldSeparator = []byte{0}

// renderSaltFields are the top-level chat-completions fields that change how
// the renderer templates the prompt. Generation-only fields (max_tokens,
// temperature, ...) are deliberately excluded: they vary per request without
// affecting the rendered prompt, and salting them would break every chain.
// When a renderer gains a new template input, add it here.
var renderSaltFields = map[string]struct{}{
	"model":                {},
	"tools":                {},
	"tool_choice":          {},
	"response_format":      {},
	"chat_template_kwargs": {},
	"reasoning_effort":     {},
	"enable_thinking":      {},
	"thinking":             {},
	"parallel_tool_calls":  {},
	"modalities":           {},
	"audio":                {},
}

// multimodalMarkers are raw-byte markers of multimodal content parts in a
// chat-completions message. Escaped text content cannot match them, because
// an escaped key appears as `\"...\"`, never as the bare quoted key scanned
// for here. A missed marker can only cause a needless synchronous render;
// a false match only skips an async warm.
var multimodalMarkers = [][]byte{
	[]byte(`"image_url"`),
	[]byte(`"input_audio"`),
	[]byte(`"input_video"`),
	[]byte(`"video_url"`),
	[]byte(`"file_id"`),
	[]byte(`"file"`),
}

// chatPrefixChain is the per-request chain of prefix nodes plus the payload's
// multimodal classification.
type chatPrefixChain struct {
	// nodes[i] commits to the salt and messages[:i]; nodes[0] commits to the
	// salt alone and is never a cache key.
	nodes []chainNode
	// textOnly reports that no message carries a known multimodal marker.
	textOnly bool
}

// chatPrefixChainFromPayload derives the chain from the payload the renderer
// receives. ok is false when the payload cannot be reduced to bytes or is not
// a JSON object carrying a messages array of objects.
func chatPrefixChainFromPayload(payload fwkrequest.RequestPayload) (chatPrefixChain, bool) {
	var raw []byte
	switch p := payload.(type) {
	case fwkrequest.RawPayload:
		raw = p
	case fwkrequest.PayloadBytes:
		raw = p
	case fwkrequest.PayloadMap:
		b, err := p.Marshal()
		if err != nil {
			return chatPrefixChain{}, false
		}
		raw = b
	default:
		return chatPrefixChain{}, false
	}
	return chatPrefixChainFromBytes(raw)
}

// chatPrefixChainFromBytes walks the wire bytes with a streaming decoder: the
// messages array is captured as raw per-message bytes (no interior parsing)
// and the render-salt fields as raw values. The chain root hashes the sorted
// salt fields so key order in the envelope cannot change it.
func chatPrefixChainFromBytes(raw []byte) (chatPrefixChain, bool) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return chatPrefixChain{}, false
	}
	var messages []json.RawMessage
	salts := make(map[string]json.RawMessage)
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return chatPrefixChain{}, false
		}
		key, ok := keyTok.(string)
		if !ok {
			return chatPrefixChain{}, false
		}
		if key == "messages" {
			var arr []json.RawMessage
			if err := dec.Decode(&arr); err != nil {
				return chatPrefixChain{}, false
			}
			messages = arr
			continue
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return chatPrefixChain{}, false
		}
		if _, affects := renderSaltFields[key]; affects {
			salts[key] = val
		}
	}
	if _, err := dec.Token(); err != nil { // closing '}'
		return chatPrefixChain{}, false
	}
	if messages == nil {
		return chatPrefixChain{}, false
	}

	root := sha256.New()
	keys := make([]string, 0, len(salts))
	for k := range salts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		root.Write([]byte(k))
		root.Write(chainFieldSeparator)
		root.Write(salts[k])
		root.Write(chainFieldSeparator)
	}
	var rootNode chainNode
	copy(rootNode[:], root.Sum(nil))

	// Each decoded message occupies at least two bytes of raw ("{}" plus a
	// separator), so a message count beyond half the input cannot be genuine.
	// The check bounds the node slice without arithmetic on len(messages),
	// which would risk overflow in the allocation size.
	if len(messages) > len(raw)/2 {
		return chatPrefixChain{}, false
	}
	nodes := make([]chainNode, 0, len(messages))
	nodes = append(nodes, rootNode)
	textOnly := true
	var length [8]byte
	for i, msg := range messages {
		if len(bytes.TrimLeft(msg, " \t\r\n")) == 0 || bytes.TrimLeft(msg, " \t\r\n")[0] != '{' {
			return chatPrefixChain{}, false
		}
		if textOnly && containsMultimodalMarker(msg) {
			textOnly = false
		}
		h := sha256.New()
		h.Write(nodes[i][:])
		binary.LittleEndian.PutUint64(length[:], uint64(len(msg)))
		h.Write(length[:])
		h.Write(msg)
		var node chainNode
		copy(node[:], h.Sum(nil))
		nodes = append(nodes, node)
	}
	return chatPrefixChain{nodes: nodes, textOnly: textOnly}, true
}

// containsMultimodalMarker reports whether raw message bytes carry a known
// multimodal content-part key.
func containsMultimodalMarker(msg json.RawMessage) bool {
	for _, marker := range multimodalMarkers {
		if bytes.Contains(msg, marker) {
			return true
		}
	}
	return false
}
