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
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"hash"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"

	"github.com/hashicorp/golang-lru/v2/expirable"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// Bidirectional KV transfer lets the prefill engine read the KV blocks the
// decode engine still holds from the previous turn of a conversation instead of
// recomputing that history. The decode engine returns the blocks' location in
// the kv_transfer_params of its response; the sidecar keeps it until the
// conversation's next request and replays it on that request's prefill leg.
//
// The NIXL pull is positional. The prefill engine copies the remote blocks over
// the leading blocks of the new prompt and never compares token contents, so
// the sidecar replays a cached entry only to a request whose messages extend,
// field for field, the messages and generated reply of the turn that produced
// it. The entry is addressed by a digest of that history and of the request
// fields that shape the prompt; no client-supplied conversation identifier is
// involved. An unrelated conversation, a different tenant (cache_salt), edited
// history, or a changed tool set hashes to another key and misses.

// kv_transfer_params fields of a NIXL decode response (vLLM
// NixlPullConnectorScheduler.request_finished) that the next turn's prefill
// request replays. reqcommon declares the remote_* fields the sidecar already
// writes itself.
const (
	requestFieldRemoteRequestID        = "remote_request_id"
	requestFieldRemoteNumTokens        = "remote_num_tokens"
	requestFieldRemoteBlocksExpiryTime = "remote_blocks_expiry_time"
	requestFieldTPSize                 = "tp_size"
	requestFieldDCPSize                = "dcp_size"
	requestFieldPPSize                 = "pp_size"
	requestFieldTransferMode           = "transfer_mode"
)

// Chat message fields shared by the request-side canonical form and the
// response-side reply reconstruction.
const (
	messageFieldName       = "name"
	messageFieldToolCalls  = "tool_calls"
	messageFieldToolCallID = "tool_call_id"
	toolCallFieldID        = "id"
	toolCallFieldType      = "type"
	toolCallFieldFunction  = "function"
	toolCallFieldName      = "name"
	toolCallFieldArguments = "arguments"
	toolCallTypeFunction   = "function"
)

// bidirectionalKVParamFields lists the decode-response fields copied into the
// prefill request. do_remote_prefill and do_remote_decode are excluded: the
// sidecar sets both for the prefill leg itself.
var bidirectionalKVParamFields = []string{
	reqcommon.FieldRemoteEngineID,
	requestFieldRemoteRequestID,
	reqcommon.FieldRemoteHost,
	reqcommon.FieldRemotePort,
	reqcommon.FieldRemoteBlockIDs,
	requestFieldRemoteNumTokens,
	requestFieldRemoteBlocksExpiryTime,
	requestFieldTPSize,
	requestFieldDCPSize,
	requestFieldPPSize,
	requestFieldTransferMode,
}

// kvReuseScopeFields are the request fields, besides the messages, that change
// the prompt tokens or the KV they produce. Two requests share cached blocks
// only when all of them are equal.
var kvReuseScopeFields = []string{
	reqcommon.FieldModel,
	"cache_salt",
	"tools",
	"tool_choice",
	"chat_template",
	"chat_template_kwargs",
	"documents",
	"reasoning_effort",
	"add_special_tokens",
	reqcommon.FieldAddGenerationPrompt,
	reqcommon.FieldContinueFinalMessage,
	reqcommon.FieldMMProcessorKwargs,
	reqcommon.FieldMediaIOKwargs,
}

// fieldTruncatePromptTokens shifts token positions when it truncates, so a
// request that sets it takes no part.
const fieldTruncatePromptTokens = "truncate_prompt_tokens"

// kvReuseEntry is one cached decode response. used makes the entry single-use
// even when two requests with identical history race for it.
type kvReuseEntry struct {
	params map[string]any
	used   atomic.Bool
}

// newKVReuseCache returns the cache of decode-side kv_transfer_params shared by
// a server and its data-parallel rank clones. Entries expire with the engine's
// decoder KV block TTL, after which the blocks they name no longer exist.
func newKVReuseCache(config Config) *expirable.LRU[string, *kvReuseEntry] {
	if !config.BidirectionalKVXfer {
		return nil
	}
	return expirable.NewLRU[string, *kvReuseEntry](config.BidirectionalCacheSize, nil, config.BidirectionalCacheTTL)
}

// sessionTargetsThisPod reports whether the request carries an EPP session
// token naming an endpoint of this pod. The EPP's encoded-endpoint strategy
// writes base64("<namespace>/<pod>-rank-<n>") for the endpoint it routed to
// (routing.EndpointName) on its response, and a client echoes it on later turns.
// Anyone can construct the token, so it marks a request as a routed follow-up
// of a conversation served here and does not authenticate the caller. Isolation
// between conversations comes from the history digest.
func (s *Server) sessionTargetsThisPod(r *http.Request) bool {
	if s.config.PodName == "" || s.config.PodNamespace == "" {
		return false
	}
	token := r.Header.Get(s.config.BidirectionalSessionHeader)
	if token == "" {
		return false
	}
	decoded, err := base64.StdEncoding.DecodeString(token)
	if err != nil {
		return false
	}
	namespace, name, found := strings.Cut(string(decoded), "/")
	if !found || namespace != s.config.PodNamespace {
		return false
	}
	pod, rank, ok := routing.ParseEndpointName(name)
	return ok && pod == s.config.PodName && rank < max(s.config.DataParallelSize, 1)
}

// kvReuse is the bidirectional KV transfer state of one chat completions request.
type kvReuse struct {
	cache *expirable.LRU[string, *kvReuseEntry]
	scope []byte
	// messages holds the decoded chat messages.
	messages []any
	// routed is true when the request carries a session token for this pod. Only
	// a routed request receives a replay. Every eligible request is stored, since
	// the first turn of a conversation precedes the response that issues the token.
	routed bool
}

// newKVReuse returns the request's bidirectional KV transfer state, or nil when
// the request cannot take part: the feature is off, the API is not Chat
// Completions, the request is not a single-choice chat with a messages array,
// or it truncates the prompt.
func (s *Server) newKVReuse(r *http.Request, body map[string]any, apiType reqcommon.APIType) *kvReuse {
	if !s.config.BidirectionalKVXfer || s.kvReuseCache == nil || s.config.MoRIIOWriteMode {
		return nil
	}
	if apiType != reqcommon.APITypeChatCompletions {
		return nil
	}
	if v, present := body["n"]; present {
		n, ok := plainJSON(v)
		if !ok || (n != nil && !isNumberOne(n)) {
			return nil
		}
	}
	if v, present := body[fieldTruncatePromptTokens]; present {
		if truncate, ok := plainJSON(v); !ok || truncate != nil {
			return nil
		}
	}
	messages, ok := decodedMessages(body[reqcommon.FieldMessages])
	if !ok || len(messages) == 0 {
		return nil
	}
	scopeFields := make(map[string]any, len(kvReuseScopeFields))
	for _, field := range kvReuseScopeFields {
		v, present := body[field]
		if !present {
			continue
		}
		decoded, ok := plainJSON(v)
		if !ok {
			return nil
		}
		scopeFields[field] = decoded
	}
	// Marshaling sorts map keys, so the digest does not depend on how the client
	// ordered or spaced the JSON.
	scope, err := json.Marshal(scopeFields)
	if err != nil {
		return nil
	}
	return &kvReuse{cache: s.kvReuseCache, scope: scope, messages: messages, routed: s.sessionTargetsThisPod(r)}
}

// plainJSON returns v as ordinary decoded JSON. decodeRequestBody leaves every
// request field it does not inspect as a json.RawMessage.
func plainJSON(v any) (any, bool) {
	raw, ok := v.(json.RawMessage)
	if !ok {
		return v, true
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	// Numbers keep their written form, so the digest does not depend on float formatting.
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return nil, false
	}
	return out, true
}

// decodedMessages returns the request's messages as decoded JSON values.
func decodedMessages(v any) ([]any, bool) {
	if raws, ok := v.([]json.RawMessage); ok {
		out := make([]any, len(raws))
		for i, raw := range raws {
			decoded, ok := plainJSON(raw)
			if !ok {
				return nil, false
			}
			out[i] = decoded
		}
		return out, true
	}
	decoded, ok := plainJSON(v)
	if !ok {
		return nil, false
	}
	messages, ok := decoded.([]any)
	return messages, ok
}

func isNumberOne(v any) bool {
	switch n := v.(type) {
	case float64:
		return n == 1
	case json.Number:
		return n.String() == "1"
	}
	return false
}

// take returns the cached decode-side params of the longest earlier turn this
// request extends, or nil. An entry is single-use: the prefill engine's read
// releases the blocks it names on the decode engine, so a second request
// replaying the same entry would point at freed blocks.
func (k *kvReuse) take() map[string]any {
	if !k.routed {
		return nil
	}
	for _, key := range k.lookupKeys() {
		if entry, ok := k.cache.Get(key); ok && entry.used.CompareAndSwap(false, true) {
			k.cache.Remove(key)
			return entry.params
		}
	}
	return nil
}

// store records the decode-side params of this turn under the digest of the
// history a follow-up request must carry: this request's messages followed by
// the reply the decoder generated.
func (k *kvReuse) store(c *decodeCapture) bool {
	params, reply, ok := c.result()
	if !ok {
		return false
	}
	h := k.newHash()
	for _, raw := range k.messages {
		if !hashMessage(h, raw) {
			return false
		}
	}
	if !hashMessage(h, reply) {
		return false
	}
	k.cache.Add(hex.EncodeToString(h.Sum(nil)), &kvReuseEntry{params: params})
	return true
}

func (k *kvReuse) newHash() hash.Hash {
	h := sha256.New()
	h.Write(k.scope)
	h.Write([]byte{'\n'})
	return h
}

// lookupKeys returns the keys under which an earlier turn of this conversation
// could have stored its params, longest history first. A follow-up request
// carries the earlier turn's messages, that turn's assistant reply, and at
// least one new message, so the candidates are the prefixes that end in an
// assistant message other than the last one.
func (k *kvReuse) lookupKeys() []string {
	h := k.newHash()
	var keys []string
	for i, raw := range k.messages {
		if i == len(k.messages)-1 {
			break
		}
		if !hashMessage(h, raw) {
			return nil
		}
		if m, _ := raw.(map[string]any); m[reqcommon.FieldRole] == roleAssistant {
			keys = append(keys, hex.EncodeToString(h.Sum(nil)))
		}
	}
	slices.Reverse(keys)
	return keys
}

// injectBidirectionalKVParams copies the allowlisted decode-response fields
// into the prefill request's kv_transfer_params.
func injectBidirectionalKVParams(dst, cached map[string]any) {
	for _, field := range bidirectionalKVParamFields {
		if v, ok := cached[field]; ok {
			dst[field] = v
		}
	}
}

// dropBidirectionalKVParams restores the prefill request's kv_transfer_params to
// the form it has without a replay. A retried prefill must not resend an entry
// whose blocks the first attempt may already have read and released.
func dropBidirectionalKVParams(kv map[string]any) {
	for _, field := range bidirectionalKVParamFields {
		switch field {
		case reqcommon.FieldRemoteEngineID, reqcommon.FieldRemoteBlockIDs, reqcommon.FieldRemoteHost, reqcommon.FieldRemotePort:
			kv[field] = nil
		default:
			delete(kv, field)
		}
	}
}

// completeKVParams validates a decode response's kv_transfer_params against what
// vLLM's NIXL pull scheduler requires to treat a prefill request as a
// decode-side block read (remote_block_ids plus remote_engine_id,
// remote_request_id, remote_host and remote_port) and returns the allowlisted
// subset.
func completeKVParams(params map[string]any) (map[string]any, bool) {
	if decode, _ := params[reqcommon.FieldDoRemoteDecode].(bool); !decode {
		return nil, false
	}
	for _, field := range []string{reqcommon.FieldRemoteEngineID, requestFieldRemoteRequestID, reqcommon.FieldRemoteHost} {
		if v, _ := params[field].(string); v == "" {
			return nil, false
		}
	}
	switch params[reqcommon.FieldRemotePort].(type) {
	case json.Number, float64:
	default:
		return nil, false
	}
	if ids, _ := params[reqcommon.FieldRemoteBlockIDs].([]any); len(ids) == 0 {
		return nil, false
	}
	out := make(map[string]any, len(bidirectionalKVParamFields))
	injectBidirectionalKVParams(out, params)
	return out, true
}

// canonicalMessage is the part of a chat message that determines the prompt
// tokens. The decoder's reply and the client's echo of it carry different extra
// fields (refusal, annotations), so both are reduced to this form before
// hashing.
type canonicalMessage struct {
	Role       string              `json:"role"`
	Name       string              `json:"name,omitempty"`
	Content    any                 `json:"content,omitempty"`
	ToolCalls  []canonicalToolCall `json:"tool_calls,omitempty"`
	ToolCallID string              `json:"tool_call_id,omitempty"`
}

type canonicalToolCall struct {
	ID        string `json:"id,omitempty"`
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

// hashMessage writes the canonical form of one chat message to h. It reports
// false for a message it cannot canonicalize, which disables reuse for the
// request.
func hashMessage(h hash.Hash, raw any) bool {
	m, ok := raw.(map[string]any)
	if !ok {
		return false
	}
	role, _ := m[reqcommon.FieldRole].(string)
	if role == "" {
		return false
	}
	c := canonicalMessage{Role: role}
	c.Name, _ = m[messageFieldName].(string)
	c.ToolCallID, _ = m[messageFieldToolCallID].(string)
	switch content := m[reqcommon.FieldContent].(type) {
	case nil:
	case string:
		if content != "" {
			c.Content = content
		}
	default:
		c.Content = content
	}
	if calls, present := m[messageFieldToolCalls]; present && calls != nil {
		list, ok := calls.([]any)
		if !ok {
			return false
		}
		for _, rawCall := range list {
			call, ok := rawCall.(map[string]any)
			if !ok {
				return false
			}
			fn, _ := call[toolCallFieldFunction].(map[string]any)
			id, _ := call[toolCallFieldID].(string)
			name, _ := fn[toolCallFieldName].(string)
			c.ToolCalls = append(c.ToolCalls, canonicalToolCall{ID: id, Name: name, Arguments: fn[toolCallFieldArguments]})
		}
	}
	b, err := json.Marshal(c)
	if err != nil {
		return false
	}
	h.Write(b)
	h.Write([]byte{'\n'})
	return true
}
