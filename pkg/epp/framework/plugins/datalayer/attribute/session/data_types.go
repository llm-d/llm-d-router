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

// Package session declares the SessionID attribute that carries per-request
// session identity for affinity scoring and filtering, and the
// SessionCacheRequest attribute that carries a session producer's engine-block
// prefixes for session prefix-cache lookup. Both are published once per
// request on the InferenceRequest attribute store.
package session

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid/constants"
)

// SessionIDDataKey identifies the session identifier published on the request
// attribute store. The default producer is the session-id-producer.
var SessionIDDataKey = plugin.NewDataKey("SessionIDDataKey", sessionidconstants.SessionIDProducerType)

// SessionID is the session identifier extracted from a request.
type SessionID string

// ReadSessionID returns the SessionID published by the default producer on the
// request attribute store, or "" and false if absent.
//
// Consumers should use this helper rather than reading the attribute directly:
// it encapsulates both the key construction and the type assertion, so a
// future change of storage location or value type does not ripple through
// every reader.
func ReadSessionID(r *fwksched.InferenceRequest) (SessionID, bool) {
	key := SessionIDDataKey.WithNonEmptyProducerName(sessionidconstants.SessionIDProducerType)
	return fwksched.ReadRequestAttribute[SessionID](r, key)
}

// SessionCacheRequestDataKey identifies the SessionCacheRequest published on
// the request attribute store. It has no default producer; the
// session-prefix-cache-producer names the producer it consumes.
var SessionCacheRequestDataKey = plugin.NewDataKey("SessionCacheRequestDataKey", "")

// SessionCacheRequest is a session producer's cache lookup for one request:
// the identity the engine reports this request's blocks under, and the
// candidate prompt prefixes whose residency the session-prefix-cache-producer
// resolves. Absence or an empty SessionID gives the request no session cache
// preference and leaves its body unchanged.
type SessionCacheRequest struct {
	// SessionID is written to the request body's session_id field, replacing
	// any client value, and returns in the KV events for the blocks this
	// request stores. A producer that must tell concurrent requests of one
	// logical session apart uses a per-request value here.
	SessionID string
	// FullReport asks the engine to report reused blocks as well as newly
	// stored ones for this request.
	FullReport bool
	// TotalTokens is the full prompt length in engine-token units, measured
	// or estimated by the producer. It bounds matched coverage and is the
	// prompt length for load accounting.
	TotalTokens int
	// Prefixes are the candidate prompt prefixes for this request, typically
	// the block paths recorded for the session's earlier turns. Each candidate
	// is resolved on its own and the best match per endpoint is published;
	// candidates are never concatenated.
	Prefixes []SessionCachePrefix
}

// SessionCachePrefix is a run of engine block hashes from the start of a
// prompt, as the engine reports them in its KV events.
type SessionCachePrefix struct {
	// CacheNamespace identifies compatible model, hash algorithm, and cache
	// salt settings. It must match the cache producer configuration.
	CacheNamespace  string
	BlockHashes     []uint64
	BlockSizeTokens int
	// Exact asserts that the request's prompt begins with these blocks, so
	// matched blocks count as cached tokens for prefill-versus-decode
	// decisions. Leave it false for a similarity estimate or an uncertain
	// continuation: the match then contributes a routing affinity score only.
	Exact bool
}
