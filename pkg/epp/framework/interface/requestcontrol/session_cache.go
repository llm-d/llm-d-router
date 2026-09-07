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

package requestcontrol

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// SessionCacheManager binds requests to externally managed sessions. Produce
// publishes SessionCacheRequest under its own producer name. Event callbacks
// may run concurrently with Produce. Session identity, continuation matching,
// report policy, and association retention belong to the manager.
type SessionCacheManager interface {
	DataProducer
	kvevents.EventConsumer
	// CacheNamespace resolves configured engine-hash compatibility for a
	// source and cache group. Empty means unconfigured. The mapping must be
	// stable until the source resets; group numbers are local to each source.
	CacheNamespace(kvevents.EventSource, *int) string
}

// SessionCacheRequestDataKey has no default producer; the manager is explicit.
var SessionCacheRequestDataKey = plugin.NewDataKey("SessionCacheRequest", "")

// SessionCacheRequest carries a manager's event stamp and cache lookup for one
// prompt. Absence or an empty stamp skips session lookup and request mutation.
type SessionCacheRequest struct {
	// Stamp is echoed in vLLM's session_id. The manager must distinguish
	// concurrent requests whose block observations cannot be combined.
	Stamp string
	// FullReport requests reused blocks as well as newly stored blocks.
	FullReport bool
	// TotalTokens is the full prompt length in engine-token units, measured
	// or estimated by the manager. It includes content beyond the known prefixes
	// and is authoritative for affinity and prefill-load accounting.
	TotalTokens int
	// Prefixes are alternative observations, never a union of session branches.
	Prefixes []SessionCachePrefix
}

// SessionCachePrefix lists consecutive engine blocks from a prompt's start.
// The manager restricts lookup to the request's tenant, model, and cache salt.
type SessionCachePrefix struct {
	// CacheNamespace identifies compatible model revisions, hash protocols
	// and seeds, block sizes, and cache-group semantics across sources.
	CacheNamespace  string
	BlockHashes     []uint64
	BlockSizeTokens int
	// Exact permits cached-token counts for P/D consumers. Leave false for
	// estimated content similarity or uncertain continuation relationships.
	Exact bool
}
