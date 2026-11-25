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
	"container/list"
	"context"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// prefixCacheConfig enables and bounds the conversation-prefix render cache.
// The cache keys rendered prompts by prefix-chain node, so a request whose
// conversation repeats a previously rendered prefix publishes the cached
// prefix tokens without a render round trip, and renders only the new suffix
// asynchronously so the next turn hits the cache in full.
type prefixCacheConfig struct {
	// Enabled turns the cache on. The default is off.
	Enabled bool `json:"enabled,omitempty"`
	// MaxEntries bounds the number of cached chain nodes (LRU). Default 4096.
	MaxEntries int `json:"maxEntries,omitempty"`
	// MaxBytes bounds the cached token IDs plus overhead (LRU). Default 256MiB.
	MaxBytes int64 `json:"maxBytes,omitempty"`
	// AsyncWarm renders the full prompt off the scheduling path after a
	// partial hit, storing the result for the session's next turn.
	AsyncWarm *asyncWarmConfig `json:"asyncWarm,omitempty"`
}

// asyncWarmConfig bounds background renders issued after a partial hit.
type asyncWarmConfig struct {
	// Concurrency caps simultaneous background renders. Default 8.
	Concurrency int `json:"concurrency,omitempty"`
	// Timeout caps one background render, e.g. "15s". Default 15s.
	Timeout string `json:"timeout,omitempty"`

	// timeout is the parsed Timeout.
	timeout time.Duration
}

const (
	defaultPrefixCacheMaxEntries = 4096
	defaultPrefixCacheMaxBytes   = 256 << 20
	defaultAsyncWarmConcurrency  = 8
	defaultAsyncWarmTimeout      = 15 * time.Second
	// renderCacheEntryOverhead accounts for the entry struct and key beyond
	// the token-ID slice it holds.
	renderCacheEntryOverhead = 128
)

func (c *prefixCacheConfig) withDefaults() *prefixCacheConfig {
	d := *c
	if d.MaxEntries <= 0 {
		d.MaxEntries = defaultPrefixCacheMaxEntries
	}
	if d.MaxBytes <= 0 {
		d.MaxBytes = defaultPrefixCacheMaxBytes
	}
	if d.AsyncWarm == nil {
		d.AsyncWarm = &asyncWarmConfig{}
	}
	if d.AsyncWarm.Concurrency <= 0 {
		d.AsyncWarm.Concurrency = defaultAsyncWarmConcurrency
	}
	warmTimeout, err := parseHTTPDuration(d.AsyncWarm.Timeout, defaultAsyncWarmTimeout)
	if err != nil {
		warmTimeout = defaultAsyncWarmTimeout
	}
	d.AsyncWarm.timeout = warmTimeout
	return &d
}

// renderCache is an LRU of rendered prompts keyed by prefix-chain node.
// Entries are immutable once stored; lookups only read the token-ID slice.
type renderCache struct {
	mu         sync.Mutex
	entries    map[chainNode]*renderCacheEntry
	lru        list.List
	bytes      int64
	maxEntries int
	maxBytes   int64
}

type renderCacheEntry struct {
	key      chainNode
	tokenIDs []uint32
	mm       *tokenization.MultiModalFeatures
	size     int64
	elem     *list.Element
}

func newRenderCache(maxEntries int, maxBytes int64) *renderCache {
	return &renderCache{
		entries:    make(map[chainNode]*renderCacheEntry),
		maxEntries: maxEntries,
		maxBytes:   maxBytes,
	}
}

func entrySize(tokenIDs []uint32) int64 {
	return int64(len(tokenIDs))*4 + renderCacheEntryOverhead
}

// lookup returns the entry for node, or nil. A hit moves the entry to the
// LRU front.
func (c *renderCache) lookup(node chainNode) *renderCacheEntry {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[node]
	if !ok {
		return nil
	}
	c.lru.MoveToFront(entry.elem)
	return entry
}

// lookupLongest returns the deepest cached node and its depth (the number of
// committed messages). Depth 0 means no hit.
func (c *renderCache) lookupLongest(nodes []chainNode) (*renderCacheEntry, int) {
	for i := len(nodes) - 1; i >= 1; i-- {
		if entry := c.lookup(nodes[i]); entry != nil {
			return entry, i
		}
	}
	return nil, 0
}

// store inserts or replaces the entry for node, evicting least-recently-used
// entries while over either bound. Replacement swaps in a new entry object
// rather than mutating the existing one: readers hold entries after the lock
// is released, so entry contents are immutable after publication.
func (c *renderCache) store(node chainNode, tokenIDs []uint32, mm *tokenization.MultiModalFeatures) {
	c.mu.Lock()
	defer c.mu.Unlock()
	size := entrySize(tokenIDs)
	if existing, ok := c.entries[node]; ok {
		c.lru.Remove(existing.elem)
		delete(c.entries, node)
		c.bytes -= existing.size
	}
	entry := &renderCacheEntry{key: node, tokenIDs: tokenIDs, mm: mm, size: size}
	entry.elem = c.lru.PushFront(entry)
	c.entries[node] = entry
	c.bytes += size
	for (len(c.entries) > c.maxEntries || c.bytes > c.maxBytes) && c.lru.Len() > 1 {
		oldest := c.lru.Back()
		victim := oldest.Value.(*renderCacheEntry)
		c.lru.Remove(oldest)
		delete(c.entries, victim.key)
		c.bytes -= victim.size
		recordPrefixCacheEvent("", prefixCacheEventEvict)
	}
}

// prefixCachedRenderer wraps a tokenizer with the conversation-prefix render
// cache. Only chat-completions rendering is chained; the other protocols pass
// through unchanged.
type prefixCachedRenderer struct {
	inner       tokenizer
	pluginName  string
	cache       *renderCache
	warmSem     chan struct{}
	warmTimeout time.Duration
}

func newPrefixCachedRenderer(inner tokenizer, pluginName string, cfg *prefixCacheConfig) *prefixCachedRenderer {
	d := cfg.withDefaults()
	return &prefixCachedRenderer{
		inner:       inner,
		pluginName:  pluginName,
		cache:       newRenderCache(d.MaxEntries, d.MaxBytes),
		warmSem:     make(chan struct{}, d.AsyncWarm.Concurrency),
		warmTimeout: d.AsyncWarm.timeout,
	}
}

// RenderChat serves chat-completions rendering from the prefix cache. A full
// hit returns the stored tokens. A partial hit returns the cached prefix
// tokens — the new suffix cannot match any pod's KV index, so downstream
// prefix matching loses at most the suffix and the last partial block — and
// renders the full prompt in the background for the next turn. A miss renders
// synchronously and stores the result.
func (r *prefixCachedRenderer) RenderChat(ctx context.Context, payload fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	chain, ok := chatPrefixChainFromPayload(payload)
	if !ok || len(chain.nodes) < 2 {
		recordPrefixCacheEvent(r.pluginName, prefixCacheEventBypass)
		return r.inner.RenderChat(ctx, payload)
	}
	if entry, depth := r.cache.lookupLongest(chain.nodes); entry != nil {
		if depth == len(chain.nodes)-1 {
			recordPrefixCacheEvent(r.pluginName, prefixCacheEventHitFull)
			return entry.tokenIDs, entry.mm, nil
		}
		recordPrefixCacheEvent(r.pluginName, prefixCacheEventHitPartial)
		if chain.textOnly && r.tryAcquireWarm() {
			go r.warmAsync(ctx, chain, payload)
		}
		return entry.tokenIDs, entry.mm, nil
	}
	recordPrefixCacheEvent(r.pluginName, prefixCacheEventMiss)
	tokenIDs, mm, err := r.inner.RenderChat(ctx, payload)
	if err == nil {
		r.cache.store(chain.nodes[len(chain.nodes)-1], tokenIDs, mm)
		recordPrefixCacheEvent(r.pluginName, prefixCacheEventStore)
	}
	return tokenIDs, mm, err
}

// Render passes through: completions prompts carry no turn boundary.
func (r *prefixCachedRenderer) Render(ctx context.Context, payload fwkrequest.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
	return r.inner.Render(ctx, payload)
}

// RenderMessages passes through: Anthropic messages are not chained yet.
func (r *prefixCachedRenderer) RenderMessages(ctx context.Context, payload fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return r.inner.RenderMessages(ctx, payload)
}

// RenderResponses passes through: Responses are not chained yet.
func (r *prefixCachedRenderer) RenderResponses(ctx context.Context, payload fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return r.inner.RenderResponses(ctx, payload)
}

// produceTimeout keeps the renderBackend's timeout delegation working through
// the wrapper.
func (r *prefixCachedRenderer) produceTimeout() time.Duration {
	if ta, ok := r.inner.(timeoutAware); ok {
		return ta.produceTimeout()
	}
	return 0
}

// tryAcquireWarm reserves a background-render slot without blocking.
func (r *prefixCachedRenderer) tryAcquireWarm() bool {
	select {
	case r.warmSem <- struct{}{}:
		return true
	default:
		return false
	}
}

// warmAsync renders the full prompt off the request path and stores the
// result under the chain's deepest node. WithoutCancel detaches the request's
// deadline and cancellation — the warm render must outlive PickEndpoint —
// while keeping its values, so the logger and Authorization header carry
// over. Multimodal payloads never reach this path; their metadata lives in
// request-scoped context values that a detached context cannot reconstruct
// once the request ends.
func (r *prefixCachedRenderer) warmAsync(ctx context.Context, chain chatPrefixChain, payload fwkrequest.RequestPayload) {
	defer func() { <-r.warmSem }()
	recordPrefixCacheEvent(r.pluginName, prefixCacheEventWarm)
	warmCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), r.warmTimeout)
	defer cancel()
	warmCtx = withUnobservedRender(warmCtx)
	tokenIDs, mm, err := r.inner.RenderChat(warmCtx, payload)
	if err != nil {
		recordPrefixCacheEvent(r.pluginName, prefixCacheEventWarmFailure)
		log.FromContext(warmCtx).V(logutil.DEBUG).Info("token-producer prefix-cache background render failed", "error", err)
		return
	}
	r.cache.store(chain.nodes[len(chain.nodes)-1], tokenIDs, mm)
	recordPrefixCacheEvent(r.pluginName, prefixCacheEventStore)
}
