/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

const (
	// defaultCacheTTL is the duration for which an endpoint candidate lookup result is considered valid.
	// This trades off "Scale-from-Zero" responsiveness (latency to see new endpoints) against Datastore lock contention.
	// 50ms aligns roughly with standard Prometheus scrape intervals or high-frequency control loops.
	defaultCacheTTL = 50 * time.Millisecond

	// cleanupInterval dictates how often we sweep the map for expired entries.
	cleanupInterval = 1 * time.Minute

	// allEndpointsCacheKey is the single cache key used now that the Delegate
	// no longer varies its result based on request metadata. The Envoy subset
	// filter moved to the gwsubset Screener, which observes request
	// metadata during screening.
	allEndpointsCacheKey = "__all_endpoints__"
)

// --- DatastoreEndpointCandidates (The Delegate) ---

// DatastoreEndpointCandidates implements contracts.EndpointCandidates by querying the EPP Datastore.
type DatastoreEndpointCandidates struct {
	datastore Datastore
}

var _ contracts.EndpointCandidates = &DatastoreEndpointCandidates{}

// NewDatastoreEndpointCandidates creates a new DatastoreEndpointCandidates.
func NewDatastoreEndpointCandidates(ds Datastore) *DatastoreEndpointCandidates {
	return &DatastoreEndpointCandidates{datastore: ds}
}

// Locate returns every endpoint candidate in the Datastore. Per-request
// filtering (e.g. the destination-endpoint-subset hint) is handled by
// the gwsubset Screener, which runs after Locate.
func (d *DatastoreEndpointCandidates) Locate(_ context.Context, _ map[string]any) []fwkdl.Endpoint {
	return d.datastore.PodList(datastore.AllPodsPredicate)
}

// --- CachedEndpointCandidates (The Decorator) ---

// cacheEntry represents a snapshot of endpoint candidate metrics at a specific point in time.
type cacheEntry struct {
	pods   []fwkdl.Endpoint
	expiry time.Time
}

// CachedEndpointCandidates is a decorator for contracts.EndpointCandidates that caches results to reduce lock contention on the
// underlying Datastore.
//
// It is designed for high-throughput paths (like the Flow Control dispatch loop) where fetching fresh data every
// millisecond is unnecessary and expensive.
type CachedEndpointCandidates struct {
	// delegate is the underlying source of truth (usually the DatastoreEndpointCandidates).
	delegate contracts.EndpointCandidates

	// ttl defines how long a cache entry remains valid.
	ttl time.Duration

	// mu protects the cache map.
	mu    sync.RWMutex
	cache map[string]cacheEntry
}

var _ contracts.EndpointCandidates = &CachedEndpointCandidates{}

// NewCachedEndpointCandidates creates a new CachedEndpointCandidates and starts a background cleanup routine.
// The provided context is used to control the lifecycle of the cleanup goroutine.
func NewCachedEndpointCandidates(ctx context.Context, delegate contracts.EndpointCandidates, ttl time.Duration) *CachedEndpointCandidates {
	if ttl <= 0 {
		ttl = defaultCacheTTL
	}

	c := &CachedEndpointCandidates{
		delegate: delegate,
		ttl:      ttl,
		cache:    make(map[string]cacheEntry),
	}

	// Start background cleanup to prevent memory leaks from unused keys.
	go c.runCleanup(ctx)

	return c
}

// Locate returns the cached endpoint candidate list, refreshing it after the TTL expires.
func (c *CachedEndpointCandidates) Locate(ctx context.Context, _ map[string]any) []fwkdl.Endpoint {
	// Fast Path: Read Lock
	c.mu.RLock()
	entry, found := c.cache[allEndpointsCacheKey]
	c.mu.RUnlock()

	if found && time.Now().Before(entry.expiry) {
		return entry.pods
	}

	// Slow Path: Write Lock with Double-Check
	// We missed the cache. Acquire write lock to update it.
	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check: Someone else might have updated the cache while we were waiting for the lock.
	entry, found = c.cache[allEndpointsCacheKey]
	if found && time.Now().Before(entry.expiry) {
		return entry.pods
	}

	// Fetch from Delegate.
	// Note: We hold the lock during the fetch. This serializes requests for the same key, preventing a "thundering herd"
	// on the underlying Datastore.
	// Since Datastore lookups are fast in-memory scans, this lock duration is acceptable.
	freshPods := c.delegate.Locate(ctx, nil)

	// Update cache.
	c.cache[allEndpointsCacheKey] = cacheEntry{
		pods:   freshPods,
		expiry: time.Now().Add(c.ttl),
	}

	return freshPods
}

// runCleanup periodically removes expired entries from the cache to prevent unbounded growth.
func (c *CachedEndpointCandidates) runCleanup(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("CachedEndpointCandidatesCleanup")
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.V(logutil.DEBUG).Info("Stopping cleanup routine")
			return
		case <-ticker.C:
			c.cleanup()
		}
	}
}

// cleanup iterates over the map and removes expired entries.
func (c *CachedEndpointCandidates) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.cache {
		if now.After(entry.expiry) {
			delete(c.cache, key)
		}
	}
}
