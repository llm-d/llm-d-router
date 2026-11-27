/*
Copyright 2025 The llm-d Authors.

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

package kvblock

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"k8s.io/apimachinery/pkg/util/sets"
)

// IndexConfig holds the configuration for the KV-block index.
// It may configure several backends such as listed within the struct.
// If multiple backends are configured, only the first one will be used.
type IndexConfig struct {
	// InMemoryConfig holds the configuration for the in-memory index.
	InMemoryConfig *InMemoryIndexConfig `json:"inMemoryConfig"`
	// RedisConfig holds the configuration for the Redis index.
	RedisConfig *RedisIndexConfig `json:"redisConfig"`
	// CostAwareMemoryConfig holds the configuration for the cost-aware memory index.
	CostAwareMemoryConfig *CostAwareMemoryIndexConfig `json:"costAwareMemoryConfig"`

	// EnableMetrics toggles whether admissions/evictions/hits/misses are
	// recorded.
	EnableMetrics bool `json:"enableMetrics"`
	// MetricsLoggingInterval defines the interval at which metrics are logged.
	// If zero, metrics logging is disabled.
	// Requires `EnableMetrics` to be true.
	MetricsLoggingInterval time.Duration `json:"metricsLoggingInterval"`
}

// DefaultIndexConfig returns a default configuration for the KV-block index.
func DefaultIndexConfig() *IndexConfig {
	return &IndexConfig{
		InMemoryConfig: DefaultInMemoryIndexConfig(),
		EnableMetrics:  false,
	}
}

// NewIndex creates a new Index instance.
func NewIndex(ctx context.Context, cfg *IndexConfig) (Index, error) {
	if cfg == nil {
		cfg = DefaultIndexConfig()
	}

	var idx Index
	var err error

	switch {
	case cfg.CostAwareMemoryConfig != nil:
		idx, err = NewCostAwareMemoryIndex(cfg.CostAwareMemoryConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create cost-aware memory index: %w", err)
		}
	case cfg.RedisConfig != nil:
		//nolint:contextcheck // NewRedisIndex does not accept context parameter
		idx, err = NewRedisIndex(cfg.RedisConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create Redis index: %w", err)
		}
	case cfg.InMemoryConfig != nil:
		idx, err = NewInMemoryIndex(cfg.InMemoryConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create in-memory index: %w", err)
		}
	default:
		return nil, fmt.Errorf("no valid index configuration provided")
	}

	// wrap in metrics only if enabled
	if cfg.EnableMetrics {
		idx = NewInstrumentedIndex(idx)
		metrics.Register()
		if cfg.MetricsLoggingInterval > 0 {
			// this is non-blocking
			metrics.StartMetricsLogging(ctx, cfg.MetricsLoggingInterval)
		}
	}

	return idx, nil
}

// Index defines the interface for a backend that manages KV-block
// indexing.
//
// An index backend is a data store that will aggregate possibly the entire
// global KV cache block index, and will be used to retrieve pod-localities
// for a given set of consecutive keys that constitute a prefix-cache hit.
// The hit may not necessarily be on all keys, but of the longest prefix match.
//
// The index backend allows efficient tracking of which vLLM engines hold which
// KV-blocks, on what device tier, and when they were last updated.
//
// Index operations are thread-safe and can be performed concurrently.
type Index interface {
	// Lookup receives a list of keys and a set of pod identifiers,
	// and retrieves the filtered pods associated with those keys.
	// The filtering is done based on the pod identifiers provided.
	// If the podIdentifierSet is empty, all pods are returned.
	//
	// It returns:
	// 1. A map where the keys are those in requestKeys and the values are pod-identifiers.
	// 2. An error if any occurred during the operation.
	Lookup(ctx context.Context, requestKeys []BlockHash, podIdentifierSet sets.Set[string]) (map[BlockHash][]PodEntry, error)
	// Add stores requestKey -> pod entries and (optionally) engineKey -> requestKey
	// mappings. If engineKeys is nil, only requestKey -> pod mappings are created
	// (used for speculative entries where engine keys are not yet known).
	//
	// When engineKeys is non-nil, the backend infers the mapping from the ratio
	// of len(engineKeys) to len(requestKeys). Both lengths derive from the same
	// token count divided by their respective block sizes, so they always divide
	// evenly. Examples with 256 tokens:
	//
	//   1:1   (engine=64, canonical=64)  -> 4 eng, 4 req  -> E0->R0, E1->R1, ...
	//   many:1 (engine=16, canonical=64) -> 16 eng, 4 req -> E0..E3->R0, E4..E7->R1, ...
	//   1:many (engine=128, canonical=64) -> 2 eng, 4 req -> E0->[R0,R1], E1->[R2,R3]
	//
	// In the many:1 case only an engine key that ends a canonical block names one.
	// The others end inside it, so their mapping holds the request key followed by
	// EmptyBlockHash and GetRequestKey rejects them: a child chained on such a key
	// would be hashed from that block's offset.
	//
	// A sub-canonical store passes EngineGeometry so the boundaries come from token
	// positions. Without it the key counts stand in, which is exact only when the
	// store ends on a canonical block boundary.
	Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry, geometry ...EngineGeometry) error
	// Evict removes keys and their associated pod entries from the index backend.
	// keyType indicates whether the keys are EngineKeys (each resolved
	// independently via the engine→request mapping) or RequestKeys (used directly).
	// The interface does not define whether a failed call leaves the index untouched
	// or partially evicted; each implementation documents its own failure semantics.
	Evict(ctx context.Context, keyType KeyType, keys []BlockHash, entries []PodEntry) error
	// GetRequestKey returns the requestKey associated with the given engineKey. An
	// engineKey that ends inside a canonical block names none and returns an error.
	GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error)
	// Clear removes all index entries for the given pod, across every device tier.
	// It backs the AllBlocksCleared KV-event (a vLLM prefix-cache reset, e.g. after
	// an RLHF weight update), which is pod-wide — vLLM emits it with no tier. Clear is
	// O(N) over the index but runs off the Lookup/Add hot path, at a coarse cadence
	// (typically once per weight sync).
	Clear(ctx context.Context, podIdentifier string) error
}

// KeyType indicates whether a key passed to Evict is an engine key or a request key.
type KeyType int

const (
	// EngineKey means the key is an engine key, as the event pool stores it,
	// that must be resolved to a request key via the engineToRequestKeys mapping.
	EngineKey KeyType = iota
	// RequestKey means the key is a request key and can be used directly.
	// This is used for speculative entries that were added without engineKey mapping.
	RequestKey
)

// BlockHash struct represents a unique identifier for a KV-cache block.
type BlockHash uint64

// EmptyBlockHash represents an invalid or uninitialized block hash.
// This serves as the "error value".
const EmptyBlockHash BlockHash = 0

// String returns a string representation of the Key.
func (c BlockHash) String() string {
	return strconv.FormatUint(uint64(c), 10)
}

// PodEntry struct represents a pod entry in the KV-block index.
type PodEntry struct {
	// PodIdentifier is the unique identifier for the pod.
	PodIdentifier string
	// DeviceTier is the tier of the device where the KV-block is stored.
	DeviceTier string
	// Speculative indicates the entry was added predictively before a KV event confirmed it.
	Speculative bool
	// HasGroup indicates GroupIdx identifies a vLLM KV cache group.
	HasGroup bool
	// GroupIdx identifies the vLLM KV cache group for HMA events.
	GroupIdx GroupID
}

// String returns a string representation of the PodEntry.
func (e *PodEntry) String() string {
	suffix := ""
	if e.Speculative {
		suffix = "[speculative]"
	}
	if e.HasGroup {
		suffix += fmt.Sprintf("[group=%d]", e.GroupIdx)
	}
	return fmt.Sprintf("%s@%s%s", e.PodIdentifier, e.DeviceTier, suffix)
}

// EngineGeometry carries the block sizes behind an Add call: the engine block
// size the event reported and the canonical block size the request keys use.
type EngineGeometry struct {
	BlockSize          int
	CanonicalBlockSize int
}

// engineToRequestMapping computes engine-key → request-key mappings from the
// lengths of both slices. With a many:1 store the block sizes name the canonical
// block a boundary engine block ends.
// Each engine key maps to an ordered slice of request keys; callers that need
// positional scores (e.g. Redis ZAdd) can iterate the slice and use the index.
// Returns an empty map if either slice is empty.
func engineToRequestMapping(engineKeys, requestKeys []BlockHash, geometry ...EngineGeometry) map[BlockHash][]BlockHash {
	mappings := make(map[BlockHash][]BlockHash, len(engineKeys))
	if len(engineKeys) == 0 || len(requestKeys) == 0 {
		return mappings
	}
	n := max(len(engineKeys), len(requestKeys))
	for i := 0; i < n; i++ {
		ek := engineKeys[i*len(engineKeys)/n]
		rk := requestKeys[i*len(requestKeys)/n]
		mappings[ek] = append(mappings[ek], rk)
	}
	// A many:1 engine block names a request key only when it ends one. A child
	// chained on a block that ends inside a canonical block would be hashed from
	// that block's offset. Mark the others for GetRequestKey; the request key they
	// map to stays in the slice so eviction still reaches it.
	if len(engineKeys) > len(requestKeys) {
		var sizes EngineGeometry
		if len(geometry) > 0 {
			sizes = geometry[0]
		}
		for i, key := range engineKeys {
			if !endsCanonicalBlock(i, len(engineKeys), len(requestKeys), sizes) {
				mappings[key] = append(mappings[key], EmptyBlockHash)
				continue
			}
			if sizes.valid() {
				// The count ratio rounds off when coveredEngineKeys keeps a block
				// past the last request key, so a boundary block takes the
				// canonical block its last token ends.
				ri := min((i+1)*sizes.BlockSize/sizes.CanonicalBlockSize-1, len(requestKeys)-1)
				mappings[key] = []BlockHash{requestKeys[ri]}
			}
		}
	}
	return mappings
}

// valid reports whether the sizes place a canonical block boundary.
func (g EngineGeometry) valid() bool {
	return g.BlockSize > 0 && g.BlockSize < g.CanonicalBlockSize
}

// boundary reports whether engine block i ends where a canonical block ends.
func (g EngineGeometry) boundary(i int) bool {
	return (i+1)*g.BlockSize%g.CanonicalBlockSize == 0
}

// endsCanonicalBlock reports whether engine block i ends where a canonical block
// ends, so that a child chained on it can be hashed from that boundary.
func endsCanonicalBlock(i, engineBlocks, requestBlocks int, sizes EngineGeometry) bool {
	if sizes.valid() {
		return sizes.boundary(i)
	}
	// Without the block sizes the key counts are the only signal, and they match
	// token positions only when the store ends on a canonical block boundary:
	// every len(engineKeys)/len(requestKeys)-th block, otherwise only the last.
	return (i+1)*requestBlocks%engineBlocks == 0
}
