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
	Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error
	// Evict removes a key and its associated pod entries from the index backend.
	// keyType indicates whether the key is an EngineKey (requires engine→request lookup)
	// or a RequestKey (used directly).
	Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error
	// GetRequestKey returns the requestKey associated with the given engineKey.
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
	// EngineKey means the key is an engine-assigned key that must be resolved
	// to a request key via the engineToRequestKeys mapping.
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

// engineToRequestMapping computes engine-key → request-key mappings using
// proportional distribution based on the lengths of both slices.
// Each engine key maps to an ordered slice of request keys; callers that need
// positional scores (e.g. Redis ZAdd) can iterate the slice and use the index.
// Returns an empty map if either slice is empty.
func engineToRequestMapping(engineKeys, requestKeys []BlockHash) map[BlockHash][]BlockHash {
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
	return mappings
}
