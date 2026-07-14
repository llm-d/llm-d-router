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
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// RedisIndexConfig holds the configuration for the RedisIndex.
type RedisIndexConfig struct {
	Address string `json:"address,omitempty"` // Redis server address
}

func DefaultRedisIndexConfig() *RedisIndexConfig {
	return &RedisIndexConfig{
		Address: "redis://127.0.0.1:6379",
	}
}

// NewRedisIndex creates a new RedisIndex instance.
func NewRedisIndex(config *RedisIndexConfig) (Index, error) {
	if config == nil {
		config = DefaultRedisIndexConfig()
	}

	// Default to the redis:// prefix for backward compatibility.
	needsPrefix := !strings.HasPrefix(config.Address, "redis://") &&
		!strings.HasPrefix(config.Address, "rediss://") &&
		!strings.HasPrefix(config.Address, "unix://")
	if needsPrefix {
		config.Address = "redis://" + config.Address
	}

	redisOpt, err := redis.ParseURL(config.Address)
	if err != nil {
		return nil, fmt.Errorf("failed to parse redis URL: %w", err)
	}

	redisClient := redis.NewClient(redisOpt)
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to redis: %w", err)
	}

	return &RedisIndex{
		RedisClient: redisClient,
	}, nil
}

// RedisIndex implements the Index interface
// using Redis as the backend for KV block indexing.
type RedisIndex struct {
	RedisClient *redis.Client
}

var _ Index = &RedisIndex{}

// pruneRequestKeyScript atomically deletes a request key hash if it contains no pods.
var pruneRequestKeyScript = redis.NewScript(`
	local hashLen = redis.call('HLEN', KEYS[1])
	if hashLen == 0 then
		redis.call('DEL', KEYS[1])
		return 1
	end
	return 0
`)

// pruneEngineKeyScript atomically deletes an engine key mapping only if all
// associated request key hashes are empty. This prevents a TOCTOU race where a
// concurrent Add could insert into a request key between checking and deleting.
// KEYS[1] = engine key ("engine:<hash>"), KEYS[2..N] = request key hashes.
var pruneEngineKeyScript = redis.NewScript(`
	for i = 2, #KEYS do
		if redis.call('HLEN', KEYS[i]) > 0 then
			return 0
		end
	end
	redis.call('DEL', KEYS[1])
	return 1
`)

// Lookup receives a list of keys and a set of pod identifiers,
// and retrieves the filtered pods associated with those keys.
// The filtering is done based on the pod identifiers provided.
// If the podIdentifierSet is empty, all pods are returned.
//
// It returns:
// 1. A map where the keys are those in (1) and the values are pod-identifiers.
// 2. An error if any occurred during the operation.
func (r *RedisIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return make(map[BlockHash][]PodEntry), nil
	}

	logger := log.FromContext(ctx).WithName("kvblock.RedisIndex.Lookup")
	podsPerKey := make(map[BlockHash][]PodEntry)

	// pipeline for single RTT
	pipe := r.RedisClient.Pipeline()
	results := make([]*redis.StringSliceCmd, len(requestKeys))

	// queue an HKeys command for each key in the pipeline
	for i, key := range requestKeys {
		// HKeys gets all field names
		results[i] = pipe.HKeys(ctx, key.String())
	}

	_, execErr := pipe.Exec(ctx)
	if execErr != nil {
		return nil, fmt.Errorf("redis pipeline execution failed: %w", execErr)
	}

	filterPods := len(podIdentifierSet) > 0 // predicate for filtering

	for idx, cmd := range results {
		key := requestKeys[idx]

		// cmd.Result() returns the slice of strings (pod IDs) which is the first layer in the mapping
		pods, cmdErr := cmd.Result()
		if cmdErr != nil {
			if !errors.Is(cmdErr, redis.Nil) {
				logger.Error(cmdErr, "failed to get pods for key", "key", key)
			}

			return podsPerKey, nil // early stop since prefix-chain breaks here
		}

		var filteredPods []PodEntry
		for _, p := range pods {
			pod, ok := decodeRedisPodField(p)
			if !ok {
				continue
			}
			if !filterPods || podIdentifierSet.Has(pod.PodIdentifier) {
				filteredPods = append(filteredPods, pod)
			}
		}

		if len(filteredPods) == 0 {
			logger.Info("no pods found for key, cutting search", "key", key)
			return podsPerKey, nil // early stop since prefix-chain breaks here
		}

		podsPerKey[key] = filteredPods
	}

	return podsPerKey, nil
}

// Add adds a set of keys and their associated pod entries to the index backend.
// If engineKeys is nil, only requestKey -> PodEntry mappings are created (no engineKey -> requestKey mapping).
// This is used for speculative entries where engine keys are not yet known.
// When engineKeys is non-nil, the mapping type is inferred from the ratio of array lengths.
func (r *RedisIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}

	pipe := r.RedisClient.Pipeline()

	// Build engine->request mappings when engine keys are provided.
	// The ratio of array lengths determines the mapping type:
	//   equal  (4 eng, 4 req) -> 1:1   E0->R0, E1->R1, ...
	//   many:1 (4 eng, 1 req) -> E0->R0, E1->R0, E2->R0, E3->R0
	//   1:many (1 eng, 4 req) -> E0->[R0, R1, R2, R3]
	if engineKeys != nil {
		mappings := engineToRequestMapping(engineKeys, requestKeys)
		for ek, rks := range mappings {
			for j, rk := range rks {
				pipe.ZAdd(ctx, redisEngineKey(ek), redis.Z{Score: float64(j), Member: rk.String()})
			}
		}
	}

	// Store requestKey -> PodEntry mappings for all request keys.
	for _, requestKey := range requestKeys {
		redisKey := requestKey.String()
		for _, entry := range entries {
			field, err := encodeRedisPodField(entry)
			if err != nil {
				return err
			}
			pipe.HSet(ctx, redisKey, field, "")
		}
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to add entries to Redis: %w", err)
	}

	return nil
}

// Evict removes a key and its associated pod entries from the index backend.
// keyType indicates whether the key is an EngineKey (requires engine→request lookup)
// or a RequestKey (used directly for speculative entries without engineKey mapping).
func (r *RedisIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	if len(entries) == 0 {
		return fmt.Errorf("no entries provided for eviction from index")
	}

	switch keyType {
	case EngineKey:
		rks, err := r.getRequestKeys(ctx, key)
		if err != nil || len(rks) == 0 {
			// Engine key not found in mapping — nothing to evict
			return nil //nolint:nilerr // intentional: missing engine key means nothing to evict
		}
		for _, rk := range rks {
			if err := r.evictPodsFromRequestKey(ctx, rk, entries); err != nil {
				return err
			}
		}
		keys := make([]string, 0, 1+len(rks))
		keys = append(keys, redisEngineKey(key))
		for _, rk := range rks {
			keys = append(keys, rk.String())
		}
		if err := pruneEngineKeyScript.Run(ctx, r.RedisClient, keys).Err(); err != nil && !errors.Is(err, redis.Nil) {
			return fmt.Errorf("failed to prune engine key mapping: %w", err)
		}
		return nil
	case RequestKey:
		return r.evictPodsFromRequestKey(ctx, key, entries)
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictPodsFromRequestKey removes the given pod entries from a single request key.
// If the pod hash becomes empty, the request key is removed.
func (r *RedisIndex) evictPodsFromRequestKey(ctx context.Context, requestKey BlockHash, entries []PodEntry) error {
	redisKey := requestKey.String()
	pipe := r.RedisClient.Pipeline()

	for _, entry := range entries {
		field, err := encodeRedisPodField(entry)
		if err != nil {
			return err
		}
		pipe.HDel(ctx, redisKey, field)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to evict entries from Redis: %w", err)
	}

	// Atomically delete the request key hash if it's now empty
	if err := pruneRequestKeyScript.Run(ctx, r.RedisClient, []string{redisKey}).Err(); err != nil {
		return fmt.Errorf("failed to prune empty request key: %w", err)
	}

	return nil
}

func encodeRedisPodField(entry PodEntry) (string, error) {
	value, err := json.Marshal(entry)
	if err != nil {
		return "", fmt.Errorf("failed to encode pod entry for Redis: %w", err)
	}
	return string(value), nil
}

func decodeRedisPodField(field string) (PodEntry, bool) {
	var entry PodEntry
	if err := json.Unmarshal([]byte(field), &entry); err != nil {
		return PodEntry{}, false
	}

	return entry, true
}

// getRequestKeys returns all request keys mapped to the given engine key.
func (r *RedisIndex) getRequestKeys(ctx context.Context, engineKey BlockHash) ([]BlockHash, error) {
	vals, err := r.RedisClient.ZRange(ctx, redisEngineKey(engineKey), 0, -1).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, nil
		}
		return nil, err
	}

	rks := make([]BlockHash, 0, len(vals))
	for _, val := range vals {
		hash, err := strconv.ParseUint(val, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid hash format: %s", val)
		}
		rks = append(rks, BlockHash(hash))
	}
	return rks, nil
}

// GetRequestKey returns the last request key (highest score) associated with the given engineKey.
func (r *RedisIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	vals, err := r.RedisClient.ZRangeArgs(ctx, redis.ZRangeArgs{
		Key:   redisEngineKey(engineKey),
		Start: 0,
		Stop:  0,
		Rev:   true,
	}).Result()
	if err != nil {
		return EmptyBlockHash, err
	}
	if len(vals) == 0 {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}

	hash, err := strconv.ParseUint(vals[0], 10, 64)
	if err != nil {
		return EmptyBlockHash, fmt.Errorf("invalid hash format: %s", vals[0])
	}
	return BlockHash(hash), nil
}

func redisEngineKey(engineKey BlockHash) string {
	return "engine:" + engineKey.String()
}

// Clear removes every hash field for the pod across all request-key hashes and
// device tiers. Each field is a JSON-encoded PodEntry, so matching decodes the
// field and compares PodIdentifier — catching every tier, group, and speculative
// variant. It pages the keyspace with SCAN (skipping engine: keys), HDELs the
// pod's fields, and prunes now-empty hashes. O(keyspace), but Clear is rare and
// off the Lookup/Add hot path. Because it deletes from the shared store, it is
// correct for multi-replica deployments with no cross-process coordination.
func (r *RedisIndex) Clear(ctx context.Context, podIdentifier string) error {
	logger := log.FromContext(ctx).WithName("kvblock.RedisIndex.Clear")

	const scanBatch int64 = 1024
	removed := 0
	var cursor uint64
	for {
		keys, next, err := r.RedisClient.Scan(ctx, cursor, "*", scanBatch).Result()
		if err != nil {
			return fmt.Errorf("clear scan failed: %w", err)
		}
		for _, key := range keys {
			if strings.HasPrefix(key, "engine:") {
				continue // engine:<hash> ZSETs hold no pod fields
			}

			fields, err := r.RedisClient.HKeys(ctx, key).Result()
			if err != nil {
				return fmt.Errorf("clear hkeys failed for %s: %w", key, err)
			}

			var stale []string
			for _, field := range fields {
				if entry, ok := decodeRedisPodField(field); ok && entry.PodIdentifier == podIdentifier {
					stale = append(stale, field)
				}
			}
			if len(stale) == 0 {
				continue
			}

			if err := r.RedisClient.HDel(ctx, key, stale...).Err(); err != nil {
				return fmt.Errorf("clear hdel failed for %s: %w", key, err)
			}
			removed += len(stale)

			if err := pruneRequestKeyScript.Run(ctx, r.RedisClient, []string{key}).Err(); err != nil &&
				!errors.Is(err, redis.Nil) {
				return fmt.Errorf("clear prune failed for %s: %w", key, err)
			}
		}

		if cursor = next; cursor == 0 {
			break
		}
	}

	logger.Info("cleared pod from index", "pod", podIdentifier, "removed", removed)
	return nil
}
