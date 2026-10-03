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

package redis

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	goredis "github.com/redis/go-redis/v9"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

func init() {
	gob.Register(&attrconcurrency.InFlightLoad{})
}

const (
	RedisStateStoreType = "redis-state-store"
	defaultAddress      = "localhost:6379"
	defaultTTL          = 180 * time.Second
)

var _ fwkdl.CrossReplicaSyncer = (*RedisStateStore)(nil)

type redisConfig struct {
	Address  string `json:"address"`
	Password string `json:"password"`
	DB       int    `json:"db"`
	TTL      string `json:"ttl"`
}

// RedisStateStore is a CrossReplicaSyncer backed by Redis for cross-replica
// state sharing. Set prepares each endpoint's aggregate, and Get serves the
// prepared value from memory.
type RedisStateStore struct {
	typedName fwkplugin.TypedName
	replicaID string
	client    *goredis.Client
	ttl       time.Duration
	cache     sync.Map
}

type aggregateCacheEntry struct {
	value     any
	expiresAt time.Time
}

func RedisStateStoreFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var cfg redisConfig
	if params != nil {
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("redis-state-store: invalid parameters: %w", err)
		}
	}
	if cfg.Address == "" {
		cfg.Address = defaultAddress
	}

	ttl := defaultTTL
	if cfg.TTL != "" {
		parsed, err := time.ParseDuration(cfg.TTL)
		if err != nil {
			return nil, fmt.Errorf("redis-state-store: invalid ttl %q: %w", cfg.TTL, err)
		}
		ttl = parsed
	}

	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "unknown"
	}

	client := goredis.NewClient(&goredis.Options{
		Addr:     cfg.Address,
		Password: cfg.Password,
		DB:       cfg.DB,
	})

	ctx := context.Background()
	if handle != nil && handle.Context() != nil {
		ctx = handle.Context()
	}
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis-state-store: failed to connect to Redis at %s: %w", cfg.Address, err)
	}

	return &RedisStateStore{
		typedName: fwkplugin.TypedName{Type: RedisStateStoreType, Name: name},
		replicaID: hostname,
		client:    client,
		ttl:       ttl,
	}, nil
}

func (s *RedisStateStore) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *RedisStateStore) hashKey(key fwkdl.StateKey, endpointID string) string {
	return string(key) + ":" + endpointID
}

// coordKey namespaces request-level values away from endpoint hashes.
func (s *RedisStateStore) coordKey(key fwkdl.StateKey, id string) string {
	return "coord:" + string(key) + ":" + id
}

type stampedValue struct {
	Value     any
	WrittenAt time.Time
}

func encodeStamped(value any, now time.Time) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(stampedValue{Value: value, WrittenAt: now}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobDecode(data []byte) (stampedValue, error) {
	var value stampedValue
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&value); err != nil {
		return stampedValue{}, err
	}
	return value, nil
}

// Set publishes this replica's value and prepares the aggregate returned by Get.
func (s *RedisStateStore) Set(ctx context.Context, key fwkdl.StateKey, endpointID string, value any, aggregate func([]any) any) error {
	logger := ctrl.LoggerFrom(ctx)
	now := time.Now()

	data, err := encodeStamped(value, now)
	if err != nil {
		return fmt.Errorf("redis-state-store: encode: %w", err)
	}

	hashKey := s.hashKey(key, endpointID)
	pipe := s.client.TxPipeline()
	pipe.HSet(ctx, hashKey, s.replicaID, data)
	pipe.HExpire(ctx, hashKey, s.ttl, s.replicaID)
	pipe.Expire(ctx, hashKey, s.ttl)
	result := pipe.HGetAll(ctx, hashKey)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("redis-state-store: set and get aggregate: %w", err)
	}

	raw, err := result.Result()
	if err != nil {
		return fmt.Errorf("redis-state-store: hgetall: %w", err)
	}
	values := make([]any, 0, len(raw))
	for field, encoded := range raw {
		stamped, err := gobDecode([]byte(encoded))
		if err != nil {
			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("redis-state-store: decode error", "field", field, "error", err)
			}
			continue
		}
		if now.Sub(stamped.WrittenAt) > s.ttl {
			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("redis-state-store: skipping stale entry", "field", field, "age", now.Sub(stamped.WrittenAt), "ttl", s.ttl)
			}
			continue
		}
		values = append(values, stamped.Value)
	}

	if len(values) == 0 {
		s.cache.Delete(hashKey)
		return nil
	}
	aggregated := aggregate(values)
	s.cache.Store(hashKey, &aggregateCacheEntry{
		value:     aggregated,
		expiresAt: now.Add(s.ttl),
	})

	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("redis-state-store: Set", "key", string(key), "endpoint", endpointID, "replica", s.replicaID, "numReplicas", len(values), "result", fmt.Sprintf("%+v", aggregated))
	}
	return nil
}

// Get returns the aggregate prepared by the most recent Set.
func (s *RedisStateStore) Get(_ context.Context, key fwkdl.StateKey, endpointID string) (any, bool, error) {
	hashKey := s.hashKey(key, endpointID)
	cached, ok := s.cache.Load(hashKey)
	if !ok {
		return nil, false, nil
	}
	entry := cached.(*aggregateCacheEntry)
	if entry.expiresAt.Before(time.Now()) {
		s.cache.CompareAndDelete(hashKey, cached)
		return nil, false, nil
	}
	return entry.value, true, nil
}

func (s *RedisStateStore) Delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
	hashKey := s.hashKey(key, endpointID)
	if err := s.client.HDel(ctx, hashKey, s.replicaID).Err(); err != nil {
		return err
	}
	s.cache.Delete(hashKey)
	return nil
}

// GetOrSet atomically returns the existing request-level value or stores candidate.
// The operation requires Redis 7.0 or newer for SET NX GET support.
func (s *RedisStateStore) GetOrSet(ctx context.Context, key fwkdl.StateKey, id string, candidate any) (any, bool, error) {
	if candidate == nil {
		return nil, false, errors.New("redis-state-store: getorset candidate must not be nil")
	}

	data, err := encodeStamped(candidate, time.Now())
	if err != nil {
		return nil, false, fmt.Errorf("redis-state-store: getorset encode: %w", err)
	}

	stored, err := s.client.SetArgs(ctx, s.coordKey(key, id), data, goredis.SetArgs{
		Mode: "NX",
		Get:  true,
		TTL:  s.ttl,
	}).Result()
	if errors.Is(err, goredis.Nil) {
		return candidate, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("redis-state-store: getorset: %w", err)
	}

	actual, err := gobDecode([]byte(stored))
	if err != nil {
		return nil, false, fmt.Errorf("redis-state-store: getorset decode: %w", err)
	}
	return actual.Value, true, nil
}
