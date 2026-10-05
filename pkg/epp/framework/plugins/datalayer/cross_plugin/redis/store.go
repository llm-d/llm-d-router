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
	"reflect"
	"sync"
	"time"

	goredis "github.com/redis/go-redis/v9"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	RedisStateStoreType = "redis-state-store"
	defaultAddress      = "localhost:6379"
	defaultTTL          = 180 * time.Second
)

var _ fwkdl.CrossReplicaSyncer = (*RedisStateStore)(nil)
var _ fwkdl.BoundCrossReplicaState = (*boundState)(nil)

type redisConfig struct {
	Address  string `json:"address"`
	Password string `json:"password"`
	DB       int    `json:"db"`
	TTL      string `json:"ttl"`
}

// RedisStateStore is a CrossReplicaSyncer backed by Redis for cross-replica
// state sharing. Set prepares each endpoint's peer aggregate, and Get combines
// it with the live local value.
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

type boundState struct {
	store     *RedisStateStore
	key       fwkdl.StateKey
	read      func(string) fwkdl.Cloneable
	aggregate func([]any) any
}

func RedisStateStoreFactory(name string, params *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
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
	if ttl <= 0 {
		return nil, fmt.Errorf("redis-state-store: ttl must be positive, got %s", ttl)
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

func (s *RedisStateStore) Bind(key fwkdl.StateKey, read func(string) fwkdl.Cloneable, aggregate func([]any) any) fwkdl.BoundCrossReplicaState {
	return &boundState{store: s, key: key, read: read, aggregate: aggregate}
}

func (s *boundState) Set(ctx context.Context, endpointID string) error {
	return s.store.set(ctx, s.key, endpointID, s.read(endpointID), s.aggregate)
}

func (s *boundState) Get(ctx context.Context, endpointID string) (any, error) {
	return s.store.get(ctx, s.key, endpointID, s.read(endpointID), s.aggregate)
}

func (s *boundState) Delete(ctx context.Context, endpointID string) error {
	return s.store.delete(ctx, s.key, endpointID)
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
	if value == nil {
		return nil, errors.New("cannot encode nil value")
	}

	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err := encoder.Encode(now); err != nil {
		return nil, err
	}
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobDecode(data []byte, prototype any) (stampedValue, error) {
	if prototype == nil {
		return stampedValue{}, errors.New("cannot decode stored value without a type prototype")
	}

	decoder := gob.NewDecoder(bytes.NewReader(data))
	var writtenAt time.Time
	if err := decoder.Decode(&writtenAt); err != nil {
		return stampedValue{}, err
	}

	valueType := reflect.TypeOf(prototype)
	target := reflect.New(valueType)
	if err := decoder.Decode(target.Interface()); err != nil {
		return stampedValue{}, err
	}
	return stampedValue{Value: target.Elem().Interface(), WrittenAt: writtenAt}, nil
}

// set publishes this replica's value and prepares the peer aggregate returned
// by get.
func (s *RedisStateStore) set(ctx context.Context, key fwkdl.StateKey, endpointID string, value any, aggregate func([]any) any) error {
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
	remoteValues := make([]any, 0, len(raw))
	var remoteExpiresAt time.Time
	for field, encoded := range raw {
		if field == s.replicaID {
			continue
		}
		stamped, err := gobDecode([]byte(encoded), value)
		if err != nil {
			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("redis-state-store: decode error", "field", field, "error", err)
			}
			continue
		}
		valueExpiresAt := stamped.WrittenAt.Add(s.ttl)
		if !valueExpiresAt.After(now) {
			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("redis-state-store: skipping stale entry", "field", field, "age", now.Sub(stamped.WrittenAt), "ttl", s.ttl)
			}
			continue
		}
		remoteValues = append(remoteValues, stamped.Value)
		if remoteExpiresAt.IsZero() || valueExpiresAt.Before(remoteExpiresAt) {
			remoteExpiresAt = valueExpiresAt
		}
	}

	if len(remoteValues) == 0 {
		s.cache.Delete(hashKey)
	} else {
		s.cache.Store(hashKey, &aggregateCacheEntry{
			value:     aggregate(remoteValues),
			expiresAt: remoteExpiresAt,
		})
	}

	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("redis-state-store: Set", "key", string(key), "endpoint", endpointID, "replica", s.replicaID, "numReplicas", len(remoteValues)+1)
	}
	return nil
}

// get combines the live local value with the peer aggregate prepared by set.
func (s *RedisStateStore) get(_ context.Context, key fwkdl.StateKey, endpointID string, local any, aggregate func([]any) any) (any, error) {
	hashKey := s.hashKey(key, endpointID)
	cached, ok := s.cache.Load(hashKey)
	if !ok {
		return aggregate([]any{local}), nil
	}
	entry := cached.(*aggregateCacheEntry)
	if !entry.expiresAt.After(time.Now()) {
		s.cache.CompareAndDelete(hashKey, cached)
		return aggregate([]any{local}), nil
	}
	return aggregate([]any{local, entry.value}), nil
}

func (s *RedisStateStore) delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
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

	actual, err := gobDecode([]byte(stored), candidate)
	if err != nil {
		return nil, false, fmt.Errorf("redis-state-store: getorset decode: %w", err)
	}
	return actual.Value, true, nil
}
