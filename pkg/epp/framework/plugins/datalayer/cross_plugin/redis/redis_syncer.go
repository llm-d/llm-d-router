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
	"maps"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	goredis "github.com/redis/go-redis/v9"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	RedisSyncerType        = "redis-syncer"
	defaultAddress         = "localhost:6379"
	defaultStateTTL        = 2 * time.Second
	defaultCoordinationTTL = 180 * time.Second
	minimumTTL             = time.Millisecond
	batchSize              = 100
	batchConcurrency       = 8
)

var _ fwkdl.CrossReplicaSyncer = (*RedisSyncer)(nil)
var _ fwkdl.CrossReplicaBatchSyncer = (*RedisSyncer)(nil)

type redisConfig struct {
	Address         string `json:"address"`
	Password        string `json:"password"`
	PasswordEnv     string `json:"passwordEnv"`
	PasswordFile    string `json:"passwordFile"`
	DB              int    `json:"db"`
	StateTTL        string `json:"stateTTL"`
	CoordinationTTL string `json:"coordinationTTL"`
}

// RedisSyncer is a CrossReplicaSyncer backed by Redis for cross-replica
// state sharing. Set prepares each endpoint's peer aggregate, and Get combines
// it with the live local value. Only a successful Set replaces the aggregate, so
// while Redis is unreachable the last known peer state keeps counting.
type RedisSyncer struct {
	typedName       fwkplugin.TypedName
	replicaID       string
	client          *goredis.Client
	stateTTL        time.Duration
	coordinationTTL time.Duration
	cache           sync.Map
}

type aggregateCacheEntry struct {
	value any
}

func parseTTL(name, raw string, defaultValue time.Duration) (time.Duration, error) {
	if raw == "" {
		return defaultValue, nil
	}
	ttl, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("redis-syncer: invalid %s %q: %w", name, raw, err)
	}
	if ttl < minimumTTL {
		return 0, fmt.Errorf("redis-syncer: %s must be at least %s, got %s", name, minimumTTL, ttl)
	}
	return ttl, nil
}

func resolvePassword(cfg redisConfig) (string, error) {
	if cfg.Password != "" {
		return "", errors.New("redis-syncer: password must use passwordEnv or passwordFile")
	}
	if cfg.PasswordEnv != "" && cfg.PasswordFile != "" {
		return "", errors.New("redis-syncer: passwordEnv and passwordFile are mutually exclusive")
	}
	if cfg.PasswordEnv != "" {
		password, ok := os.LookupEnv(cfg.PasswordEnv)
		if !ok {
			return "", fmt.Errorf("redis-syncer: password environment variable %q is not set", cfg.PasswordEnv)
		}
		return password, nil
	}
	if cfg.PasswordFile != "" {
		password, err := os.ReadFile(cfg.PasswordFile)
		if err != nil {
			return "", fmt.Errorf("redis-syncer: read password file %q: %w", cfg.PasswordFile, err)
		}
		return strings.TrimRight(string(password), "\r\n"), nil
	}
	return "", nil
}

func RedisSyncerFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var cfg redisConfig
	if params != nil {
		params.DisallowUnknownFields()
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("redis-syncer: invalid parameters: %w", err)
		}
	}
	if cfg.Address == "" {
		cfg.Address = defaultAddress
	}

	stateTTL, err := parseTTL("stateTTL", cfg.StateTTL, defaultStateTTL)
	if err != nil {
		return nil, err
	}
	coordinationTTL, err := parseTTL("coordinationTTL", cfg.CoordinationTTL, defaultCoordinationTTL)
	if err != nil {
		return nil, err
	}
	password, err := resolvePassword(cfg)
	if err != nil {
		return nil, err
	}
	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "unknown"
	}
	replicaID := hostname + "-" + uuid.NewString()

	client := goredis.NewClient(&goredis.Options{
		Addr:                  cfg.Address,
		Password:              password,
		DB:                    cfg.DB,
		ContextTimeoutEnabled: true,
	})
	if handle != nil && handle.Context() != nil {
		context.AfterFunc(handle.Context(), func() { _ = client.Close() })
	}

	return &RedisSyncer{
		typedName:       fwkplugin.TypedName{Type: RedisSyncerType, Name: name},
		replicaID:       replicaID,
		client:          client,
		stateTTL:        stateTTL,
		coordinationTTL: coordinationTTL,
	}, nil
}

func (s *RedisSyncer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *RedisSyncer) hashKey(key fwkdl.StateKey, endpointID string) string {
	return string(key) + ":" + endpointID
}

// coordKey namespaces request-level values away from endpoint hashes.
func (s *RedisSyncer) coordKey(key fwkdl.StateKey, id string) string {
	return "coord:" + string(key) + ":" + id
}

func encodeValue(value any) ([]byte, error) {
	if value == nil {
		return nil, errors.New("cannot encode nil value")
	}

	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobDecode(data []byte, prototype any) (any, error) {
	if prototype == nil {
		return nil, errors.New("cannot decode stored value without a type prototype")
	}

	decoder := gob.NewDecoder(bytes.NewReader(data))
	valueType := reflect.TypeOf(prototype)
	target := reflect.New(valueType)
	if err := decoder.Decode(target.Interface()); err != nil {
		return nil, err
	}
	return target.Elem().Interface(), nil
}

// set publishes this replica's value and prepares the peer aggregate returned
// by get.
func (s *RedisSyncer) set(ctx context.Context, key fwkdl.StateKey, endpointID string, value any, aggregate func([]any) any) error {
	logger := ctrl.LoggerFrom(ctx)

	data, err := encodeValue(value)
	if err != nil {
		return fmt.Errorf("redis-syncer: encode: %w", err)
	}

	hashKey := s.hashKey(key, endpointID)
	pipe := s.client.TxPipeline()
	pipe.HSet(ctx, hashKey, s.replicaID, data)
	pipe.HPExpire(ctx, hashKey, s.stateTTL, s.replicaID)
	pipe.PExpire(ctx, hashKey, s.stateTTL)
	result := pipe.HGetAll(ctx, hashKey)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("redis-syncer: set and get aggregate; Redis 7.4 or newer with HPEXPIRE is required: %w", err)
	}

	raw, err := result.Result()
	if err != nil {
		return fmt.Errorf("redis-syncer: hgetall: %w", err)
	}
	peerCount := s.cachePeerAggregate(ctx, hashKey, value, raw, aggregate)

	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("redis-syncer: Set", "key", string(key), "endpoint", endpointID, "replica", s.replicaID, "numReplicas", peerCount+1)
	}
	return nil
}

func (s *RedisSyncer) cachePeerAggregate(ctx context.Context, hashKey string, value any, raw map[string]string, aggregate func([]any) any) int {
	logger := ctrl.LoggerFrom(ctx)
	remoteValues := make([]any, 0, len(raw))
	for _, field := range slices.Sorted(maps.Keys(raw)) {
		if field == s.replicaID {
			continue
		}
		encoded := raw[field]
		remoteValue, err := gobDecode([]byte(encoded), value)
		if err != nil {
			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("redis-syncer: decode error", "field", field, "error", err)
			}
			continue
		}
		remoteValues = append(remoteValues, remoteValue)
	}

	if len(remoteValues) == 0 {
		s.cache.Delete(hashKey)
	} else {
		s.cache.Store(hashKey, &aggregateCacheEntry{value: aggregate(remoteValues)})
	}

	return len(remoteValues)
}

func (s *RedisSyncer) Set(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) error {
	return s.set(ctx, spec.StateKey, endpointID, spec.Read(endpointID), spec.Aggregate)
}

func (s *RedisSyncer) SetBatch(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointIDs []string, timeout time.Duration) map[string]error {
	failures := make(map[string]error)
	slots := make(chan struct{}, batchConcurrency)
	var mu sync.Mutex
	var wg sync.WaitGroup
	remaining := endpointIDs
dispatch:
	for start := 0; start < len(endpointIDs); start += batchSize {
		if ctx.Err() != nil {
			break
		}
		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
			break dispatch
		}
		end := min(start+batchSize, len(endpointIDs))
		chunk := endpointIDs[start:end]
		remaining = endpointIDs[end:]
		wg.Go(func() {
			defer func() { <-slots }()
			chunkCtx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			errs := s.setBatchChunk(chunkCtx, spec, chunk)
			mu.Lock()
			maps.Copy(failures, errs)
			mu.Unlock()
		})
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		for _, endpointID := range remaining {
			failures[endpointID] = err
		}
	}
	return failures
}

func (s *RedisSyncer) setBatchChunk(ctx context.Context, spec fwkdl.CrossReplicaSpec, chunk []string) map[string]error {
	failures := make(map[string]error)
	if err := ctx.Err(); err != nil {
		for _, endpointID := range chunk {
			failures[endpointID] = err
		}
		return failures
	}
	logger := ctrl.LoggerFrom(ctx)

	pipe := s.client.TxPipeline()
	type publication struct {
		endpointID string
		value      any
		commands   []goredis.Cmder
		result     *goredis.MapStringStringCmd
	}
	publications := make([]publication, 0, len(chunk))
	for _, endpointID := range chunk {
		value := spec.Read(endpointID)
		data, err := encodeValue(value)
		if err != nil {
			failures[endpointID] = fmt.Errorf("redis-syncer: encode: %w", err)
			continue
		}
		hashKey := s.hashKey(spec.StateKey, endpointID)
		hset := pipe.HSet(ctx, hashKey, s.replicaID, data)
		hpexpire := pipe.HPExpire(ctx, hashKey, s.stateTTL, s.replicaID)
		pexpire := pipe.PExpire(ctx, hashKey, s.stateTTL)
		result := pipe.HGetAll(ctx, hashKey)
		publications = append(publications, publication{
			endpointID: endpointID, value: value,
			commands: []goredis.Cmder{hset, hpexpire, pexpire, result}, result: result,
		})
	}
	if len(publications) == 0 {
		return failures
	}
	commands, execErr := pipe.Exec(ctx)
	if execFailedWholeChunk(execErr, commands) {
		for _, publication := range publications {
			failures[publication.endpointID] = fmt.Errorf("redis-syncer: set and get aggregate; Redis 7.4 or newer with HPEXPIRE is required: %w", execErr)
		}
		return failures
	}
	for _, publication := range publications {
		var err error
		for _, cmd := range publication.commands {
			if cmd.Err() != nil {
				err = cmd.Err()
				break
			}
		}
		if err != nil {
			failures[publication.endpointID] = fmt.Errorf("redis-syncer: set and get aggregate; Redis 7.4 or newer with HPEXPIRE is required: %w", err)
			continue
		}
		peerCount := s.cachePeerAggregate(ctx, s.hashKey(spec.StateKey, publication.endpointID), publication.value, publication.result.Val(), spec.Aggregate)
		if v := logger.V(logutil.DEBUG); v.Enabled() {
			v.Info("redis-syncer: Set", "key", string(spec.StateKey), "endpoint", publication.endpointID, "replica", s.replicaID, "numReplicas", peerCount+1)
		}
	}
	return failures
}

// execFailedWholeChunk reports whether a TxPipeline Exec error means every
// command in the transaction failed. go-redis returns the first command error
// from Exec: when that error is a Redis error attached to one command (for
// example WRONGTYPE on one key), the other commands still ran and only that
// endpoint failed. EXECABORT, transport errors, context errors, and Redis
// errors attached to no command mean the transaction did not run or its
// outcome is unknown, so the whole chunk fails and caches stay.
func execFailedWholeChunk(execErr error, cmds []goredis.Cmder) bool {
	if execErr == nil {
		return false
	}
	var redisErr goredis.Error
	if !errors.As(execErr, &redisErr) || goredis.HasErrorPrefix(execErr, "EXECABORT") {
		return true
	}
	return !slices.ContainsFunc(cmds, func(cmd goredis.Cmder) bool {
		return errors.Is(cmd.Err(), execErr)
	})
}

// get combines the live local value with the peer aggregate prepared by set.
func (s *RedisSyncer) get(_ context.Context, key fwkdl.StateKey, endpointID string, local any, aggregate func([]any) any) any {
	hashKey := s.hashKey(key, endpointID)
	cached, ok := s.cache.Load(hashKey)
	if !ok {
		return local
	}
	return aggregate([]any{local, cached.(*aggregateCacheEntry).value})
}

func (s *RedisSyncer) Get(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) (any, bool, error) {
	return s.get(ctx, spec.StateKey, endpointID, spec.Read(endpointID), spec.Aggregate), true, nil
}

func (s *RedisSyncer) delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
	hashKey := s.hashKey(key, endpointID)
	s.cache.Delete(hashKey)
	return s.client.HDel(ctx, hashKey, s.replicaID).Err()
}

func (s *RedisSyncer) Delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
	return s.delete(ctx, key, endpointID)
}

// GetOrSet atomically returns the existing request-level value or stores candidate.
// The operation requires Redis 7.0 or newer for SET NX GET support.
func (s *RedisSyncer) GetOrSet(ctx context.Context, key fwkdl.StateKey, id string, candidate any) (any, bool, error) {
	if candidate == nil {
		return nil, false, errors.New("redis-syncer: getorset candidate must not be nil")
	}

	data, err := encodeValue(candidate)
	if err != nil {
		return nil, false, fmt.Errorf("redis-syncer: getorset encode: %w", err)
	}

	stored, err := s.client.SetArgs(ctx, s.coordKey(key, id), data, goredis.SetArgs{
		Mode: "NX",
		Get:  true,
		TTL:  s.coordinationTTL,
	}).Result()
	if errors.Is(err, goredis.Nil) {
		return candidate, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("redis-syncer: getorset: %w", err)
	}

	actual, err := gobDecode([]byte(stored), candidate)
	if err != nil {
		return nil, false, fmt.Errorf("redis-syncer: getorset decode: %w", err)
	}
	return actual, true, nil
}
