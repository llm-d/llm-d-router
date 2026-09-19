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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

const (
	testEndpointID = "vortex/backend-rank-0"
	testReplicaID  = "epp-a"
	testTTL        = time.Minute
)

var testStateKey = fwkdl.StateKey("inflight")

type commandRecorder struct {
	commands []goredis.Cmder
	err      error
}

func (r *commandRecorder) DialHook(next goredis.DialHook) goredis.DialHook {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		return next(ctx, network, addr)
	}
}

func (r *commandRecorder) ProcessHook(_ goredis.ProcessHook) goredis.ProcessHook {
	return func(_ context.Context, cmd goredis.Cmder) error {
		r.commands = append(r.commands, cmd)
		return r.err
	}
}

func (r *commandRecorder) ProcessPipelineHook(_ goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return func(_ context.Context, commands []goredis.Cmder) error {
		r.commands = append(r.commands, commands...)
		return r.err
	}
}

type hgetallCounter struct {
	count atomic.Int64
}

func (c *hgetallCounter) DialHook(next goredis.DialHook) goredis.DialHook { return next }

func (c *hgetallCounter) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook {
	return func(ctx context.Context, cmd goredis.Cmder) error {
		c.record(cmd)
		return next(ctx, cmd)
	}
}

func (c *hgetallCounter) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []goredis.Cmder) error {
		for _, cmd := range cmds {
			c.record(cmd)
		}
		return next(ctx, cmds)
	}
}

func (c *hgetallCounter) record(cmd goredis.Cmder) {
	if cmd.Name() == "hgetall" {
		c.count.Add(1)
	}
}

func newTestStore(t testing.TB, server *miniredis.Miniredis, replicaID string) (*RedisStateStore, *hgetallCounter) {
	t.Helper()
	counter := &hgetallCounter{}
	client := goredis.NewClient(&goredis.Options{Addr: server.Addr()})
	client.AddHook(counter)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	return &RedisStateStore{
		replicaID: replicaID,
		client:    client,
		ttl:       testTTL,
	}, counter
}

func seedReplica(t testing.TB, server *miniredis.Miniredis, key fwkdl.StateKey, endpointID, replicaID string, value any, writtenAt time.Time) {
	t.Helper()
	data, err := encodeStamped(value, writtenAt)
	require.NoError(t, err)
	server.HSet(string(key)+":"+endpointID, replicaID, string(data))
}

func sumInts(values []any) any {
	total := 0
	for _, value := range values {
		total += value.(int)
	}
	return total
}

func getSum(t testing.TB, store *RedisStateStore, key fwkdl.StateKey, endpointID string) (int, bool) {
	t.Helper()
	value, ok, err := store.Get(context.Background(), key, endpointID)
	require.NoError(t, err)
	if !ok {
		return 0, false
	}
	return value.(int), true
}

func TestSetPreparesAggregateAndGetReadsMemory(t *testing.T) {
	server := miniredis.RunT(t)
	store, counter := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, testEndpointID, "epp-b", 7, time.Now())

	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	require.Equal(t, int64(1), counter.count.Load())

	for range 2 {
		total, ok := getSum(t, store, testStateKey, testEndpointID)
		require.True(t, ok)
		require.Equal(t, 11, total)
	}
	require.Equal(t, int64(1), counter.count.Load(), "Get must not read Redis")
}

func TestSetRefreshesPreparedAggregate(t *testing.T) {
	server := miniredis.RunT(t)
	store, counter := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, testEndpointID, "epp-b", 7, time.Now())
	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	seedReplica(t, server, testStateKey, testEndpointID, "epp-b", 9, time.Now())
	total, ok := getSum(t, store, testStateKey, testEndpointID)
	require.True(t, ok)
	require.Equal(t, 11, total)

	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	total, ok = getSum(t, store, testStateKey, testEndpointID)
	require.True(t, ok)
	require.Equal(t, 13, total)
	require.Equal(t, int64(2), counter.count.Load())
}

func TestConcurrentSet(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	const callers = 16
	errs := make(chan error, callers)

	var wg sync.WaitGroup
	for i := range callers {
		wg.Go(func() {
			errs <- store.Set(context.Background(), testStateKey, fmt.Sprintf("endpoint-%d", i), i, sumInts)
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
}

func TestGetMissesBeforeSet(t *testing.T) {
	server := miniredis.RunT(t)
	store, counter := newTestStore(t, server, testReplicaID)

	value, ok, err := store.Get(context.Background(), testStateKey, testEndpointID)

	require.NoError(t, err)
	require.False(t, ok)
	require.Nil(t, value)
	require.Zero(t, counter.count.Load())
}

func TestSetSkipsStaleAndUndecodableFields(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, testEndpointID, "stale", 100, time.Now().Add(-2*testTTL))
	seedReplica(t, server, testStateKey, testEndpointID, "fresh", 7, time.Now())
	server.HSet(store.hashKey(testStateKey, testEndpointID), "corrupt", "not-gob")

	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	total, ok := getSum(t, store, testStateKey, testEndpointID)
	require.True(t, ok)
	require.Equal(t, 11, total)
}

func TestPreparedAggregateExpires(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	hashKey := store.hashKey(testStateKey, testEndpointID)
	store.cache.Store(hashKey, &aggregateCacheEntry{value: 4, expiresAt: time.Now().Add(-time.Second)})

	value, ok, err := store.Get(context.Background(), testStateKey, testEndpointID)

	require.NoError(t, err)
	require.False(t, ok)
	require.Nil(t, value)
	_, cached := store.cache.Load(hashKey)
	require.False(t, cached)
}

func TestDeleteInvalidatesAggregateAndReplicaField(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	require.NoError(t, store.Delete(context.Background(), testStateKey, testEndpointID))

	_, ok := getSum(t, store, testStateKey, testEndpointID)
	require.False(t, ok)
	exists, err := store.client.HExists(context.Background(), store.hashKey(testStateKey, testEndpointID), testReplicaID).Result()
	require.NoError(t, err)
	require.False(t, exists)
}

func TestSetWritesAndReadsAggregateInOneTransaction(t *testing.T) {
	recorder := &commandRecorder{}
	client := goredis.NewClient(&goredis.Options{})
	client.AddHook(recorder)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	store := &RedisStateStore{replicaID: testReplicaID, client: client, ttl: testTTL}
	hashKey := store.hashKey(testStateKey, testEndpointID)

	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	require.Len(t, recorder.commands, 6)
	require.Equal(t, "multi", recorder.commands[0].Name())
	require.Equal(t, "hset", recorder.commands[1].Name())
	require.Equal(t, []any{"HEXPIRE", hashKey, int64(60), "FIELDS", 1, testReplicaID}, recorder.commands[2].Args())
	require.Equal(t, "expire", recorder.commands[3].Name())
	require.Equal(t, "hgetall", recorder.commands[4].Name())
	require.Equal(t, "exec", recorder.commands[5].Name())
}

func TestSetAppliesFieldTTL(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	hashKey := store.hashKey(testStateKey, testEndpointID)

	require.NoError(t, store.Set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	server.HSet(hashKey, "peer", "value")
	server.SetTTL(hashKey, 2*testTTL)
	server.FastForward(testTTL + time.Second)

	exists, err := store.client.HExists(context.Background(), hashKey, testReplicaID).Result()
	require.NoError(t, err)
	require.False(t, exists)
	exists, err = store.client.HExists(context.Background(), hashKey, "peer").Result()
	require.NoError(t, err)
	require.True(t, exists)
}

func TestGetOrSetReturnsFirstValueAcrossStores(t *testing.T) {
	server := miniredis.RunT(t)
	storeA, _ := newTestStore(t, server, "epp-a")
	storeB, _ := newTestStore(t, server, "epp-b")
	ctx := context.Background()

	actual, existed, err := storeA.GetOrSet(ctx, "request", "request-id", "a")
	require.NoError(t, err)
	require.False(t, existed)
	require.Equal(t, "a", actual)

	actual, existed, err = storeB.GetOrSet(ctx, "request", "request-id", "b")
	require.NoError(t, err)
	require.True(t, existed)
	require.Equal(t, "a", actual)
}

func TestGetOrSetIsLinearizable(t *testing.T) {
	server := miniredis.RunT(t)
	const callers = 16
	type result struct {
		actual  any
		existed bool
		err     error
	}
	results := make(chan result, callers)

	var wg sync.WaitGroup
	for i := range callers {
		store, _ := newTestStore(t, server, fmt.Sprintf("epp-%d", i))
		wg.Go(func() {
			actual, existed, err := store.GetOrSet(
				context.Background(), "request", "request-id", store.replicaID,
			)
			results <- result{actual: actual, existed: existed, err: err}
		})
	}
	wg.Wait()
	close(results)

	winners := 0
	var winner any
	allResults := make([]result, 0, callers)
	for result := range results {
		require.NoError(t, result.err)
		allResults = append(allResults, result)
		if !result.existed {
			winners++
			winner = result.actual
		}
	}
	require.Equal(t, 1, winners)
	for _, result := range allResults {
		require.Equal(t, winner, result.actual)
	}
}

func TestGetOrSetDoesNotCollideWithEndpointHashes(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	ctx := context.Background()

	require.NoError(t, store.Set(ctx, "request", "request-id", 4, sumInts))
	actual, existed, err := store.GetOrSet(ctx, "request", "request-id", "decision")
	require.NoError(t, err)
	require.False(t, existed)
	require.Equal(t, "decision", actual)

	total, ok := getSum(t, store, "request", "request-id")
	require.True(t, ok)
	require.Equal(t, 4, total)
}

func TestGetOrSetExpires(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	store.ttl = time.Second
	ctx := context.Background()

	_, existed, err := store.GetOrSet(ctx, "request", "request-id", "first")
	require.NoError(t, err)
	require.False(t, existed)

	server.FastForward(2 * store.ttl)
	actual, existed, err := store.GetOrSet(ctx, "request", "request-id", "second")
	require.NoError(t, err)
	require.False(t, existed)
	require.Equal(t, "second", actual)
}

func TestGetOrSetPropagatesRedisErrors(t *testing.T) {
	recorder := &commandRecorder{err: errors.New("redis unavailable")}
	client := goredis.NewClient(&goredis.Options{})
	client.AddHook(recorder)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	store := &RedisStateStore{client: client, ttl: testTTL}

	actual, existed, err := store.GetOrSet(context.Background(), "request", "request-id", "candidate")

	require.ErrorContains(t, err, "redis unavailable")
	require.False(t, existed)
	require.Nil(t, actual)
}

func TestGetOrSetRejectsNilCandidate(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)

	actual, existed, err := store.GetOrSet(context.Background(), "request", "request-id", nil)

	require.ErrorContains(t, err, "candidate must not be nil")
	require.False(t, existed)
	require.Nil(t, actual)
}

func TestGetOrSetUsesOneAtomicRedisCommand(t *testing.T) {
	recorder := &commandRecorder{err: goredis.Nil}
	client := goredis.NewClient(&goredis.Options{})
	client.AddHook(recorder)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	store := &RedisStateStore{client: client, ttl: testTTL}

	actual, existed, err := store.GetOrSet(context.Background(), "request", "request-id", "candidate")

	require.NoError(t, err)
	require.False(t, existed)
	require.Equal(t, "candidate", actual)
	require.Len(t, recorder.commands, 1)
	args := recorder.commands[0].Args()
	require.Len(t, args, 7)
	require.Equal(t, "set", args[0])
	require.Equal(t, store.coordKey("request", "request-id"), args[1])
	_, ok := args[2].([]byte)
	require.True(t, ok)
	require.Equal(t, []any{"ex", int64(60), "NX", "get"}, args[3:])
}

func TestFactoryConfiguration(t *testing.T) {
	server := miniredis.RunT(t)
	testCases := []struct {
		name    string
		params  string
		wantErr string
		wantTTL time.Duration
	}{
		{
			name:    "defaults",
			params:  fmt.Sprintf("{\"address\": %q}", server.Addr()),
			wantTTL: defaultTTL,
		},
		{
			name:    "explicit TTL",
			params:  fmt.Sprintf("{\"address\": %q, \"ttl\": \"30s\"}", server.Addr()),
			wantTTL: 30 * time.Second,
		},
		{
			name:    "invalid TTL",
			params:  fmt.Sprintf("{\"address\": %q, \"ttl\": \"soon\"}", server.Addr()),
			wantErr: "invalid ttl",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(tc.params)), nil)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			store := plugin.(*RedisStateStore)
			t.Cleanup(func() { require.NoError(t, store.client.Close()) })
			require.Equal(t, tc.wantTTL, store.ttl)
			require.Equal(t, RedisStateStoreType, store.TypedName().Type)
			require.Equal(t, "redis", store.TypedName().Name)
		})
	}
}
