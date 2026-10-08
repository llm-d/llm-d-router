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
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	testEndpointID      = "vortex/backend-rank-0"
	testReplicaID       = "epp-a"
	testStateTTL        = time.Minute
	testCoordinationTTL = 3 * time.Minute
)

var testStateKey = fwkdl.StateKey("inflight")

type customValue struct {
	Count int
}

type cloneableInt int

func (v cloneableInt) Clone() fwkdl.Cloneable { return v }

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

type miniredisHExpireCompatibilityHook struct{}

func (h *miniredisHExpireCompatibilityHook) DialHook(next goredis.DialHook) goredis.DialHook {
	return next
}

func (h *miniredisHExpireCompatibilityHook) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook {
	return func(ctx context.Context, cmd goredis.Cmder) error {
		rewriteHPExpireForMiniredis(cmd)
		return next(ctx, cmd)
	}
}

func (h *miniredisHExpireCompatibilityHook) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []goredis.Cmder) error {
		for _, cmd := range cmds {
			rewriteHPExpireForMiniredis(cmd)
		}
		return next(ctx, cmds)
	}
}

func rewriteHPExpireForMiniredis(cmd goredis.Cmder) {
	if !strings.EqualFold(cmd.Name(), "HPEXPIRE") {
		return
	}
	args := cmd.Args()
	ttlMillis, ok := args[2].(int64)
	if !ok {
		return
	}
	args[0] = "HEXPIRE"
	args[2] = (ttlMillis + 999) / 1000
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
	client.AddHook(&miniredisHExpireCompatibilityHook{})
	client.AddHook(counter)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	return &RedisStateStore{
		replicaID:       replicaID,
		client:          client,
		stateTTL:        testStateTTL,
		coordinationTTL: testCoordinationTTL,
	}, counter
}

func seedReplica(t testing.TB, server *miniredis.Miniredis, key fwkdl.StateKey, replicaID string, value any) {
	t.Helper()
	data, err := encodeValue(value)
	require.NoError(t, err)
	server.HSet(string(key)+":"+testEndpointID, replicaID, string(data))
}

func sumInts(values []any) any {
	total := 0
	for _, value := range values {
		total += value.(int)
	}
	return total
}

func sumCloneableInts(values []any) any {
	var total cloneableInt
	for _, value := range values {
		total += value.(cloneableInt)
	}
	return total
}

func sumCustomValues(values []any) any {
	total := &customValue{}
	for _, value := range values {
		total.Count += value.(*customValue).Count
	}
	return total
}

func getSum(t testing.TB, store *RedisStateStore, key fwkdl.StateKey, endpointID string, local any) int {
	t.Helper()
	value := store.get(context.Background(), key, endpointID, local, sumInts)
	return value.(int)
}

func TestSetPreparesPeerAggregateAndGetCombinesLiveLocal(t *testing.T) {
	server := miniredis.RunT(t)
	store, counter := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, "epp-b", cloneableInt(7))
	local := cloneableInt(4)
	var aggregateCalls atomic.Int64
	aggregate := func(values []any) any {
		aggregateCalls.Add(1)
		return sumCloneableInts(values)
	}
	spec := fwkdl.CrossReplicaSpec{
		StateKey:  testStateKey,
		Read:      func(string) fwkdl.Cloneable { return local },
		Aggregate: aggregate,
	}

	require.NoError(t, store.Set(context.Background(), spec, testEndpointID))
	require.Equal(t, int64(1), counter.count.Load())
	require.Equal(t, int64(1), aggregateCalls.Load())
	cached, ok := store.cache.Load(store.hashKey(testStateKey, testEndpointID))
	require.True(t, ok)
	require.Equal(t, cloneableInt(7), cached.(*aggregateCacheEntry).value)

	value, ok, err := store.Get(context.Background(), spec, testEndpointID)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, cloneableInt(11), value)

	local = 5
	value, ok, err = store.Get(context.Background(), spec, testEndpointID)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, cloneableInt(12), value)

	require.Equal(t, int64(1), counter.count.Load(), "Get must not read Redis")
	require.Equal(t, int64(3), aggregateCalls.Load())
}

func TestGetAggregatesLiveLocalWithoutPeers(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	hashKey := store.hashKey(testStateKey, testEndpointID)
	store.cache.Store(hashKey, &aggregateCacheEntry{value: 7, expiresAt: time.Now().Add(testStateTTL)})

	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	_, cached := store.cache.Load(hashKey)
	require.False(t, cached)
	total := getSum(t, store, testStateKey, testEndpointID, 5)
	require.Equal(t, 5, total)
}

func TestSetAndGetConcurrent(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	spec := fwkdl.CrossReplicaSpec{
		StateKey:  testStateKey,
		Read:      func(string) fwkdl.Cloneable { return cloneableInt(1) },
		Aggregate: sumCloneableInts,
	}

	const iterations = 50
	errs := make(chan error, 4*iterations)
	var wg sync.WaitGroup
	for range 2 {
		wg.Go(func() {
			for range iterations {
				errs <- store.Set(context.Background(), spec, testEndpointID)
				_, _, err := store.Get(context.Background(), spec, testEndpointID)
				errs <- err
			}
		})
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		require.NoError(t, err)
	}
}

func TestSetSupportsUnregisteredConcreteValue(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, "epp-b", &customValue{Count: 7})

	require.NoError(t, store.set(
		context.Background(), testStateKey, testEndpointID, &customValue{Count: 4}, sumCustomValues,
	))
	value := store.get(
		context.Background(), testStateKey, testEndpointID,
		&customValue{Count: 4}, sumCustomValues,
	)

	require.Equal(t, &customValue{Count: 11}, value)
}

func TestEncodeValueUsesConcreteValueStream(t *testing.T) {
	data, err := encodeValue(&customValue{Count: 7})
	require.NoError(t, err)

	decoder := gob.NewDecoder(bytes.NewReader(data))
	var value customValue
	require.NoError(t, decoder.Decode(&value))
	require.Equal(t, customValue{Count: 7}, value)
}

func TestEncodeValueSupportsInFlightLoad(t *testing.T) {
	want := &attrconcurrency.InFlightLoad{Tokens: 7, Requests: 2}
	data, err := encodeValue(want)
	require.NoError(t, err)

	actual, err := gobDecode(data, &attrconcurrency.InFlightLoad{})
	require.NoError(t, err)
	require.Equal(t, want, actual)
}

func TestSetRefreshesPreparedAggregate(t *testing.T) {
	server := miniredis.RunT(t)
	store, counter := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, "epp-b", 7)
	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	seedReplica(t, server, testStateKey, "epp-b", 9)
	total := getSum(t, store, testStateKey, testEndpointID, 4)
	require.Equal(t, 11, total)

	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	total = getSum(t, store, testStateKey, testEndpointID, 4)
	require.Equal(t, 13, total)
	require.Equal(t, int64(2), counter.count.Load())
}

func TestSetAggregatesPeersInReplicaIDOrder(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, "epp-z", "third")
	seedReplica(t, server, testStateKey, "epp-b", "first")
	seedReplica(t, server, testStateKey, "epp-m", "second")
	aggregate := func(values []any) any {
		parts := make([]string, 0, len(values))
		for _, value := range values {
			parts = append(parts, value.(string))
		}
		return strings.Join(parts, ",")
	}

	for range 20 {
		require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, "local", aggregate))
		cached, ok := store.cache.Load(store.hashKey(testStateKey, testEndpointID))
		require.True(t, ok)
		require.Equal(t, "first,second,third", cached.(*aggregateCacheEntry).value)
	}
}

func TestConcurrentSet(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	const callers = 16
	errs := make(chan error, callers)

	var wg sync.WaitGroup
	for i := range callers {
		wg.Go(func() {
			errs <- store.set(context.Background(), testStateKey, fmt.Sprintf("endpoint-%d", i), i, sumInts)
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
}

func TestGetReturnsLiveLocalBeforeSet(t *testing.T) {
	server := miniredis.RunT(t)
	store, counter := newTestStore(t, server, testReplicaID)

	value := store.get(context.Background(), testStateKey, testEndpointID, 4, sumInts)

	require.Equal(t, 4, value)
	require.Zero(t, counter.count.Load())
}

func TestSetSkipsUndecodableFields(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, "fresh", 7)
	server.HSet(store.hashKey(testStateKey, testEndpointID), "corrupt", "not-gob")

	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	total := getSum(t, store, testStateKey, testEndpointID, 4)
	require.Equal(t, 11, total)
}

func TestExpiredPeerAggregateFallsBackToLiveLocal(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	hashKey := store.hashKey(testStateKey, testEndpointID)
	store.cache.Store(hashKey, &aggregateCacheEntry{
		value:     11,
		expiresAt: time.Now().Add(-time.Second),
	})

	value := store.get(context.Background(), testStateKey, testEndpointID, 4, sumInts)

	require.Equal(t, 4, value)
	_, cached := store.cache.Load(hashKey)
	require.False(t, cached)
}

func TestSetExpiresAggregatesFromLocalRefreshTime(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	hashKey := store.hashKey(testStateKey, testEndpointID)
	seedReplica(t, server, testStateKey, "epp-b", 7)
	seedReplica(t, server, testStateKey, "epp-c", 9)

	refreshedAt := time.Now()
	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	cached, ok := store.cache.Load(hashKey)
	require.True(t, ok)
	entry := cached.(*aggregateCacheEntry)

	require.WithinDuration(t, refreshedAt.Add(testStateTTL), entry.expiresAt, 100*time.Millisecond)
}

func TestDeleteInvalidatesAggregateAndReplicaField(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	seedReplica(t, server, testStateKey, "epp-b", 7)
	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	require.NoError(t, store.Delete(context.Background(), testStateKey, testEndpointID))

	_, ok := store.cache.Load(store.hashKey(testStateKey, testEndpointID))
	require.False(t, ok)
	exists, err := store.client.HExists(context.Background(), store.hashKey(testStateKey, testEndpointID), testReplicaID).Result()
	require.NoError(t, err)
	require.False(t, exists)
}

func TestDeleteInvalidatesAggregateWhenRedisFails(t *testing.T) {
	recorder := &commandRecorder{err: errors.New("redis unavailable")}
	client := goredis.NewClient(&goredis.Options{})
	client.AddHook(recorder)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	store := &RedisStateStore{replicaID: testReplicaID, client: client, stateTTL: testStateTTL}
	hashKey := store.hashKey(testStateKey, testEndpointID)
	store.cache.Store(hashKey, &aggregateCacheEntry{value: 7, expiresAt: time.Now().Add(testStateTTL)})

	err := store.Delete(context.Background(), testStateKey, testEndpointID)

	require.ErrorContains(t, err, "redis unavailable")
	_, ok := store.cache.Load(hashKey)
	require.False(t, ok)
}

func TestSetWritesAndReadsAggregateInOneTransaction(t *testing.T) {
	recorder := &commandRecorder{}
	client := goredis.NewClient(&goredis.Options{})
	client.AddHook(recorder)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	store := &RedisStateStore{replicaID: testReplicaID, client: client, stateTTL: testStateTTL}
	hashKey := store.hashKey(testStateKey, testEndpointID)

	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	require.Len(t, recorder.commands, 6)
	require.Equal(t, "multi", recorder.commands[0].Name())
	require.Equal(t, "hset", recorder.commands[1].Name())
	require.Equal(t, []any{"HPEXPIRE", hashKey, int64(60000), "FIELDS", 1, testReplicaID}, recorder.commands[2].Args())
	require.Equal(t, "pexpire", recorder.commands[3].Name())
	require.Equal(t, []any{"pexpire", hashKey, int64(60000)}, recorder.commands[3].Args())
	require.Equal(t, "hgetall", recorder.commands[4].Name())
	require.Equal(t, "exec", recorder.commands[5].Name())
}

func TestSetAppliesFieldTTL(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	hashKey := store.hashKey(testStateKey, testEndpointID)

	require.NoError(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	server.HSet(hashKey, "peer", "value")
	server.SetTTL(hashKey, 2*testStateTTL)
	server.FastForward(testStateTTL + time.Second)

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

func TestGetOrSetSupportsUnregisteredConcreteValue(t *testing.T) {
	server := miniredis.RunT(t)
	storeA, _ := newTestStore(t, server, "epp-a")
	storeB, _ := newTestStore(t, server, "epp-b")
	ctx := context.Background()

	actual, existed, err := storeA.GetOrSet(ctx, "request", "request-id", &customValue{Count: 1})
	require.NoError(t, err)
	require.False(t, existed)
	require.Equal(t, &customValue{Count: 1}, actual)

	actual, existed, err = storeB.GetOrSet(ctx, "request", "request-id", &customValue{Count: 2})
	require.NoError(t, err)
	require.True(t, existed)
	require.Equal(t, &customValue{Count: 1}, actual)
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

	peer, err := encodeValue(7)
	require.NoError(t, err)
	server.HSet(store.hashKey("request", "request-id"), "epp-b", string(peer))
	require.NoError(t, store.set(ctx, "request", "request-id", 4, sumInts))
	actual, existed, err := store.GetOrSet(ctx, "request", "request-id", "decision")
	require.NoError(t, err)
	require.False(t, existed)
	require.Equal(t, "decision", actual)

	total := getSum(t, store, "request", "request-id", 4)
	require.Equal(t, 11, total)
}

func TestGetOrSetExpires(t *testing.T) {
	server := miniredis.RunT(t)
	store, _ := newTestStore(t, server, testReplicaID)
	store.coordinationTTL = time.Second
	ctx := context.Background()

	_, existed, err := store.GetOrSet(ctx, "request", "request-id", "first")
	require.NoError(t, err)
	require.False(t, existed)

	server.FastForward(2 * store.coordinationTTL)
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
	store := &RedisStateStore{client: client, coordinationTTL: testCoordinationTTL}

	actual, existed, err := store.GetOrSet(context.Background(), "request", "request-id", "candidate")

	require.ErrorContains(t, err, "redis unavailable")
	require.False(t, existed)
	require.Nil(t, actual)
}

func TestSetRecoversAfterRedisBecomesAvailable(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	address := listener.Addr().String()
	require.NoError(t, listener.Close())

	client := goredis.NewClient(&goredis.Options{
		Addr:         address,
		DialTimeout:  100 * time.Millisecond,
		ReadTimeout:  100 * time.Millisecond,
		WriteTimeout: 100 * time.Millisecond,
		MaxRetries:   0,
	})
	client.AddHook(&miniredisHExpireCompatibilityHook{})
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	store := &RedisStateStore{replicaID: testReplicaID, client: client, stateTTL: testStateTTL}

	require.Error(t, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))

	server := miniredis.NewMiniRedis()
	require.NoError(t, server.StartAddr(address))
	t.Cleanup(server.Close)
	seedReplica(t, server, testStateKey, "epp-b", 7)

	require.EventuallyWithT(t, func(collect *assert.CollectT) {
		assert.NoError(collect, store.set(context.Background(), testStateKey, testEndpointID, 4, sumInts))
	}, time.Second, 10*time.Millisecond)
	total := getSum(t, store, testStateKey, testEndpointID, 4)
	require.Equal(t, 11, total)
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
	store := &RedisStateStore{client: client, coordinationTTL: testCoordinationTTL}

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
	require.Equal(t, []any{"ex", int64(180), "NX", "get"}, args[3:])
}

func TestFactoryConfiguration(t *testing.T) {
	server := miniredis.RunT(t)
	testCases := []struct {
		name                string
		params              string
		wantErr             string
		wantStateTTL        time.Duration
		wantCoordinationTTL time.Duration
	}{
		{
			name:                "defaults",
			params:              fmt.Sprintf("{\"address\": %q}", server.Addr()),
			wantStateTTL:        defaultStateTTL,
			wantCoordinationTTL: defaultCoordinationTTL,
		},
		{
			name:                "explicit TTLs",
			params:              fmt.Sprintf("{\"address\": %q, \"stateTTL\": \"500ms\", \"coordinationTTL\": \"30s\"}", server.Addr()),
			wantStateTTL:        500 * time.Millisecond,
			wantCoordinationTTL: 30 * time.Second,
		},
		{
			name:    "invalid state TTL",
			params:  fmt.Sprintf("{\"address\": %q, \"stateTTL\": \"soon\"}", server.Addr()),
			wantErr: "invalid stateTTL",
		},
		{
			name:    "invalid coordination TTL",
			params:  fmt.Sprintf("{\"address\": %q, \"coordinationTTL\": \"soon\"}", server.Addr()),
			wantErr: "invalid coordinationTTL",
		},
		{
			name:    "zero state TTL",
			params:  fmt.Sprintf("{\"address\": %q, \"stateTTL\": \"0s\"}", server.Addr()),
			wantErr: "stateTTL must be at least 1ms",
		},
		{
			name:    "sub-millisecond coordination TTL",
			params:  fmt.Sprintf("{\"address\": %q, \"coordinationTTL\": \"500us\"}", server.Addr()),
			wantErr: "coordinationTTL must be at least 1ms",
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
			require.Equal(t, tc.wantStateTTL, store.stateTTL)
			require.Equal(t, tc.wantCoordinationTTL, store.coordinationTTL)
			require.Equal(t, RedisStateStoreType, store.TypedName().Type)
			require.Equal(t, "redis", store.TypedName().Name)
		})
	}
}

func TestFactoryDoesNotRequireRedisAvailability(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	address := listener.Addr().String()
	require.NoError(t, listener.Close())

	plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(
		fmt.Sprintf(`{"address": %q}`, address),
	)), nil)

	require.NoError(t, err)
	store := plugin.(*RedisStateStore)
	t.Cleanup(func() { require.NoError(t, store.client.Close()) })
}

func TestFactoryClosesClientWithHandleContext(t *testing.T) {
	server := miniredis.RunT(t)
	ctx, cancel := context.WithCancel(context.Background())
	handle := fwkplugin.NewEppHandle(ctx, nil)
	plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(
		fmt.Sprintf(`{"address": %q}`, server.Addr()),
	)), handle)
	require.NoError(t, err)
	store := plugin.(*RedisStateStore)

	cancel()

	require.Eventually(t, func() bool {
		return errors.Is(store.client.Ping(context.Background()).Err(), goredis.ErrClosed)
	}, time.Second, 10*time.Millisecond)
}

func TestFactoryGeneratesDistinctReplicaIDs(t *testing.T) {
	server := miniredis.RunT(t)
	newStore := func() *RedisStateStore {
		plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(
			fmt.Sprintf(`{"address": %q}`, server.Addr()),
		)), nil)
		require.NoError(t, err)
		store := plugin.(*RedisStateStore)
		t.Cleanup(func() { require.NoError(t, store.client.Close()) })
		return store
	}

	storeA := newStore()
	storeB := newStore()

	require.NotEqual(t, storeA.replicaID, storeB.replicaID)
}
