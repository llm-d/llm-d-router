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
	"errors"
	"fmt"
	"io"
	"net"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	miniredisserver "github.com/alicebob/miniredis/v2/server"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

type batchPipelineHook struct {
	run func(context.Context, []goredis.Cmder, goredis.ProcessPipelineHook) error
}

func (h batchPipelineHook) DialHook(next goredis.DialHook) goredis.DialHook          { return next }
func (h batchPipelineHook) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook { return next }
func (h batchPipelineHook) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []goredis.Cmder) error {
		if len(cmds) == 0 || cmds[0].Name() != "multi" {
			return next(ctx, cmds)
		}
		return h.run(ctx, cmds, next)
	}
}

func batchIntSpec() fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey:  testStateKey,
		Read:      func(string) fwkdl.Cloneable { return cloneableInt(4) },
		Aggregate: sumCloneableInts,
	}
}

func batchEndpointIDs(n int) []string {
	ids := make([]string, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("ns/endpoint-%d", i)
	}
	return ids
}

func TestSetBatchMatchesSetAcrossChunks(t *testing.T) {
	server := miniredis.RunT(t)
	serial, _ := newTestSyncer(t, server, testReplicaID)
	batch, _ := newTestSyncer(t, server, testReplicaID)
	ids := batchEndpointIDs(205)
	original := slices.Clone(ids)
	spec := batchIntSpec()
	spec.Aggregate = func(values []any) any { return slices.Clone(values) }
	for i, id := range ids {
		key := serial.hashKey(spec.StateKey, id)
		serial.cache.Store(key, &aggregateCacheEntry{value: "old"})
		batch.cache.Store(key, &aggregateCacheEntry{value: "old"})
		if i%3 != 0 {
			for _, peer := range []struct {
				id    string
				value cloneableInt
			}{{"epp-z", 9}, {"epp-b", 7}} {
				data, err := encodeValue(peer.value)
				require.NoError(t, err)
				server.HSet(key, peer.id, string(data))
			}
		}
		server.HSet(key, "corrupt", "not-gob")
		require.NoError(t, serial.Set(context.Background(), spec, id))
	}
	var mu sync.Mutex
	var sizes []int
	batch.client.AddHook(batchPipelineHook{run: func(ctx context.Context, cmds []goredis.Cmder, next goredis.ProcessPipelineHook) error {
		assert.Equal(t, "multi", cmds[0].Name())
		assert.Equal(t, "exec", cmds[len(cmds)-1].Name())
		mu.Lock()
		sizes = append(sizes, (len(cmds)-2)/4)
		mu.Unlock()
		return next(ctx, cmds)
	}})
	require.Empty(t, batch.SetBatch(context.Background(), spec, ids, time.Second))
	require.ElementsMatch(t, []int{100, 100, 5}, sizes)
	require.Equal(t, original, ids)
	for i, id := range ids {
		key := serial.hashKey(spec.StateKey, id)
		want, wantOK := serial.cache.Load(key)
		got, gotOK := batch.cache.Load(key)
		require.Equal(t, wantOK, gotOK, id)
		if i%3 == 0 {
			require.False(t, gotOK, id)
			continue
		}
		require.Equal(t, []any{cloneableInt(7), cloneableInt(9)}, got.(*aggregateCacheEntry).value)
		require.Equal(t, want.(*aggregateCacheEntry).value, got.(*aggregateCacheEntry).value)
	}
	require.Empty(t, batch.SetBatch(context.Background(), spec, nil, time.Second))
	require.Len(t, sizes, 3)
}

func TestSetBatchEndpointFailures(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	ids := batchEndpointIDs(3)
	spec := batchIntSpec()
	spec.Read = func(id string) fwkdl.Cloneable {
		if id == ids[0] {
			return nil
		}
		return cloneableInt(4)
	}
	require.NoError(t, server.Set(syncer.hashKey(spec.StateKey, ids[1]), "wrong-type"))
	old := &aggregateCacheEntry{value: cloneableInt(99)}
	for _, id := range ids {
		syncer.cache.Store(syncer.hashKey(spec.StateKey, id), old)
	}
	failures := syncer.SetBatch(context.Background(), spec, ids, time.Second)
	require.Len(t, failures, 2)
	require.ErrorContains(t, failures[ids[0]], "encode")
	require.ErrorContains(t, failures[ids[1]], "WRONGTYPE")
	require.False(t, server.Exists(syncer.hashKey(spec.StateKey, ids[0])))
	require.NotEmpty(t, server.HGet(syncer.hashKey(spec.StateKey, ids[2]), testReplicaID))
	for _, id := range ids[:2] {
		entry, ok := syncer.cache.Load(syncer.hashKey(spec.StateKey, id))
		require.True(t, ok)
		require.Same(t, old, entry)
	}
	_, ok := syncer.cache.Load(syncer.hashKey(spec.StateKey, ids[2]))
	require.False(t, ok)
}

func TestSetBatchChunkFailuresKeepCaches(t *testing.T) {
	for _, mode := range []string{"execabort", "transport", "unattributed"} {
		t.Run(mode, func(t *testing.T) {
			server := miniredis.RunT(t)
			syncer, _ := newTestSyncer(t, server, testReplicaID)
			ids := batchEndpointIDs(3)
			old := &aggregateCacheEntry{value: cloneableInt(99)}
			for _, id := range ids {
				syncer.cache.Store(syncer.hashKey(testStateKey, id), old)
			}
			var returned error
			syncer.client.AddHook(batchPipelineHook{run: func(ctx context.Context, cmds []goredis.Cmder, next goredis.ProcessPipelineHook) error {
				switch mode {
				case "execabort":
					cmds[1] = goredis.NewIntCmd(ctx, "hset", "malformed")
				case "transport":
					returned = io.ErrUnexpectedEOF
					for _, cmd := range cmds {
						cmd.SetErr(returned)
					}
					return returned
				}
				returned = next(ctx, cmds)
				if mode == "unattributed" {
					for _, cmd := range cmds {
						assert.NoError(t, cmd.Err())
					}
					returned = goredis.ErrCrossSlot
				}
				return returned
			}})
			failures := syncer.SetBatch(context.Background(), batchIntSpec(), ids, time.Second)
			require.Len(t, failures, len(ids))
			require.Error(t, returned)
			if mode == "execabort" {
				require.True(t, goredis.HasErrorPrefix(returned, "EXECABORT"))
			}
			for _, id := range ids {
				require.ErrorIs(t, failures[id], returned)
				entry, ok := syncer.cache.Load(syncer.hashKey(testStateKey, id))
				require.True(t, ok)
				require.Same(t, old, entry)
				if mode != "unattributed" {
					require.False(t, server.Exists(syncer.hashKey(testStateKey, id)))
				}
			}
		})
	}
}

type batchRetryConn struct {
	net.Conn
	armed    *atomic.Bool
	executed <-chan struct{}
}

func (c batchRetryConn) Write(data []byte) (int, error) {
	n, err := c.Conn.Write(data)
	if err == nil && c.armed.CompareAndSwap(true, false) {
		select {
		case <-c.executed:
		case <-time.After(time.Second):
		}
		return n, io.ErrUnexpectedEOF
	}
	return n, err
}

func TestSetBatchRetriedWriteKeepsCaches(t *testing.T) {
	server := miniredis.RunT(t)
	var armed atomic.Bool
	var execs atomic.Int64
	executed := make(chan struct{}, 2)
	server.Server().SetPreHook(func(_ *miniredisserver.Peer, cmd string, _ ...string) bool {
		if cmd == "EXEC" {
			execs.Add(1)
			executed <- struct{}{}
		}
		return false
	})
	client := goredis.NewClient(&goredis.Options{
		Addr: server.Addr(), MaxRetries: 1, MinRetryBackoff: time.Millisecond, MaxRetryBackoff: time.Millisecond,
		Dialer: func(ctx context.Context, network, addr string) (net.Conn, error) {
			conn, err := (&net.Dialer{}).DialContext(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			return batchRetryConn{Conn: conn, armed: &armed, executed: executed}, nil
		},
	})
	client.AddHook(&miniredisHExpireCompatibilityHook{})
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	require.NoError(t, client.Ping(context.Background()).Err())
	syncer := &RedisSyncer{replicaID: testReplicaID, client: client, stateTTL: testStateTTL}
	ids := batchEndpointIDs(3)
	old := &aggregateCacheEntry{value: cloneableInt(99)}
	for _, id := range ids {
		syncer.cache.Store(syncer.hashKey(testStateKey, id), old)
	}
	client.AddHook(batchPipelineHook{run: func(ctx context.Context, cmds []goredis.Cmder, next goredis.ProcessPipelineHook) error {
		err := next(ctx, cmds)
		assert.ErrorIs(t, err, io.ErrUnexpectedEOF)
		for _, cmd := range cmds[1 : len(cmds)-1] {
			assert.NoError(t, cmd.Err())
		}
		return err
	}})
	armed.Store(true)
	failures := syncer.SetBatch(context.Background(), batchIntSpec(), ids, time.Second)
	require.Len(t, failures, len(ids))
	require.Eventually(t, func() bool { return execs.Load() == 2 }, time.Second, time.Millisecond)
	for _, id := range ids {
		require.ErrorIs(t, failures[id], io.ErrUnexpectedEOF)
		entry, ok := syncer.cache.Load(syncer.hashKey(testStateKey, id))
		require.True(t, ok)
		require.Same(t, old, entry)
	}
}

func TestSetBatchReportsUnattemptedAfterDeadline(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	ids := batchEndpointIDs(batchConcurrency*batchSize + 5)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	var calls atomic.Int64
	syncer.client.AddHook(batchPipelineHook{run: func(ctx context.Context, cmds []goredis.Cmder, next goredis.ProcessPipelineHook) error {
		calls.Add(1)
		firstChunk := cmds[1].Args()[1] == syncer.hashKey(testStateKey, ids[0])
		var err error
		if firstChunk {
			err = next(ctx, cmds)
		}
		<-ctx.Done()
		if firstChunk {
			return err
		}
		return ctx.Err()
	}})
	failures := syncer.SetBatch(ctx, batchIntSpec(), ids, time.Second)
	require.Equal(t, int64(batchConcurrency), calls.Load())
	require.Len(t, failures, len(ids)-batchSize)
	for _, id := range ids[:batchSize] {
		require.NotContains(t, failures, id)
	}
	for _, id := range ids[batchSize:] {
		require.ErrorIs(t, failures[id], context.DeadlineExceeded)
		require.False(t, server.Exists(syncer.hashKey(testStateKey, id)))
	}
	calls.Store(0)
	failures = syncer.SetBatch(ctx, batchIntSpec(), ids, time.Second)
	require.Len(t, failures, len(ids))
	require.Zero(t, calls.Load())
}

func TestSetBatchChunkTimeoutFailsOnlySlowChunks(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	ids := batchEndpointIDs(batchConcurrency*batchSize + 5)
	// Stall every chunk except the tail: the tail can only publish if its own
	// chunk gets a fresh timeout after waiting for a slot.
	syncer.client.AddHook(batchPipelineHook{run: func(ctx context.Context, cmds []goredis.Cmder, next goredis.ProcessPipelineHook) error {
		if cmds[1].Args()[1] != syncer.hashKey(testStateKey, ids[batchConcurrency*batchSize]) {
			time.Sleep(500 * time.Millisecond)
		}
		return next(ctx, cmds)
	}})
	failures := syncer.SetBatch(context.Background(), batchIntSpec(), ids, 200*time.Millisecond)
	require.Len(t, failures, batchConcurrency*batchSize)
	for _, id := range ids[:batchConcurrency*batchSize] {
		require.ErrorIs(t, failures[id], context.DeadlineExceeded)
	}
	for _, id := range ids[batchConcurrency*batchSize:] {
		require.NotContains(t, failures, id)
		require.True(t, server.Exists(syncer.hashKey(testStateKey, id)))
	}
}

func TestSetBatchAppliesFieldAndKeyTTLs(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	ids := batchEndpointIDs(101)
	require.Empty(t, syncer.SetBatch(context.Background(), batchIntSpec(), ids, time.Second))
	for _, id := range ids {
		key := syncer.hashKey(testStateKey, id)
		require.Equal(t, testStateTTL, server.TTL(key))
		server.HSet(key, "peer", "value")
		server.SetTTL(key, 2*testStateTTL)
	}
	server.FastForward(testStateTTL + time.Second)
	for _, id := range ids {
		key := syncer.hashKey(testStateKey, id)
		require.Empty(t, server.HGet(key, testReplicaID))
		require.Equal(t, "value", server.HGet(key, "peer"))
	}
}

func TestSetBatchConcurrentContributors(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	ids := batchEndpointIDs(101)
	failures := make(chan map[string]error, 2)
	var wg sync.WaitGroup
	for _, key := range []fwkdl.StateKey{testStateKey, "prefix"} {
		wg.Go(func() {
			spec := batchIntSpec()
			spec.StateKey = key
			failures <- syncer.SetBatch(context.Background(), spec, ids, time.Second)
		})
	}
	wg.Wait()
	close(failures)
	for errs := range failures {
		require.Empty(t, errs)
	}
	for _, key := range []fwkdl.StateKey{testStateKey, "prefix"} {
		for _, id := range ids {
			require.NotEmpty(t, server.HGet(syncer.hashKey(key, id), testReplicaID))
		}
	}
}

func TestSetBatchAllEncodeFailures(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	spec := batchIntSpec()
	spec.Read = func(string) fwkdl.Cloneable { return nil }
	syncer.client.AddHook(batchPipelineHook{run: func(context.Context, []goredis.Cmder, goredis.ProcessPipelineHook) error {
		t.Error("no encodable endpoints should send a pipeline")
		return errors.New("unexpected pipeline")
	}})
	ids := batchEndpointIDs(101)
	require.Len(t, syncer.SetBatch(context.Background(), spec, ids, time.Second), len(ids))
}

func TestSetBatchConcurrentChunksPreserveEndpointResults(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	ids := batchEndpointIDs(505)
	original := slices.Clone(ids)
	spec := batchIntSpec()
	spec.Read = func(id string) fwkdl.Cloneable {
		if id == ids[0] {
			return nil
		}
		return cloneableInt(4)
	}
	peer, err := encodeValue(cloneableInt(7))
	require.NoError(t, err)
	old := &aggregateCacheEntry{value: cloneableInt(99)}
	for _, id := range ids {
		key := syncer.hashKey(spec.StateKey, id)
		server.HSet(key, "peer", string(peer))
		syncer.cache.Store(key, old)
	}
	server.Del(syncer.hashKey(spec.StateKey, ids[104]))
	require.NoError(t, server.Set(syncer.hashKey(spec.StateKey, ids[104]), "wrong-type"))
	failures := syncer.SetBatch(context.Background(), spec, ids, time.Second)
	require.Len(t, failures, 2)
	require.ErrorContains(t, failures[ids[0]], "encode")
	require.ErrorContains(t, failures[ids[104]], "WRONGTYPE")
	require.Equal(t, original, ids)
	for _, id := range ids {
		entry, ok := syncer.cache.Load(syncer.hashKey(spec.StateKey, id))
		require.True(t, ok)
		if failures[id] != nil {
			require.Same(t, old, entry)
		} else {
			require.Equal(t, cloneableInt(7), entry.(*aggregateCacheEntry).value)
		}
	}
}

func TestSetBatchConcurrentChunksBoundWorkAfterCancellation(t *testing.T) {
	server := miniredis.RunT(t)
	syncer, _ := newTestSyncer(t, server, testReplicaID)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{}, batchConcurrency)
	var calls atomic.Int64
	var reads atomic.Int64
	syncer.client.AddHook(batchPipelineHook{run: func(ctx context.Context, _ []goredis.Cmder, _ goredis.ProcessPipelineHook) error {
		calls.Add(1)
		started <- struct{}{}
		<-ctx.Done()
		return ctx.Err()
	}})
	spec := batchIntSpec()
	spec.Read = func(string) fwkdl.Cloneable { reads.Add(1); return cloneableInt(4) }
	ids := batchEndpointIDs(batchConcurrency*batchSize + 5)
	old := &aggregateCacheEntry{value: cloneableInt(99)}
	for _, id := range ids {
		syncer.cache.Store(syncer.hashKey(spec.StateKey, id), old)
	}
	done := make(chan map[string]error, 1)
	go func() { done <- syncer.SetBatch(ctx, spec, ids, time.Second) }()
	for range batchConcurrency {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("concurrent chunks did not start")
		}
	}
	require.Equal(t, int64(batchConcurrency), calls.Load())
	cancel()
	select {
	case failures := <-done:
		require.Len(t, failures, len(ids))
		for _, id := range ids {
			require.ErrorIs(t, failures[id], context.Canceled)
			entry, ok := syncer.cache.Load(syncer.hashKey(spec.StateKey, id))
			require.True(t, ok)
			require.Same(t, old, entry)
		}
	case <-time.After(time.Second):
		t.Fatal("canceled chunks did not finish")
	}
	require.Equal(t, int64(batchConcurrency), calls.Load())
	require.Equal(t, int64(batchConcurrency*batchSize), reads.Load())
}
