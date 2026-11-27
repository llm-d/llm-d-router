// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
	"github.com/stretchr/testify/require"
)

type indexConsumers []*kvevents.Pool

func (c indexConsumers) ProcessEvents(ctx context.Context, source kvevents.EventSource, batch kvevents.EventBatch) error {
	for _, pool := range c {
		if err := pool.ProcessEvents(ctx, source, batch); err != nil {
			return err
		}
	}
	return nil
}
func (c indexConsumers) Reset(ctx context.Context, source kvevents.EventSource) error {
	for _, pool := range c {
		if err := pool.Reset(ctx, source); err != nil {
			return err
		}
	}
	return nil
}

func TestDecodedConsumersRecoverAndEvictTogether(t *testing.T) {
	for _, snapshot := range []bool{false, true} {
		name := "replay"
		if snapshot {
			name = "snapshot"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			var consumers indexConsumers
			var indexes []kvblock.Index
			for range 2 {
				index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
				require.NoError(t, err)
				tokens, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
				require.NoError(t, err)
				pool, err := kvevents.NewPool(kvevents.DefaultConfig(), index, tokens, nil)
				require.NoError(t, err)
				defer pool.Shutdown(ctx)
				consumers = append(consumers, pool)
				indexes = append(indexes, index)
			}
			transport, err := kvevents.NewConsumerPool(kvevents.DefaultConfig(), engineadapter.NewVLLMAdapter(), consumers)
			require.NoError(t, err)
			transport.Start(ctx)
			defer transport.Shutdown(ctx)
			pub := zmq4.NewPub(ctx)
			endpoint := listenEphemeral(t, pub)
			defer pub.Close()
			replayEndpoint, snapshotEndpoint := "", ""
			var server *snapshotServer
			var buffer *replayBuffer
			if snapshot {
				server, snapshotEndpoint = startSnapshotServer(t, ctx)
				server.set(4, buildDistinctBlockStoredPayload(t, 100))
			} else {
				buffer, replayEndpoint = startReplayBuffer(t, ctx)
				buffer.set(replayMessage{seq: 4, payload: buildDistinctBlockStoredPayload(t, 100)})
			}
			manager := kvevents.NewSubscriberManager(transport)
			defer manager.Shutdown(ctx)
			require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "10.0.0.1:8000", endpoint, replayEndpoint, snapshotEndpoint, "kv@", true))
			h := &replayHarness{ctx: ctx, pub: pub, topic: []byte("kv@wire:8000@TestModel")}
			resident := func(hash uint64, want bool) bool {
				for _, index := range indexes {
					key, err := index.GetRequestKey(ctx, kvevents.ModelEngineKey("TestModel", hash))
					if err != nil {
						if !want {
							continue
						}
						return false
					}
					hits, err := index.Lookup(ctx, []kvblock.BlockHash{key}, nil)
					if err != nil || (len(hits[key]) > 0) != want {
						return false
					}
				}
				return true
			}
			require.Eventually(t, func() bool {
				h.send(t, 4, buildEventBatchPayload(t))
				return resident(100, true)
			}, 5*time.Second, 20*time.Millisecond)
			h.send(t, 5, buildDistinctBlockStoredPayload(t, 200))
			require.Eventually(t, func() bool { return resident(200, true) }, time.Second, time.Millisecond)
			h.send(t, 6, buildBlockRemovedPayload(t, 100))
			require.Eventually(t, func() bool { return resident(100, false) }, time.Second, time.Millisecond)
			if snapshot {
				server.set(8, buildDistinctBlockStoredPayload(t, 300))
			} else {
				buffer.set(replayMessage{seq: 7, payload: buildBlockRemovedPayload(t, 200)}, replayMessage{seq: 8, payload: buildDistinctBlockStoredPayload(t, 300)})
			}
			h.send(t, 8, buildDistinctBlockStoredPayload(t, 300))
			require.Eventually(t, func() bool { return resident(200, false) && resident(300, true) }, 5*time.Second, time.Millisecond)
			require.True(t, manager.RemoveSubscriber(ctx, "pod"))
			require.Eventually(t, func() bool { return resident(300, false) }, time.Second, time.Millisecond)
		})
	}
}

type countingConsumer struct {
	batches, resets, resyncResets, resyncAborts, resyncBatches atomic.Int32
	resyncThrough                                              atomic.Uint64
}

func (c *countingConsumer) ProcessEvents(_ context.Context, source kvevents.EventSource, _ kvevents.EventBatch) error {
	if source.Resync {
		c.resyncBatches.Add(1)
		return nil
	}
	c.batches.Add(1)
	return nil
}

func (c *countingConsumer) Reset(_ context.Context, source kvevents.EventSource) error {
	if source.ResyncAborted {
		c.resyncAborts.Add(1)
		return nil
	}
	if source.Resync {
		c.resyncResets.Add(1)
		c.resyncThrough.Store(source.Sequence)
		return nil
	}
	c.resets.Add(1)
	return nil
}

func TestResyncRequestReloadsSnapshotOnce(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	consumer := &countingConsumer{}
	transport, err := kvevents.NewConsumerPool(kvevents.DefaultConfig(), engineadapter.NewVLLMAdapter(), consumer)
	require.NoError(t, err)
	transport.Start(ctx)
	defer transport.Shutdown(ctx)
	pub := zmq4.NewPub(ctx)
	endpoint := listenEphemeral(t, pub)
	defer pub.Close()
	server, snapshotEndpoint := startSnapshotServer(t, ctx)
	manager := kvevents.NewSubscriberManager(transport)
	defer manager.Shutdown(ctx)
	require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "10.0.0.1:8000", endpoint, "", snapshotEndpoint, "kv@", true))
	h := &replayHarness{ctx: ctx, pub: pub, topic: []byte("kv@wire:8000@model")}
	var seq uint64
	send := func() {
		seq++
		server.set(int64(seq), buildEventBatchPayload(t)) //nolint:gosec // test sequence
		h.send(t, seq, buildEventBatchPayload(t))
	}
	require.Eventually(t, func() bool { send(); return server.requests.Load() == 1 }, 5*time.Second, 20*time.Millisecond)
	// Attachment and the snapshot load each reset the consumer.
	require.Eventually(t, func() bool { return consumer.resets.Load() == 2 }, 5*time.Second, 20*time.Millisecond)

	requested := seq
	manager.RequestResync("10.0.0.1:8000")
	require.Eventually(t, func() bool { send(); return server.requests.Load() == 2 }, 5*time.Second, 20*time.Millisecond)
	// The resync resets only the consumers that requested it, once, and
	// marks the snapshot's batches for them.
	require.Eventually(t, func() bool { return consumer.resyncBatches.Load() == 1 }, 5*time.Second, 20*time.Millisecond)
	require.Equal(t, int32(1), consumer.resyncResets.Load())
	require.Equal(t, int32(2), consumer.resets.Load())
	through := consumer.resyncThrough.Load()
	// The cut can equal requested: the frame at requested may still be queued
	// when the resync runs on it, before the next send moves the server on.
	require.True(t, through >= requested && through <= seq,
		"the rebuild covers the snapshot cut, got %d, requested %d, seq %d", through, requested, seq)

	// A request inside the cooldown waits.
	manager.RequestResync("10.0.0.1:8000")
	for range 5 {
		send()
		time.Sleep(20 * time.Millisecond)
	}
	require.Equal(t, int32(2), server.requests.Load())
	require.Equal(t, int32(1), consumer.resyncResets.Load())
}

func TestFailedResyncKeepsEndpointState(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	consumer := &countingConsumer{}
	transport, err := kvevents.NewConsumerPool(kvevents.DefaultConfig(), engineadapter.NewVLLMAdapter(), consumer)
	require.NoError(t, err)
	transport.Start(ctx)
	defer transport.Shutdown(ctx)
	pub := zmq4.NewPub(ctx)
	endpoint := listenEphemeral(t, pub)
	defer pub.Close()
	buffer, replayEndpoint := startReplayBuffer(t, ctx)
	buffer.set(replayMessage{seq: 1, payload: buildEventBatchPayload(t)})
	manager := kvevents.NewSubscriberManager(transport)
	defer manager.Shutdown(ctx)
	require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "10.0.0.1:8000", endpoint, replayEndpoint, "", "kv@", true))
	h := &replayHarness{ctx: ctx, pub: pub, topic: []byte("kv@wire:8000@model")}
	require.Eventually(t, func() bool { return consumer.resets.Load() == 1 }, 5*time.Second, 20*time.Millisecond)
	seq := uint64(1)
	require.Eventually(t, func() bool {
		seq++
		buffer.set(replayMessage{seq: 1, payload: buildEventBatchPayload(t)}, replayMessage{seq: seq, payload: buildEventBatchPayload(t)})
		h.send(t, seq, buildEventBatchPayload(t))
		return buffer.requests.Load() >= 1
	}, 5*time.Second, 20*time.Millisecond)

	buffer.fail.Store(true)
	before := consumer.resets.Load()
	manager.RequestResync("10.0.0.1:8000")
	for range 5 {
		seq++
		h.send(t, seq, buildEventBatchPayload(t))
		time.Sleep(20 * time.Millisecond)
	}
	require.GreaterOrEqual(t, buffer.requests.Load(), int32(2), "the resync requested a replay")
	require.Equal(t, before, consumer.resets.Load(), "a failed resync resets no consumer")
	require.Zero(t, consumer.resyncResets.Load(), "no rebuild data arrived, so nothing is reset")
}

func TestPartialResyncReplayIsAborted(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	consumer := &countingConsumer{}
	transport, err := kvevents.NewConsumerPool(kvevents.DefaultConfig(), engineadapter.NewVLLMAdapter(), consumer)
	require.NoError(t, err)
	transport.Start(ctx)
	defer transport.Shutdown(ctx)
	pub := zmq4.NewPub(ctx)
	endpoint := listenEphemeral(t, pub)
	defer pub.Close()
	buffer, replayEndpoint := startReplayBuffer(t, ctx)
	var history []replayMessage
	publish := func(seq uint64) {
		history = append(history, replayMessage{seq: seq, payload: buildEventBatchPayload(t)})
		buffer.set(history...)
	}
	publish(1)
	manager := kvevents.NewSubscriberManager(transport)
	defer manager.Shutdown(ctx)
	require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "10.0.0.1:8000", endpoint, replayEndpoint, "", "kv@", true))
	h := &replayHarness{ctx: ctx, pub: pub, topic: []byte("kv@wire:8000@model")}
	require.Eventually(t, func() bool { return buffer.requests.Load() == 1 }, 5*time.Second, 20*time.Millisecond)
	seq := uint64(1)
	for range 3 {
		seq++
		publish(seq)
		h.send(t, seq, buildEventBatchPayload(t))
		time.Sleep(20 * time.Millisecond)
	}
	resets := consumer.resets.Load()

	// The buffer serves sequence 1 and then skips one, so the replay fails
	// after its first batch.
	buffer.set(history[0], history[2])
	manager.RequestResync("10.0.0.1:8000")
	require.Eventually(t, func() bool {
		seq++
		h.send(t, seq, buildEventBatchPayload(t))
		return consumer.resyncAborts.Load() == 1
	}, 5*time.Second, 20*time.Millisecond, "the consumers holding a partial rebuild are told the resync was aborted")
	require.Equal(t, int32(1), consumer.resyncResets.Load())
	require.Equal(t, int32(1), consumer.resyncBatches.Load())
	require.Equal(t, resets, consumer.resets.Load(), "the abort leaves the other consumers' state alone")
}

func TestResyncReplayReachesOnlyResyncedConsumers(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	consumer := &countingConsumer{}
	transport, err := kvevents.NewConsumerPool(kvevents.DefaultConfig(), engineadapter.NewVLLMAdapter(), consumer)
	require.NoError(t, err)
	transport.Start(ctx)
	defer transport.Shutdown(ctx)
	pub := zmq4.NewPub(ctx)
	endpoint := listenEphemeral(t, pub)
	defer pub.Close()
	buffer, replayEndpoint := startReplayBuffer(t, ctx)
	var history []replayMessage
	publish := func(seq uint64) {
		history = append(history, replayMessage{seq: seq, payload: buildEventBatchPayload(t)})
		buffer.set(history...)
	}
	publish(1)
	manager := kvevents.NewSubscriberManager(transport)
	defer manager.Shutdown(ctx)
	require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "10.0.0.1:8000", endpoint, replayEndpoint, "", "kv@", true))
	h := &replayHarness{ctx: ctx, pub: pub, topic: []byte("kv@wire:8000@model")}
	require.Eventually(t, func() bool { return buffer.requests.Load() == 1 }, 5*time.Second, 20*time.Millisecond)
	seq := uint64(1)
	send := func() {
		seq++
		publish(seq)
		h.send(t, seq, buildEventBatchPayload(t))
	}
	for range 3 {
		send()
		time.Sleep(20 * time.Millisecond)
	}
	resets := consumer.resets.Load()

	manager.RequestResync("10.0.0.1:8000")
	require.Eventually(t, func() bool { send(); return buffer.requests.Load() == 2 }, 5*time.Second, 20*time.Millisecond)
	require.Eventually(t, func() bool { return consumer.resyncResets.Load() == 1 }, 5*time.Second, 20*time.Millisecond)
	through := consumer.resyncThrough.Load()
	require.Positive(t, through)
	require.Eventually(t, func() bool { return consumer.resyncBatches.Load() == int32(through) }, //nolint:gosec // test sequence
		5*time.Second, 20*time.Millisecond, "replayed batches up to the stream position reach only the resynced consumers")
	require.Equal(t, resets, consumer.resets.Load(), "the resync leaves the other consumers' state alone")
}

func TestConsumerReceivesBatchZeroAfterRestartWithoutReplay(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	consumer := &countingConsumer{}
	transport, err := kvevents.NewConsumerPool(kvevents.DefaultConfig(), engineadapter.NewVLLMAdapter(), consumer)
	require.NoError(t, err)
	transport.Start(ctx)
	defer transport.Shutdown(ctx)
	pub := zmq4.NewPub(ctx)
	endpoint := listenEphemeral(t, pub)
	defer pub.Close()
	manager := kvevents.NewSubscriberManager(transport)
	defer manager.Shutdown(ctx)
	require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "10.0.0.1:8000", endpoint, "", "", "kv@", true))
	h := &replayHarness{ctx: ctx, pub: pub, topic: []byte("kv@wire:8000@model")}
	require.Eventually(t, func() bool {
		h.send(t, 0, buildEventBatchPayload(t))
		return consumer.batches.Load() > 0
	}, 5*time.Second, 20*time.Millisecond)
	// Let batches sent while the subscriber connected drain.
	time.Sleep(200 * time.Millisecond)
	batches := consumer.batches.Load()

	// The engine restarted after publishing one batch, so its next batch is 0 again.
	h.send(t, 0, buildEventBatchPayload(t))
	require.Eventually(t, func() bool { return consumer.batches.Load() == batches+1 },
		time.Second, time.Millisecond, "the consumer receives the restarted engine's batch 0")
}
