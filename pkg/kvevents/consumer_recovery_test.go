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

type countingConsumer struct{ resets atomic.Int32 }

func (c *countingConsumer) ProcessEvents(context.Context, kvevents.EventSource, kvevents.EventBatch) error {
	return nil
}

func (c *countingConsumer) Reset(context.Context, kvevents.EventSource) error {
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
		server.set(int64(seq)) //nolint:gosec // test sequence
		h.send(t, seq, buildEventBatchPayload(t))
	}
	require.Eventually(t, func() bool { send(); return server.requests.Load() == 1 }, 5*time.Second, 20*time.Millisecond)
	// Attachment and the snapshot load each reset the consumer.
	require.Eventually(t, func() bool { return consumer.resets.Load() == 2 }, 5*time.Second, 20*time.Millisecond)

	manager.RequestResync("10.0.0.1:8000")
	require.Eventually(t, func() bool { send(); return server.requests.Load() == 2 }, 5*time.Second, 20*time.Millisecond)
	// The resync resets the consumer before the rebuild and again when the snapshot loads.
	require.Eventually(t, func() bool { return consumer.resets.Load() == 4 }, 5*time.Second, 20*time.Millisecond)

	// A request inside the cooldown waits.
	manager.RequestResync("10.0.0.1:8000")
	for range 5 {
		send()
		time.Sleep(20 * time.Millisecond)
	}
	require.Equal(t, int32(2), server.requests.Load())
	require.Equal(t, int32(4), consumer.resets.Load())
}
