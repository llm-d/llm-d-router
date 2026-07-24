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
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"net"
	"sync/atomic"
	"testing"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

// buildEventBatchPayload constructs a minimal valid msgpack EventBatch payload.
func buildEventBatchPayload(t *testing.T) []byte {
	t.Helper()

	// AllBlocksCleared event — simplest valid event, processed without side-effects.
	allCleared := []any{string(kvevents.EventTypeAllBlocksCleared)}
	rawEvent, err := msgpack.Marshal(allCleared)
	require.NoError(t, err)

	// EventBatch is array-encoded: [TS, Events, DataParallelRank]
	batch := []any{
		1234567890.0,                   // TS
		[]msgpack.RawMessage{rawEvent}, // Events
		nil,                            // DataParallelRank
	}

	var buf bytes.Buffer
	enc := msgpack.NewEncoder(&buf)
	enc.UseArrayEncodedStructs(true)
	require.NoError(t, enc.Encode(batch))
	return buf.Bytes()
}

// TestZMQPubSub verifies that the pure-Go ZMQ library correctly implements
// the PUB/SUB pattern used by the zmqSubscriber.
func TestZMQPubSub(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	endpoint := "tcp://127.0.0.1:15558"
	filter := "kv@"

	// Subscriber binds (Listen), publisher connects (Dial).
	sub := zmq4.NewSub(ctx)
	defer sub.Close()
	require.NoError(t, sub.Listen(endpoint))
	require.NoError(t, sub.SetOption(zmq4.OptionSubscribe, filter))

	// Give subscriber time to bind.
	time.Sleep(50 * time.Millisecond)

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(endpoint))

	// Give the connection time to establish.
	time.Sleep(50 * time.Millisecond)

	// Build a 3-frame message: [topic, seqBytes, payload]
	topic := "kv@10.0.0.1@TestModel"
	seqBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seqBytes, 42)
	payload := []byte("hello")

	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqBytes, payload)))

	// Receive with timeout.
	recvDone := make(chan zmq4.Msg, 1)
	go func() {
		msg, err := sub.Recv()
		if err == nil {
			recvDone <- msg
		}
	}()

	select {
	case msg := <-recvDone:
		require.Len(t, msg.Frames, 3)
		assert.Equal(t, topic, string(msg.Frames[0]))
		assert.Equal(t, seqBytes, msg.Frames[1])
		assert.Equal(t, payload, msg.Frames[2])
	case <-ctx.Done():
		t.Fatal("timeout waiting for ZMQ message")
	}
}

// TestZMQSubscriber_ReceivesMessages verifies the full message path:
// publisher → zmqSubscriber → pool (end-to-end without mocks).
func TestZMQSubscriber_ReceivesMessages(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Setup pool.
	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	// Start subscriber — remote=false means it binds (Listen).
	endpoint := "tcp://127.0.0.1:15559"
	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", endpoint, "", "kv@", false)
	require.NoError(t, err)

	// Give subscriber time to bind.
	time.Sleep(100 * time.Millisecond)

	// Publisher dials into the subscriber's bound address.
	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(endpoint))

	// Give the connection time to establish and subscription filter to propagate.
	time.Sleep(100 * time.Millisecond)

	// Send a valid 3-frame ZMQ message.
	topic := "kv@10.0.0.1@TestModel"
	seqBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seqBytes, 1)
	payload := buildEventBatchPayload(t)

	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqBytes, payload)))

	// Allow time for the message to be received and processed.
	time.Sleep(200 * time.Millisecond)

	subManager.Shutdown(ctx)
}

// TestZMQSubscriber_ShortSequenceFrameSkipped verifies that a message with a
// truncated sequence frame (< 8 bytes) is skipped instead of panicking.
func TestZMQSubscriber_ShortSequenceFrameSkipped(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Setup pool.
	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	// Pick an available ephemeral port to avoid conflicts with parallel tests or CI.
	ln, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "127.0.0.1:0")
	require.NoError(t, err)
	endpoint := fmt.Sprintf("tcp://%s", ln.Addr().String())
	ln.Close()
	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", endpoint, "", "kv@", false)
	require.NoError(t, err)
	time.Sleep(100 * time.Millisecond)

	// Publisher dials into the subscriber's bound address.
	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(endpoint))
	time.Sleep(100 * time.Millisecond)

	// Send malformed messages with a truncated sequence frame (3 bytes instead of 8).
	// Before the fix this would panic with index-out-of-range in binary.BigEndian.Uint64.
	// Retry sending for a short window to mitigate ZMQ "slow joiner" behavior where
	// early sends can be dropped before the subscription is fully established.
	shortSeq := []byte{0x01, 0x02, 0x03}
	sendDeadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(sendDeadline) {
		require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte("kv@10.0.0.1@TestModel"), shortSeq, []byte("bad"))))
		time.Sleep(10 * time.Millisecond)
	}

	// Allow a brief moment for any in-flight message to be processed before shutdown.
	time.Sleep(100 * time.Millisecond)

	// If we reach here without a panic, the short frame was correctly skipped.
	subManager.Shutdown(ctx)
}

// ephemeralPort returns an available TCP port on localhost.
func ephemeralPort(t *testing.T) string {
	t.Helper()
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := ln.Addr().String()
	ln.Close()
	return fmt.Sprintf("tcp://%s", addr)
}

// seqFrame encodes a uint64 as an 8-byte big-endian frame.
func seqFrame(seq uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, seq)
	return b
}

// endSeqFrame returns the end-of-replay marker sequence.
func endSeqFrame() []byte {
	return seqFrame(math.MaxUint64)
}

// startMockRouter runs a ROUTER socket that serves replay requests.
// It responds with events for sequences in [startSeq, startSeq+count),
// then sends the end-of-replay marker.
func startMockRouter(t *testing.T, ctx context.Context, endpoint string,
	payloadFn func() []byte, replayCount int,
) *atomic.Int32 {
	t.Helper()
	received := &atomic.Int32{}
	topic := "kv@10.0.0.1@TestModel"

	router := zmq4.NewRouter(ctx)
	require.NoError(t, router.Listen(endpoint))
	t.Cleanup(func() { router.Close() })

	go func() {
		for {
			msg, err := router.Recv()
			if err != nil {
				return
			}
			received.Add(1)

			// ROUTER receives [clientID, empty, startSeqBytes]
			frames := msg.Frames
			if len(frames) != 3 {
				continue
			}
			clientID := frames[0]
			startSeqBytes := frames[2]
			startSeq := binary.BigEndian.Uint64(startSeqBytes)

			for i := range replayCount {
				seq := startSeq + uint64(i) // #nosec G115 -- test data, i is small
				err := router.Send(zmq4.NewMsgFrom(
					clientID, []byte{}, []byte(topic), seqFrame(seq), payloadFn(),
				))
				if err != nil {
					return
				}
			}
			if err := router.Send(zmq4.NewMsgFrom(
				clientID, []byte{}, []byte{}, endSeqFrame(), []byte{},
			)); err != nil {
				return
			}
		}
	}()

	return received
}

// TestZMQSubscriber_ReplayOnGap verifies that when a sequence gap is detected,
// the subscriber requests replay and processes the missing events.
func TestZMQSubscriber_ReplayOnGap(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	pubEndpoint := ephemeralPort(t)
	replayEndpoint := ephemeralPort(t)

	topic := "kv@10.0.0.1@TestModel"
	payload := buildEventBatchPayload(t)

	// Start mock ROUTER that serves 4 replay events (seq 1-4).
	replayRequests := startMockRouter(t, ctx, replayEndpoint,
		func() []byte { return payload }, 4)

	time.Sleep(50 * time.Millisecond)

	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", pubEndpoint, replayEndpoint, "kv@", false)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		return replayRequests.Load() >= 1
	}, 5*time.Second, 50*time.Millisecond, "proactive replay on connect expected")

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(pubEndpoint))
	time.Sleep(100 * time.Millisecond)

	// Send seq 0 — already covered by proactive replay, no gap.
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(0), payload)))
	time.Sleep(200 * time.Millisecond)

	// Send seq 5 — gap from lastSeq (set by proactive replay). Triggers another replay.
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(5), payload)))

	require.Eventually(t, func() bool {
		return replayRequests.Load() >= 2
	}, 5*time.Second, 50*time.Millisecond, "proactive + gap replay expected")

	subManager.Shutdown(ctx)
}

// TestZMQSubscriber_ProactiveReplayOnConnect verifies that replay fires
// immediately on subscriber connect, without waiting for a live event.
func TestZMQSubscriber_ProactiveReplayOnConnect(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	pubEndpoint := ephemeralPort(t)
	replayEndpoint := ephemeralPort(t)

	payload := buildEventBatchPayload(t)

	replayRequests := startMockRouter(t, ctx, replayEndpoint,
		func() []byte { return payload }, 5)

	time.Sleep(50 * time.Millisecond)

	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", pubEndpoint, replayEndpoint, "kv@", false)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		return replayRequests.Load() >= 1
	}, 5*time.Second, 50*time.Millisecond, "proactive replay should fire on connect without live events")

	subManager.Shutdown(ctx)
}

// TestZMQSubscriber_SuccessfulReplayDoesNotBlockNextGap verifies that a
// successful replay does not trigger a cooldown, so a subsequent gap is
// replayed immediately.
func TestZMQSubscriber_SuccessfulReplayDoesNotBlockNextGap(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	pubEndpoint := ephemeralPort(t)
	replayEndpoint := ephemeralPort(t)

	topic := "kv@10.0.0.1@TestModel"
	payload := buildEventBatchPayload(t)

	replayRequests := startMockRouter(t, ctx, replayEndpoint,
		func() []byte { return payload }, 3)

	time.Sleep(50 * time.Millisecond)

	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", pubEndpoint, replayEndpoint, "kv@", false)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		return replayRequests.Load() >= 1
	}, 5*time.Second, 50*time.Millisecond, "proactive replay expected")

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(pubEndpoint))
	time.Sleep(100 * time.Millisecond)

	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(10), payload)))

	require.Eventually(t, func() bool {
		return replayRequests.Load() >= 2
	}, 5*time.Second, 50*time.Millisecond, "successful replay should not block subsequent gap replay")

	subManager.Shutdown(ctx)
}

// TestZMQSubscriber_ReplayDisabled verifies that with an empty replay endpoint,
// gaps are not replayed and no panic occurs.
func TestZMQSubscriber_ReplayDisabled(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	pubEndpoint := ephemeralPort(t)

	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", pubEndpoint, "", "kv@", false)
	require.NoError(t, err)
	time.Sleep(100 * time.Millisecond)

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(pubEndpoint))
	time.Sleep(100 * time.Millisecond)

	topic := "kv@10.0.0.1@TestModel"
	payload := buildEventBatchPayload(t)

	// Send seq 0, then seq 5 — gap exists but replay is disabled.
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(0), payload)))
	time.Sleep(100 * time.Millisecond)
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(5), payload)))
	time.Sleep(200 * time.Millisecond)

	// No panic, no hang — test passes if we reach here.
	subManager.Shutdown(ctx)
}

// TestZMQSubscriber_ReplayDoesNotCauseSpuriousReReplay verifies that when
// replay returns events beyond the gap-triggering sequence, subsequent live
// events do not trigger another replay.
func TestZMQSubscriber_ReplayDoesNotCauseSpuriousReReplay(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	pubEndpoint := ephemeralPort(t)
	replayEndpoint := ephemeralPort(t)

	topic := "kv@10.0.0.1@TestModel"
	payload := buildEventBatchPayload(t)

	// Mock ROUTER returns 100 events (seq 1-100) on replay request.
	replayRequests := startMockRouter(t, ctx, replayEndpoint,
		func() []byte { return payload }, 100)

	time.Sleep(50 * time.Millisecond)

	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", pubEndpoint, replayEndpoint, "kv@", false)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		return replayRequests.Load() >= 1
	}, 5*time.Second, 50*time.Millisecond, "proactive replay expected")

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(pubEndpoint))
	time.Sleep(100 * time.Millisecond)

	// seq 0 and seq 50 are both below lastSeq (set to 99 by proactive replay
	// of 100 events). No gap, no additional replay.
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(0), payload)))
	time.Sleep(200 * time.Millisecond)
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(50), payload)))
	time.Sleep(200 * time.Millisecond)

	assert.Equal(t, int32(1), replayRequests.Load(), "no additional replay for seqs covered by proactive replay")

	subManager.Shutdown(ctx)
}

// TestZMQSubscriber_FailedReplayDegradeGracefully verifies that a bad
// replay endpoint does not crash or hang the subscriber — cooldown
// prevents repeated failed attempts.
func TestZMQSubscriber_FailedReplayDegradeGracefully(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	pubEndpoint := ephemeralPort(t)
	// Use a port with nothing listening — replay Dial will fail.
	badReplayEndpoint := ephemeralPort(t)

	topic := "kv@10.0.0.1@TestModel"
	payload := buildEventBatchPayload(t)

	subManager := kvevents.NewSubscriberManager(pool)
	err = subManager.EnsureSubscriber(ctx, "test-pod", pubEndpoint, badReplayEndpoint, "kv@", false)
	require.NoError(t, err)

	// Wait for proactive replay to fail (bad endpoint).
	time.Sleep(500 * time.Millisecond)

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(pubEndpoint))
	time.Sleep(100 * time.Millisecond)

	// seq 0 — establishes lastSeq. Cooldown blocks replay, event processed.
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(0), payload)))
	time.Sleep(200 * time.Millisecond)

	// seq 5 — gap detected but cooldown still active from proactive failure.
	// Event processed normally since canAttemptReplay() is false.
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqFrame(5), payload)))
	time.Sleep(200 * time.Millisecond)

	// No panic, no hang — cooldown prevented replay attempt on bad endpoint.
	subManager.Shutdown(ctx)
}
