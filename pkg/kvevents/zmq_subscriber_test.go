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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

// buildEventBatchPayload constructs a minimal valid msgpack EventBatch payload.
func buildEventBatchPayload(t *testing.T) []byte {
	t.Helper()

	return buildEventPayload(t, []any{string(kvevents.EventTypeAllBlocksCleared)})
}

func buildEventPayload(t *testing.T, events ...[]any) []byte {
	t.Helper()

	rawEvents := make([]msgpack.RawMessage, 0, len(events))
	for _, event := range events {
		rawEvent, err := msgpack.Marshal(event)
		require.NoError(t, err)
		rawEvents = append(rawEvents, rawEvent)
	}

	// EventBatch is array-encoded: [TS, Events, DataParallelRank]
	batch := []any{
		1234567890.0, // TS
		rawEvents,    // Events
		nil,          // DataParallelRank
	}

	var buf bytes.Buffer
	enc := msgpack.NewEncoder(&buf)
	enc.UseArrayEncodedStructs(true)
	require.NoError(t, enc.Encode(batch))
	return buf.Bytes()
}

func buildDistinctBlockStoredPayload(t *testing.T, blockHash uint64) []byte {
	t.Helper()

	tokens := make([]uint32, 64)
	for i := range tokens {
		tokens[i] = uint32(blockHash) + uint32(i) + 1 //#nosec G115 -- test data is small
	}
	return buildEventPayload(t, []any{
		string(kvevents.EventTypeBlockStored),
		[]uint64{blockHash},
		uint64(0),
		tokens,
		64,
	})
}

func buildBlockRemovedPayload(t *testing.T, blockHash uint64) []byte {
	t.Helper()
	return buildEventPayload(t, []any{
		string(kvevents.EventTypeBlockRemoved),
		[]uint64{blockHash},
	})
}

func buildBlockStoredEventBatchPayload(t *testing.T, blockHashBase uint64, dataParallelRank int) []byte {
	t.Helper()

	tokens := make([]uint32, 64)
	blockHashes := make([]any, 4)
	for i := range tokens {
		tokens[i] = uint32(i + 1) //#nosec G115 -- test data
	}
	for i := range blockHashes {
		blockHashes[i] = blockHashBase + uint64(i) //#nosec G115 -- test data
	}

	blockStored := []any{
		string(kvevents.EventTypeBlockStored),
		blockHashes,
		uint64(0),
		tokens,
		16,
		nil,
		"gpu",
		nil,
		nil,
	}
	payload, err := msgpack.Marshal([]any{
		1234567890.0,
		[]any{blockStored},
		dataParallelRank,
	})
	require.NoError(t, err)
	return payload
}

func availableEndpoint(t *testing.T, ctx context.Context) string {
	t.Helper()

	ln, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "127.0.0.1:0")
	require.NoError(t, err)
	endpoint := fmt.Sprintf("tcp://%s", ln.Addr().String())
	require.NoError(t, ln.Close())
	return endpoint
}

// listenEphemeral binds sock to a kernel-assigned loopback port and returns its
// endpoint. Binding port 0 directly leaves no window for another process to
// take the port, unlike reserving one with availableEndpoint and binding later.
func listenEphemeral(t *testing.T, sock zmq4.Socket) string {
	t.Helper()
	require.NoError(t, sock.Listen("tcp://127.0.0.1:0"))
	return "tcp://" + sock.Addr().String()
}

func seqFrame(seq uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, seq)
	return b
}

type replayMessage struct {
	seq     uint64
	payload []byte
}

func newTestPool(t *testing.T, ctx context.Context) (kvblock.Index, kvblock.TokenProcessor, *kvevents.Pool) {
	t.Helper()

	index, err := kvblock.NewIndex(ctx, kvblock.DefaultIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool, err := kvevents.NewPool(kvevents.DefaultConfig(), index, tokenProcessor, engineadapter.NewVLLMAdapter())
	require.NoError(t, err)
	pool.Start(ctx)
	t.Cleanup(func() {
		shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		pool.Shutdown(shutdownCtx)
	})
	return index, tokenProcessor, pool
}

func shutdownSubscriberManager(ctx context.Context, sm *kvevents.SubscriberManager) {
	shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	sm.Shutdown(shutdownCtx)
}

const (
	testReplayAttemptIdleTimeout = 150 * time.Millisecond
	testMaxNoProgressAttempts    = 3
)

type replayBuffer struct {
	mu           sync.RWMutex
	messages     []replayMessage
	fail         atomic.Bool
	partialOnce  atomic.Bool
	partialAfter int
	messageDelay time.Duration
	silent       bool
	requests     atomic.Int32
	lastStartSeq atomic.Uint64
	requestTimes []time.Time
}

func (b *replayBuffer) requestTime(idx int) time.Time {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if idx < 0 || idx >= len(b.requestTimes) {
		return time.Time{}
	}
	return b.requestTimes[idx]
}

func (b *replayBuffer) set(messages ...replayMessage) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.messages = append([]replayMessage(nil), messages...)
}

func startReplayBuffer(t *testing.T, ctx context.Context) (*replayBuffer, string) {
	t.Helper()
	buffer := &replayBuffer{}
	topic := []byte("kv@10.0.0.1:8000@TestModel")

	router := zmq4.NewRouter(ctx)
	endpoint := listenEphemeral(t, router)
	t.Cleanup(func() { router.Close() })

	// A send failure means only that this client hung up, so it abandons the
	// response rather than the loop: subscribers close a replay socket as soon
	// as their attempt idle timer fires, and later requests must still be served.
	serve := func(msg zmq4.Msg) {
		buffer.mu.Lock()
		buffer.requestTimes = append(buffer.requestTimes, time.Now())
		buffer.mu.Unlock()
		buffer.requests.Add(1)
		if len(msg.Frames) != 3 {
			return
		}
		clientID := msg.Frames[0]
		if buffer.fail.Load() {
			_ = router.Send(zmq4.NewMsgFrom(clientID, []byte{}, []byte("malformed")))
			return
		}

		startSeq := binary.BigEndian.Uint64(msg.Frames[2])
		buffer.lastStartSeq.Store(startSeq)
		buffer.mu.RLock()
		messages := append([]replayMessage(nil), buffer.messages...)
		messageDelay := buffer.messageDelay
		silent := buffer.silent
		partialAfter := buffer.partialAfter
		buffer.mu.RUnlock()
		if silent {
			return
		}
		partial := buffer.partialOnce.CompareAndSwap(true, false)
		sent := 0
		for _, replay := range messages {
			if replay.seq < startSeq {
				continue
			}
			if messageDelay > 0 {
				time.Sleep(messageDelay)
			}
			if err := router.Send(zmq4.NewMsgFrom(
				clientID, []byte{}, topic, seqFrame(replay.seq), replay.payload,
			)); err != nil {
				return
			}
			sent++
			if partial && sent == partialAfter {
				break
			}
		}
		if partial && sent == partialAfter {
			return
		}
		_ = router.Send(zmq4.NewMsgFrom(
			clientID, []byte{}, []byte{}, seqFrame(math.MaxUint64), []byte{},
		))
	}

	go func() {
		for {
			msg, err := router.Recv()
			if err != nil {
				return
			}
			serve(msg)
		}
	}()

	return buffer, endpoint
}

// TestZMQPubSub verifies that the pure-Go ZMQ library correctly implements
// the PUB/SUB pattern used by the zmqSubscriber.
func TestZMQPubSub(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	filter := "kv@"

	// Subscriber binds (Listen), publisher connects (Dial).
	sub := zmq4.NewSub(ctx)
	defer sub.Close()
	endpoint := listenEphemeral(t, sub)
	require.NoError(t, sub.SetOption(zmq4.OptionSubscribe, filter))

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(endpoint))

	// Build a 3-frame message: [topic, seqBytes, payload]
	topic := "kv@10.0.0.1@TestModel"
	seqBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seqBytes, 42)
	payload := []byte("hello")

	// Receive with timeout, retrying send to handle ZMQ slow-joiner propagation.
	recvDone := make(chan zmq4.Msg, 1)
	go func() {
		msg, err := sub.Recv()
		if err == nil {
			recvDone <- msg
		}
	}()

	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqBytes, payload)))
	for {
		select {
		case msg := <-recvDone:
			require.Len(t, msg.Frames, 3)
			assert.Equal(t, topic, string(msg.Frames[0]))
			assert.Equal(t, seqBytes, msg.Frames[1])
			assert.Equal(t, payload, msg.Frames[2])
			return
		case <-ticker.C:
			require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte(topic), seqBytes, payload)))
		case <-ctx.Done():
			t.Fatal("timeout waiting for ZMQ message")
		}
	}
}

// TestZMQSubscriber_ReceivesMessages verifies the full message path:
// publisher -> zmqSubscriber -> pool (end-to-end without mocks).
func TestZMQSubscriber_ReceivesMessages(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)

	index, _, pool := newTestPool(t, ctx)

	// Start subscriber - remote=false means it binds (Listen).
	endpoint := availableEndpoint(t, ctx)
	subManager := kvevents.NewSubscriberManager(pool)
	t.Cleanup(func() { shutdownSubscriberManager(ctx, subManager) })
	require.NoError(t, subManager.EnsureSubscriber(ctx, "test-pod", "", endpoint, "", "", "kv@", false))

	// Publisher dials into the subscriber's bound address.
	pub := zmq4.NewPub(ctx)
	t.Cleanup(func() { _ = pub.Close() })
	require.NoError(t, pub.Dial(endpoint))

	// Send a valid 3-frame ZMQ message, retrying until the subscription propagates.
	topic := []byte("kv@10.0.0.1@TestModel")
	payload := buildDistinctBlockStoredPayload(t, 100)
	var seq uint64
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		seq++
		require.NoError(c, pub.Send(zmq4.NewMsgFrom(topic, seqFrame(seq), payload)))
		key, err := index.GetRequestKey(ctx, kvevents.ModelEngineKey("TestModel", 100))
		require.NoError(c, err)
		hits, err := index.Lookup(ctx, []kvblock.BlockHash{key}, nil)
		require.NoError(c, err)
		require.Len(c, hits[key], 1)
		assert.Equal(c, "10.0.0.1", hits[key][0].PodIdentifier)
	}, 5*time.Second, 20*time.Millisecond)
}

func TestZMQSubscribers_SameTopicUsesServingEndpointIdentity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)

	index, tokenProcessor, pool := newTestPool(t, ctx)

	subManager := kvevents.NewSubscriberManager(pool)
	t.Cleanup(func() { shutdownSubscriberManager(ctx, subManager) })

	sourceEndpoints := []string{"10.0.0.1:8000", "10.0.0.1:8003"}
	zmqEndpoints := []string{availableEndpoint(t, ctx), availableEndpoint(t, ctx)}
	for i := range sourceEndpoints {
		require.NoError(t, subManager.EnsureSubscriber(
			ctx,
			fmt.Sprintf("test-rank-%d", i),
			sourceEndpoints[i],
			zmqEndpoints[i],
			"",
			"",
			"kv@",
			false,
		))
	}
	time.Sleep(100 * time.Millisecond)

	publishers := make([]zmq4.Socket, len(zmqEndpoints))
	for i, endpoint := range zmqEndpoints {
		publishers[i] = zmq4.NewPub(ctx)
		defer publishers[i].Close()
		require.NoError(t, publishers[i].Dial(endpoint))
	}
	time.Sleep(100 * time.Millisecond)

	topic := []byte("kv@10.0.0.1:8000@TestModel")
	payloads := [][]byte{
		buildBlockStoredEventBatchPayload(t, 100, 8),
		buildBlockStoredEventBatchPayload(t, 200, 11),
	}
	tokens := make([]uint32, 64)
	for i := range tokens {
		tokens[i] = uint32(i + 1) //#nosec G115 -- test data
	}
	keys, err := tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "TestModel", nil)
	require.NoError(t, err)
	require.NotEmpty(t, keys)
	firstKey := keys[0]

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for i, publisher := range publishers {
			seq := make([]byte, 8)
			binary.BigEndian.PutUint64(seq, uint64(i+1))
			require.NoError(t, publisher.Send(zmq4.NewMsgFrom(topic, seq, payloads[i])))
		}

		time.Sleep(50 * time.Millisecond)
		result, lookupErr := index.Lookup(ctx, keys, nil)
		require.NoError(t, lookupErr)
		if len(result[firstKey]) != 2 {
			continue
		}

		got := []string{
			result[firstKey][0].PodIdentifier,
			result[firstKey][1].PodIdentifier,
		}
		assert.ElementsMatch(t, sourceEndpoints, got)
		return
	}

	t.Fatal("timed out waiting for blocks from both serving endpoints")
}

// TestZMQSubscriber_ShortSequenceFrameSkipped verifies that a message with a
// truncated sequence frame (< 8 bytes) is skipped and the subscriber keeps
// processing the messages that follow it.
func TestZMQSubscriber_ShortSequenceFrameSkipped(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)

	index, _, pool := newTestPool(t, ctx)

	endpoint := availableEndpoint(t, ctx)
	subManager := kvevents.NewSubscriberManager(pool)
	t.Cleanup(func() { shutdownSubscriberManager(ctx, subManager) })
	require.NoError(t, subManager.EnsureSubscriber(ctx, "test-pod", "", endpoint, "", "", "kv@", false))

	// Publisher dials into the subscriber's bound address.
	pub := zmq4.NewPub(ctx)
	t.Cleanup(func() { _ = pub.Close() })
	require.NoError(t, pub.Dial(endpoint))

	topic := []byte("kv@10.0.0.1@TestModel")
	shortSeq := []byte{0x01, 0x02, 0x03}
	var seq uint64
	// Each tick publishes a truncated-sequence message ahead of a valid one.
	sendUntilIndexed := func(hash uint64) {
		payload := buildDistinctBlockStoredPayload(t, hash)
		require.EventuallyWithT(t, func(c *assert.CollectT) {
			seq++
			require.NoError(c, pub.Send(zmq4.NewMsgFrom(topic, shortSeq, []byte("bad"))))
			require.NoError(c, pub.Send(zmq4.NewMsgFrom(topic, seqFrame(seq), payload)))
			_, err := index.GetRequestKey(ctx, kvevents.ModelEngineKey("TestModel", hash))
			require.NoError(c, err)
		}, 5*time.Second, 20*time.Millisecond)
	}

	// Indexing the first block shows the subscription is established, so the
	// truncated messages sent for the second block reach the subscriber.
	sendUntilIndexed(100)
	sendUntilIndexed(200)
}

type replayHarness struct {
	ctx        context.Context
	index      kvblock.Index
	buffer     *replayBuffer
	pub        zmq4.Socket
	subManager *kvevents.SubscriberManager
	topic      []byte
}

func newReplayHarness(t *testing.T, messages []replayMessage, fail bool) *replayHarness {
	return newReplayHarnessWithBehavior(t, messages, fail, 0, 0, false)
}

func newReplayHarnessWithPartial(
	t *testing.T, messages []replayMessage, fail bool, partialAfter int,
) *replayHarness {
	return newReplayHarnessWithBehavior(t, messages, fail, partialAfter, 0, false)
}

func newReplayHarnessWithBehavior(
	t *testing.T, messages []replayMessage, fail bool, partialAfter int,
	messageDelay time.Duration, silent bool, opts ...kvevents.SubscriberManagerOption,
) *replayHarness {
	t.Helper()
	smOpts := append([]kvevents.SubscriberManagerOption{
		kvevents.WithReplayTimers(kvevents.ReplayTimers{
			AttemptIdleTimeout: testReplayAttemptIdleTimeout,
			RetryBackoff:       20 * time.Millisecond,
		}),
	}, opts...)
	return newReplayHarnessWithOptions(t, messages, fail, partialAfter, messageDelay, silent, smOpts...)
}

func newReplayHarnessWithOptions(
	t *testing.T, messages []replayMessage, fail bool, partialAfter int,
	messageDelay time.Duration, silent bool, opts ...kvevents.SubscriberManagerOption,
) *replayHarness {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	t.Cleanup(cancel)

	index, _, pool := newTestPool(t, ctx)

	// The subscriber connects, as production subscribers with a replay endpoint
	// do, so every listener in the harness binds its own port.
	pub := zmq4.NewPub(ctx)
	pubEndpoint := listenEphemeral(t, pub)
	buffer, replayEndpoint := startReplayBuffer(t, ctx)
	buffer.set(messages...)
	buffer.fail.Store(fail)
	buffer.mu.Lock()
	buffer.partialAfter = partialAfter
	buffer.messageDelay = messageDelay
	buffer.silent = silent
	buffer.mu.Unlock()
	buffer.partialOnce.Store(partialAfter > 0)

	subManager := kvevents.NewSubscriberManager(pool, opts...)
	t.Cleanup(func() {
		_ = pub.Close()
		shutdownSubscriberManager(ctx, subManager)
	})
	require.NoError(t, subManager.EnsureSubscriber(
		ctx, "test-pod", "10.0.0.1:8000", pubEndpoint, replayEndpoint, "", "kv@", true))
	require.Eventually(t, func() bool { return buffer.requests.Load() >= 1 },
		5*time.Second, 50*time.Millisecond, "proactive replay expected")
	time.Sleep(100 * time.Millisecond)

	return &replayHarness{
		ctx:        ctx,
		index:      index,
		buffer:     buffer,
		pub:        pub,
		subManager: subManager,
		topic:      []byte("kv@10.0.0.1:8000@TestModel"),
	}
}

func TestZMQSubscriber_ProactiveReplayResumesAfterPartialResponse(t *testing.T) {
	h := newReplayHarnessWithPartial(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)},
		{seq: 1, payload: buildDistinctBlockStoredPayload(t, 200)},
		{seq: 2, payload: buildDistinctBlockStoredPayload(t, 300)},
	}, false, 2)

	require.Eventually(t, func() bool { return h.buffer.requests.Load() == 2 },
		5*time.Second, 50*time.Millisecond, "partial replay should resume on a new request")
	require.Equal(t, uint64(2), h.buffer.lastStartSeq.Load())
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)
}

func TestZMQSubscriber_ProactiveReplayAcceptsEndAfterProgress(t *testing.T) {
	h := newReplayHarnessWithPartial(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)},
		{seq: 1, payload: buildDistinctBlockStoredPayload(t, 200)},
		{seq: 2, payload: buildDistinctBlockStoredPayload(t, 300)},
	}, false, 3)

	require.Eventually(t, func() bool { return h.buffer.requests.Load() == 2 },
		5*time.Second, 50*time.Millisecond, "replay should request the terminal marker after interruption")
	require.Equal(t, uint64(3), h.buffer.lastStartSeq.Load())
	time.Sleep(300 * time.Millisecond)
	key, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
	require.NoError(t, err)
	hits, err := h.index.Lookup(h.ctx, []kvblock.BlockHash{key}, nil)
	require.NoError(t, err)
	require.Len(t, hits[key], 1, "terminal marker after replay progress must preserve the rebuilt index")
}

func TestZMQSubscriber_ColdReplayRetainedHistoryIndexesLiveEvents(t *testing.T) {
	h := newReplayHarness(t, []replayMessage{
		{seq: 20000, payload: buildDistinctBlockStoredPayload(t, 100)},
		{seq: 20001, payload: buildDistinctBlockStoredPayload(t, 200)},
	}, false)

	require.Eventually(t, func() bool {
		_, firstErr := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		_, secondErr := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 200))
		return firstErr == nil && secondErr == nil
	}, 5*time.Second, 50*time.Millisecond)
	h.send(t, 20002, buildDistinctBlockStoredPayload(t, 300))
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond, "live events must advance after a retained replay window")
	require.Equal(t, int32(1), h.buffer.requests.Load())
}

func TestZMQSubscriber_ColdReplayRetainedHistoryResumes(t *testing.T) {
	h := newReplayHarnessWithPartial(t, []replayMessage{
		{seq: 20000, payload: buildDistinctBlockStoredPayload(t, 100)},
		{seq: 20001, payload: buildDistinctBlockStoredPayload(t, 200)},
		{seq: 20002, payload: buildDistinctBlockStoredPayload(t, 300)},
	}, false, 2)
	require.Eventually(t, func() bool { return h.buffer.requests.Load() == 2 },
		5*time.Second, 50*time.Millisecond)
	require.Equal(t, uint64(20002), h.buffer.lastStartSeq.Load())
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)
}

func TestZMQSubscriber_GapFillRejectsMissingRequestedSequence(t *testing.T) {
	h := newReplayHarness(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)},
	}, false)
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)
	key, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
	require.NoError(t, err)
	h.buffer.set(replayMessage{seq: 3, payload: buildDistinctBlockStoredPayload(t, 300)})
	h.send(t, 3, buildDistinctBlockStoredPayload(t, 300))
	require.Eventually(t, func() bool { return h.buffer.requests.Load() == 2 },
		5*time.Second, 50*time.Millisecond)
	require.Equal(t, uint64(1), h.buffer.lastStartSeq.Load())
	require.Eventually(t, func() bool {
		hits, lookupErr := h.index.Lookup(h.ctx, []kvblock.BlockHash{key}, nil)
		return lookupErr == nil && len(hits[key]) == 0
	}, 5*time.Second, 50*time.Millisecond, "a missing gap sequence must invalidate stale cache state")
	_, err = h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
	require.Error(t, err)
}

func TestZMQSubscriber_ColdReplayRetainedHistoryRejectsInternalGap(t *testing.T) {
	h := newReplayHarness(t, []replayMessage{
		{seq: 20000, payload: buildDistinctBlockStoredPayload(t, 100)},
		{seq: 20002, payload: buildDistinctBlockStoredPayload(t, 300)},
	}, false)
	require.Eventually(t, func() bool {
		key, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		if err != nil {
			return false
		}
		hits, err := h.index.Lookup(h.ctx, []kvblock.BlockHash{key}, nil)
		return err == nil && len(hits[key]) == 0
	}, 5*time.Second, 50*time.Millisecond, "a gap inside the retained window must clear partial state")
	_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
	require.Error(t, err)
}

func TestZMQSubscriber_ProactiveReplayClearsPartialHistoryOnGap(t *testing.T) {
	h := newReplayHarness(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)},
		{seq: 2, payload: buildDistinctBlockStoredPayload(t, 300)},
	}, false)

	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)
	require.Eventually(t, func() bool {
		key, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		if err != nil {
			return false
		}
		hits, err := h.index.Lookup(h.ctx, []kvblock.BlockHash{key}, nil)
		return err == nil && len(hits[key]) == 0
	}, 5*time.Second, 50*time.Millisecond, "incomplete replay must clear partial index state")
}

func TestZMQSubscriber_ProactiveReplayUsesInactivityDeadline(t *testing.T) {
	const attemptIdleTimeout = 500 * time.Millisecond
	h := newReplayHarnessWithBehavior(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)},
		{seq: 1, payload: buildDistinctBlockStoredPayload(t, 200)},
		{seq: 2, payload: buildDistinctBlockStoredPayload(t, 300)},
	}, false, 0, 200*time.Millisecond, false,
		kvevents.WithReplayTimers(kvevents.ReplayTimers{AttemptIdleTimeout: attemptIdleTimeout}))

	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)
	firstReq := h.buffer.requestTime(0)
	require.False(t, firstReq.IsZero())
	require.Greater(t, time.Since(firstReq), attemptIdleTimeout)
	require.Equal(t, int32(1), h.buffer.requests.Load(), "progressing replay must not reconnect")
}

func TestZMQSubscriber_ProactiveReplayStopsAfterNoProgress(t *testing.T) {
	h := newReplayHarnessWithBehavior(t, nil, false, 0, 0, true)

	require.Eventually(t, func() bool {
		return h.buffer.requests.Load() == testMaxNoProgressAttempts
	}, 8*time.Second, 50*time.Millisecond)
	time.Sleep(testReplayAttemptIdleTimeout + 100*time.Millisecond)
	require.Equal(t, int32(testMaxNoProgressAttempts), h.buffer.requests.Load())
}

func TestZMQSubscriber_WithReplayTimers(t *testing.T) {
	t.Run("short_cooldown_allows_retry_after_expiry", func(t *testing.T) {
		h := newReplayHarnessWithBehavior(
			t, nil, true, 0, 0, false,
			kvevents.WithReplayTimers(kvevents.ReplayTimers{
				Cooldown: 150 * time.Millisecond,
			}),
		)

		h.buffer.set(replayMessage{seq: 0, payload: buildDistinctBlockStoredPayload(t, 200)})
		h.buffer.fail.Store(false)

		payload := buildEventBatchPayload(t)
		seq := uint64(5)
		require.EventuallyWithT(t, func(c *assert.CollectT) {
			require.NoError(c, h.pub.Send(zmq4.NewMsgFrom(h.topic, seqFrame(seq), payload)))
			seq++
			assert.GreaterOrEqual(c, h.buffer.requests.Load(), int32(2))
			assert.True(c, h.indexed(200))
		}, 2*time.Second, 30*time.Millisecond, "replay must retry after configured cooldown expires")
	})

	t.Run("short_timeout_aborts_stalled_replay", func(t *testing.T) {
		h := newReplayHarnessWithBehavior(
			t, nil, false, 0, 0, true,
			kvevents.WithReplayTimers(kvevents.ReplayTimers{
				Timeout:            150 * time.Millisecond,
				AttemptIdleTimeout: 2 * time.Second,
			}),
		)

		payload := buildDistinctBlockStoredPayload(t, 100)
		seq := uint64(5)
		require.EventuallyWithT(t, func(c *assert.CollectT) {
			require.NoError(c, h.pub.Send(zmq4.NewMsgFrom(h.topic, seqFrame(seq), payload)))
			seq++
			assert.True(c, h.indexed(100))
		}, 1500*time.Millisecond, 30*time.Millisecond, "replay must abort once Timeout expires")
		assert.Equal(t, int32(1), h.buffer.requests.Load())
	})

	t.Run("nil_option_and_non_positive_timeout_and_idle_fall_back_to_defaults", func(t *testing.T) {
		h := newReplayHarnessWithOptions(
			t,
			[]replayMessage{{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)}},
			false, 0, 50*time.Millisecond, false,
			nil,
			kvevents.WithReplayTimers(kvevents.ReplayTimers{
				Timeout:            -time.Second,
				AttemptIdleTimeout: 0,
				RetryBackoff:       -time.Millisecond,
				Cooldown:           -time.Second,
			}),
		)

		require.Eventually(t, func() bool { return h.indexed(100) },
			5*time.Second, 20*time.Millisecond, "non-positive timeout and idle timeout must fall back to defaults")
		assert.Equal(t, int32(1), h.buffer.requests.Load())
	})

	t.Run("non_positive_retry_backoff_falls_back_to_default", func(t *testing.T) {
		h := newReplayHarnessWithOptions(
			t,
			[]replayMessage{
				{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)},
				{seq: 1, payload: buildDistinctBlockStoredPayload(t, 200)},
			},
			false, 1, 0, false,
			kvevents.WithReplayTimers(kvevents.ReplayTimers{
				AttemptIdleTimeout: 50 * time.Millisecond,
				RetryBackoff:       -time.Millisecond,
			}),
		)

		require.Eventually(t, func() bool {
			return h.buffer.requests.Load() == 2 && h.indexed(200)
		}, 5*time.Second, 20*time.Millisecond)
		elapsed := h.buffer.requestTime(1).Sub(h.buffer.requestTime(0))
		require.GreaterOrEqual(t, elapsed, 120*time.Millisecond,
			"non-positive retry backoff must fall back to default")
	})

	t.Run("non_positive_cooldown_falls_back_to_default", func(t *testing.T) {
		h := newReplayHarnessWithOptions(
			t, nil, true, 0, 0, false,
			kvevents.WithReplayTimers(kvevents.ReplayTimers{
				Cooldown: -time.Second,
			}),
		)

		h.send(t, 5, buildDistinctBlockStoredPayload(t, 100))
		h.send(t, 6, buildDistinctBlockStoredPayload(t, 200))
		require.Eventually(t, func() bool { return h.indexed(100) && h.indexed(200) },
			5*time.Second, 50*time.Millisecond)
		assert.Equal(t, int32(1), h.buffer.requests.Load(), "non-positive cooldown must fall back to default")
	})
}

func (h *replayHarness) send(t *testing.T, seq uint64, payload []byte) {
	t.Helper()
	require.NoError(t, h.pub.Send(zmq4.NewMsgFrom(h.topic, seqFrame(seq), payload)))
}

func TestZMQSubscriber_ProactiveReplayRebuildsServingEndpoint(t *testing.T) {
	h := newReplayHarness(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 300)},
	}, false)

	require.Eventually(t, func() bool {
		key, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		if err != nil {
			return false
		}
		hits, err := h.index.Lookup(h.ctx, []kvblock.BlockHash{key}, nil)
		return err == nil && len(hits[key]) == 1 &&
			hits[key][0].PodIdentifier == "10.0.0.1:8000"
	}, 5*time.Second, 50*time.Millisecond)
}

func TestZMQSubscriber_GapReplayDoesNotDuplicateTriggeringEvent(t *testing.T) {
	h := newReplayHarness(t, nil, false)
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 100))
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)

	h.buffer.set(
		replayMessage{seq: 1, payload: buildDistinctBlockStoredPayload(t, 200)},
		replayMessage{seq: 2, payload: buildDistinctBlockStoredPayload(t, 300)},
	)
	h.send(t, 2, buildDistinctBlockStoredPayload(t, 300))
	require.Eventually(t, func() bool { return h.buffer.requests.Load() == 2 },
		5*time.Second, 50*time.Millisecond, "gap replay expected")
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)

	h.send(t, 3, buildBlockRemovedPayload(t, 300))
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		return err != nil
	}, 5*time.Second, 50*time.Millisecond,
		"one remove must evict a block stored once on the wire")
}

func TestZMQSubscriber_DropsPostGapEventsDuringReplayCooldown(t *testing.T) {
	h := newReplayHarness(t, nil, true)
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 100))
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond)

	h.send(t, 2, buildDistinctBlockStoredPayload(t, 300))
	time.Sleep(300 * time.Millisecond)
	_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
	require.Error(t, err, "event past an unrecovered gap must not reach the index")
}

func TestZMQSubscriber_UnavailableReplayIndexesLiveEvents(t *testing.T) {
	h := newReplayHarness(t, nil, true)
	h.send(t, 5, buildDistinctBlockStoredPayload(t, 100))
	h.send(t, 6, buildDistinctBlockStoredPayload(t, 200))
	require.Eventually(t, func() bool { return h.indexed(100) && h.indexed(200) },
		5*time.Second, 50*time.Millisecond, "live frames must be indexed while the replay is unavailable")
	assert.Equal(t, int32(1), h.buffer.requests.Load(), "the replay is retried only after the cooldown")
}

func TestZMQSubscriber_SequenceResetClearsAndRebuildsPod(t *testing.T) {
	h := newReplayHarness(t, nil, false)
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 100))
	h.send(t, 1, buildDistinctBlockStoredPayload(t, 200))
	require.Eventually(t, func() bool {
		_, firstErr := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
		_, secondErr := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 200))
		return firstErr == nil && secondErr == nil
	}, 5*time.Second, 50*time.Millisecond)
	oldRequestKey, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
	require.NoError(t, err)

	h.buffer.set(replayMessage{seq: 0, payload: buildDistinctBlockStoredPayload(t, 300)})
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 300))
	require.Eventually(t, func() bool { return h.buffer.requests.Load() == 2 },
		5*time.Second, 50*time.Millisecond, "full replay after sequence reset expected")
	require.Eventually(t, func() bool {
		oldHits, lookupErr := h.index.Lookup(h.ctx, []kvblock.BlockHash{oldRequestKey}, nil)
		_, newErr := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 300))
		return lookupErr == nil && len(oldHits[oldRequestKey]) == 0 && newErr == nil
	}, 5*time.Second, 50*time.Millisecond,
		"restart must replace stale pod state with the replayed epoch")
}

func TestZMQSubscriber_SequenceResetWithoutReplayClearsPod(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	t.Cleanup(cancel)

	index, _, pool := newTestPool(t, ctx)

	pubEndpoint := availableEndpoint(t, ctx)
	subManager := kvevents.NewSubscriberManager(pool)
	t.Cleanup(func() { shutdownSubscriberManager(ctx, subManager) })
	require.NoError(t, subManager.EnsureSubscriber(
		ctx, "test-pod", "10.0.0.1:8000", pubEndpoint, "", "", "kv@", false))

	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(t, pub.Dial(pubEndpoint))
	time.Sleep(100 * time.Millisecond)

	topic := []byte("kv@10.0.0.1:8000@TestModel")
	send := func(seq uint64, payload []byte) {
		require.NoError(t, pub.Send(zmq4.NewMsgFrom(topic, seqFrame(seq), payload)))
	}
	send(0, buildDistinctBlockStoredPayload(t, 100))
	send(1, buildDistinctBlockStoredPayload(t, 200))
	require.Eventually(t, func() bool {
		_, firstErr := index.GetRequestKey(ctx, kvevents.ModelEngineKey("TestModel", 100))
		_, secondErr := index.GetRequestKey(ctx, kvevents.ModelEngineKey("TestModel", 200))
		return firstErr == nil && secondErr == nil
	}, 5*time.Second, 50*time.Millisecond)
	oldRequestKey, err := index.GetRequestKey(ctx, kvevents.ModelEngineKey("TestModel", 100))
	require.NoError(t, err)

	// The engine restarted at the same address: its publisher numbers from zero again.
	send(0, buildDistinctBlockStoredPayload(t, 300))
	require.Eventually(t, func() bool {
		oldHits, lookupErr := index.Lookup(ctx, []kvblock.BlockHash{oldRequestKey}, nil)
		_, newErr := index.GetRequestKey(ctx, kvevents.ModelEngineKey("TestModel", 300))
		return lookupErr == nil && len(oldHits[oldRequestKey]) == 0 && newErr == nil
	}, 5*time.Second, 50*time.Millisecond,
		"restart must drop the pod's stale blocks without a replay endpoint")
}

func TestZMQSubscriber_ReplayedLiveEventsDoNotTriggerAnotherReplay(t *testing.T) {
	payload := buildEventBatchPayload(t)
	h := newReplayHarness(t, []replayMessage{
		{seq: 0, payload: payload},
		{seq: 1, payload: payload},
		{seq: 2, payload: payload},
	}, false)

	h.send(t, 0, payload)
	h.send(t, 2, payload)
	time.Sleep(300 * time.Millisecond)
	assert.Equal(t, int32(1), h.buffer.requests.Load())
}

// snapshotServer answers snapshot requests with its current cut and batches.
type snapshotServer struct {
	mu        sync.Mutex
	cut       int64
	publisher []byte
	batches   [][]byte
	requests  atomic.Int32
}

func (s *snapshotServer) set(cut int64, batches ...[]byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cut, s.batches = cut, batches
}

func (s *snapshotServer) setPublisher(publisher byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.publisher = publisherID(publisher)
}

// publisherID is the 16-byte identity a vLLM publisher that serves snapshots
// sends with every batch and snapshot reply.
func publisherID(publisher byte) []byte {
	return bytes.Repeat([]byte{publisher}, 16)
}

// identified appends a publisher identity to a batch, as a vLLM publisher that
// serves snapshots encodes live and replayed batches.
func identified(t *testing.T, payload []byte, publisher byte) []byte {
	t.Helper()
	var fields []msgpack.RawMessage
	require.NoError(t, msgpack.Unmarshal(payload, &fields))
	id, err := msgpack.Marshal(publisherID(publisher))
	require.NoError(t, err)
	out, err := msgpack.Marshal(append(fields, id))
	require.NoError(t, err)
	return out
}

func startSnapshotServer(t *testing.T, ctx context.Context) (*snapshotServer, string) {
	t.Helper()
	server := &snapshotServer{publisher: publisherID(0)}
	rep := zmq4.NewRep(ctx)
	endpoint := listenEphemeral(t, rep)
	t.Cleanup(func() { rep.Close() })
	go func() {
		for {
			if _, err := rep.Recv(); err != nil {
				return
			}
			server.requests.Add(1)
			server.mu.Lock()
			frames := [][]byte{seqFrame(uint64(server.cut)), server.publisher} //nolint:gosec // signed on the wire
			frames = append(frames, server.batches...)
			server.mu.Unlock()
			if err := rep.Send(zmq4.NewMsgFrom(frames...)); err != nil {
				return
			}
		}
	}()
	return server, endpoint
}

func newSnapshotHarness(t *testing.T, cut int64, batches ...[]byte) (*replayHarness, *snapshotServer) {
	t.Helper()
	return newSnapshotReplayHarness(t, nil, cut, batches...)
}

// newSnapshotReplayHarness also serves replay when replay is not nil.
func newSnapshotReplayHarness(
	t *testing.T, replay []replayMessage, cut int64, batches ...[]byte,
) (*replayHarness, *snapshotServer) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	t.Cleanup(cancel)

	index, _, pool := newTestPool(t, ctx)

	pub := zmq4.NewPub(ctx)
	pubEndpoint := listenEphemeral(t, pub)
	server, snapshotEndpoint := startSnapshotServer(t, ctx)
	server.set(cut, batches...)
	replayEndpoint := ""
	var buffer *replayBuffer
	if replay != nil {
		buffer, replayEndpoint = startReplayBuffer(t, ctx)
		buffer.set(replay...)
	}

	subManager := kvevents.NewSubscriberManager(pool)
	t.Cleanup(func() {
		_ = pub.Close()
		shutdownSubscriberManager(ctx, subManager)
	})
	require.NoError(t, subManager.EnsureSubscriber(
		ctx, t.Name(), "10.0.0.1:8000", pubEndpoint, replayEndpoint, snapshotEndpoint, "kv@", true))
	time.Sleep(100 * time.Millisecond)

	return &replayHarness{
		ctx: ctx, index: index, buffer: buffer, pub: pub, subManager: subManager,
		topic: []byte("kv@10.0.0.1:8000@TestModel"),
	}, server
}

func (h *replayHarness) indexed(hash uint64) bool {
	_, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", hash))
	return err == nil
}

func TestZMQSubscriber_SnapshotLoadsStateAndResumesAfterCut(t *testing.T) {
	h, server := newSnapshotHarness(t, 5, buildDistinctBlockStoredPayload(t, 100))

	// The first frame is inside the snapshot: it triggers the load and is not applied.
	h.send(t, 5, buildDistinctBlockStoredPayload(t, 999))
	require.Eventually(t, func() bool { return h.indexed(100) }, 5*time.Second, 50*time.Millisecond)
	h.send(t, 6, buildDistinctBlockStoredPayload(t, 200))
	require.Eventually(t, func() bool { return h.indexed(200) }, 5*time.Second, 50*time.Millisecond)
	assert.False(t, h.indexed(999), "a frame at or before the cut is already in the snapshot")
	assert.Equal(t, int32(1), server.requests.Load())
}

func TestZMQSubscriber_SnapshotRebuildsPodOnGap(t *testing.T) {
	h, server := newSnapshotHarness(t, 5, buildDistinctBlockStoredPayload(t, 100))
	h.send(t, 5, buildEventBatchPayload(t))
	require.Eventually(t, func() bool { return h.indexed(100) }, 5*time.Second, 50*time.Millisecond)

	oldRequestKey, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
	require.NoError(t, err)

	server.set(8, buildDistinctBlockStoredPayload(t, 300))
	h.send(t, 8, buildEventBatchPayload(t))
	require.Eventually(t, func() bool {
		oldHits, lookupErr := h.index.Lookup(h.ctx, []kvblock.BlockHash{oldRequestKey}, nil)
		return lookupErr == nil && len(oldHits[oldRequestKey]) == 0 && h.indexed(300)
	}, 5*time.Second, 50*time.Millisecond, "a gap must replace the pod's state with a new snapshot")
	assert.Equal(t, int32(2), server.requests.Load())
}

func TestZMQSubscriber_UnavailableSnapshotIndexesLiveEvents(t *testing.T) {
	h, server := newSnapshotHarness(t, -2)
	h.send(t, 5, buildDistinctBlockStoredPayload(t, 100))
	h.send(t, 6, buildDistinctBlockStoredPayload(t, 200))
	require.Eventually(t, func() bool { return h.indexed(100) && h.indexed(200) },
		5*time.Second, 50*time.Millisecond, "live frames must be indexed while the snapshot is unavailable")
	assert.Equal(t, int32(1), server.requests.Load(), "the snapshot is retried only after the cooldown")
}

func TestZMQSubscriber_SnapshotSequenceResetReloadsPod(t *testing.T) {
	h, server := newSnapshotHarness(t, 5, buildDistinctBlockStoredPayload(t, 100))
	h.send(t, 5, buildEventBatchPayload(t))
	require.Eventually(t, func() bool { return h.indexed(100) }, 5*time.Second, 50*time.Millisecond)
	oldRequestKey, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
	require.NoError(t, err)

	// The engine restarted at the same address: its publisher numbers from zero again.
	server.set(0, buildDistinctBlockStoredPayload(t, 300))
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 999))
	require.Eventually(t, func() bool {
		oldHits, lookupErr := h.index.Lookup(h.ctx, []kvblock.BlockHash{oldRequestKey}, nil)
		return lookupErr == nil && len(oldHits[oldRequestKey]) == 0 && h.indexed(300)
	}, 5*time.Second, 50*time.Millisecond, "a sequence reset must replace the pod's state with a new snapshot")
	assert.False(t, h.indexed(999), "a frame at or before the cut is already in the snapshot")
	assert.Equal(t, int32(2), server.requests.Load())
}

func TestZMQSubscriber_SnapshotMetrics(t *testing.T) {
	liveOnly := func() float64 {
		var m dto.Metric
		require.NoError(t, metrics.LiveOnlySubscribers.Write(&m))
		return m.GetGauge().GetValue()
	}
	loads := func() uint64 {
		var m dto.Metric
		require.NoError(t, metrics.SnapshotLoadDuration.Write(&m))
		return m.GetHistogram().GetSampleCount()
	}
	before := loads()
	baseline := liveOnly()

	h, server := newSnapshotHarness(t, -2)
	h.send(t, 5, buildDistinctBlockStoredPayload(t, 100))
	require.Eventually(t, func() bool { return h.indexed(100) && liveOnly() == baseline+1 },
		5*time.Second, 50*time.Millisecond, "a subscriber whose snapshot failed must be counted as live-only")
	assert.Equal(t, before, loads(), "an unavailable snapshot is not a load")

	// The engine restarted and serves a snapshot.
	server.set(0, buildDistinctBlockStoredPayload(t, 300))
	h.send(t, 0, buildEventBatchPayload(t))
	require.Eventually(t, func() bool { return h.indexed(300) && liveOnly() == baseline },
		5*time.Second, 50*time.Millisecond, "a loaded snapshot must leave the live-only count")
	assert.Equal(t, before+1, loads())

	// The snapshot is unavailable when a gap rebuilds the pod.
	server.set(-2)
	h.send(t, 3, buildDistinctBlockStoredPayload(t, 400))
	require.Eventually(t, func() bool { return h.indexed(400) && liveOnly() == baseline+1 },
		5*time.Second, 50*time.Millisecond, "a failed rebuild must count the subscriber as live-only again")
	assert.Equal(t, before+1, loads())

	require.True(t, h.subManager.RemoveSubscriber(h.ctx, t.Name()))
	require.Eventually(t, func() bool { return liveOnly() == baseline },
		5*time.Second, 50*time.Millisecond, "a removed live-only subscriber must leave the count")

	// With a replay buffer, the snapshot and the replay both fail on join.
	h, _ = newSnapshotReplayHarness(t, []replayMessage{}, -2)
	h.buffer.fail.Store(true)
	h.send(t, 5, buildDistinctBlockStoredPayload(t, 100))
	require.Eventually(t, func() bool { return h.indexed(100) && liveOnly() == baseline+1 },
		5*time.Second, 50*time.Millisecond, "a subscriber whose snapshot and replay failed must be counted as live-only")

	// The engine restarted and its replay buffer serves the stream; the frame is inside the replay.
	h.buffer.fail.Store(false)
	h.buffer.set(replayMessage{seq: 0, payload: buildDistinctBlockStoredPayload(t, 200)})
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 999))
	require.Eventually(t, func() bool { return h.indexed(200) && liveOnly() == baseline },
		5*time.Second, 50*time.Millisecond, "a completed replay must leave the live-only count")
	assert.False(t, h.indexed(999), "a frame at or before the replayed sequence is already indexed")

	// A failed gap replay starts the cooldown, so the next frame is indexed live.
	h.buffer.fail.Store(true)
	h.send(t, 2, buildEventBatchPayload(t))
	h.send(t, 3, buildDistinctBlockStoredPayload(t, 300))
	require.Eventually(t, func() bool { return h.indexed(300) && liveOnly() == baseline+1 },
		5*time.Second, 50*time.Millisecond, "a failed gap replay must count the subscriber as live-only again")

	// A restart from sequence 0 whose rebuild fails still indexes a complete stream.
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 500))
	require.Eventually(t, func() bool { return h.indexed(500) && liveOnly() == baseline },
		5*time.Second, 50*time.Millisecond, "a restart from sequence 0 must leave the live-only count")
	assert.Equal(t, before+1, loads())
}

func TestZMQSubscriber_SnapshotPreferredOverReplay(t *testing.T) {
	h, server := newSnapshotReplayHarness(t,
		[]replayMessage{{seq: 0, payload: buildDistinctBlockStoredPayload(t, 300)}},
		5, buildDistinctBlockStoredPayload(t, 100))
	h.send(t, 5, buildEventBatchPayload(t))
	require.Eventually(t, func() bool { return h.indexed(100) }, 5*time.Second, 50*time.Millisecond,
		"the snapshot must load on the first live frame")
	assert.False(t, h.indexed(300), "the replay must not run when the snapshot loads")
	assert.Equal(t, int32(1), server.requests.Load())
	assert.Equal(t, int32(0), h.buffer.requests.Load())
}

func TestZMQSubscriber_FailedSnapshotFallsBackToReplay(t *testing.T) {
	h, server := newSnapshotReplayHarness(t,
		[]replayMessage{{seq: 0, payload: buildDistinctBlockStoredPayload(t, 100)}}, -2)
	h.send(t, 1, buildDistinctBlockStoredPayload(t, 200))
	require.Eventually(t, func() bool { return h.indexed(100) && h.indexed(200) },
		5*time.Second, 50*time.Millisecond, "an unavailable snapshot must fall back to a full replay")
	oldRequestKey, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
	require.NoError(t, err)

	// The engine restarted and its snapshot is unavailable, so the pod is rebuilt from replay.
	h.buffer.set(replayMessage{seq: 0, payload: buildDistinctBlockStoredPayload(t, 300)})
	h.send(t, 0, buildDistinctBlockStoredPayload(t, 300))
	require.Eventually(t, func() bool {
		oldHits, lookupErr := h.index.Lookup(h.ctx, []kvblock.BlockHash{oldRequestKey}, nil)
		return lookupErr == nil && len(oldHits[oldRequestKey]) == 0 && h.indexed(300)
	}, 5*time.Second, 50*time.Millisecond, "a failed snapshot must fall back to a full replay")
	assert.Equal(t, int32(2), server.requests.Load())
	assert.Equal(t, int32(2), h.buffer.requests.Load())
}

func TestZMQSubscriber_PublisherChangeReloadsSnapshot(t *testing.T) {
	h, server := newSnapshotHarness(t, 5, buildDistinctBlockStoredPayload(t, 100))
	server.setPublisher(1)
	h.send(t, 5, identified(t, buildEventBatchPayload(t), 1))
	require.Eventually(t, func() bool { return h.indexed(100) }, 5*time.Second, 50*time.Millisecond)
	oldRequestKey, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 100))
	require.NoError(t, err)

	// A restarted engine whose first frame continues the old sequence.
	server.setPublisher(2)
	server.set(6, buildDistinctBlockStoredPayload(t, 300))
	h.send(t, 6, identified(t, buildDistinctBlockStoredPayload(t, 999), 2))
	require.Eventually(t, func() bool {
		oldHits, lookupErr := h.index.Lookup(h.ctx, []kvblock.BlockHash{oldRequestKey}, nil)
		return lookupErr == nil && len(oldHits[oldRequestKey]) == 0 && h.indexed(300)
	}, 5*time.Second, 50*time.Millisecond, "a new publisher identity must replace the pod's state")
	assert.False(t, h.indexed(999), "a frame at or before the cut is already in the snapshot")
	assert.Equal(t, int32(2), server.requests.Load())
}

func TestZMQSubscriber_SnapshotFromAnotherPublisherIsNotApplied(t *testing.T) {
	// The engine restarted between the live frame and the snapshot request.
	h, server := newSnapshotHarness(t, 2, buildDistinctBlockStoredPayload(t, 100))
	server.setPublisher(2)
	h.send(t, 5, identified(t, buildDistinctBlockStoredPayload(t, 200), 1))
	require.Eventually(t, func() bool { return h.indexed(200) }, 5*time.Second, 50*time.Millisecond,
		"live frames must be indexed while no matching snapshot is loaded")
	assert.False(t, h.indexed(100), "a snapshot from another publisher must not be applied")
	oldRequestKey, err := h.index.GetRequestKey(h.ctx, kvevents.ModelEngineKey("TestModel", 200))
	require.NoError(t, err)

	h.send(t, 3, identified(t, buildDistinctBlockStoredPayload(t, 300), 2))
	require.Eventually(t, func() bool {
		oldHits, lookupErr := h.index.Lookup(h.ctx, []kvblock.BlockHash{oldRequestKey}, nil)
		return lookupErr == nil && len(oldHits[oldRequestKey]) == 0 && h.indexed(100) && h.indexed(300)
	}, 5*time.Second, 50*time.Millisecond, "the restarted publisher's snapshot must replace the old blocks")
	assert.Equal(t, int32(2), server.requests.Load())
}
