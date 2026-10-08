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

package kvevents

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5/msgpcode"
	"go.opentelemetry.io/otel/attribute"
	"golang.org/x/sync/semaphore"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

const (
	retryInterval               = 5 * time.Second
	replayTimeout               = 2 * time.Minute
	replayAttemptIdleTimeout    = 2 * time.Second
	replayRetryBackoff          = 100 * time.Millisecond
	replayCooldown              = 30 * time.Second
	maxConcurrentReplay         = 8
	maxReplayNoProgressAttempts = 3
	snapshotTimeout             = 30 * time.Second
	publisherIDSize             = 16
)

var processReplayLimiter = semaphore.NewWeighted(maxConcurrentReplay)

// zmqSubscriber connects to a ZMQ publisher and forwards messages to a pool.
type zmqSubscriber struct {
	pool           *Pool
	podIdentifier  string
	sourceEndpoint string
	endpoint       string
	replayEndpoint string
	// snapshotEndpoint serves the publisher's current cache state, used in
	// place of a full replay.
	snapshotEndpoint string
	remote           bool
	topicFilter      string
	queueMu          sync.Mutex
	retired          bool

	// Replay state persists across reconnections within subscriber lifetime.
	lastSeq           uint64
	hasLastSeq        bool
	lastLiveSeq       uint64
	hasLastLiveSeq    bool
	lastReplayFailure time.Time
	// publisherID identifies the engine process behind the live stream; it
	// changes only when the engine restarts.
	publisherID []byte
}

// newZMQSubscriber creates a new ZMQ subscriber.
func newZMQSubscriber(
	pool *Pool,
	podIdentifier, sourceEndpoint, endpoint, replayEndpoint, snapshotEndpoint, topicFilter string,
	remote bool,
) *zmqSubscriber {
	return &zmqSubscriber{
		pool:             pool,
		podIdentifier:    podIdentifier,
		sourceEndpoint:   sourceEndpoint,
		endpoint:         endpoint,
		replayEndpoint:   replayEndpoint,
		snapshotEndpoint: snapshotEndpoint,
		remote:           remote,
		topicFilter:      topicFilter,
	}
}

// parseEventFrame validates and extracts a live or replayed event frame.
// The returned sequence number is vLLM's per-pod event counter, which never
// approaches int64 overflow.
//
//nolint:gocritic // unnamedResult conflicts with nonamedreturns
func parseEventFrame(frames [][]byte) (string, uint64, []byte, bool) {
	if len(frames) != 3 || len(frames[1]) < 8 {
		return "", 0, nil, false
	}
	return string(frames[0]), binary.BigEndian.Uint64(frames[1]), frames[2], true
}

// batchPublisherID returns the identity a vLLM publisher that serves snapshots
// appends to each batch, [ts, events, data_parallel_rank, publisher_id], or nil
// for a batch without one. As the last element, a 16-byte publisher_id is the
// batch's final bytes, so the events are not decoded here.
func batchPublisherID(payload []byte) []byte {
	const binHeaderSize = 2
	if len(payload) < 1+binHeaderSize+publisherIDSize || payload[0] != msgpcode.FixedArrayLow|4 {
		return nil
	}
	tail := payload[len(payload)-binHeaderSize-publisherIDSize:]
	if tail[0] != msgpcode.Bin8 || tail[1] != publisherIDSize {
		return nil
	}
	return tail[binHeaderSize:]
}

// Start connects to a ZMQ PUB socket as a SUB, receives messages,
// wraps them in RawMessage structs, and pushes them into the pool.
// This loop will run until the provided context is canceled.
func (z *zmqSubscriber) Start(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("zmq-subscriber")

	for {
		select {
		case <-ctx.Done():
			logger.Info("shutting down zmq-subscriber")
			return
		default:
			// We run the subscriber in a separate function to handle socket
			// setup/teardown and connection retries cleanly.
			z.runSubscriber(ctx)
			// wait before retrying, unless the context has been canceled.
			select {
			case <-time.After(retryInterval):
				metrics.SubscriberReconnections.WithLabelValues(z.podIdentifier).Inc()
				logger.Info("retrying zmq-subscriber")
			case <-ctx.Done():
				logger.Info("shutting down zmq-subscriber")
				return
			}
		}
	}
}

// runSubscriber connects to the ZMQ PUB socket, subscribes to the topic filter,
// and listens for messages.
func (z *zmqSubscriber) runSubscriber(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("zmq-subscriber")

	// Disable zmq4's automatic reconnect to avoid a data race in the library:
	// when autoReconnect is true, scheduleRmConn calls Dial which writes
	// socket state without proper locking, racing with Close().
	// Reconnection is already handled by the outer retry loop in Start().
	sub := zmq4.NewSub(ctx)
	defer sub.Close()

	// Bind for local endpoints, connect for remote ones.
	if !z.remote {
		if err := sub.Listen(z.endpoint); err != nil {
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "bind").Inc()
			logger.Error(err, "Failed to bind subscriber socket", "endpoint", z.endpoint)
			return
		}
		logger.Info("Bound subscriber socket", "endpoint", z.endpoint)
	} else {
		if err := sub.Dial(z.endpoint); err != nil {
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "connect").Inc()
			logger.Error(err, "Failed to connect subscriber socket", "endpoint", z.endpoint)
			return
		}
		logger.Info("Connected subscriber socket", "endpoint", z.endpoint)
	}

	if err := sub.SetOption(zmq4.OptionSubscribe, z.topicFilter); err != nil {
		metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "subscribe").Inc()
		logger.Error(err, "Failed to subscribe to topic filter", "topic", z.topicFilter)
		return
	}

	// Rebuild the index from buffered events without waiting for live traffic.
	if z.replayEndpoint != "" && !z.hasLastSeq && z.canAttemptReplay() {
		logger.Info("Requesting proactive replay on connect",
			"endpoint", z.endpoint, "replayEndpoint", z.replayEndpoint)
		z.requestReplay(ctx, 0)
	}

	debugLogger := logger.V(logging.DEBUG)
	for {
		msg, err := sub.Recv()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "recv").Inc()
			debugLogger.Error(err, "Failed to receive message from zmq subscriber", "endpoint", z.endpoint)
			return
		}
		metrics.MessagesReceived.WithLabelValues(z.podIdentifier).Inc()
		topic, seq, payload, ok := parseEventFrame(msg.Frames)
		if !ok {
			debugLogger.Error(nil, "Malformed event frame",
				"frameCount", len(msg.Frames), "endpoint", z.endpoint)
			continue
		}

		if z.replayEndpoint == "" && z.snapshotEndpoint == "" {
			// A per-endpoint subscriber reads a single publisher, so a backwards
			// sequence means the engine restarted with an empty cache. The bound
			// global socket interleaves many publishers, so its sequences move
			// backwards without a restart.
			if z.sourceEndpoint != "" && z.hasLastLiveSeq && seq < z.lastLiveSeq {
				logger.Info("Detected event sequence reset, clearing pod state",
					"lastLiveSeq", z.lastLiveSeq, "currentSeq", seq,
					"endpoint", z.endpoint)
				z.resetForSource(topic)
			}
			z.lastLiveSeq = seq
			z.hasLastLiveSeq = true
			z.addTask(ctx, topic, seq, payload)
			continue
		}

		restarted := z.hasLastLiveSeq && seq < z.lastLiveSeq
		if z.snapshotEndpoint != "" {
			if id := batchPublisherID(payload); id != nil && !bytes.Equal(id, z.publisherID) {
				restarted = restarted || z.publisherID != nil
				z.publisherID = bytes.Clone(id)
			}
		}

		replayAttempted := false
		if restarted {
			logger.Info("Detected publisher restart, rebuilding index",
				"lastLiveSeq", z.lastLiveSeq, "currentSeq", seq,
				"endpoint", z.endpoint)
			z.resetForSource(topic)
			z.lastSeq = 0
			z.hasLastSeq = false
			z.hasLastLiveSeq = false
			z.lastReplayFailure = time.Time{}
			replayAttempted = true
			z.rebuild(ctx, topic)
		}

		if z.hasLastLiveSeq && seq == z.lastLiveSeq {
			continue
		}
		z.lastLiveSeq = seq
		z.hasLastLiveSeq = true

		if z.hasLastSeq && seq <= z.lastSeq {
			continue
		}

		if z.hasLastSeq && seq > z.lastSeq+1 {
			missed := seq - z.lastSeq - 1
			if !z.canAttemptReplay() {
				debugLogger.Info("Dropping event while replay is in cooldown",
					"lastSeq", z.lastSeq, "currentSeq", seq, "missed", missed,
					"endpoint", z.endpoint)
				continue
			}
			logger.Info("Detected gap in event sequence, requesting replay",
				"lastSeq", z.lastSeq, "currentSeq", seq, "missed", missed,
				"endpoint", z.endpoint)
			replayAttempted = true
			if z.replayEndpoint == "" {
				// Without a replay buffer the publisher is rebuilt from its snapshot.
				z.resetForSource(topic)
				z.hasLastSeq = false
				z.rebuild(ctx, topic)
			} else if !z.requestReplay(ctx, z.lastSeq+1) {
				continue
			}
		}

		if !z.hasLastSeq && seq > 0 {
			rebuilt := false
			if !replayAttempted && z.canAttemptReplay() {
				logger.Info("Joining mid-stream, requesting full replay",
					"currentSeq", seq, "endpoint", z.endpoint)
				rebuilt = z.rebuild(ctx, topic)
			}
			if !rebuilt {
				// Without a base the pod is indexed from live events alone,
				// as it is with neither endpoint set.
				if z.snapshotEndpoint != "" {
					z.addTask(ctx, topic, seq, payload)
				}
				continue
			}
		}

		if z.hasLastSeq {
			if seq <= z.lastSeq || seq > z.lastSeq+1 {
				continue
			}
		}

		debugLogger.V(logging.TRACE).Info("Received message from zmq subscriber",
			"topic", topic, "seq", seq, "payloadSize", len(payload))
		z.addTask(ctx, topic, seq, payload)
		z.lastSeq = seq
		z.hasLastSeq = true
	}
}

// addTask hands a received message to the pool, carrying the receive span's
// identity so processing rejoins this trace across the worker queue. The span
// starts after Recv returns so it measures handoff work rather than the idle
// wait for the next message.
func (z *zmqSubscriber) addTask(ctx context.Context, topic string, seq uint64, payload []byte) {
	// Spans route through the pool so a single Config.Tracing decision governs
	// every stage of the pipeline.
	_, span := z.pool.startSpan(ctx, "events_receive", consumerSpanOptions)
	defer span.End()
	if span.IsRecording() {
		//#nosec -- seq is vLLM's per-pod event counter; see parseEventFrame doc
		seqAttr := int64(seq)
		attrs := []attribute.KeyValue{
			semconv.LLMDKVCacheEventsTopic(topic),
			semconv.LLMDKVCacheEventsSequence(seqAttr),
			semconv.LLMDKVCacheEventsPayloadSizeBytes(len(payload)),
		}
		// Empty unless the subscriber was created by pod discovery.
		if z.sourceEndpoint != "" {
			attrs = append(attrs, semconv.LLMDKVCacheEventsSourceEndpoint(z.sourceEndpoint))
		}
		span.SetAttributes(attrs...)
	}

	msg := &RawMessage{
		Topic:          topic,
		Sequence:       seq,
		Payload:        payload,
		SourceEndpoint: z.sourceEndpoint,
	}
	// carried is bound inside the branch on purpose. Taking &sc directly makes
	// sc escape, so it heap-allocates on every message including the ones the
	// disabled path never traces.
	if sc := span.SpanContext(); sc.IsValid() {
		carried := sc
		msg.SpanContext = &carried
	}
	z.enqueue(msg)
}

func (z *zmqSubscriber) enqueue(msg *RawMessage) {
	z.queueMu.Lock()
	defer z.queueMu.Unlock()
	if z.retired {
		return
	}
	z.pool.AddTask(msg)
}

func (z *zmqSubscriber) resetForSource(topic string) {
	z.enqueue(&RawMessage{Topic: topic, SourceEndpoint: z.sourceEndpoint, reset: true})
}

// retire prevents any later messages from this subscriber from being queued.
// When resetSource is true, it queues a reset after all messages accepted before
// retirement. The pool shards both messages and the reset by source endpoint,
// so the worker processes them in that order.
func (z *zmqSubscriber) retire(resetSource bool) {
	z.queueMu.Lock()
	defer z.queueMu.Unlock()
	z.retired = true
	if resetSource && z.sourceEndpoint != "" {
		z.pool.AddTask(&RawMessage{
			Topic:          z.topicFilter,
			SourceEndpoint: z.sourceEndpoint,
			reset:          true,
			retire:         true,
		})
	}
}

func (z *zmqSubscriber) canAttemptReplay() bool {
	return z.lastReplayFailure.IsZero() || time.Since(z.lastReplayFailure) >= replayCooldown
}

func (z *zmqSubscriber) invalidateReplay(topic string) {
	z.resetForSource(topic)
	z.lastSeq = 0
	z.hasLastSeq = false
	z.lastReplayFailure = time.Now()
}

// rebuild loads the publisher's current state from its snapshot, and falls
// back to replaying its buffered events from the start.
func (z *zmqSubscriber) rebuild(ctx context.Context, topic string) bool {
	if z.snapshotEndpoint != "" && z.requestSnapshot(ctx, topic) {
		return true
	}
	return z.replayEndpoint != "" && z.requestReplay(ctx, 0)
}

// requestSnapshot queues the publisher's snapshot, event batches that rebuild
// its cache state up to a sequence cut, and resumes the stream after the cut.
// A failure starts the replay cooldown.
func (z *zmqSubscriber) requestSnapshot(ctx context.Context, topic string) bool {
	logger := log.FromContext(ctx).WithName("zmq-snapshot")
	started := time.Now()
	cut, publisherID, batches, err := z.fetchSnapshot(ctx)
	if err == nil && z.publisherID != nil && !bytes.Equal(publisherID, z.publisherID) {
		err = fmt.Errorf("snapshot from publisher %x while the live stream is from %x", publisherID, z.publisherID)
	}
	if err != nil {
		z.lastReplayFailure = time.Now()
		metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "snapshot").Inc()
		logger.Error(err, "Failed to load snapshot", "snapshotEndpoint", z.snapshotEndpoint)
		return false
	}
	// The snapshot replaces events indexed while no snapshot was available.
	z.resetForSource(topic)
	z.publisherID = bytes.Clone(publisherID)
	for _, batch := range batches {
		z.addTask(ctx, topic, 0, batch)
	}
	// A cut of -1 is a publisher that has not published yet.
	z.lastSeq, z.hasLastSeq = 0, false
	if cut >= 0 {
		z.lastSeq, z.hasLastSeq = uint64(cut), true
	}
	z.lastReplayFailure = time.Time{}
	logger.Info("Snapshot loaded", "cut", cut, "batches", len(batches),
		"duration", time.Since(started), "snapshotEndpoint", z.snapshotEndpoint)
	return true
}

// fetchSnapshot requests the snapshot. The reply frames are an 8-byte signed
// sequence cut, the 16-byte publisher identity and the event batches; a cut
// below -1 means the publisher cannot serve one.
//
//nolint:gocritic // unnamedResult conflicts with nonamedreturns
func (z *zmqSubscriber) fetchSnapshot(ctx context.Context) (int64, []byte, [][]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, snapshotTimeout)
	defer cancel()
	if err := processReplayLimiter.Acquire(ctx, 1); err != nil {
		return 0, nil, nil, fmt.Errorf("waiting for capacity: %w", err)
	}
	defer processReplayLimiter.Release(1)

	req := zmq4.NewReq(ctx, zmq4.WithTimeout(snapshotTimeout), zmq4.WithDialerMaxRetries(0))
	defer req.Close()
	if err := req.Dial(z.snapshotEndpoint); err != nil {
		return 0, nil, nil, err
	}
	if err := req.Send(zmq4.NewMsgString("snapshot")); err != nil {
		return 0, nil, nil, err
	}
	msg, err := req.Recv()
	if err != nil {
		return 0, nil, nil, err
	}
	if len(msg.Frames) < 2 || len(msg.Frames[0]) != 8 || len(msg.Frames[1]) != publisherIDSize {
		return 0, nil, nil, fmt.Errorf("malformed snapshot reply with %d frames", len(msg.Frames))
	}
	cut := int64(binary.BigEndian.Uint64(msg.Frames[0])) //nolint:gosec // signed on the wire
	if cut < -1 {
		return 0, nil, nil, fmt.Errorf("snapshot unavailable")
	}
	return cut, msg.Frames[1], msg.Frames[2:], nil
}

// requestReplay requests buffered events starting from startSeq.
func (z *zmqSubscriber) requestReplay(ctx context.Context, startSeq uint64) bool {
	logger := log.FromContext(ctx).WithName("zmq-replay")
	debugLogger := logger.V(logging.DEBUG)

	replayCtx, cancel := context.WithTimeout(ctx, replayTimeout)
	defer cancel()

	replayed := 0
	nextSeq := startSeq
	attempt := 0
	noProgressAttempts := 0
	for {
		if replayCtx.Err() != nil {
			z.invalidateReplay(z.topicFilter)
			logger.Info("Replay timed out",
				"replayed", replayed, "attempts", attempt,
				"replayEndpoint", z.replayEndpoint)
			return false
		}
		attempt++

		attemptCtx, attemptCancel := context.WithCancel(replayCtx)
		dealer := zmq4.NewDealer(attemptCtx, zmq4.WithTimeout(replayAttemptIdleTimeout))
		if err := dealer.Dial(z.replayEndpoint); err != nil {
			attemptCancel()
			z.invalidateReplay(z.topicFilter)
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "replay-connect").Inc()
			logger.Error(err, "Failed to connect replay socket",
				"replayEndpoint", z.replayEndpoint)
			return false
		}
		if replayCtx.Err() != nil {
			_ = dealer.Close()
			attemptCancel()
			continue
		}
		waitStarted := time.Now()
		if err := processReplayLimiter.Acquire(replayCtx, 1); err != nil {
			_ = dealer.Close()
			attemptCancel()
			z.invalidateReplay(z.topicFilter)
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "replay-capacity").Inc()
			logger.Info("Replay timed out waiting for process capacity",
				"waitDuration", time.Since(waitStarted),
				"replayEndpoint", z.replayEndpoint)
			return false
		}
		if waitDuration := time.Since(waitStarted); waitDuration >= time.Second {
			logger.Info("Replay admitted after waiting for process capacity",
				"waitDuration", waitDuration, "replayEndpoint", z.replayEndpoint)
		}

		seqBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(seqBytes, nextSeq)
		if err := dealer.SendMulti(zmq4.NewMsgFrom([]byte{}, seqBytes)); err != nil {
			_ = dealer.Close()
			attemptCancel()
			processReplayLimiter.Release(1)
			z.invalidateReplay(z.topicFilter)
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "replay-send").Inc()
			logger.Error(err, "Failed to send replay request",
				"nextSeq", nextSeq, "replayEndpoint", z.replayEndpoint)
			return false
		}

		idleTimer := time.AfterFunc(replayAttemptIdleTimeout, attemptCancel)
		attemptReplayed := 0
		complete := false
		expectedSeq := nextSeq
		var receiveErr error
		var terminalErr error
		for {
			msg, err := dealer.Recv()
			if err != nil {
				receiveErr = err
				break
			}
			idleTimer.Reset(replayAttemptIdleTimeout)

			frames := msg.Frames
			if len(frames) > 0 && len(frames[0]) == 0 {
				frames = frames[1:]
			}
			if len(frames) == 3 && len(frames[2]) == 0 {
				complete = true
				break
			}

			topic, seq, payload, ok := parseEventFrame(frames)
			if !ok {
				terminalErr = fmt.Errorf("malformed replay frame with %d frames", len(frames))
				break
			}
			if seq != expectedSeq {
				terminalErr = fmt.Errorf("incomplete replay: expected sequence %d, got %d", expectedSeq, seq)
				break
			}

			z.addTask(ctx, topic, seq, payload)
			z.lastSeq = seq
			z.hasLastSeq = true
			replayed++
			attemptReplayed++
			expectedSeq++
		}

		idleTimer.Stop()
		_ = dealer.Close()
		attemptCancel()
		processReplayLimiter.Release(1)
		if terminalErr != nil {
			z.invalidateReplay(z.topicFilter)
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "replay-incomplete").Inc()
			logger.Error(terminalErr, "Replay response is incomplete",
				"attempt", attempt, "replayed", replayed,
				"nextSeq", nextSeq, "replayEndpoint", z.replayEndpoint)
			return false
		}
		if complete {
			if replayed == 0 && startSeq > 0 {
				err := fmt.Errorf("incomplete replay: sequence %d was not available", startSeq)
				z.invalidateReplay(z.topicFilter)
				metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "replay-incomplete").Inc()
				logger.Error(err, "Replay response is incomplete",
					"attempt", attempt, "replayed", replayed,
					"nextSeq", nextSeq, "replayEndpoint", z.replayEndpoint)
				return false
			}
			z.lastReplayFailure = time.Time{}
			logger.Info("Replay complete", "replayed", replayed,
				"attempts", attempt, "startSeq", startSeq,
				"replayEndpoint", z.replayEndpoint)
			return true
		}
		if replayCtx.Err() != nil {
			continue
		}

		if attemptReplayed == 0 {
			noProgressAttempts++
			if noProgressAttempts >= maxReplayNoProgressAttempts {
				if receiveErr == nil {
					receiveErr = fmt.Errorf("replay response ended without progress")
				}
				z.invalidateReplay(z.topicFilter)
				metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "replay-no-progress").Inc()
				logger.Error(receiveErr, "Replay stopped after no progress",
					"attempts", attempt, "replayed", replayed,
					"nextSeq", nextSeq, "replayEndpoint", z.replayEndpoint)
				return false
			}
		} else {
			noProgressAttempts = 0
			nextSeq = expectedSeq
		}
		debugLogger.Info("Replay response interrupted, resuming",
			"attempt", attempt, "attemptReplayed", attemptReplayed,
			"replayed", replayed, "nextSeq", nextSeq, "error", receiveErr,
			"replayEndpoint", z.replayEndpoint)

		select {
		case <-time.After(replayRetryBackoff):
		case <-replayCtx.Done():
		}
	}
}
