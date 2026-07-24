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
	"context"
	"encoding/binary"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const (
	retryInterval       = 5 * time.Second
	replayTimeout       = 30 * time.Second
	replaySocketTimeout = 10 * time.Second
	replayCooldown      = 30 * time.Second
)

// zmqSubscriber connects to a ZMQ publisher and forwards messages to a pool.
type zmqSubscriber struct {
	pool           *Pool
	endpoint       string
	replayEndpoint string
	remote         bool
	topicFilter    string

	// Replay state — persists across reconnections within subscriber lifetime.
	lastSeq           uint64
	hasLastSeq        bool
	lastReplayFailure time.Time
}

// newZMQSubscriber creates a new ZMQ subscriber.
func newZMQSubscriber(pool *Pool, endpoint, replayEndpoint, topicFilter string, remote bool) *zmqSubscriber {
	return &zmqSubscriber{
		pool:           pool,
		endpoint:       endpoint,
		replayEndpoint: replayEndpoint,
		remote:         remote,
		topicFilter:    topicFilter,
	}
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
				logger.Info("retrying zmq-subscriber")
			case <-ctx.Done():
				logger.Info("shutting down zmq-subscriber")
				return
			}
		}
	}
}

// parseEventFrame validates and extracts topic, sequence, and payload from a
// 3-frame ZMQ message. Returns false if the frame is malformed.
//
//nolint:gocritic // unnamedResult conflicts with nonamedreturns
func parseEventFrame(frames [][]byte) (string, uint64, []byte, bool) {
	if len(frames) != 3 {
		return "", 0, nil, false
	}
	seqBytes := frames[1]
	if len(seqBytes) < 8 {
		return "", 0, nil, false
	}
	return string(frames[0]), binary.BigEndian.Uint64(seqBytes), frames[2], true
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
			logger.Error(err, "Failed to bind subscriber socket", "endpoint", z.endpoint)
			return
		}
		logger.Info("Bound subscriber socket", "endpoint", z.endpoint)
	} else {
		if err := sub.Dial(z.endpoint); err != nil {
			logger.Error(err, "Failed to connect subscriber socket", "endpoint", z.endpoint)
			return
		}
		logger.Info("Connected subscriber socket", "endpoint", z.endpoint)
	}

	if err := sub.SetOption(zmq4.OptionSubscribe, z.topicFilter); err != nil {
		logger.Error(err, "Failed to subscribe to topic filter", "topic", z.topicFilter)
		return
	}

	// Proactive replay on connect: rebuild the index from buffered events
	// without waiting for a live event to arrive.
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
				return // context cancelled, clean shutdown
			}
			debugLogger.Error(err, "Failed to receive message from zmq subscriber", "endpoint", z.endpoint)
			return // exit to trigger reconnect
		}
		parts := msg.Frames
		topic, seq, payload, ok := parseEventFrame(parts)
		if !ok {
			debugLogger.Error(nil, "Malformed event frame", "frameCount", len(parts), "endpoint", z.endpoint)
			continue
		}

		// Gap detection: request replay for missed events.
		if z.replayEndpoint != "" && z.canAttemptReplay() {
			if z.hasLastSeq && seq > z.lastSeq+1 {
				missed := seq - z.lastSeq - 1
				logger.Info("Detected gap in event sequence, requesting replay",
					"lastSeq", z.lastSeq, "currentSeq", seq, "missed", missed,
					"endpoint", z.endpoint)
				if !z.requestReplay(ctx, z.lastSeq+1) {
					// Replay failed — do not advance lastSeq past the gap.
					continue
				}
			} else if !z.hasLastSeq && seq > 0 {
				logger.Info("Joining mid-stream, requesting full replay",
					"currentSeq", seq, "endpoint", z.endpoint)
				z.requestReplay(ctx, 0)
			}
		}

		debugLogger.V(logging.TRACE).Info("Received message from zmq subscriber",
			"topic", topic,
			"seq", seq,
			"payloadSize", len(payload))

		z.pool.AddTask(&RawMessage{
			Topic:    topic,
			Sequence: seq,
			Payload:  payload,
		})

		// Freeze lastSeq during cooldown so the gap is preserved for
		// retry — blocks past a gap are dropped (broken parent chain)
		// and can only be recovered via replay.
		if z.replayEndpoint == "" || z.canAttemptReplay() {
			if !z.hasLastSeq || seq > z.lastSeq {
				z.lastSeq = seq
				z.hasLastSeq = true
			}
		}
	}
}

// canAttemptReplay returns true if enough time has passed since the last
// replay failure.
func (z *zmqSubscriber) canAttemptReplay() bool {
	return z.lastReplayFailure.IsZero() || time.Since(z.lastReplayFailure) >= replayCooldown
}

// requestReplay connects to the vLLM ROUTER socket via a DEALER and requests
// all buffered events starting from startSeq. Replayed events are fed into
// the pool via AddTask, identical to live events. Returns true on success.
func (z *zmqSubscriber) requestReplay(ctx context.Context, startSeq uint64) bool {
	logger := log.FromContext(ctx).WithName("zmq-replay")

	// replayCtx bounds the entire operation including Recv, which blocks on
	// the socket's parent context. WithTimeout on the DEALER only bounds Send.
	replayCtx, cancel := context.WithTimeout(ctx, replayTimeout)
	defer cancel()

	dealer := zmq4.NewDealer(replayCtx, zmq4.WithTimeout(replaySocketTimeout))
	defer dealer.Close()

	if err := dealer.Dial(z.replayEndpoint); err != nil {
		z.lastReplayFailure = time.Now()
		logger.Error(err, "Failed to connect replay socket",
			"replayEndpoint", z.replayEndpoint)
		return false
	}

	// DEALER must prepend an empty delimiter so the ROUTER sees
	// [client_id, empty, startSeq] — matching the REQ envelope format.
	seqBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seqBytes, startSeq)
	if err := dealer.SendMulti(zmq4.NewMsgFrom([]byte{}, seqBytes)); err != nil {
		z.lastReplayFailure = time.Now()
		logger.Error(err, "Failed to send replay request",
			"startSeq", startSeq, "replayEndpoint", z.replayEndpoint)
		return false
	}

	replayed := 0
	for {
		select {
		case <-replayCtx.Done():
			z.lastReplayFailure = time.Now()
			logger.Info("Replay timed out",
				"replayed", replayed, "replayEndpoint", z.replayEndpoint)
			return false
		default:
		}

		msg, err := dealer.Recv()
		if err != nil {
			z.lastReplayFailure = time.Now()
			logger.Error(err, "Failed to receive replay message",
				"replayed", replayed, "replayEndpoint", z.replayEndpoint)
			return false
		}

		frames := msg.Frames
		// DEALER receives [empty_delim, topic, seq, payload].
		// Strip the leading empty delimiter to get [topic, seq, payload].
		if len(frames) > 0 && len(frames[0]) == 0 {
			frames = frames[1:]
		}

		// End of replay: empty payload.
		if len(frames) == 3 && len(frames[2]) == 0 {
			break
		}

		replayTopic, replaySeq, replayPayload, ok := parseEventFrame(frames)
		if !ok {
			z.lastReplayFailure = time.Now()
			logger.Error(nil, "Malformed replay frame",
				"frameCount", len(frames), "replayed", replayed)
			return false
		}

		z.pool.AddTask(&RawMessage{
			Topic:    replayTopic,
			Sequence: replaySeq,
			Payload:  replayPayload,
		})

		if replaySeq > z.lastSeq || !z.hasLastSeq {
			z.lastSeq = replaySeq
			z.hasLastSeq = true
		}

		replayed++
	}

	logger.Info("Replay complete",
		"replayed", replayed, "startSeq", startSeq,
		"replayEndpoint", z.replayEndpoint)
	return true
}
