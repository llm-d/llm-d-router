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
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

const (
	// How long to wait before retrying to connect.
	retryInterval = 5 * time.Second
)

// zmqSubscriber connects to a ZMQ publisher and forwards messages to a pool.
type zmqSubscriber struct {
	pool          *Pool
	podIdentifier string
	endpoint      string
	remote        bool
	topicFilter   string
}

// newZMQSubscriber creates a new ZMQ subscriber.
func newZMQSubscriber(pool *Pool, podIdentifier, endpoint, topicFilter string, remote bool) *zmqSubscriber {
	return &zmqSubscriber{
		pool:          pool,
		podIdentifier: podIdentifier,
		endpoint:      endpoint,
		remote:        remote,
		topicFilter:   topicFilter,
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

	debugLogger := logger.V(logging.DEBUG)

	for {
		msg, err := sub.Recv()
		if err != nil {
			if ctx.Err() != nil {
				return // context cancelled, clean shutdown
			}
			metrics.ZMQErrors.WithLabelValues(z.podIdentifier, "recv").Inc()
			debugLogger.Error(err, "Failed to receive message from zmq subscriber", "endpoint", z.endpoint)
			return // exit to trigger reconnect
		}
		metrics.MessagesReceived.WithLabelValues(z.podIdentifier).Inc()
		parts := msg.Frames
		if len(parts) != 3 {
			debugLogger.Error(nil, "Unexpected frame count", "got", len(parts), "want", 3)
			continue
		}
		topic := string(parts[0])
		seqBytes := parts[1]
		payload := parts[2]

		if len(seqBytes) < 8 {
			debugLogger.Error(nil, "Sequence frame too short", "got", len(seqBytes), "want", 8, "topic", topic, "endpoint", z.endpoint)
			continue
		}
		seq := binary.BigEndian.Uint64(seqBytes)

		debugLogger.V(logging.TRACE).Info("Received message from zmq subscriber",
			"topic", topic,
			"seq", seq,
			"payloadSize", len(payload))

		z.pool.AddTask(&RawMessage{
			Topic:    topic,
			Sequence: seq,
			Payload:  payload,
		})
	}
}
