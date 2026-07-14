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
	"encoding/binary"
	"testing"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/stretchr/testify/require"
)

// BenchmarkZMQSubscriber_Throughput measures raw pub→sub message throughput
// using the pure-Go ZMQ library.
func BenchmarkZMQSubscriber_Throughput(b *testing.B) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	endpoint := "tcp://127.0.0.1:15560"
	filter := "kv@"

	// Subscriber binds.
	sub := zmq4.NewSub(ctx)
	defer sub.Close()
	require.NoError(b, sub.Listen(endpoint))
	require.NoError(b, sub.SetOption(zmq4.OptionSubscribe, filter))

	// Give subscriber time to bind.
	time.Sleep(50 * time.Millisecond)

	// Publisher connects.
	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	require.NoError(b, pub.Dial(endpoint))

	// Give connection time to establish.
	time.Sleep(50 * time.Millisecond)

	topic := []byte("kv@10.0.0.1@BenchModel")
	seqBytes := make([]byte, 8)
	payload := make([]byte, 256) // simulate a small event batch

	// Receive goroutine counts messages.
	received := make(chan struct{}, b.N+1)
	go func() {
		for {
			_, err := sub.Recv()
			if err != nil {
				return
			}
			received <- struct{}{}
		}
	}()

	b.ResetTimer()
	start := time.Now()

	for i := 0; i < b.N; i++ {
		binary.BigEndian.PutUint64(seqBytes, uint64(i)) //nolint:gosec // i is a non-negative loop counter, conversion is safe
		if err := pub.Send(zmq4.NewMsgFrom(topic, seqBytes, payload)); err != nil {
			b.Fatalf("send failed: %v", err)
		}
	}

	// Wait for all messages to be received (with timeout).
	deadline := time.After(30 * time.Second)
	for i := 0; i < b.N; i++ {
		select {
		case <-received:
		case <-deadline:
			b.Fatalf("timeout waiting for messages (%d/%d received)", i, b.N)
		}
	}

	elapsed := time.Since(start)
	b.StopTimer()

	b.ReportMetric(float64(b.N)/elapsed.Seconds(), "msg/s")
}
