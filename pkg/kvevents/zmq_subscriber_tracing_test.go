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

package kvevents //nolint:testpackage // tests drive unexported addTask

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// drainOne pulls the single queued message off whichever shard received it.
func drainOne(t *testing.T, pool *Pool) *RawMessage {
	t.Helper()
	for _, q := range pool.queues {
		if q.Len() == 0 {
			continue
		}
		msg, shutdown := q.Get()
		require.False(t, shutdown)
		q.Done(msg)
		return msg
	}
	t.Fatal("no message was enqueued")
	return nil
}

// addTask is the producing half of the receive-to-process trace link: it must
// both emit the span and stamp its identity onto the queued message.
func TestAddTask_EmitsReceiveSpanAndCarriesItsContext(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	pool := newTracingPool(t)
	z := newZMQSubscriber(pool, "pod-1", "10.0.0.1:8003", "tcp://10.0.0.1:5557", "", "kv@", false)

	z.addTask(context.Background(), "kv@10.0.0.1:8000@test-model", 42, []byte{1, 2, 3})

	span := findEventSpan(t, recorder, "events_receive")
	attrs := eventSpanAttrs(span)
	assert.Equal(t, "kv@10.0.0.1:8000@test-model", attrs["llm_d.kv_cache.events.topic"].AsString())
	assert.Equal(t, int64(42), attrs["llm_d.kv_cache.events.sequence"].AsInt64())
	assert.Equal(t, int64(3), attrs["llm_d.kv_cache.events.payload_size_bytes"].AsInt64())
	assert.Equal(t, "10.0.0.1:8003", attrs["llm_d.kv_cache.events.source_endpoint"].AsString())

	// The queued message must carry the span's identity, or processing starts a
	// detached trace no matter how well the consuming side re-parents.
	msg := drainOne(t, pool)
	assert.Equal(t, span.SpanContext().TraceID(), msg.SpanContext.TraceID())
	assert.Equal(t, span.SpanContext().SpanID(), msg.SpanContext.SpanID())
}

// End to end across the queue boundary: the span the subscriber created must be
// the parent of the span the worker creates.
func TestAddTask_ReceiveSpanParentsProcessSpan(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	pool := newTracingPool(t)
	pool.adapter = &sourceEndpointAdapter{}
	z := newZMQSubscriber(pool, "pod-1", "", "tcp://10.0.0.1:5557", "", "kv@", false)

	z.addTask(context.Background(), "kv@10.0.0.1:8000@test-model", 7, []byte{1})
	pool.processRawMessage(context.Background(), drainOne(t, pool))

	receive := findEventSpan(t, recorder, "events_receive")
	process := findEventSpan(t, recorder, "events_process")
	require.Equal(t, receive.SpanContext().TraceID(), process.SpanContext().TraceID())
	assert.Equal(t, receive.SpanContext().SpanID(), process.Parent().SpanID())
}
