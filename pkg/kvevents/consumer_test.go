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

package kvevents //nolint:testpackage // exercises stream lifecycle internals

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type recordingConsumer struct {
	operations []string
	sources    []EventSource
	batches    []EventBatch
	err        error
}

func (c *recordingConsumer) ProcessEvents(_ context.Context, source EventSource, batch EventBatch) error {
	c.operations = append(c.operations, "batch")
	c.sources = append(c.sources, source)
	c.batches = append(c.batches, batch)
	return c.err
}

func (c *recordingConsumer) Reset(_ context.Context, source EventSource) error {
	c.operations = append(c.operations, "reset:"+source.Endpoint)
	return nil
}

func TestConsumerPoolOrderedEventsAndReset(t *testing.T) {
	c := &recordingConsumer{}
	p, err := NewConsumerPool(DefaultConfig(), &sourceEndpointAdapter{}, c)
	require.NoError(t, err)
	t.Cleanup(func() { p.Shutdown(t.Context()) })
	for _, msg := range []*RawMessage{
		{Payload: []byte{1}, SourceEndpoint: "serving:8000", Sequence: 4},
		{SourceEndpoint: "serving:8000", reset: true},
		{Payload: []byte{2}, SourceEndpoint: "serving:8000", Sequence: 5},
	} {
		p.processRawMessage(t.Context(), msg)
	}
	assert.Equal(t, []string{"batch", "reset:serving:8000", "batch"}, c.operations)
	assert.Equal(t, EventSource{Endpoint: "serving:8000", ModelName: "test-model", Sequence: 4}, c.sources[0])
	stored := c.batches[0].Events[0].(*BlockStoredEvent)
	assert.Zero(t, stored.BlockSize, "consumer sees events even when the token path would reject them")
	c.err = errors.New("consumer failed")
	p.processRawMessage(t.Context(), &RawMessage{Payload: []byte{3}, SourceEndpoint: "serving:8000"})
	assert.Equal(t, []string{"batch", "reset:serving:8000"}, c.operations[3:])
}

func TestConsumerSubscriberInvalidatesWithoutReplay(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Concurrency = 1
	p, err := NewConsumerPool(cfg, &sourceEndpointAdapter{}, &recordingConsumer{})
	require.NoError(t, err)
	t.Cleanup(func() { p.Shutdown(t.Context()) })
	z := newZMQSubscriber(p, "pod", "serving:8000", "", "", "", "", true)
	require.True(t, z.acceptLiveWithoutReplay("topic", 4))
	require.True(t, z.acceptLiveWithoutReplay("topic", 5))
	require.False(t, z.acceptLiveWithoutReplay("topic", 5))
	assert.Zero(t, p.queues[0].Len())
	require.True(t, z.acceptLiveWithoutReplay("topic", 7))
	require.True(t, z.acceptLiveWithoutReplay("topic", 1))
	assert.Equal(t, 2, p.queues[0].Len())
	for range 2 {
		msg, shutdown := p.queues[0].Get()
		require.False(t, shutdown)
		assert.True(t, msg.reset)
		assert.Equal(t, "serving:8000", msg.SourceEndpoint)
		p.queues[0].Done(msg)
	}
}

func TestConsumerSubscriberDetachDrainsBeforeReset(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Concurrency = 1
	p, err := NewConsumerPool(cfg, &sourceEndpointAdapter{}, &recordingConsumer{})
	require.NoError(t, err)
	t.Cleanup(func() { p.Shutdown(t.Context()) })
	sm := NewSubscriberManager(p)
	done := make(chan struct{})
	sm.subscribers["pod"] = &subscriberEntry{
		sourceEndpoint: "serving:8000", done: done,
		subscriber: newZMQSubscriber(p, "pod", "serving:8000", "", "", "", "", true),
		cancel: func() {
			p.AddTask(&RawMessage{Payload: []byte{1}, SourceEndpoint: "serving:8000"})
			close(done)
		},
	}
	sm.RemoveSubscriber(t.Context(), "pod")
	require.Equal(t, 2, p.queues[0].Len())
	for _, reset := range []bool{false, true} {
		msg, shutdown := p.queues[0].Get()
		require.False(t, shutdown)
		assert.Equal(t, reset, msg.reset)
		p.queues[0].Done(msg)
	}
}

func TestConsumerSubscriberAttachInvalidatesRetainedState(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Concurrency = 1
	p, err := NewConsumerPool(cfg, &sourceEndpointAdapter{}, &recordingConsumer{})
	require.NoError(t, err)
	t.Cleanup(func() { p.Shutdown(t.Context()) })
	sm := NewSubscriberManager(p)
	// A bound socket waits without producing connection errors.
	require.NoError(t, sm.EnsureSubscriber(t.Context(), "pod", "serving:8000", "tcp://127.0.0.1:0", "", "", "", false))
	require.Equal(t, 1, p.queues[0].Len())
	msg, shutdown := p.queues[0].Get()
	require.False(t, shutdown)
	assert.True(t, msg.reset)
	assert.Equal(t, "serving:8000", msg.SourceEndpoint)
	p.queues[0].Done(msg)
	sm.RemoveSubscriber(t.Context(), "pod")
}

func TestConsumerSubscriberRetirementDoesNotWaitForSocketClose(t *testing.T) {
	for _, replace := range []bool{false, true} {
		name := "remove"
		if replace {
			name = "replace"
		}
		t.Run(name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.Concurrency = 1
			p, err := NewConsumerPool(cfg, &sourceEndpointAdapter{}, &recordingConsumer{})
			require.NoError(t, err)
			t.Cleanup(func() { p.Shutdown(t.Context()) })
			sm := NewSubscriberManager(p)
			t.Cleanup(func() { sm.Shutdown(t.Context()) })
			done := make(chan struct{})
			t.Cleanup(func() { close(done) })
			old := newZMQSubscriber(p, "pod", "serving:8000", "old", "", "", "", true)
			sm.subscribers["pod"] = &subscriberEntry{
				subscriber: old, endpoint: "old", sourceEndpoint: "serving:8000",
				done: done, cancel: func() {},
			}
			old.addTask(t.Context(), "topic", 1, []byte{1})
			completed := make(chan error, 1)
			go func() {
				if replace {
					// A bound socket waits without generating reconnect resets.
					completed <- sm.EnsureSubscriber(t.Context(), "pod", "serving:8000", "tcp://127.0.0.1:0", "", "", "", false)
				} else {
					sm.RemoveSubscriber(t.Context(), "pod")
					completed <- nil
				}
			}()
			select {
			case err := <-completed:
				require.NoError(t, err)
			case <-time.After(time.Second):
				t.Fatal("endpoint reconciliation waited for the retired socket to close")
			}
			// Late events and reconnect resets from the retired subscriber must
			// not alter residency after its removal or replacement.
			old.addTask(t.Context(), "topic", 2, []byte{2})
			old.invalidateReplay("topic")
			want := []bool{false, true}
			if replace {
				sm.subscribers["pod"].subscriber.addTask(t.Context(), "topic", 3, []byte{3})
				want = append(want, true, false)
			}
			require.Equal(t, len(want), p.queues[0].Len())
			for _, reset := range want {
				msg, shutdown := p.queues[0].Get()
				require.False(t, shutdown)
				assert.Equal(t, reset, msg.reset)
				assert.Equal(t, "serving:8000", msg.SourceEndpoint)
				if !reset {
					assert.NotEqual(t, uint64(2), msg.Sequence)
				}
				p.queues[0].Done(msg)
			}
		})
	}
}

func TestIndexConsumerPreservesDecodedBatch(t *testing.T) {
	p, _, _ := newTestPool(t, 16)
	defer p.Shutdown(t.Context())
	group := 0
	source := EventSource{Endpoint: "serving:8000", ModelName: "model"}
	stored := &BlockStoredEvent{BlockHashes: []uint64{10}, Tokens: makeTokens(16), BlockSize: 16, GroupIdx: &group, KVCacheSpecKind: KVCacheSpecKindFullAttention}
	require.NoError(t, p.ProcessEvents(t.Context(), source, EventBatch{Events: []GenericEvent{stored}}))
	inferred := &BlockStoredEvent{BlockHashes: []uint64{20}, Tokens: makeTokens(16), BlockSize: 16, GroupIdx: &group}
	require.NoError(t, p.ProcessEvents(t.Context(), source, EventBatch{Events: []GenericEvent{inferred}}))
	require.Empty(t, inferred.KVCacheSpecKind, "inference must remain private to the index consumer")
	require.NoError(t, p.Reset(t.Context(), source))
	_, found := p.GroupCatalog().Get(source.Endpoint, 0)
	require.True(t, found, "stream recovery preserves learned group formats")
	source.Retired = true
	require.NoError(t, p.Reset(t.Context(), source))
	_, found = p.GroupCatalog().Get(source.Endpoint, 0)
	require.False(t, found, "endpoint retirement discards learned formats")
}
