package kvevents

import (
	"context"
	"errors"
	"testing"

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

func (c *recordingConsumer) Reset(_ context.Context, endpoint string) error {
	c.operations = append(c.operations, "reset:"+endpoint)
	return nil
}

func TestConsumerPoolOrderedEventsAndReset(t *testing.T) {
	c := &recordingConsumer{}
	p := NewConsumerPool(DefaultConfig(), &sourceEndpointAdapter{}, c)
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
	p := NewConsumerPool(cfg, &sourceEndpointAdapter{}, &recordingConsumer{})
	t.Cleanup(func() { p.Shutdown(t.Context()) })
	z := newZMQSubscriber(p, "pod", "serving:8000", "", "", "", true)
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
	p := NewConsumerPool(cfg, &sourceEndpointAdapter{}, &recordingConsumer{})
	t.Cleanup(func() { p.Shutdown(t.Context()) })
	sm := NewSubscriberManager(p)
	done := make(chan struct{})
	sm.subscribers["pod"] = &subscriberEntry{
		sourceEndpoint: "serving:8000", done: done,
		subscriber: newZMQSubscriber(p, "pod", "serving:8000", "", "", "", true),
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
	p := NewConsumerPool(cfg, &sourceEndpointAdapter{}, &recordingConsumer{})
	t.Cleanup(func() { p.Shutdown(t.Context()) })
	sm := NewSubscriberManager(p)
	// No socket is needed to verify ordering before the subscriber starts.
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	require.NoError(t, sm.EnsureSubscriber(ctx, "pod", "serving:8000", "", "", "", true))
	require.Equal(t, 1, p.queues[0].Len())
	msg, shutdown := p.queues[0].Get()
	require.False(t, shutdown)
	assert.True(t, msg.reset)
	assert.Equal(t, "serving:8000", msg.SourceEndpoint)
	p.queues[0].Done(msg)
	sm.RemoveSubscriber(t.Context(), "pod")
}
