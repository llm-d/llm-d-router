// Copyright 2026 The llm-d Authors.
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
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// recordedMessages returns the messages a recording holds, in record order.
func recordedMessages(t *testing.T, path string) []*RawMessage {
	t.Helper()
	f, err := os.Open(path)
	require.NoError(t, err)
	defer f.Close()
	var msgs []*RawMessage
	require.NoError(t, forEachRecord(f, func(msg *RawMessage) error {
		msgs = append(msgs, msg)
		return nil
	}))
	return msgs
}

// recordingPath returns a fresh temp file path for a recording sink.
func recordingPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "events.rec")
}

// recordingPool builds a pool that records its received wire events to path.
func recordingPool(t *testing.T, path string, adapter EngineAdapter) *Pool {
	t.Helper()
	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: 16, HashSeed: "test",
	})
	require.NoError(t, err)
	cfg := DefaultConfig()
	cfg.EventRecordingPath = path
	pool, err := NewPool(cfg, idx, tp, adapter)
	require.NoError(t, err)
	return pool
}

// A recording round-trips every field of the messages the pool received,
// in order, and skips subscriber-inferred control messages.
func TestEventRecorderRoundTrip(t *testing.T) {
	path := recordingPath(t)
	pool := recordingPool(t, path, stubAdapter{})
	messages := []*RawMessage{
		{Topic: "kv@pod-1@model-1", Sequence: 7, Payload: []byte{1, 2, 3}, SourceEndpoint: "tcp://10.0.0.1:5557"},
		{Topic: "kv@pod-1@model-1", Sequence: 8, Payload: []byte{4}},
		{Topic: "kv@pod-2@model-1", Sequence: 1, Payload: []byte{5, 6}, SourceEndpoint: "tcp://10.0.0.2:5557"},
		// Control messages are transport state, not wire events.
		{Topic: "kv@", reset: true},
		{Topic: "kv@", reset: true, retire: true},
	}
	for _, msg := range messages {
		pool.AddTask(msg)
	}
	pool.Shutdown(context.Background())

	got := recordedMessages(t, path)
	require.Len(t, got, 3, "control messages must not be recorded")
	for i, want := range messages[:3] {
		assert.Equal(t, want.Topic, got[i].Topic)
		assert.Equal(t, want.Sequence, got[i].Sequence)
		assert.Equal(t, want.Payload, got[i].Payload)
		assert.Equal(t, want.SourceEndpoint, got[i].SourceEndpoint)
	}
}

// A message appended after reopening the same sink extends the recording.
func TestEventRecorderAppendsAcrossSessions(t *testing.T) {
	path := recordingPath(t)
	pool := recordingPool(t, path, stubAdapter{})
	pool.AddTask(&RawMessage{Topic: "kv@pod-1", Sequence: 1, Payload: []byte{1}})
	pool.Shutdown(context.Background())

	pool = recordingPool(t, path, stubAdapter{})
	pool.AddTask(&RawMessage{Topic: "kv@pod-1", Sequence: 2, Payload: []byte{2}})
	pool.Shutdown(context.Background())

	got := recordedMessages(t, path)
	require.Len(t, got, 2)
	assert.Equal(t, uint64(1), got[0].Sequence)
	assert.Equal(t, uint64(2), got[1].Sequence)
}

// countingAdapter counts the messages the pool decoded.
type countingAdapter struct {
	stubAdapter
	calls atomic.Int64
}

func (a *countingAdapter) ParseMessage(msg *RawMessage) (string, string, EventBatch, error) {
	a.calls.Add(1)
	return a.stubAdapter.ParseMessage(msg)
}

// ReplayFile feeds a recording back through the pool's processing path:
// every recorded message is decoded by the adapter and drained by the
// workers, and the replay itself is not re-recorded.
func TestReplayFileFeedsPool(t *testing.T) {
	path := recordingPath(t)
	pool := recordingPool(t, path, stubAdapter{})
	const messages = 5
	for i := range uint64(messages) {
		pool.AddTask(&RawMessage{Topic: "kv@pod-1@model-1", Sequence: i, Payload: []byte{byte(i)}})
	}
	pool.Shutdown(context.Background())
	sizeAfterRecording, err := os.Stat(path)
	require.NoError(t, err)

	adapter := &countingAdapter{}
	replayPool := recordingPool(t, path, adapter)
	replayPool.Start(context.Background())
	require.NoError(t, replayPool.ReplayFile(path))
	require.Eventually(t, func() bool {
		return replayPool.queueDepth.Load() == 0
	}, 5*time.Second, 10*time.Millisecond)
	replayPool.Shutdown(context.Background())

	assert.Equal(t, int64(messages), adapter.calls.Load(), "every recorded message must be processed")
	sizeAfterReplay, err := os.Stat(path)
	require.NoError(t, err)
	assert.Equal(t, sizeAfterRecording.Size(), sizeAfterReplay.Size(), "replayed messages must not be re-recorded")
}

// A recording cut short mid-record is an error, and the records before the
// cut are still processed.
func TestReplayFileTruncatedTail(t *testing.T) {
	path := recordingPath(t)
	pool := recordingPool(t, path, stubAdapter{})
	pool.AddTask(&RawMessage{Topic: "kv@pod-1", Sequence: 1, Payload: []byte{1, 2, 3}})
	pool.AddTask(&RawMessage{Topic: "kv@pod-1", Sequence: 2, Payload: []byte{4, 5, 6}})
	pool.Shutdown(context.Background())

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	// Cut the last record's payload short.
	truncated := filepath.Join(t.TempDir(), "truncated.rec")
	require.NoError(t, os.WriteFile(truncated, data[:len(data)-2], 0o600))

	adapter := &countingAdapter{}
	replayPool := recordingPool(t, truncated, adapter)
	replayPool.Start(context.Background())
	require.Error(t, replayPool.ReplayFile(truncated), "a truncated final record must fail the replay")
	require.Eventually(t, func() bool {
		return replayPool.queueDepth.Load() == 0
	}, 5*time.Second, 10*time.Millisecond)
	replayPool.Shutdown(context.Background())
	assert.Equal(t, int64(1), adapter.calls.Load(), "the complete first record must still be processed")
}
