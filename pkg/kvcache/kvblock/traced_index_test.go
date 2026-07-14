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

package kvblock_test

import (
	"context"
	"errors"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestNewTracedIndex(t *testing.T) {
	// Create a base in-memory index
	baseIdx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	// Wrap it with tracing
	tracedIdx := kvblock.NewTracedIndex(baseIdx)
	require.NotNil(t, tracedIdx)
}

func TestTracedIndexBehavior(t *testing.T) {
	ctx := context.Background()

	// Create a base in-memory index
	baseIdx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	// Wrap it with tracing
	tracedIdx := kvblock.NewTracedIndex(baseIdx)

	// Test Add operation
	engineKey := kvblock.BlockHash(123)
	requestKey := kvblock.BlockHash(789)
	entries := []kvblock.PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "cpu"},
	}

	err = tracedIdx.Add(ctx, []kvblock.BlockHash{engineKey}, []kvblock.BlockHash{requestKey}, entries)
	require.NoError(t, err)

	// Test Lookup operation with tracing
	result, err := tracedIdx.Lookup(ctx, []kvblock.BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result[requestKey], 2)

	// Test Evict operation
	err = tracedIdx.Evict(ctx, engineKey, kvblock.EngineKey, []kvblock.PodEntry{entries[0]})
	require.NoError(t, err)

	// Verify eviction worked (pod1 should be removed, pod2 should remain)
	result, err = tracedIdx.Lookup(ctx, []kvblock.BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, result[requestKey], 1)
	require.Equal(t, "pod2", result[requestKey][0].PodIdentifier)
}

func TestTracedIndexCacheHitMetrics(t *testing.T) {
	ctx := context.Background()

	baseIdx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	tracedIdx := kvblock.NewTracedIndex(baseIdx)

	// Add some data
	engineKeys := []kvblock.BlockHash{kvblock.BlockHash(1)}
	requestKeys := []kvblock.BlockHash{kvblock.BlockHash(2)}
	entries := []kvblock.PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}

	err = tracedIdx.Add(ctx, engineKeys, requestKeys, entries)
	require.NoError(t, err)

	// Lookup should succeed and record cache hit
	result, err := tracedIdx.Lookup(ctx, requestKeys, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, result[requestKeys[0]], 1)

	// Lookup non-existent key should record cache miss
	nonExistentKeys := []kvblock.BlockHash{kvblock.BlockHash(999)}
	result, err = tracedIdx.Lookup(ctx, nonExistentKeys, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, result[nonExistentKeys[0]], 0)
}

func TestTracedIndexAddAndEvictSpans(t *testing.T) {
	ctx := context.Background()
	spanRecorder := setupSpanRecorder(t)

	baseIdx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	tracedIdx := kvblock.NewTracedIndex(baseIdx)

	engineKey := kvblock.BlockHash(123)
	requestKey := kvblock.BlockHash(789)
	entries := []kvblock.PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "cpu"},
	}

	err = tracedIdx.Add(ctx, []kvblock.BlockHash{engineKey}, []kvblock.BlockHash{requestKey}, entries)
	require.NoError(t, err)

	err = tracedIdx.Evict(ctx, engineKey, kvblock.EngineKey, []kvblock.PodEntry{entries[0]})
	require.NoError(t, err)

	spans := spanRecorder.Ended()
	addSpan := spanByName(t, spans, "llm_d.kv_cache.index.add")
	addAttrs := spanAttributes(addSpan)
	require.Equal(t, int64(1), addAttrs["llm_d.kv_cache.index.add.engine_key_count"].AsInt64())
	require.Equal(t, int64(1), addAttrs["llm_d.kv_cache.index.add.request_key_count"].AsInt64())
	require.Equal(t, int64(2), addAttrs["llm_d.kv_cache.index.add.pod_entry_count"].AsInt64())
	require.Equal(t, int64(2), addAttrs["llm_d.kv_cache.index.add.device_tier_count"].AsInt64())

	evictSpan := spanByName(t, spans, "llm_d.kv_cache.index.evict")
	evictAttrs := spanAttributes(evictSpan)
	require.Equal(t, "engine", evictAttrs["llm_d.kv_cache.index.evict.key_type"].AsString())
	require.Equal(t, int64(1), evictAttrs["llm_d.kv_cache.index.evict.pod_entry_count"].AsInt64())
	require.Equal(t, int64(1), evictAttrs["llm_d.kv_cache.index.evict.device_tier_count"].AsInt64())
}

func TestTracedIndexAddAndEvictSpansRecordErrors(t *testing.T) {
	ctx := context.Background()
	spanRecorder := setupSpanRecorder(t)

	expectedErr := errors.New("index operation failed")
	tracedIdx := kvblock.NewTracedIndex(&failingIndex{err: expectedErr})

	err := tracedIdx.Add(ctx, nil, []kvblock.BlockHash{1},
		[]kvblock.PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}})
	require.ErrorIs(t, err, expectedErr)

	err = tracedIdx.Evict(ctx, kvblock.BlockHash(1), kvblock.RequestKey,
		[]kvblock.PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}})
	require.ErrorIs(t, err, expectedErr)

	spans := spanRecorder.Ended()
	addSpan := spanByName(t, spans, "llm_d.kv_cache.index.add")
	require.Equal(t, codes.Error, addSpan.Status().Code)
	require.Equal(t, expectedErr.Error(), addSpan.Status().Description)

	evictSpan := spanByName(t, spans, "llm_d.kv_cache.index.evict")
	require.Equal(t, codes.Error, evictSpan.Status().Code)
	require.Equal(t, expectedErr.Error(), evictSpan.Status().Description)
}

func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()

	spanRecorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRecorder))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)

	t.Cleanup(func() {
		otel.SetTracerProvider(previous)
		require.NoError(t, provider.Shutdown(context.Background()))
	})

	return spanRecorder
}

func spanByName(t *testing.T, spans []sdktrace.ReadOnlySpan, name string) sdktrace.ReadOnlySpan {
	t.Helper()

	for _, span := range spans {
		if span.Name() == name {
			return span
		}
	}

	require.Failf(t, "missing span", "span %q not found", name)
	return nil
}

func spanAttributes(span sdktrace.ReadOnlySpan) map[string]attribute.Value {
	attrs := make(map[string]attribute.Value)
	for _, attr := range span.Attributes() {
		attrs[string(attr.Key)] = attr.Value
	}
	return attrs
}

type failingIndex struct {
	err error
}

func (f *failingIndex) Lookup(
	context.Context,
	[]kvblock.BlockHash,
	sets.Set[string],
) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
	return nil, f.err
}

func (f *failingIndex) Add(context.Context, []kvblock.BlockHash, []kvblock.BlockHash, []kvblock.PodEntry) error {
	return f.err
}

func (f *failingIndex) Evict(context.Context, kvblock.BlockHash, kvblock.KeyType, []kvblock.PodEntry) error {
	return f.err
}

func (f *failingIndex) GetRequestKey(context.Context, kvblock.BlockHash) (kvblock.BlockHash, error) {
	return kvblock.EmptyBlockHash, f.err
}

func (f *failingIndex) Clear(context.Context, string) error {
	return f.err
}
