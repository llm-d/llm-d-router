/*
Copyright 2025 The llm-d Authors.

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

package kvblock_test

import (
	"context"
	"errors"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

func createInstrumentedIndexForTesting(t *testing.T) Index {
	t.Helper()
	cfg := DefaultInMemoryIndexConfig()
	cfg.PodCacheSize = 1000 // for testConcurrentOperations
	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)
	instrumented := NewInstrumentedIndex(index)
	assert.NotNil(t, instrumented)
	return instrumented
}

func TestNewInstrumentedIndex(t *testing.T) {
	// Wrap with instrumentation
	instrumented := createInstrumentedIndexForTesting(t)
	// Verify it implements Index interface
	assert.Implements(t, (*Index)(nil), instrumented)
}

func TestInstrumentedIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createInstrumentedIndexForTesting)
}

func TestInstrumentedIndexCountsOnlySuccessfulOperations(t *testing.T) {
	ctx := context.Background()

	requestKeys := []BlockHash{1, 2, 3}
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}

	admissionsBefore := testutil.ToFloat64(metrics.Admissions)
	evictionsBefore := testutil.ToFloat64(metrics.Evictions)

	// Failed operations must not move the admission/eviction counters: the
	// metric help ("Total number of KV-block admissions/evictions") and
	// docs/metrics.md ("Blocks admitted/evicted from the index") count blocks
	// that actually changed the index.
	failing := NewInstrumentedIndex(&failingIndex{err: errors.New("index unavailable")})
	require.Error(t, failing.Add(ctx, nil, requestKeys, entries))
	require.Error(t, failing.Evict(ctx, BlockHash(1), EngineKey, entries))

	assert.InDelta(t, admissionsBefore, testutil.ToFloat64(metrics.Admissions), 1e-9,
		"failed admissions must not be counted")
	assert.InDelta(t, evictionsBefore, testutil.ToFloat64(metrics.Evictions), 1e-9,
		"failed evictions must not be counted")

	// Successful operations must still count every block.
	succeeding := NewInstrumentedIndex(&failingIndex{})
	require.NoError(t, succeeding.Add(ctx, nil, requestKeys, entries))
	require.NoError(t, succeeding.Evict(ctx, BlockHash(1), EngineKey, entries))

	assert.InDelta(t, admissionsBefore+float64(len(requestKeys)), testutil.ToFloat64(metrics.Admissions), 1e-9,
		"successful admissions must be counted")
	assert.InDelta(t, evictionsBefore+float64(len(entries)), testutil.ToFloat64(metrics.Evictions), 1e-9,
		"successful evictions must be counted")
}
