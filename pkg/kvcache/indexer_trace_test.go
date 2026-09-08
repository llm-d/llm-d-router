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

package kvcache_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		otel.SetTracerProvider(previous)
		require.NoError(t, provider.Shutdown(context.Background()))
	})
	return recorder
}

func spanAttributes(t *testing.T, recorder *tracetest.SpanRecorder, name string) map[string]attribute.Value {
	t.Helper()
	for _, span := range recorder.Ended() {
		if span.Name() == name {
			attrs := make(map[string]attribute.Value)
			for _, attr := range span.Attributes() {
				attrs[string(attr.Key)] = attr.Value
			}
			return attrs
		}
	}
	require.Failf(t, "span not recorded", "no ended span named %q", name)
	return nil
}

// blocks_found counts requested keys held by any candidate pod, not the
// longest chain: pod-a holds only the first key and pod-b only the second,
// so both keys are found while the longest chain is one block.
func TestScoreTokensBlockHitTelemetry(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	tp := &mockTokenProcessor{blockKeys: u64ToBlockKeys([]uint64{10, 20})}
	indexer := newTestIndexer(t, tp)
	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		20: {{PodIdentifier: testPodB, DeviceTier: "gpu"}},
	})

	scores, err := indexer.ScoreTokens(ctx, []uint32{1, 2}, testModel, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, map[string]float64{testPodA: 1.0}, scores)

	scoreAttrs := spanAttributes(t, recorder, "score_tokens")
	assert.Equal(t, int64(2), scoreAttrs["llm_d.kv_cache.blocks_found"].AsInt64())
	assert.InDelta(t, 1.0, scoreAttrs["llm_d.kv_cache.block_hit_ratio"].AsFloat64(), 0.0001)

	matchAttrs := spanAttributes(t, recorder, "match_block_keys")
	assert.Equal(t, int64(2), matchAttrs["llm_d.kv_cache.prefix_match.keys_found"].AsInt64())
	assert.Equal(t, int64(1), matchAttrs["llm_d.kv_cache.prefix_match.longest_chain"].AsInt64())
	assert.Equal(t, int64(1), matchAttrs["llm_d.kv_cache.prefix_match.pods_matched"].AsInt64())
}

// emptyEntryIndex reports one requested key as an entry with no pods, the
// miss shape Index.Lookup's contract allows. InMemoryIndex reaches it under
// a race: Lookup reads cache.Len() and cache.Keys() as separate locked
// calls, so evicting the last entry between them leaves the unfiltered
// branch assigning an empty slice. The stub makes that deterministic.
type emptyEntryIndex struct {
	kvblock.Index
	emptyKey kvblock.BlockHash
}

func (e *emptyEntryIndex) Lookup(ctx context.Context, requestKeys []kvblock.BlockHash,
	podIdentifierSet sets.Set[string],
) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
	found, err := e.Index.Lookup(ctx, requestKeys, podIdentifierSet)
	if err != nil {
		return nil, err
	}
	found[e.emptyKey] = nil
	return found, nil
}

// A key returned with no pods is a miss, so it counts towards neither
// blocks_found nor the hit ratio.
func TestScoreTokensBlockHitTelemetryIgnoresEmptyEntries(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	keys := u64ToBlockKeys([]uint64{10, 20})
	backing, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	populateIndex(t, backing, map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	tp := &mockTokenProcessor{blockKeys: keys}
	indexer := kvcache.NewIndexerForTest(tp,
		&emptyEntryIndex{Index: backing, emptyKey: keys[1]},
		kvcache.DefaultKVCacheBackendConfig())

	_, err = indexer.ScoreTokens(ctx, []uint32{1, 2}, testModel, nil, nil)
	require.NoError(t, err)

	scoreAttrs := spanAttributes(t, recorder, "score_tokens")
	assert.Equal(t, int64(1), scoreAttrs["llm_d.kv_cache.blocks_found"].AsInt64())
	assert.InDelta(t, 0.5, scoreAttrs["llm_d.kv_cache.block_hit_ratio"].AsFloat64(), 0.0001)

	matchAttrs := spanAttributes(t, recorder, "match_block_keys")
	assert.Equal(t, int64(1), matchAttrs["llm_d.kv_cache.prefix_match.keys_found"].AsInt64())
}
