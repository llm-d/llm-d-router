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

package kvevents //nolint:testpackage // drives the unexported event path

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// B1 rank-ownership contract.
//
// A shared-frontend engine (one HTTP port, several internal execution ranks)
// publishes one KV-event stream per rank on a per-rank socket while every rank
// reports the same serving endpoint. The rank therefore has to travel with the
// event: reading ownership from the subscriber endpoint alone collapses every
// rank of a shared frontend into one indistinguishable cache owner (#2306).
//
// These fixtures use the event path the pool already exposes. The wire-level
// rank distinction is covered in pkg/kvevents/engineadapter.

const (
	b1Endpoint = "10.0.0.1:8000"
	b1Model    = "test-model"
	b1Block    = 16
)

// b1StoreAdapter reports one BlockStored of the configured engine hashes with
// the configured rank (nil = the publisher sent no rank).
type b1StoreAdapter struct {
	rank *int
	keys []uint64
}

func (a *b1StoreAdapter) ParseMessage(*RawMessage) (string, string, EventBatch, error) {
	return b1Endpoint, b1Model, EventBatch{
		Events: []GenericEvent{&BlockStoredEvent{
			BlockHashes: a.keys,
			Tokens:      makeTokens(b1Block * len(a.keys)),
			BlockSize:   b1Block,
		}},
		DataParallelRank: a.rank,
	}, nil
}

func (a *b1StoreAdapter) ShardingKey(*RawMessage) string { return b1Endpoint }

// b1RemoveAdapter reports one BlockRemoved of the configured engine hashes.
type b1RemoveAdapter struct {
	rank *int
	keys []uint64
}

func (a *b1RemoveAdapter) ParseMessage(*RawMessage) (string, string, EventBatch, error) {
	return b1Endpoint, b1Model, EventBatch{
		Events:           []GenericEvent{&BlockRemovedEvent{BlockHashes: a.keys}},
		DataParallelRank: a.rank,
	}, nil
}

func (a *b1RemoveAdapter) ShardingKey(*RawMessage) string { return b1Endpoint }

func b1Rank(n int) *int { return &n }

// b1RequestKey returns the request key the index uses for one canonical block
// of makeTokens(b1Block).
func b1RequestKey(t *testing.T, tp kvblock.TokenProcessor) kvblock.BlockHash {
	t.Helper()
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, makeTokens(b1Block), b1Model, nil)
	require.NoError(t, err)
	require.Len(t, keys, 1)
	return keys[0]
}

func b1Owners(t *testing.T, ctx context.Context, idx kvblock.Index,
	key kvblock.BlockHash,
) []kvblock.PodEntry {
	t.Helper()
	result, err := idx.Lookup(ctx, []kvblock.BlockHash{key}, nil)
	require.NoError(t, err)
	return result[key]
}

// B1-U02: two ranks behind one serving endpoint must be two owners of the same
// block, not one.
func TestB1_OwnershipIsPerRankWithinOneEndpoint(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, b1Block)
	hashes := []uint64{0xB101}

	for _, rank := range []int{0, 1} {
		pool.adapter = &b1StoreAdapter{rank: b1Rank(rank), keys: hashes}
		pool.processRawMessage(ctx, &RawMessage{
			Topic: "kv@" + b1Endpoint + "@" + b1Model, Payload: []byte{byte(rank)},
		})
	}

	owners := b1Owners(t, ctx, idx, b1RequestKey(t, tp))
	assert.Len(t, owners, 2,
		"rank 0 and rank 1 of one serving endpoint are two cache owners, got %v", owners)
}

// B1-U02: the same rank number behind two endpoints stays separable.
func TestB1_OwnershipIsPerEndpointForTheSameRank(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, b1Block)

	for _, endpoint := range []string{"10.0.0.1:8000", "10.0.0.2:8000"} {
		pool.adapter = &b1StoreAdapter{rank: b1Rank(0), keys: []uint64{0xB102}}
		pool.processRawMessage(ctx, &RawMessage{
			Topic: "kv@" + b1Endpoint + "@" + b1Model, Payload: []byte{1},
			SourceEndpoint: endpoint,
		})
	}

	owners := b1Owners(t, ctx, idx, b1RequestKey(t, tp))
	require.Len(t, owners, 2)
	assert.ElementsMatch(t,
		[]string{"10.0.0.1:8000", "10.0.0.2:8000"},
		[]string{owners[0].PodIdentifier, owners[1].PodIdentifier},
		"rank 0 of two endpoints must stay queryable per endpoint")
}

// B1-U03: a remove announced by one rank must not evict the copy another rank
// still holds.
func TestB1_RemoveOnAnotherRankDoesNotEvict(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, b1Block)
	hashes := []uint64{0xB103}
	key := b1RequestKey(t, tp)

	pool.adapter = &b1StoreAdapter{rank: b1Rank(0), keys: hashes}
	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@x@y", Payload: []byte{1}})
	require.Len(t, b1Owners(t, ctx, idx, key), 1)

	pool.adapter = &b1RemoveAdapter{rank: b1Rank(1), keys: hashes}
	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@x@y", Payload: []byte{1}})
	assert.Len(t, b1Owners(t, ctx, idx, key), 1,
		"rank 1 removing a hash must not evict the copy rank 0 still holds")

	pool.adapter = &b1RemoveAdapter{rank: b1Rank(0), keys: hashes}
	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@x@y", Payload: []byte{1}})
	assert.Empty(t, b1Owners(t, ctx, idx, key),
		"the owning rank's remove must still evict")
}

// B1-U03: the pool must feed the event's rank into the reference-count scope.
// Today the scope is built with the no-rank sentinel for every event, so a rank
// that never stored a hash shares the refcount of the rank that did.
func TestB1_DedupScopeUsesTheEventRank(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, _ := newTestPool(t, b1Block)
	hashes := []uint64{0xB104}

	pool.adapter = &b1StoreAdapter{rank: b1Rank(0), keys: hashes}
	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@x@y", Payload: []byte{1}})

	rank0 := blockScope{b1Endpoint, "gpu", noGroupIdx, 0}
	rank1 := blockScope{b1Endpoint, "gpu", noGroupIdx, 1}

	pool.dedup.mu.Lock()
	defer pool.dedup.mu.Unlock()
	assert.Equal(t, 1, pool.dedup.refs[b1Endpoint][rank0.key(hashes[0])],
		"the store must be attributed to the reporting rank")
	assert.NotContains(t, pool.dedup.refs[b1Endpoint], rank1.key(hashes[0]),
		"a rank that reported nothing must not share the other rank's count")
}

// B1-U03: AllBlocksCleared is scoped to the stream that reported it.
func TestB1_ResetIsScopedToTheReportingStream(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, b1Block)
	other := "10.0.0.2:8000"
	key := b1RequestKey(t, tp)
	hashes := []uint64{0xB105}

	for _, endpoint := range []string{b1Endpoint, other} {
		pool.adapter = &b1StoreAdapter{rank: b1Rank(0), keys: hashes}
		pool.processRawMessage(ctx, &RawMessage{
			Topic: "kv@x@y", Payload: []byte{1}, SourceEndpoint: endpoint})
	}
	require.Len(t, b1Owners(t, ctx, idx, key), 2)

	pool.processRawMessage(ctx, &RawMessage{
		Topic: "kv@x@y", SourceEndpoint: b1Endpoint, reset: true})

	owners := b1Owners(t, ctx, idx, key)
	require.Len(t, owners, 1, "only the reporting endpoint is cleared")
	assert.Equal(t, other, owners[0].PodIdentifier)
}

// B1-U05: deployments that predate rank awareness keep working unchanged.
func TestB1_LegacyPathsWithoutRank(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	t.Run("direct per-rank ports keep endpoint-scoped ownership", func(t *testing.T) {
		pool, idx, tp := newTestPool(t, b1Block)
		for _, endpoint := range []string{"10.0.0.1:8000", "10.0.0.1:8001"} {
			pool.adapter = &b1StoreAdapter{rank: nil, keys: []uint64{0xB107}}
			pool.processRawMessage(ctx, &RawMessage{
				Topic: "kv@" + b1Endpoint + "@" + b1Model, Payload: []byte{1},
				SourceEndpoint: endpoint,
			})
		}

		owners := b1Owners(t, ctx, idx, b1RequestKey(t, tp))
		require.Len(t, owners, 2)
		assert.ElementsMatch(t,
			[]string{"10.0.0.1:8000", "10.0.0.1:8001"},
			[]string{owners[0].PodIdentifier, owners[1].PodIdentifier})
	})

	t.Run("a rank-less store and remove still evict", func(t *testing.T) {
		pool, idx, tp := newTestPool(t, b1Block)
		hashes := []uint64{0xB108}
		key := b1RequestKey(t, tp)

		pool.adapter = &b1StoreAdapter{rank: nil, keys: hashes}
		pool.processRawMessage(ctx, &RawMessage{Topic: "kv@x@y", Payload: []byte{1}})
		require.Len(t, b1Owners(t, ctx, idx, key), 1)

		pool.adapter = &b1RemoveAdapter{rank: nil, keys: hashes}
		pool.processRawMessage(ctx, &RawMessage{Topic: "kv@x@y", Payload: []byte{1}})
		assert.Empty(t, b1Owners(t, ctx, idx, key),
			"an unranked store and remove reference-count against each other")
	})

	t.Run("a rank-less stream is not merged with a ranked one", func(t *testing.T) {
		pool, _, _ := newTestPool(t, b1Block)
		hashes := []uint64{0xB109}

		pool.adapter = &b1StoreAdapter{rank: nil, keys: hashes}
		pool.processRawMessage(ctx, &RawMessage{Topic: "kv@x@y", Payload: []byte{1}})

		unranked := blockScope{b1Endpoint, "gpu", noGroupIdx, noDataParallelRank}
		pool.dedup.mu.Lock()
		defer pool.dedup.mu.Unlock()
		assert.Equal(t, 1, pool.dedup.refs[b1Endpoint][unranked.key(hashes[0])],
			"an unranked publisher keeps the unranked scope")
	})
}

// B1-I01: replaying a rank-carrying fixture must connect the candidate identity
// and the index query. The query the EPP serves from must be able to tell the
// two owners apart.
func TestB1_ReplayFixtureKeepsRankThroughTheIndexQuery(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, b1Block)
	hashes := []uint64{0xB10A}

	for _, rank := range []int{0, 1} {
		// One subscriber per rank: same serving endpoint, per-rank socket.
		pool.adapter = &b1StoreAdapter{rank: b1Rank(rank), keys: hashes}
		pool.processRawMessage(ctx, &RawMessage{
			Topic: "kv@" + b1Endpoint + "@" + b1Model, Payload: []byte{byte(rank)},
			SourceEndpoint: b1Endpoint,
		})
	}

	owners := b1Owners(t, ctx, idx, b1RequestKey(t, tp))
	require.Len(t, owners, 2,
		"a replayed per-rank fixture reaches the index as two owners, got %v", owners)
}
