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

package kvevents

import (
	"context"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

const anchorTestModel = "test-model"

// trunkHashBase is the hash base of the chain every test conversation starts on.
const trunkHashBase = 1000

// radixNodeStore builds the BlockStored event a radix-tree engine emits for a
// node covering tokens[start:end): one hash per page, the parent being the
// trunk page that precedes start. A page hash is hashBase plus the page's
// position, so pages shared by two branches carry the same hash.
func radixNodeStore(tokens []uint32, start, end, pageSize int, hashBase uint64) *BlockStoredEvent {
	pages := (end - start) / pageSize
	hashes := make([]uint64, pages)
	for i := range hashes {
		hashes[i] = hashBase + uint64(start/pageSize+i) //#nosec G115 -- test data
	}
	var parent uint64
	if start > 0 {
		parent = trunkHashBase + uint64(start/pageSize-1) //#nosec G115 -- test data
	}
	return &BlockStoredEvent{
		BlockHashes: hashes,
		Tokens:      tokens[start : start+pages*pageSize],
		ParentHash:  parent,
		BlockSize:   pageSize,
	}
}

// radixNodeRemove builds the BlockRemoved event for the single-token pages of
// the trunk node covering [start, end).
func radixNodeRemove(start, end int) *BlockRemovedEvent {
	hashes := make([]uint64, end-start)
	for i := range hashes {
		hashes[i] = trunkHashBase + uint64(start+i) //#nosec G115 -- test data
	}
	return &BlockRemovedEvent{BlockHashes: hashes}
}

// indexedPrefixBlocks returns how many leading canonical blocks of tokens the
// index holds, and how many canonical blocks tokens span.
func indexedPrefixBlocks(t *testing.T, idx kvblock.Index, tp kvblock.TokenProcessor,
	tokens []uint32, features []*kvblock.BlockExtraFeatures,
) (indexed, total int) {
	t.Helper()
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, anchorTestModel, features)
	require.NoError(t, err)
	result, err := idx.Lookup(context.Background(), keys, nil)
	require.NoError(t, err)
	for _, key := range keys {
		if len(result[key]) == 0 {
			break
		}
		indexed++
	}
	return indexed, len(keys)
}

func processStores(t *testing.T, pool *Pool, events ...GenericEvent) {
	t.Helper()
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool.processEventBatch(ctx, &EventBatch{Events: events}, "pod-a", anchorTestModel)
}

func TestSubCanonicalStores_MultiTurnChain(t *testing.T) {
	tests := []struct {
		name      string
		canonical int
		pageSize  int
		bounds    []int // node boundaries, in tokens
		want      int   // canonical blocks indexed
	}{
		{name: "page 1 under canonical 16", canonical: 16, pageSize: 1, bounds: []int{0, 100, 180, 300}, want: 18},
		{name: "page 1 under canonical 64", canonical: 64, pageSize: 1, bounds: []int{0, 100, 180, 300}, want: 4},
		{name: "page 16 under canonical 64", canonical: 64, pageSize: 16, bounds: []int{0, 96, 176, 288}, want: 4},
		{name: "nodes shorter than a canonical block", canonical: 16, pageSize: 1, bounds: []int{0, 5, 10, 15, 20, 25, 30, 35, 40}, want: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, idx, tp := newTestPool(t, tt.canonical)
			conversation := makeTokens(tt.bounds[len(tt.bounds)-1])
			for i := 0; i+1 < len(tt.bounds); i++ {
				processStores(t, pool, radixNodeStore(conversation, tt.bounds[i], tt.bounds[i+1], tt.pageSize, trunkHashBase))
			}
			indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, nil)
			assert.Equal(t, tt.want, total)
			assert.Equal(t, tt.want, indexed)
		})
	}
}

// A request that shares a prefix ending mid canonical block splits the radix
// node there. The new branch chains from the shared tokens of that block.
func TestSubCanonicalStores_BranchAtMidBlock(t *testing.T) {
	const canonical = 16
	pool, idx, tp := newTestPool(t, canonical)
	first := makeTokens(120)
	second := makeTokens(120)
	for i := 50; i < len(second); i++ {
		second[i] += 10000
	}
	processStores(t, pool,
		radixNodeStore(first, 0, 120, 1, trunkHashBase),
		radixNodeStore(second, 50, 120, 1, 5000),
	)

	for name, tokens := range map[string][]uint32{"first": first, "second": second} {
		indexed, total := indexedPrefixBlocks(t, idx, tp, tokens, nil)
		assert.Equal(t, total, indexed, "%s branch", name)
	}

	secondKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, second, anchorTestModel, nil)
	require.NoError(t, err)
	// The page that ends a canonical block names that block's request key.
	for pos := 63; pos < len(secondKeys)*canonical; pos += canonical {
		requestKey, err := idx.GetRequestKey(context.Background(), ModelEngineKey(anchorTestModel, 5000+uint64(pos))) //#nosec G115 -- test data
		require.NoError(t, err, "page %d", pos)
		assert.Equal(t, secondKeys[pos/canonical], requestKey, "page %d", pos)
	}
}

// A cache salt arrives on the first page of a root store. It taints the first
// canonical block even when a later store completes that block.
func TestSubCanonicalStores_SaltedRootShorterThanBlock(t *testing.T) {
	const canonical = 16
	pool, idx, tp := newTestPool(t, canonical)
	conversation := makeTokens(40)
	root := radixNodeStore(conversation, 0, 10, 1, trunkHashBase)
	root.ExtraKeys = make([][]any, len(root.BlockHashes))
	root.ExtraKeys[0] = []any{"salt-a"}
	processStores(t, pool, root, radixNodeStore(conversation, 10, 40, 1, trunkHashBase))

	salted := []*kvblock.BlockExtraFeatures{{MMHashes: []kvblock.MMHash{{Hash: "salt-a"}}}, nil}
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, salted)
	assert.Equal(t, 2, total)
	assert.Equal(t, 2, indexed)

	indexed, _ = indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Zero(t, indexed, "an unsalted request matched a salted chain")
}

func TestSubCanonicalStores_Removal(t *testing.T) {
	const canonical = 16
	conversation := makeTokens(300)
	stores := func() []GenericEvent {
		return []GenericEvent{
			radixNodeStore(conversation, 0, 100, 1, trunkHashBase),
			radixNodeStore(conversation, 100, 180, 1, trunkHashBase),
			radixNodeStore(conversation, 180, 300, 1, trunkHashBase),
		}
	}

	t.Run("removing a leaf evicts the block it completed", func(t *testing.T) {
		pool, idx, tp := newTestPool(t, canonical)
		processStores(t, pool, stores()...)
		processStores(t, pool, radixNodeRemove(180, 300))
		// The leaf starts at 180; the block holding tokens 176..191 goes with it.
		indexed, _ := indexedPrefixBlocks(t, idx, tp, conversation, nil)
		assert.Equal(t, 11, indexed)

		processStores(t, pool, radixNodeStore(conversation, 180, 300, 1, trunkHashBase))
		indexed, _ = indexedPrefixBlocks(t, idx, tp, conversation, nil)
		assert.Equal(t, 18, indexed, "a re-stored leaf chains from its parent again")
	})

	t.Run("a duplicate store needs a matching number of removes", func(t *testing.T) {
		pool, idx, tp := newTestPool(t, canonical)
		processStores(t, pool, stores()...)
		processStores(t, pool, radixNodeStore(conversation, 180, 300, 1, trunkHashBase))
		processStores(t, pool, radixNodeRemove(180, 300))
		indexed, _ := indexedPrefixBlocks(t, idx, tp, conversation, nil)
		assert.Equal(t, 18, indexed)
		processStores(t, pool, radixNodeRemove(180, 300))
		indexed, _ = indexedPrefixBlocks(t, idx, tp, conversation, nil)
		assert.Equal(t, 11, indexed)
	})

	t.Run("a child of a removed page is skipped", func(t *testing.T) {
		pool, idx, tp := newTestPool(t, canonical)
		skipped := metrics.KVEventStoresSkipped.WithLabelValues(cacheKindLabel(""), "unresolved_sub_canonical_parent")
		before := counterValue(t, skipped)
		processStores(t, pool, stores()[0], radixNodeRemove(0, 100), stores()[1])
		assert.Equal(t, 1.0, counterValue(t, skipped)-before)
		indexed, _ := indexedPrefixBlocks(t, idx, tp, conversation, nil)
		assert.Zero(t, indexed)
	})

	t.Run("a cleared pod forgets its chains", func(t *testing.T) {
		pool, idx, tp := newTestPool(t, canonical)
		skipped := metrics.KVEventStoresSkipped.WithLabelValues(cacheKindLabel(""), "unresolved_sub_canonical_parent")
		before := counterValue(t, skipped)
		processStores(t, pool, stores()[0], &AllBlocksClearedEvent{}, stores()[1])
		assert.Equal(t, 1.0, counterValue(t, skipped)-before)
		indexed, _ := indexedPrefixBlocks(t, idx, tp, conversation, nil)
		assert.Zero(t, indexed)
	})
}

// Extra features on engine blocks that end up in a residual taint the canonical
// block a later store completes, alongside that store's own features.
func TestSubCanonicalStores_FeaturesAcrossStores(t *testing.T) {
	const canonical = 16
	pool, idx, tp := newTestPool(t, canonical)
	conversation := makeTokens(40)
	root := radixNodeStore(conversation, 0, 20, 1, trunkHashBase)
	root.ExtraKeys = make([][]any, len(root.BlockHashes))
	root.ExtraKeys[18] = []any{"image-1"}
	child := radixNodeStore(conversation, 20, 40, 1, trunkHashBase)
	child.ExtraKeys = make([][]any, len(child.BlockHashes))
	child.ExtraKeys[5] = []any{"image-2"}
	processStores(t, pool, root, child)

	features := []*kvblock.BlockExtraFeatures{nil, {MMHashes: []kvblock.MMHash{{Hash: "image-1"}, {Hash: "image-2"}}}}
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, features)
	assert.Equal(t, 2, total)
	assert.Equal(t, 2, indexed)
}

// An engine may remove a block before the blocks that follow it on the chain.
// The canonical block a later store completed over that block goes with it.
func TestSubCanonicalStores_AncestorRemovedFirst(t *testing.T) {
	const canonical = 16
	pool, idx, tp := newTestPool(t, canonical)
	first := makeTokens(180)
	second := makeTokens(180)
	for i := 100; i < len(second); i++ {
		second[i] += 10000
	}
	processStores(t, pool,
		radixNodeStore(first, 0, 100, 1, trunkHashBase),
		radixNodeStore(first, 100, 180, 1, trunkHashBase),
		radixNodeStore(second, 100, 180, 1, 5000),
	)
	for name, tokens := range map[string][]uint32{"first": first, "second": second} {
		indexed, total := indexedPrefixBlocks(t, idx, tp, tokens, nil)
		require.Equal(t, total, indexed, "%s branch", name)
	}

	// Token 98 sits in the block holding tokens 96..111 of both branches.
	processStores(t, pool, radixNodeRemove(98, 99))
	for name, tokens := range map[string][]uint32{"first": first, "second": second} {
		indexed, _ := indexedPrefixBlocks(t, idx, tp, tokens, nil)
		assert.Equal(t, 6, indexed, "%s branch", name)
	}
}

// Removing a store drops what it registered on the blocks it chained from.
func TestSubCanonicalStores_RemovedChildLeavesNoDependents(t *testing.T) {
	pool, _, _ := newTestPool(t, 16)
	conversation := makeTokens(180)
	processStores(t, pool,
		radixNodeStore(conversation, 0, 100, 1, trunkHashBase),
		radixNodeStore(conversation, 100, 180, 1, trunkHashBase),
	)
	// The child completes the block holding tokens 96..111 over tokens 96..99.
	require.Len(t, pool.anchors.pods["pod-a"].dependents, 4)

	processStores(t, pool, radixNodeRemove(100, 180))
	assert.Empty(t, pool.anchors.pods["pod-a"].dependents)
}

// A block removed from one device tier and still held on another keeps its
// anchor, so a later store can chain from it.
func TestSubCanonicalStores_BlockHeldOnAnotherTier(t *testing.T) {
	const canonical = 16
	pool, idx, tp := newTestPool(t, canonical)
	conversation := makeTokens(300)
	offloaded := radixNodeStore(conversation, 100, 180, 1, trunkHashBase)
	offloaded.DeviceTier = "CPU_PINNED"
	processStores(t, pool,
		radixNodeStore(conversation, 0, 100, 1, trunkHashBase),
		radixNodeStore(conversation, 100, 180, 1, trunkHashBase),
		offloaded,
		radixNodeRemove(100, 180),
		radixNodeStore(conversation, 180, 300, 1, trunkHashBase),
	)
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Equal(t, total, indexed)
}

// A device-tier update announces blocks without tokens. It holds their anchors
// the same way a store with tokens does.
func TestSubCanonicalStores_BlockHeldByDeviceTierUpdate(t *testing.T) {
	pool, idx, tp := newTestPool(t, 16)
	conversation := makeTokens(180)
	root := radixNodeStore(conversation, 0, 100, 1, trunkHashBase)
	offloaded := &BlockStoredEvent{BlockHashes: root.BlockHashes, BlockSize: 1, DeviceTier: "CPU"}
	offloadRemoved := radixNodeRemove(0, 100)
	offloadRemoved.DeviceTier = "CPU"
	processStores(t, pool,
		root,
		offloaded,
		offloadRemoved,
		radixNodeStore(conversation, 100, 180, 1, trunkHashBase),
	)
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Equal(t, total, indexed)
}

// A remove on a device tier that never announced the block releases nothing.
func TestSubCanonicalStores_RemoveWithoutStoreKeepsAnchor(t *testing.T) {
	pool, idx, tp := newTestPool(t, 16)
	conversation := makeTokens(180)
	unmatched := radixNodeRemove(0, 100)
	unmatched.DeviceTier = "CPU"
	processStores(t, pool,
		radixNodeStore(conversation, 0, 100, 1, trunkHashBase),
		unmatched,
		radixNodeStore(conversation, 100, 180, 1, trunkHashBase),
	)
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Equal(t, total, indexed)
}

// A block held on two device tiers is removed from each in turn. The canonical
// block a later store completed over it leaves the tier that held it.
func TestSubCanonicalStores_AncestorRemovedPerTier(t *testing.T) {
	pool, idx, tp := newTestPool(t, 16)
	conversation := makeTokens(180)
	offloaded := radixNodeStore(conversation, 0, 100, 1, trunkHashBase)
	offloaded.DeviceTier = "CPU"
	processStores(t, pool,
		radixNodeStore(conversation, 0, 100, 1, trunkHashBase),
		offloaded,
		radixNodeStore(conversation, 100, 180, 1, trunkHashBase),
	)
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, nil)
	require.Equal(t, total, indexed)

	// Token 98 sits in the block holding tokens 96..111, indexed on GPU only.
	offloadRemoved := radixNodeRemove(98, 99)
	offloadRemoved.DeviceTier = "CPU"
	processStores(t, pool, offloadRemoved)
	indexed, _ = indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Equal(t, total, indexed)

	processStores(t, pool, radixNodeRemove(98, 99))
	indexed, _ = indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Equal(t, 6, indexed)
}

// A block removed and stored again belongs to a later event than its
// neighbors. Removing it still evicts the canonical block completed over it.
func TestSubCanonicalStores_AncestorRemovedAfterRestore(t *testing.T) {
	pool, idx, tp := newTestPool(t, 16)
	conversation := makeTokens(36)
	second := &BlockRemovedEvent{BlockHashes: []uint64{trunkHashBase + 1}}
	processStores(t, pool,
		radixNodeStore(conversation, 0, 12, 4, trunkHashBase),
		second,
		radixNodeStore(conversation, 4, 12, 4, trunkHashBase),
		radixNodeStore(conversation, 12, 36, 4, trunkHashBase),
	)
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, nil)
	require.Equal(t, 2, total)
	require.Equal(t, total, indexed)

	processStores(t, pool, second)
	indexed, _ = indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Zero(t, indexed)
}

// A store chains only from a block of its own block size.
func TestSubCanonicalStores_ParentOfAnotherBlockSize(t *testing.T) {
	pool, idx, tp := newTestPool(t, 16)
	conversation := makeTokens(40)
	skipped := metrics.KVEventStoresSkipped.WithLabelValues(cacheKindLabel(""), "unresolved_sub_canonical_parent")
	before := counterValue(t, skipped)
	child := radixNodeStore(conversation, 20, 40, 4, 5000)
	child.ParentHash = trunkHashBase + 19
	processStores(t, pool, radixNodeStore(conversation, 0, 20, 1, trunkHashBase), child)
	assert.Equal(t, 1.0, counterValue(t, skipped)-before)
	indexed, _ := indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Equal(t, 1, indexed)
}

// A subscriber reset clears the pod's anchors with its index entries.
func TestSubCanonicalStores_SubscriberReset(t *testing.T) {
	pool, _, _ := newTestPool(t, 16)
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	processStores(t, pool, radixNodeStore(makeTokens(100), 0, 100, 1, trunkHashBase))
	_, found := pool.anchors.lookup("pod-a", ModelEngineKey(anchorTestModel, trunkHashBase+99))
	require.True(t, found)

	pool.processRawMessage(ctx, &RawMessage{SourceEndpoint: "pod-a", reset: true})
	_, found = pool.anchors.lookup("pod-a", ModelEngineKey(anchorTestModel, trunkHashBase+99))
	assert.False(t, found)
}

// A pod that recomputes from an engine block inside a canonical block keys the
// new blocks from that canonical block's start, so a second pod continuing
// from the canonical boundary keeps its keys.
func TestSubCanonicalStores_RecomputeFromInteriorParent(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 64)
	tokens := makeTokens(128)
	store := func(pod string, ev GenericEvent) {
		pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{ev}}, pod, anchorTestModel)
	}
	store("pod-a", &BlockStoredEvent{BlockHashes: makeEngineKeys(4, 100), Tokens: tokens[:64], BlockSize: 16})
	store("pod-b", &BlockStoredEvent{BlockHashes: makeEngineKeys(4, 100), Tokens: tokens[:64], BlockSize: 16})
	store("pod-a", &BlockRemovedEvent{BlockHashes: []uint64{103}})
	store("pod-a", &BlockStoredEvent{BlockHashes: makeEngineKeys(5, 103), Tokens: tokens[48:], ParentHash: 102, BlockSize: 16})
	store("pod-b", &BlockStoredEvent{BlockHashes: makeEngineKeys(4, 104), Tokens: tokens[64:], ParentHash: 103, BlockSize: 16})

	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, anchorTestModel, nil)
	require.NoError(t, err)
	require.Len(t, keys, 2)
	for i, key := range keys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{key}, nil)
		require.NoError(t, err)
		pods := make([]string, 0, len(result[key]))
		for _, entry := range result[key] {
			pods = append(pods, entry.PodIdentifier)
		}
		assert.ElementsMatch(t, []string{"pod-a", "pod-b"}, pods, "canonical block %d", i)
	}
}

// An index consumer chains sub-canonical stores like a subscribing pool, and
// a stream reset drops its anchors.
func TestSubCanonicalStores_IndexConsumer(t *testing.T) {
	const canonical = 16
	_, idx, tp := newTestPool(t, canonical)
	consumer := NewIndexConsumer(idx, tp)
	source := EventSource{Endpoint: "pod-a", ModelName: anchorTestModel}
	conversation := makeTokens(300)
	for _, bounds := range [][2]int{{0, 100}, {100, 180}, {180, 300}} {
		require.NoError(t, consumer.ProcessEvents(t.Context(), source, EventBatch{Events: []GenericEvent{
			radixNodeStore(conversation, bounds[0], bounds[1], 1, trunkHashBase),
		}}))
	}
	indexed, total := indexedPrefixBlocks(t, idx, tp, conversation, nil)
	assert.Equal(t, total, indexed)

	require.NoError(t, consumer.Reset(t.Context(), source))
	_, found := consumer.anchors.lookup("pod-a", ModelEngineKey(anchorTestModel, trunkHashBase+299))
	assert.False(t, found)
}

// chainNode is one stored event of a randomized hash chain: it continues from
// engine block parentBlock of parent, or starts a chain when parent is nil.
type chainNode struct {
	parent      *chainNode
	parentBlock int
	tokens      []uint32
	hashes      []uint64
}

// path returns the tokens and engine hashes from the chain's start to the end
// of n.
func (n *chainNode) path(pageSize int) ([]uint32, []uint64) {
	if n.parent == nil {
		return n.tokens, n.hashes
	}
	tokens, hashes := n.parent.path(pageSize)
	blocks := len(hashes) - len(n.parent.hashes) + n.parentBlock + 1
	return slices.Concat(tokens[:blocks*pageSize], n.tokens), slices.Concat(hashes[:blocks], n.hashes)
}

// Random stores and removes on two device tiers, with duplicate stores,
// partial removes, removes that match no store and stores repeated after a
// remove. A tier holds a canonical block in the index only while the engine
// holds every engine block of it on that tier, and always does when none of
// them was removed from the tier.
func TestSubCanonicalStores_RandomizedTiers(t *testing.T) {
	const canonical = 16
	tiers := []string{"gpu", "cpu"}
	for _, pageSize := range []int{1, 2, 4, 8} {
		t.Run(fmt.Sprintf("page-%d", pageSize), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(uint64(pageSize), 7)) //#nosec G404,G115 -- test data
			for sequence := range 150 {
				pool, idx, tp := newTestPool(t, canonical)
				ctx := context.Background()
				var nodes []*chainNode
				var nextToken uint32
				var nextHash uint64
				var opsLog []string
				held := map[string]map[uint64]int{"gpu": {}, "cpu": {}}
				removed := map[string]map[uint64]bool{"gpu": {}, "cpu": {}}
				heldOn := func(tier string, hashes []uint64) bool {
					return !slices.ContainsFunc(hashes, func(h uint64) bool { return held[tier][h] == 0 })
				}

				for step := range 60 {
					tier := tiers[rng.IntN(len(tiers))]
					var node *chainNode
					if len(nodes) < 8 && (len(nodes) == 0 || rng.IntN(4) == 0) {
						node = &chainNode{}
						if len(nodes) > 0 && rng.IntN(5) > 0 {
							node.parent = nodes[rng.IntN(len(nodes))]
							node.parentBlock = rng.IntN(len(node.parent.hashes))
						}
						for range 1 + rng.IntN(3*canonical/pageSize) {
							nextHash++
							node.hashes = append(node.hashes, nextHash)
							for range pageSize {
								nextToken++
								node.tokens = append(node.tokens, nextToken)
							}
						}
						nodes = append(nodes, node)
					} else {
						node = nodes[rng.IntN(len(nodes))]
					}

					if rng.IntN(2) == 0 {
						// An engine stores a node on a tier that holds the blocks before it.
						_, pathHashes := node.path(pageSize)
						if !heldOn(tier, pathHashes[:len(pathHashes)-len(node.hashes)]) {
							continue
						}
						ev := &BlockStoredEvent{
							BlockHashes: node.hashes, Tokens: node.tokens, BlockSize: pageSize, DeviceTier: tier,
						}
						if node.parent != nil {
							ev.ParentHash = node.parent.hashes[node.parentBlock]
						}
						pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{ev}}, "pod-a", anchorTestModel)
						opsLog = append(opsLog, fmt.Sprintf("store %s node=%d parent=%d hashes=%v", tier, slices.Index(nodes, node), ev.ParentHash, node.hashes))
						for _, h := range node.hashes {
							held[tier][h]++
						}
					} else {
						from := rng.IntN(len(node.hashes))
						hashes := node.hashes[from : from+1+rng.IntN(len(node.hashes)-from)]
						pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
							&BlockRemovedEvent{BlockHashes: hashes, DeviceTier: tier},
						}}, "pod-a", anchorTestModel)
						opsLog = append(opsLog, fmt.Sprintf("remove %s %v", tier, hashes))
						for _, h := range hashes {
							if held[tier][h] > 0 {
								held[tier][h]--
								removed[tier][h] = removed[tier][h] || held[tier][h] == 0
							}
						}
					}

					for nodeIndex, n := range nodes {
						tokens, hashes := n.path(pageSize)
						keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, anchorTestModel, nil)
						require.NoError(t, err)
						for block, key := range keys {
							entries, err := idx.Lookup(ctx, []kvblock.BlockHash{key}, nil)
							require.NoError(t, err)
							engineBlocks := hashes[block*canonical/pageSize : (block+1)*canonical/pageSize]
							for _, tier := range tiers {
								indexed := slices.ContainsFunc(entries[key], func(e kvblock.PodEntry) bool {
									return e.DeviceTier == tier
								})
								wholeBlockHeld := heldOn(tier, engineBlocks)
								neverRemoved := !slices.ContainsFunc(engineBlocks, func(h uint64) bool { return removed[tier][h] })
								if indexed && !wholeBlockHeld {
									t.Fatalf("sequence %d step %d: node %d block %d is indexed on %s without its engine blocks",
										sequence, step, nodeIndex, block, tier)
								}
								if !indexed && wholeBlockHeld && neverRemoved {
									t.Fatalf("sequence %d step %d: node %d block %d is held on %s and not indexed (blocks %v)\n%s",
										sequence, step, nodeIndex, block, tier, engineBlocks, strings.Join(opsLog, "\n"))
								}
							}
						}
					}
				}
			}
		})
	}
}
