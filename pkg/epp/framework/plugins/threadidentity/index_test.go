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

package threadidentity

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const testBlockSize = 64

// tokens builds a deterministic token stream: n tokens tagged by seed, so two
// streams share a prefix exactly as far as they share a seed.
func tokens(seed uint32, n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = seed*1_000_003 + uint32(i)
	}
	return out
}

// grown extends a stream with n more tokens (an appended turn).
func grown(base []uint32, n int, seed uint32) []uint32 {
	return append(append([]uint32{}, base...), tokens(seed, n)...)
}

// chainOf hashes token streams into an identity chain, exactly as the arrival
// hook does (one prompt per list position).
func chainOf(t *testing.T, prompts ...[]uint32) chain {
	t.Helper()
	promptTokens := make([]fwkrequest.PromptTokens, len(prompts))
	for i, p := range prompts {
		promptTokens[i] = fwkrequest.PromptTokens{TokenIDs: p}
	}
	req := &fwksched.InferenceRequest{Body: &fwkrequest.InferenceRequestBody{
		TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: promptTokens},
	}}
	return identityChain(promptBlocks(context.Background(), req, testBlockSize), testBlockSize)
}

// arr builds an arrival from token streams (multi-prompt when several are given).
func arr(t *testing.T, scope string, roles []string, marker string, prompts ...[]uint32) Arrival {
	t.Helper()
	return Arrival{Chain: chainOf(t, prompts...), Scope: scope, Roles: roles, Marker: marker}
}

var clockZero = time.Unix(1_700_000_000, 0)

func newTestIndex() *threadIndex {
	return newThreadIndex(testBlockSize, time.Hour, 1000)
}

func turn(ix *threadIndex, at time.Time, a Arrival) ResolveResult {
	return ix.Resolve(at, a, true, true, true)
}

// complete resolves an arrival and finishes it: dispatched, then settled, so the
// thread is free for the next turn.
func complete(t *testing.T, ix *threadIndex, at time.Time, a Arrival) ResolveResult {
	t.Helper()
	res := turn(ix, at, a)
	ix.Dispatch(res.ID, a.Marker, at)
	ix.Settle(res.ID, a.Marker, at)
	return res
}

// requireInvariants asserts every live record is indexed exactly once.
func requireInvariants(t *testing.T, ix *threadIndex) {
	t.Helper()
	require.Empty(t, ix.checkIndexInvariants())
}

// TestChain_FullBlocksStopAtThePartialBlock: a chain's full-block count excludes
// the trailing partial block, which is what the tolerance measures against.
func TestChain_FullBlocksStopAtThePartialBlock(t *testing.T) {
	// 2*B + 10 tokens: two full blocks, then a partial.
	c := chainOf(t, tokens(1, 2*testBlockSize+10))
	assert.Equal(t, 3, len(c.blocks))
	assert.Equal(t, 2, c.fullBlocks)
	assert.Equal(t, 2*testBlockSize+10, c.tokens)
	assert.Equal(t, c.blocks[2], c.lastHash())

	// Block-aligned: every block is full, no partial.
	c = chainOf(t, tokens(1, 2*testBlockSize))
	assert.Equal(t, 2, len(c.blocks))
	assert.Equal(t, 2, c.fullBlocks)
}

// TestChain_MultiPromptConcatenates: all prompts count, their chains in order.
func TestChain_MultiPromptConcatenates(t *testing.T) {
	first, second := tokens(1, testBlockSize), tokens(2, testBlockSize/2)
	both := chainOf(t, first, second)
	justFirst := chainOf(t, first)
	require.Equal(t, 1, len(justFirst.blocks))

	assert.Equal(t, 2, len(both.blocks), "the second prompt contributes its block")
	assert.Equal(t, both.blocks[0], justFirst.blocks[0], "the first prompt's chain is unchanged")
	// The trailing prompt ends mid-block, so the last block is partial.
	assert.Equal(t, 1, both.fullBlocks)
	assert.Equal(t, testBlockSize+testBlockSize/2, both.tokens)
}

// TestChain_MultiPromptRechainsAcrossTheBoundary: F1 — a later prompt's blocks are
// re-chained onto the chain before them, so two requests that share a later prompt
// but differ before it do NOT share that prompt's block keys. Without the re-chain
// (prefixhash seeds each prompt from the model and salt independently) a shared
// prompt is a shared key set, and the resolve below reads it as a continuation.
func TestChain_MultiPromptRechainsAcrossTheBoundary(t *testing.T) {
	a, b, c := tokens(1, 2*testBlockSize), tokens(2, 2*testBlockSize), tokens(3, 2*testBlockSize)

	ac, bc := chainOf(t, a, c), chainOf(t, b, c)
	require.Len(t, ac.blocks, 4)
	require.Len(t, bc.blocks, 4)

	// The first prompt keeps the prefix scorer's own hashes, exactly as if it were
	// the whole request (single-prompt requests carry no re-chain at all).
	assert.Equal(t, chainOf(t, a).blocks, ac.blocks[:2])

	for k := range ac.blocks {
		assert.NotEqual(t, ac.blocks[k], bc.blocks[k], "block %d must not be shared", k)
	}

	// The re-chain is deterministic and content-addressed: the same two prompts
	// yield the same chain, so a repeat still continues.
	assert.Equal(t, ac.blocks, chainOf(t, a, c).blocks)

	// An extension of the SECOND prompt re-chains consistently: the shared prefix of
	// C keeps its ids, so a grown multi-prompt request still continues.
	extended := chainOf(t, a, append(append([]uint32{}, c...), tokens(4, 10)...))
	require.Len(t, extended.blocks, 5)
	assert.Equal(t, ac.blocks, extended.blocks[:4])
}

// TestResolve_MultiPromptSharingALaterPromptIsNotAContinuation is F1's gate case:
// tips [A,C] and [B,C] with |A|=|B| must NOT continue one another.
func TestResolve_MultiPromptSharingALaterPromptIsNotAContinuation(t *testing.T) {
	ix := newTestIndex()
	a, b, c := tokens(1, 2*testBlockSize), tokens(2, 2*testBlockSize), tokens(3, 2*testBlockSize)

	first := complete(t, ix, clockZero, arr(t, "", nil, "r1", a, c))
	require.Equal(t, OutcomeNew, first.Outcome)

	second := complete(t, ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", b, c))
	assert.Equal(t, OutcomeNew, second.Outcome,
		"a different first prompt is a different conversation, however much of C it shares")
	assert.NotEqual(t, first.ID, second.ID)
	assert.Zero(t, second.MatchedBlocks, "nothing of the other conversation's chain may be verified")
	requireInvariants(t, ix)

	// The identical pair still continues, in both directions of arrival order.
	third := complete(t, ix, clockZero.Add(2*time.Second), arr(t, "", nil, "r3", b, c))
	assert.Equal(t, OutcomeMatched, third.Outcome)
	assert.Equal(t, second.ID, third.ID)
	requireInvariants(t, ix)
}

// TestResolve_Continuation: an appended turn continues the thread.
func TestResolve_Continuation(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 3*testBlockSize)

	first := complete(t, ix, clockZero, arr(t, "", nil, "r1", tip))
	require.Equal(t, OutcomeNew, first.Outcome)

	next := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeMatched, next.Outcome)
	assert.Equal(t, first.ID, next.ID, "a continuation keeps the thread id")
	assert.Equal(t, 2, next.MatchedBlocks, "the verified prefix stops one full block short of the tip")
	requireInvariants(t, ix)
}

// TestResolve_TailToleranceIsOneFullBlock: a request that differs from the tip
// only in the tip's last full block (plus its partial) still continues — the
// documented B..2B-1 tokens of slack.
func TestResolve_TailToleranceIsOneFullBlock(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 4*testBlockSize)
	first := complete(t, ix, clockZero, arr(t, "", nil, "r1", tip))

	// Diverge from the first token of the tip's LAST full block: the maximum
	// tolerated divergence, a full block plus the partial block (B..2B-1 tokens).
	divergeAt := 3 * testBlockSize
	req := grown(tip[:divergeAt], 3*testBlockSize, 9)
	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", req))
	assert.Equal(t, OutcomeMatched, res.Outcome)
	assert.Equal(t, first.ID, res.ID)
	assert.Equal(t, 3, res.MatchedBlocks, "the verified prefix stops before the tip's last full block")
	requireInvariants(t, ix)
}

// TestResolve_TwoBlocksApartIsANewThread: divergence in the tip's second-to-last
// full block is a different thread.
func TestResolve_TwoBlocksApartIsANewThread(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 4*testBlockSize)
	first := complete(t, ix, clockZero, arr(t, "", nil, "r1", tip))

	divergeAt := 2 * testBlockSize // the tip's second-to-last full block (3 full blocks kept)
	req := grown(tip[:divergeAt], 3*testBlockSize, 9)
	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", req))
	assert.Equal(t, OutcomeNew, res.Outcome)
	assert.NotEqual(t, first.ID, res.ID)
	requireInvariants(t, ix)
}

// TestResolve_OneFullBlockTipGetsNoTolerance: with a single full block the empty
// prefix would match every request, so that block must match.
func TestResolve_OneFullBlockTipGetsNoTolerance(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, testBlockSize+5) // one full block, then a partial
	complete(t, ix, clockZero, arr(t, "", nil, "r1", tip))

	// Diverging inside the tip's only full block is a new thread.
	req := grown(tip[:testBlockSize/2], 2*testBlockSize, 9)
	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", req))
	assert.Equal(t, OutcomeNew, res.Outcome, "a one-full-block tip verifies its block")

	// Extending it (the block intact) continues.
	ix2 := newTestIndex()
	base := complete(t, ix2, clockZero, arr(t, "", nil, "r3", tip))
	ext := turn(ix2, clockZero.Add(time.Second), arr(t, "", nil, "r4", grown(tip, 200, 3)))
	assert.Equal(t, OutcomeMatched, ext.Outcome)
	assert.Equal(t, base.ID, ext.ID)
	requireInvariants(t, ix)
	requireInvariants(t, ix2)
}

// TestResolve_SubBlockTipIsNotContinuable: a request shorter than one block has
// no full block, so it cannot be continued and is not indexed.
func TestResolve_SubBlockTipIsNotContinuable(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, testBlockSize-10)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))
	require.Equal(t, OutcomeNew, first.Outcome)

	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", grown(tip, 5, 2)))
	assert.Equal(t, OutcomeNew, res.Outcome, "a sub-block tip re-keys once")
	assert.NotEqual(t, first.ID, res.ID)
	require.Empty(t, ix.byPrefix, "a sub-block tip holds no index key")
	requireInvariants(t, ix)
}

// TestResolve_DescendingProbeFindsATipWithNoShorterTip: the tip index is not
// prefix-closed — a tip at k=7 with nothing at k=5..6 must still be found.
func TestResolve_DescendingProbeFindsATipWithNoShorterTip(t *testing.T) {
	ix := newTestIndex()
	// A tip with 8 full blocks: its verified prefix is 7 blocks.
	tip := tokens(1, 8*testBlockSize)
	first := complete(t, ix, clockZero, arr(t, "", nil, "r1", tip))
	require.Equal(t, OutcomeNew, first.Outcome)
	require.Len(t, ix.byPrefix, 1)
	require.Equal(t, 7, ix.threads[first.ID].prefixLen)

	// A request with 12 full blocks: probes k=12,11,10,9 (all misses) then k=7.
	req := grown(tip, 4*testBlockSize, 2)
	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", req))
	assert.Equal(t, OutcomeMatched, res.Outcome)
	assert.Equal(t, first.ID, res.ID)
	assert.Equal(t, 7, res.MatchedBlocks)
	requireInvariants(t, ix)
}

// TestResolve_ForkIfBusy: extending a busy tip forks a sibling thread.
func TestResolve_ForkIfBusy(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))
	require.Equal(t, OutcomeNew, first.Outcome)
	require.True(t, ix.Dispatch(first.ID, "r1", clockZero))

	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeForked, res.Outcome)
	assert.NotEqual(t, first.ID, res.ID)
	assert.Equal(t, 1, ix.threads[first.ID].forks)
	requireInvariants(t, ix)
}

// TestResolve_SamePromptWhileBusyIsASibling: two identical prompts in flight are
// two siblings, not one continuation.
func TestResolve_SamePromptWhileBusyIsASibling(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))
	require.True(t, ix.Dispatch(first.ID, "r1", clockZero))

	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", tip))
	assert.Equal(t, OutcomeForked, res.Outcome)
	assert.NotEqual(t, first.ID, res.ID)
}

// TestResolve_RetryOfStillQueuedTurn: S2 — an identical duplicate of a tip with
// nothing IN FLIGHT is a client retry and continues.
func TestResolve_RetryOfStillQueuedTurn(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))
	require.Equal(t, OutcomeNew, first.Outcome)

	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", tip))
	assert.Equal(t, OutcomeMatched, res.Outcome)
	assert.True(t, res.Retry)
	assert.Equal(t, first.ID, res.ID)

	// With the rule off, the same duplicate forks.
	ix2 := newTestIndex()
	queued := ix2.Resolve(clockZero, arr(t, "", nil, "r1", tip), true, true, false)
	dup := ix2.Resolve(clockZero.Add(time.Second), arr(t, "", nil, "r2", tip), true, true, false)
	assert.Equal(t, OutcomeForked, dup.Outcome)
	assert.NotEqual(t, queued.ID, dup.ID)
}

// TestResolve_S2ExactKeyIncludesThePartialBlock: S2's key is (token count, last
// hash INCLUDING the partial block), so a same-length request whose tail differs
// is not a retry — it is a sibling, and it forks.
func TestResolve_S2ExactKeyIncludesThePartialBlock(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize+10)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))
	require.Equal(t, OutcomeNew, first.Outcome)

	// Same token length, different tail (inside the tip's partial block): the
	// tolerance still matches the thread, but the bodies differ, so S2 must not
	// call it a retry.
	other := append(append([]uint32{}, tip[:len(tip)-5]...), tokens(9, 5)...)
	res := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", other))
	assert.Equal(t, OutcomeForked, res.Outcome, "a differing tail is a sibling, not a retry")
	assert.False(t, res.Retry)
	assert.NotEqual(t, first.ID, res.ID)
	requireInvariants(t, ix)
}

// TestResolve_StickyBusyGuard: H1 — the request carries the tip's reply, so the
// busy tip is proven complete and the request continues.
func TestResolve_StickyBusyGuard(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	roles := []string{"user"}
	first := turn(ix, clockZero, arr(t, "", roles, "r1", tip))
	require.True(t, ix.Dispatch(first.ID, "r1", clockZero))

	// The request's second message is the assistant reply to the tip.
	res := turn(ix, clockZero.Add(time.Second),
		arr(t, "", []string{"user", "assistant", "user"}, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeMatched, res.Outcome)
	assert.True(t, res.StickyBusy)
	assert.Equal(t, first.ID, res.ID)
}

// TestResolve_StickyBusyGuard_NotProven: a request that extends a busy tip with a
// user message first has no proof of completion and forks.
func TestResolve_StickyBusyGuard_NotProven(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", []string{"user"}, "r1", tip))
	require.True(t, ix.Dispatch(first.ID, "r1", clockZero))

	res := turn(ix, clockZero.Add(time.Second),
		arr(t, "", []string{"user", "user"}, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeForked, res.Outcome)
	assert.False(t, res.StickyBusy)
}

// TestResolve_ForkOnBusyDisabled: the measurement arm where every match continues.
func TestResolve_ForkOnBusyDisabled(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := ix.Resolve(clockZero, arr(t, "", []string{"user"}, "r1", tip), false, true, true)
	require.True(t, ix.Dispatch(first.ID, "r1", clockZero))

	res := ix.Resolve(clockZero.Add(time.Second),
		arr(t, "", []string{"user", "user"}, "r2", grown(tip, 100, 2)), false, true, true)
	assert.Equal(t, OutcomeMatched, res.Outcome)
	assert.Equal(t, first.ID, res.ID)
}

// TestResolve_SessionScope: tips of another session never match; an unscoped tip
// matches any session.
func TestResolve_SessionScope(t *testing.T) {
	tip := tokens(1, 2*testBlockSize)

	// Another session's tip is invisible.
	ix := newTestIndex()
	complete(t, ix, clockZero, arr(t, "session-a", nil, "r1", tip))
	other := turn(ix, clockZero.Add(time.Second), arr(t, "session-b", nil, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeNew, other.Outcome)

	// A request without a session header matches a scoped tip (no header → no
	// scoping, the design's accepted floor).
	ix2 := newTestIndex()
	free := complete(t, ix2, clockZero, arr(t, "session-a", nil, "r1", tip))
	unscoped := turn(ix2, clockZero.Add(time.Second), arr(t, "", nil, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeMatched, unscoped.Outcome)
	assert.Equal(t, free.ID, unscoped.ID)

	// A fork inherits the parent's scope when the request carries none.
	ix3 := newTestIndex()
	parent := turn(ix3, clockZero, arr(t, "session-a", nil, "r1", tip))
	require.True(t, ix3.Dispatch(parent.ID, "r1", clockZero))
	child := turn(ix3, clockZero.Add(time.Second), arr(t, "", nil, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeForked, child.Outcome)
	assert.Equal(t, "session-a", ix3.threads[child.ID].scope)
}

// TestResolve_NoChainIsANewThread: a request with no identity (a failed render)
// starts its own thread and never continues anything.
func TestResolve_NoChainIsANewThread(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))

	res := turn(ix, clockZero.Add(time.Second), Arrival{Scope: "", Marker: "r2"})
	assert.Equal(t, OutcomeNew, res.Outcome)
	assert.NotEqual(t, first.ID, res.ID)

	// And it is not continuable in turn.
	again := turn(ix, clockZero.Add(2*time.Second), Arrival{Scope: "", Marker: "r3"})
	assert.Equal(t, OutcomeNew, again.Outcome)
	assert.NotEqual(t, res.ID, again.ID)
	requireInvariants(t, ix)
}

// TestMint_NeverReusesALiveID: a look-alike re-mint bumps the ordinal instead of
// overwriting a live thread.
func TestMint_NeverReusesALiveID(t *testing.T) {
	ix := newTestIndex()
	a := turn(ix, clockZero, arr(t, "", nil, "r1", tokens(1, 2*testBlockSize)))
	require.True(t, ix.Dispatch(a.ID, "r1", clockZero))
	b := turn(ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", tokens(1, 2*testBlockSize)))
	require.NotEqual(t, a.ID, b.ID)
	require.Len(t, ix.threads, 2)
	requireInvariants(t, ix)
}

// TestMarkers_ArePerRequestID: dispatch and settle move exactly one marker.
func TestMarkers_ArePerRequestID(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))

	rec := ix.threads[first.ID]
	require.Equal(t, map[string]bool{"r1": true}, keySet(rec.pending))

	require.True(t, ix.Dispatch(first.ID, "r1", clockZero))
	assert.Empty(t, rec.pending)
	assert.Equal(t, map[string]bool{"r1": true}, keySet(rec.inflight))

	require.True(t, ix.Settle(first.ID, "r1", clockZero))
	assert.False(t, rec.outstanding())
}

// TestMarkers_ExpireOnTheRequestTimeout: a marker that never settles expires, so
// a rejected request cannot keep its thread busy forever.
func TestMarkers_ExpireOnTheRequestTimeout(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))
	require.True(t, ix.threads[first.ID].outstanding())

	later := clockZero.Add(2 * time.Hour)
	res := turn(ix, later, arr(t, "", nil, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeMatched, res.Outcome, "the expired marker freed the thread")
}

// TestNoTimeGC_DormantThreadKeepsItsID: there is no idle GC (stage 0's p99.9
// inter-turn gap is 3,036s).
func TestNoTimeGC_DormantThreadKeepsItsID(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, arr(t, "", nil, "r1", tip))
	require.True(t, ix.Settle(first.ID, "r1", clockZero))

	later := clockZero.Add(6 * time.Hour)
	res := turn(ix, later, arr(t, "", nil, "r2", grown(tip, 100, 2)))
	assert.Equal(t, OutcomeMatched, res.Outcome)
	assert.Equal(t, first.ID, res.ID)
}

// TestCap_EvictsIdleThreadsButNeverOutstandingOnes: the count cap is the only
// eviction path.
func TestCap_EvictsIdleThreadsButNeverOutstandingOnes(t *testing.T) {
	ix := newThreadIndex(testBlockSize, time.Hour, 2)
	busy := turn(ix, clockZero, arr(t, "", nil, "r1", tokens(1, 2*testBlockSize)))
	require.True(t, ix.Dispatch(busy.ID, "r1", clockZero))
	idleA := complete(t, ix, clockZero.Add(time.Second), arr(t, "", nil, "r2", tokens(2, 2*testBlockSize)))
	idleB := complete(t, ix, clockZero.Add(time.Minute), arr(t, "", nil, "r3", tokens(3, 2*testBlockSize)))
	require.Len(t, ix.threads, 3)

	require.Equal(t, 1, ix.EnforceCap(clockZero.Add(time.Hour)))
	_, busyKept := ix.threads[busy.ID]
	_, oldestIdleKept := ix.threads[idleA.ID]
	_, newerIdleKept := ix.threads[idleB.ID]
	assert.True(t, busyKept, "an outstanding thread is never evicted")
	assert.False(t, oldestIdleKept, "the idlest non-outstanding thread goes first")
	assert.True(t, newerIdleKept)
	requireInvariants(t, ix)
}

// TestSnapshot_Bounded: the dump reports the table and is bounded.
func TestSnapshot_Bounded(t *testing.T) {
	ix := newThreadIndex(testBlockSize, time.Hour, 0)
	for i := 0; i < maxSnapshotRows+20; i++ {
		turn(ix, clockZero.Add(time.Duration(i)*time.Second), arr(t, "", nil, fmt.Sprintf("r%d", i), tokens(uint32(i+1), testBlockSize*2)))
	}
	snap := ix.Snapshot(clockZero.Add(time.Hour))
	assert.Equal(t, maxSnapshotRows+20, snap.Threads)
	assert.Len(t, snap.Rows, maxSnapshotRows)
	assert.Equal(t, testBlockSize, snap.BlockSize)
}

// TestResolve_Concurrent: the hooks are concurrent; the index stays consistent.
func TestResolve_Concurrent(t *testing.T) {
	ix := newTestIndex()
	base := tokens(1, 2*testBlockSize)
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			marker := fmt.Sprintf("r%d", i)
			req := grown(base, 10+i, uint32(i+100))
			res := ix.Resolve(clockZero, arr(t, "", nil, marker, req), true, true, true)
			ix.Dispatch(res.ID, marker, clockZero)
			ix.Settle(res.ID, marker, clockZero)
		}(i)
	}
	wg.Wait()
	requireInvariants(t, ix)
}

func keySet(m map[string]time.Time) map[string]bool {
	out := make(map[string]bool, len(m))
	for k := range m {
		out[k] = true
	}
	return out
}

// TestCap_EvictsDeterministicallyOnATie: the cap's victim order is TOTAL. The victims come
// from a map, so an equal lastSeen leaves the choice to map iteration order — two replays of
// one scenario then drop different threads FOR GOOD, and nothing downstream can tell which.
func TestCap_EvictsDeterministicallyOnATie(t *testing.T) {
	// twenty identical-lastSeen idle threads against a cap of one: exactly one survives, and
	// it must be the same one every time
	run := func() string {
		ix := newThreadIndex(testBlockSize, time.Hour, 1)
		for i := 0; i < 20; i++ {
			complete(t, ix, clockZero, arr(t, "", nil, fmt.Sprintf("r%d", i), tokens(uint32(i+1), 2*testBlockSize)))
		}
		require.Len(t, ix.threads, 20)
		require.Equal(t, 19, ix.EnforceCap(clockZero.Add(time.Minute)))
		require.Len(t, ix.threads, 1)
		for id := range ix.threads {
			return id
		}
		return ""
	}
	first := run()
	for i := 0; i < 5; i++ {
		if again := run(); again != first {
			t.Fatalf("run %d kept a different thread across the cap (%s vs %s): the victim order is not total", i, again, first)
		}
	}
}
