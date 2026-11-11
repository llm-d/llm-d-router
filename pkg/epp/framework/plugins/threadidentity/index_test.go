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

// THE CONTINUATION RULE, in its three answers: a turn that extends a tip continues the thread, a
// turn that extends a tip while the thread is BUSY forks, and a request with no identity at all is
// a new thread.

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const testBlockSize = 64

var clockZero = time.Unix(1_700_000_000, 0)

// tokens builds a deterministic token stream: two streams share a prefix exactly as far as they
// share a seed.
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

// chainOf hashes token streams into an identity chain, exactly as the arrival hook does.
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

func newTestIndex() *threadIndex { return newThreadIndex(testBlockSize, time.Hour, 1000) }

// turn resolves one arrival (the rule's defaults: fork when busy, the sticky guard, the retry rule).
func turn(ix *threadIndex, at time.Time, a Arrival) ResolveResult {
	return ix.Resolve(at, a, true, true, true)
}

// complete resolves an arrival and finishes it: dispatched, then settled.
func complete(t *testing.T, ix *threadIndex, at time.Time, a Arrival) ResolveResult {
	t.Helper()
	res := turn(ix, at, a)
	ix.Dispatch(res.ID, a.Marker, at)
	ix.Settle(res.ID, a.Marker, at)
	return res
}

func requireInvariants(t *testing.T, ix *threadIndex) {
	t.Helper()
	require.Empty(t, ix.checkIndexInvariants())
}

// TestResolve_Continuation: a turn that extends a finished thread's tip continues it — the same id,
// with the verified prefix stopping one full block short of the tip.
func TestResolve_Continuation(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 3*testBlockSize)

	first := complete(t, ix, clockZero, Arrival{Chain: chainOf(t, tip), Marker: "r1"})
	require.Equal(t, OutcomeNew, first.Outcome)

	next := turn(ix, clockZero.Add(time.Second), Arrival{Chain: chainOf(t, grown(tip, 100, 2)), Marker: "r2"})
	assert.Equal(t, OutcomeMatched, next.Outcome)
	assert.Equal(t, first.ID, next.ID, "a continuation keeps the thread id")
	assert.Equal(t, 2, next.MatchedBlocks, "the verified prefix stops one full block short of the tip")
	requireInvariants(t, ix)
}

// TestResolve_ForkIfBusy: the same extension while the thread still has the request in flight is a
// FORK — a sub-agent, not a continuation.
func TestResolve_ForkIfBusy(t *testing.T) {
	ix := newTestIndex()
	tip := tokens(1, 2*testBlockSize)
	first := turn(ix, clockZero, Arrival{Chain: chainOf(t, tip), Marker: "r1"})
	require.Equal(t, OutcomeNew, first.Outcome)
	require.True(t, ix.Dispatch(first.ID, "r1", clockZero))

	res := turn(ix, clockZero.Add(time.Second), Arrival{Chain: chainOf(t, grown(tip, 100, 2)), Marker: "r2"})
	assert.Equal(t, OutcomeForked, res.Outcome)
	assert.NotEqual(t, first.ID, res.ID)
	assert.Equal(t, 1, ix.threads[first.ID].forks)
	requireInvariants(t, ix)
}

// TestResolve_NoChainIsANewThread: a request with no identity is a thread of its own, and it never
// becomes anybody's continuation.
func TestResolve_NoChainIsANewThread(t *testing.T) {
	ix := newTestIndex()
	first := turn(ix, clockZero, Arrival{Chain: chainOf(t, tokens(1, 2*testBlockSize)), Marker: "r1"})

	res := turn(ix, clockZero.Add(time.Second), Arrival{Marker: "r2"})
	assert.Equal(t, OutcomeNew, res.Outcome)
	assert.NotEqual(t, first.ID, res.ID)

	again := turn(ix, clockZero.Add(2*time.Second), Arrival{Marker: "r3"})
	assert.Equal(t, OutcomeNew, again.Outcome)
	assert.NotEqual(t, res.ID, again.ID)
	requireInvariants(t, ix)
}
