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

package groupsched

// THE FINGERPRINT CONTRACT — the blocks the scheduler charges are the blocks thread identity
// published, byte for byte.
//
// The scheduler no longer hashes anything: thread identity hashes the rendered request once, at
// arrival, derives its thread from the result, and publishes the same per-prompt block keys. If
// those two ever disagreed, every cost in the policy would be computed from a chain the queue key
// knows nothing about — silently, because both are just numbers. These tests hold the two
// together: the attribute equals a fresh hash of the request's own tokens, for every shape in the
// corpus.

import (
	"context"
	"fmt"
	"testing"
	"time"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	"github.com/stretchr/testify/require"
)

// fingerprintCorpus is the request shapes the policy has to charge correctly: a fresh thread, a
// long continuation, a mid-block tail, a many-turn conversation, and a MULTI-PROMPT request
// (whose prompts are hashed independently, so they exercise the per-prompt shape of the
// attribute rather than one flat list).
func fingerprintCorpus(t *testing.T) map[string]*fwksched.InferenceRequest {
	t.Helper()
	corpus := map[string]*fwksched.InferenceRequest{
		"first turn":     chatRequest("a", 1, 500),
		"long":           chatRequest("b", 1, 5000),
		"partial tail":   chatRequest("c", 1, 63),
		"many turns":     chatRequest("d", 9, 300),
		"one token less": chatRequest("e", 1, 255),
	}
	multi := chatRequest("f", 1, 200)
	multi.Body.TokenizedRequest = &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{
		{TokenIDs: promptTokensFor(300)},
		{TokenIDs: promptTokensFor(70)},
	}}
	corpus["multi-prompt"] = multi
	return corpus
}

// promptTokensFor is a deterministic token stream of n ids.
func promptTokensFor(n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(1000 + (i*37)%9999)
	}
	return out
}

// TestPublishedFingerprintIsTheRequestsOwn: the attribute thread identity publishes is exactly
// what hashing the rendered request yields — every prompt, at the plugin's block size.
func TestPublishedFingerprintIsTheRequestsOwn(t *testing.T) {
	bi := newThreadIdentity(t)
	for name, req := range fingerprintCorpus(t) {
		t.Run(name, func(t *testing.T) {
			// the multi-prompt request brings its own tokens; the others are rendered here
			preRendered := req.Body.TokenizedRequest != nil
			require.NoError(t, bi.RequestHeader(context.Background(), req))
			if !preRendered {
				require.NotNil(t, req.Body.TokenizedRequest, "the arrival hook rendered the request")
			}

			published := chainsOfAttribute(t, req)
			wantHashes, wantTokens := prefixhash.GetBlockHashesWithPromptTokens(context.Background(), req, bi.BlockSize(), 1<<20)
			require.Len(t, published, len(wantHashes), "one entry per prompt")
			for i := range wantHashes {
				require.Equal(t, wantHashes[i], published[i].Keys, "prompt %d: the published keys are the request's own", i)
				require.Equal(t, wantTokens[i], published[i].Tokens, "prompt %d: the token count is the prompt's", i)
			}
		})
	}
}

// TestPublishedFingerprintIsUnchangedByChargeChains: reading the attribute returns the very
// slices the plugin published — no copy, no re-hash — so two reads in one pass see the same
// array. This is what makes Pick allocation-free on the block path.
func TestPublishedFingerprintIsUnchangedByChargeChains(t *testing.T) {
	bi := newThreadIdentity(t)
	req := chatRequest("g", 2, 400)
	require.NoError(t, bi.RequestHeader(context.Background(), req))

	first := chargeChains(req)
	second := chargeChains(req)
	require.NotEmpty(t, first)
	require.Equal(t, len(first), len(second))
	for i := range first {
		require.Equal(t, &first[i].Keys[0], &second[i].Keys[0], "prompt %d: the same backing array", i)
	}
}

// TestNoFingerprintMeansNoChain: a request that never reached thread identity carries no chain,
// so a head built from it is charged nothing rather than charged a chain we invented.
func TestNoFingerprintMeansNoChain(t *testing.T) {
	require.Nil(t, chargeChains(chatRequest("h", 1, 400)))
	require.Nil(t, chargeChains(nil))
}

// benchmarkRequests is N requests with long prompts, each already through thread identity —
// the shape Pick actually meets: real chains on every head, one per thread.
func benchmarkRequests(b *testing.B, n int) []*fwksched.InferenceRequest {
	b.Helper()
	bi := newThreadIdentity(b)
	reqs := make([]*fwksched.InferenceRequest, 0, n)
	for i := 0; i < n; i++ {
		req := chatRequest(fmt.Sprintf("bench%d", i), 3, 2000)
		if err := bi.RequestHeader(context.Background(), req); err != nil {
			b.Fatalf("thread identity: %v", err)
		}
		reqs = append(reqs, req)
	}
	return reqs
}

// recomputeChains is what Pick used to do at every pass: hash the request's tokens and copy them
// into the scheduler's own Chain type. It is kept HERE, as the benchmark's reference, because the
// measurement only means something against the work it replaced.
func recomputeChains(request *fwksched.InferenceRequest) []Chain {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return nil
	}
	hashed, promptTokens := prefixhash.GetBlockHashesWithPromptTokens(context.Background(), request, 64, 1<<20)
	chains := make([]Chain, 0, len(hashed))
	for i, prompt := range hashed {
		keys := make([]Hash, len(prompt))
		copy(keys, prompt)
		chains = append(chains, Chain{Keys: keys, Tokens: promptTokens[i]})
	}
	return chains
}

// BenchmarkFingerprintRecomputed is the block path BEFORE: hash every queued request at every
// pass, and copy the keys into the scheduler's slices.
func BenchmarkFingerprintRecomputed(b *testing.B) {
	reqs := benchmarkRequests(b, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, req := range reqs {
			_ = recomputeChains(req)
		}
	}
}

// BenchmarkFingerprintPublished is the block path AFTER: read the attribute the arrival hook
// already published. No hashing, no copying.
func BenchmarkFingerprintPublished(b *testing.B) {
	reqs := benchmarkRequests(b, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, req := range reqs {
			_ = chargeChains(req)
		}
	}
}

// BenchmarkPassWithLongPrompts measures a whole decision pass over 64 long-prompt heads on one
// learned engine: what Pick costs when the blocks are handed over rather than computed.
func BenchmarkPassWithLongPrompts(b *testing.B) {
	reqs := benchmarkRequests(b, 64)
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		b.Fatal(err)
	}
	if err := s.SetNumBlocksFor("e0", 64, 4096); err != nil {
		b.Fatal(err)
	}
	now := time.Unix(1793000000, 0).UTC()
	hs := make([]Head, 0, len(reqs))
	for _, req := range reqs {
		hs = append(hs, Head{
			Thread:     req.FairnessID,
			ReqID:      req.RequestID,
			FairnessID: req.FairnessID,
			Returner:   isReturner(req),
			Chains:     chargeChains(req),
			Arrival:    now,
		})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Pass(hs, now)
	}
}
