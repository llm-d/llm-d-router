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

// THE SEAT BUDGET, exactly: what a head costs is everything beyond the engine's resident prefix,
// and a seat that does not fit charges nothing.

import (
	"testing"
)

func newIdx(t *testing.T) *BlockIndex {
	t.Helper()
	idx, err := NewBlockIndex(64)
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

// keys is n block keys drawn from seed's key space: the same seed is the same prefix (a fork, or a
// turn that grows), a different seed shares nothing.
func keys(n int, seed Hash) []Hash {
	out := make([]Hash, 0, n)
	prev := seed
	for i := 0; i < n; i++ {
		prev = prev*0x9E3779B97F4A7C15 ^ Hash(i+1) ^ 0xC2B2AE3D27D4EB4F
		out = append(out, prev)
	}
	return out
}

func seedFor(id string) Hash {
	h := Hash(1469598103934665603)
	for i := 0; i < len(id); i++ {
		h = (h ^ Hash(id[i])) * 1099511628211
	}
	return h
}

// chain is a one-prompt chain of n blocks of id's own content.
func chain(id string, n int) []Chain {
	return []Chain{{Keys: keys(n, seedFor(id)), Tokens: n * 64}}
}

// chainWithPrefix is a chain that REUSES another thread's first prefixBlocks keys.
func chainWithPrefix(prefix string, prefixBlocks int, own string, ownBlocks int) []Chain {
	k := append(keys(prefixBlocks, seedFor(prefix)), keys(ownBlocks, seedFor(own))...)
	return []Chain{{Keys: k, Tokens: (prefixBlocks + ownBlocks) * 64}}
}

func mustSeat(t *testing.T, idx *BlockIndex, id string, chains []Chain, budget int) int {
	t.Helper()
	ok, cost := idx.TrySeat(id, chains, budget)
	if !ok {
		t.Fatalf("TrySeat(%s) rejected (cost %d)", id, cost)
	}
	return cost
}

func TestFailClosedOnUnknownBlockSize(t *testing.T) {
	if _, err := NewBlockIndex(0); err == nil {
		t.Fatal("block size 0 must fail closed")
	}
	if _, err := NewBlockIndex(-1); err == nil {
		t.Fatal("negative block size must fail closed")
	}
	if got := newIdx(t).blockSizeTokens; got != 64 {
		t.Fatalf("block size: %d want 64", got)
	}
}

// TestCostIsTheColdPartOfTheChain: a head's cost is everything beyond the engine's resident
// PREFIX — the shared prefix is free, and the partial last block is paid for (it must be computed).
func TestCostIsTheColdPartOfTheChain(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 4), 1000)
	if got := idx.Cost(chainWithPrefix("A", 3, "B", 2)); got != 2 {
		t.Fatalf("a head sharing 3 of a's 4 blocks costs %d, want 2 (1 new shared tail + 1 private)", got)
	}
	if got := idx.Cost(chain("A", 4)); got != 0 {
		t.Fatalf("a fully resident chain costs %d, want 0", got)
	}
	if got := idx.ColdTokens(chain("A", 4)); got != 0 {
		t.Fatalf("a fully resident chain has %d cold tokens, want 0", got)
	}
	// the partial tail: 100 tokens at 64/block is 2 blocks, the second partial
	partial := []Chain{{Keys: keys(2, seedFor("P")), Tokens: 100}}
	if got := idx.Cost(partial); got != 2 {
		t.Fatalf("a 100-token prompt costs %d blocks, want 2 (the partial block counts)", got)
	}
	if got := idx.ColdTokens(partial); got != 100 {
		t.Fatalf("a 100-token prompt is %d cold tokens, want 100", got)
	}
}

// TestTrySeatAtomicBudget: one number, one moment — TrySeat's cost is the same computation the
// placement module makes, and a rejected seat charges nothing.
func TestTrySeatAtomicBudget(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 4), 100)
	// budget 6: b shares 2 of a's blocks and adds 2 → exactly 6
	ok, cost := idx.TrySeat("b", chainWithPrefix("A", 2, "B", 2), 6)
	if !ok || cost != 2 {
		t.Fatalf("seat b: ok=%v cost=%d want true/2", ok, cost)
	}
	ok, cost = idx.TrySeat("c", chain("C", 5), 6)
	if ok || cost != 5 {
		t.Fatalf("seat c: ok=%v cost=%d want false/5", ok, cost)
	}
	if _, ok := idx.seats["c"]; ok {
		t.Fatal("a rejected seat must not be seated")
	}
	if got := idx.UniqueBlocks(); got != 6 {
		t.Fatalf("a rejected seat charged %d blocks, want the 6 already held", got)
	}
}

// TestExitingMovesStayNotUsed: an exiting thread's blocks are still pinned by the engine, so they
// stay on the USED side and leave the STAY side — that difference is what enforcement acts on.
func TestExitingMovesStayNotUsed(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 3), 1000)
	used, stay := idx.UniqueBlocks(), idx.StayBlocks()
	idx.MarkExiting("a")
	if idx.UniqueBlocks() != used {
		t.Fatalf("exiting moved the used side: %d -> %d", used, idx.UniqueBlocks())
	}
	if idx.StayBlocks() != stay-3 {
		t.Fatalf("exiting left %d stay blocks, want %d (they are no longer staying)", idx.StayBlocks(), stay-3)
	}
	if !idx.Exiting("a") {
		t.Fatal("the thread must read as exiting")
	}
}
