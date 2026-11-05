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

// THE BLOCK INDEX — one engine's model of which KV blocks it holds.

import (
	"errors"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
)

// ErrBlockSizeUnknown is the fail-closed startup error: the block size must be measured, never
// assumed (a hardcoded default once admitted four times the cache).
var ErrBlockSizeUnknown = errors.New("groupsched: engine block size not set (fail closed)")

// Hash is one chained block hash (prefixhash's key space). It is an ALIAS, not a defined type:
// the block keys arrive from thread identity's published attribute and are used without a copy.
type Hash = prefixhash.BlockHash

// Chain is one prompt's EXACT prefill chain: the block keys of its tokens (the last block is
// partial, so its key is private to that exact token sequence) and the prompt's token count.
// It is thread identity's published PromptBlocks by identity, so the scheduler hands the
// attribute's own slices to the index — a Pick pass copies no block keys. One render and one
// hash per request, at arrival; the queue key and the costs come from the same blocks.
//
// The scheduler NEVER writes these slices: they are shared read-only with the publisher (see
// PromptBlocks), so Keys is read, ranged over and compared, never appended to or sorted.
type Chain = threadidentity.PromptBlocks

// BlockIndex is one engine's KV-block accounting. It has NO lock: every call comes from the
// scheduler under its own mutex (see state.go's convention).
type BlockIndex struct {
	blockSizeTokens int

	blocks     map[Hash]int
	blocksStay map[Hash]int
	// seats holds every seated thread's record.
	seats map[string]*seatBlocks
}

// seatBlocks is one seated thread's footprint: its newest chains (the turns of its thread
// extend one another, so the newest chain is its set) and the keys actually charged for it.
type seatBlocks struct {
	chains []Chain
	// charged is the deduplicated union of the chains' keys: what the two refcounts were
	// incremented by at the seat, and what the unseat releases.
	charged map[Hash]struct{}
	exiting bool
}

// NewBlockIndex returns the index for an engine whose KV blocks hold blockSizeTokens tokens.
func NewBlockIndex(blockSizeTokens int) (*BlockIndex, error) {
	if blockSizeTokens <= 0 {
		return nil, fmt.Errorf("%w: got %d", ErrBlockSizeUnknown, blockSizeTokens)
	}
	return &BlockIndex{
		blockSizeTokens: blockSizeTokens,
		blocks:          make(map[Hash]int),
		blocksStay:      make(map[Hash]int),
		seats:           make(map[string]*seatBlocks),
	}, nil
}

// chargeSet deduplicates the union of the chains' keys.
func chargeSet(chains []Chain) map[Hash]struct{} {
	var out map[Hash]struct{}
	for _, c := range chains {
		if out == nil {
			out = make(map[Hash]struct{}, len(c.Keys))
		}
		for _, h := range c.Keys {
			out[h] = struct{}{}
		}
	}
	return out
}

// residentPrefix is how many of the chain's blocks this engine already holds — the
// resident PREFIX, which is a binary search because residency is prefix-closed.
func (b *BlockIndex) residentPrefix(keys []Hash) int {
	lo, hi := 0, len(keys)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if _, held := b.blocks[keys[mid]]; held {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// Cost is the exact admission cost of a thread on this engine, in blocks: how many NEW
// blocks the seat would add to the engine's used side.
func (b *BlockIndex) Cost(chains []Chain) int {
	n := 0
	for h := range chargeSet(chains) {
		if _, held := b.blocks[h]; !held {
			n++
		}
	}
	return n
}

// ColdTokens is the same cost in TOKENS (the cold valve's unit): the prompt's tokens
// beyond the resident prefix.
func (b *BlockIndex) ColdTokens(chains []Chain) int64 {
	var n int64
	for _, c := range chains {
		p := b.residentPrefix(c.Keys)
		if p == len(c.Keys) {
			continue // nothing to prefill
		}
		n += int64(c.Tokens - p*b.blockSizeTokens)
	}
	return n
}

// TrySeat seats a thread if its exact cost fits the budget, and inserts its chains in
// the same critical section: the fit test and the seat can never disagree, because there is
// one number and one moment.
func (b *BlockIndex) TrySeat(id string, chains []Chain, budgetBlocks int) (bool, int) {
	if _, seated := b.seats[id]; seated {
		return true, 0
	}
	cost := b.Cost(chains)
	if len(b.blocks)+cost > budgetBlocks {
		return false, cost
	}
	b.insert(id, chains)
	return true, cost
}

// insert creates the seat's record and charges its keys to both counters.
func (b *BlockIndex) insert(id string, chains []Chain) {
	s := &seatBlocks{chains: chains, charged: chargeSet(chains)}
	b.seats[id] = s
	for h := range s.charged {
		b.blocks[h]++
		b.blocksStay[h]++
	}
}

// SeatForce seats a thread WITHOUT the budget check: the blocks are taken as already
// present on this engine (an observed placement said so — the engine, not the model, is
// authoritative about what it holds).
func (b *BlockIndex) SeatForce(id string, chains []Chain) {
	if _, seated := b.seats[id]; seated {
		b.Grow(id, chains) // already charging this thread: the newest chain replaces its set
		return
	}
	b.insert(id, chains)
}

// Grow makes a seated thread's chains the NEWEST ones it has: a thread's turns extend
// one another, so the newest chain is its set, and the difference is charged (or released, for
// the partial block the growing tail replaced).
func (b *BlockIndex) Grow(id string, chains []Chain) {
	s, ok := b.seats[id]
	if !ok || sameChains(s.chains, chains) {
		return
	}
	old := s.charged
	s.chains = chains
	s.charged = chargeSet(chains)
	b.applyDelta(s, old)
}

// sameChains reports whether two chain sets are the SAME CHAINS, not merely the same shape.
func sameChains(a, c []Chain) bool {
	if len(a) != len(c) {
		return false
	}
	for i := range a {
		if len(a[i].Keys) != len(c[i].Keys) || a[i].Tokens != c[i].Tokens {
			return false
		}
		// both empty: one prompt with no blocks, nothing to compare
		if len(a[i].Keys) == 0 {
			continue
		}
		if a[i].Keys[len(a[i].Keys)-1] != c[i].Keys[len(c[i].Keys)-1] {
			return false
		}
	}
	return true
}

// applyDelta charges the keys the thread gained and releases the keys it lost.
func (b *BlockIndex) applyDelta(s *seatBlocks, old map[Hash]struct{}) {
	for h := range s.charged {
		if _, had := old[h]; !had {
			b.blocks[h]++
			if !s.exiting {
				b.blocksStay[h]++
			}
		}
	}
	for h := range old {
		if _, has := s.charged[h]; !has {
			b.dec(b.blocks, h)
			if !s.exiting {
				b.dec(b.blocksStay, h)
			}
		}
	}
}

// Unseat releases a thread's seat and drops its record: this is the ONE release point,
// and the record does not outlive the seat (a re-seat costs the thread nothing to rebuild —
// its head carries the chain).
func (b *BlockIndex) Unseat(id string) {
	s, ok := b.seats[id]
	if !ok {
		return
	}
	for h := range s.charged {
		b.dec(b.blocks, h)
		if !s.exiting {
			b.dec(b.blocksStay, h) // an exiting thread left the STAY side at its mark
		}
	}
	delete(b.seats, id)
}

// dec decrements a refcount, panicking on underflow: a refcount below zero is a programming
// error, and a silent one would poison every later decision.
func (b *BlockIndex) dec(m map[Hash]int, h Hash) {
	c, ok := m[h]
	if !ok || c <= 0 {
		panic(fmt.Sprintf("groupsched: refcount underflow for hash %d", h))
	}
	if c == 1 {
		delete(m, h)
	} else {
		m[h] = c - 1
	}
}

// MarkExiting marks a thread as on its way out: it is not dispatched and not re-seated,
// and it leaves at its next drain.
func (b *BlockIndex) MarkExiting(id string) {
	s, ok := b.seats[id]
	if !ok || s.exiting {
		return
	}
	s.exiting = true
	for h := range s.charged {
		b.dec(b.blocksStay, h)
	}
}

// Exiting reports the thread's exiting state.
func (b *BlockIndex) Exiting(id string) bool {
	s, ok := b.seats[id]
	return ok && s.exiting
}

// ---- the counted views ----

// UniqueBlocks is the USED side: every seated thread's blocks, exiting included.
func (b *BlockIndex) UniqueBlocks() int { return len(b.blocks) }

// StayBlocks is the STAY side: the blocks of the threads that are not exiting.
func (b *BlockIndex) StayBlocks() int { return len(b.blocksStay) }

// ExitingBlocks is the part of the USED side that belongs only to exiting threads (used
// − stay): the overage enforcement must NOT evict again, and admission still refuses.
func (b *BlockIndex) ExitingBlocks() int { return len(b.blocks) - len(b.blocksStay) }

// ChargedBlocks is the number of DISTINCT blocks this seat holds on the engine: the
// footprint a single thread costs the budget.
func (b *BlockIndex) ChargedBlocks(id string) int {
	if s, ok := b.seats[id]; ok {
		return len(s.charged)
	}
	return 0
}

// Chains is a seated thread's chains: what a rebind carries to the engine that is
// observed to be serving it.
func (b *BlockIndex) Chains(id string) []Chain {
	s, ok := b.seats[id]
	if !ok {
		return nil
	}
	return s.chains
}
