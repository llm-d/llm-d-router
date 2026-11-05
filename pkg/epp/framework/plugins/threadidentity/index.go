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

// Package threadidentity derives the flow-control queue key from the request
// body: the arrival hook renders the request with the token producer, hashes it
// into the prefix scorer's own block chain, and matches it against the live
// thread table, so the fairness id is a derived THREAD id instead of a
// client-supplied session id.
//
// A thread is a chain of requests, each continuing the previous; its tip is the
// latest request. An arrival CONTINUES a tip, FORKS it (it extends a busy tip) or
// starts a NEW thread. The session header is a separate attribute, never a queue
// key.
//
// The identity is the ENGINE's token stream, not a normalized view of the body:
// a thread is a KV-cache lineage, so a request continues a tip exactly when the
// cache would serve it the same prefix. That is why the block chain is the prefix
// scorer's own (prefixhash, model+salt seeded, chained) and why a mid-session
// tool-set change or a re-rendered history SPLITS a thread — the tokens really do
// diverge there.
//
// Busy is marked at ARRIVAL, not at dispatch: under load the queueing interval is
// most of a busy period, so a thread that looked free while its turn sat in the
// admission queue would hand a sibling the parent's queue. A thread is retained
// by content: there is no idle GC (stage 0 measured a p99.9 inter-turn gap of
// 3,036s, so any silence timeout mints false NEW threads for dormant clients);
// only the table cap evicts.
//
// Error asymmetry: a missed continuation (a false NEW thread) re-fragments a
// client into newcomer churn, while a false continuation only shares a queue
// between two requests of the same thread. Every rule resolves toward
// continuation — including the tail tolerance in setTipLocked.
package threadidentity

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
)

// idHexLen is the number of hex characters a derived id keeps: an id can become a metric label,
// so its length is a cardinality decision, not a cosmetic one.
const (
	// idHexLen keeps thread ids short: an id can become a metric label value.
	idHexLen = 10
	// maxPendingPerThread bounds one thread's arrival markers that have not
	// dispatched, so a leaked marker cannot wedge or grow the thread.
	maxPendingPerThread = 64
	// maxSnapshotRows bounds the StateDumper table.
	maxSnapshotRows = 200
	// maxMintAttempts bounds the ordinal bumps used to avoid reusing a live id.
	maxMintAttempts = 1 << 12
	// maxChainBlocks bounds a chain's length. It is effectively unbounded — a
	// truncated chain would make long requests look alike — and only guards a
	// corrupt token count.
	maxChainBlocks = 1 << 20
)

// mintFallbackSeq makes a minted id unique within this process when the ordinal
// space is exhausted (see mintLocked).
var mintFallbackSeq atomic.Uint64

// Outcome labels how Resolve identified a request — the three arrival outcomes.
type Outcome string

// OutcomeMatched opens the outcome kinds, one per arrival. This one is the "continue": the request
// continued a thread, so it keeps that thread's queue key.
const (
	// OutcomeMatched: the request continued a thread (the "continue" outcome).
	OutcomeMatched Outcome = "matched"
	// OutcomeForked: the request extended a BUSY tip, so it started a new thread
	// (a parallel sibling inheriting the tip's context).
	OutcomeForked Outcome = "forked"
	// OutcomeNew: nothing matched, so the request started a new thread.
	OutcomeNew Outcome = "new"
)

// chain is a request's identity in the prefix scorer's hash space.
type chain struct {
	// blocks holds every block hash in order: the per-prompt chains
	// concatenated, each prompt's trailing partial block included, and every
	// block of a LATER prompt re-chained onto the chain before it (see
	// identityChain) so a prompt shared by two requests is not a set of shared
	// keys. All prompts count (the faithful mapping of v1's strings.Join).
	//
	// A multi-prompt request's earlier prompts keep their interior partial block
	// in the chain. That is stricter than the tip tolerance below, not looser in
	// the direction that matters: a continuation has to reproduce them exactly.
	blocks []prefixhash.BlockHash
	// fullBlocks counts the blocks that cover a whole block of tokens — every
	// block but the LAST prompt's trailing partial one. Only the end of a chain
	// is a tip, so only the end needs the distinction.
	fullBlocks int
	// tokens is the request's prompt token count.
	tokens int
}

// lastHash is the tip commitment: the last block hash, partial block included.
// Together with tokens it is the S2 exact key — two requests with the same pair
// rendered the same prompt.
func (c chain) lastHash() prefixhash.BlockHash {
	if len(c.blocks) == 0 {
		return 0
	}
	return c.blocks[len(c.blocks)-1]
}

// threadRecord is one live thread: its tip and its liveness.
type threadRecord struct {
	id    string
	scope string // client session id at creation, "" when the client sent none

	// The tip. (fullBlocks, lastHash, tokens) is the S2 exact key: two requests
	// with the same triple are identical up to the whole rendered prompt.
	fullBlocks int
	lastHash   prefixhash.BlockHash
	tokens     int
	// prefixLen is how many leading blocks CONTINUATION verifies — the tip's
	// chain except its last full block — and prefixKey is that prefix's last
	// hash, the tip index key. prefixLen == 0 means the tip is not continuable
	// (it is shorter than one full block).
	prefixLen int
	prefixKey prefixhash.BlockHash

	// msgCount is the tip request's message count: H1 reads the role at that
	// index of a later request.
	msgCount int

	// pending holds the arrival markers of requests that have not dispatched
	// (they are in admission control), inflight those that have. Keyed by
	// request id, so a cancellation, a rejection or a retry adds or removes
	// exactly one entry — a FIFO counter cannot say which request ended.
	pending  map[string]time.Time
	inflight map[string]time.Time

	lastSeen time.Time
	created  time.Time
	turns    int
	forks    int
}

// idle reports the thread's silence (no arrival, dispatch or settle).
func (r *threadRecord) idle(now time.Time) time.Duration { return now.Sub(r.lastSeen) }

// markPending records one arrival marker, keeping the thread alive.
func (r *threadRecord) markPending(marker string, now time.Time) {
	if r.pending == nil {
		r.pending = make(map[string]time.Time, 2)
	}
	if _, dup := r.pending[marker]; !dup && len(r.pending) >= maxPendingPerThread {
		// A lost marker errs toward continuation, which is the safe direction.
		return
	}
	r.pending[marker] = now
}

// markDispatched moves a marker from pending to in flight.
func (r *threadRecord) markDispatched(marker string, now time.Time) {
	if r.inflight == nil {
		r.inflight = make(map[string]time.Time, 2)
	}
	delete(r.pending, marker)
	r.inflight[marker] = now
}

// markSettled removes a marker from in flight (response complete or aborted).
func (r *threadRecord) markSettled(marker string) { delete(r.inflight, marker) }

// pruneMarkers drops markers older than ttl: a request rejected before dispatch
// or one that never reported an end of stream must not keep its thread busy
// forever. Pruning errs toward continuation.
func (r *threadRecord) pruneMarkers(now time.Time, ttl time.Duration) {
	if ttl <= 0 {
		return
	}
	for _, set := range []map[string]time.Time{r.pending, r.inflight} {
		for marker, at := range set {
			if now.Sub(at) > ttl {
				delete(set, marker)
			}
		}
	}
}

// outstanding reports whether the thread has a request in flight or pending —
// the continuation rule's busy predicate.
func (r *threadRecord) outstanding() bool {
	return len(r.inflight) > 0 || len(r.pending) > 0
}

// Arrival is one request's identity inputs.
type Arrival struct {
	// Chain is the request's identity chain; the zero value means the request
	// has no identity (a failed render), which never continues anything.
	Chain chain
	// Scope is the client session id, "" when the client sent none.
	Scope string
	// Roles is the request's message roles in wire order (nil for body shapes
	// without messages — H1 then cannot fire).
	Roles []string
	// Marker is the request's id: the handle that later removes this arrival
	// from the thread's outstanding set.
	Marker string
}

// ResolveResult is the outcome of one arrival.
type ResolveResult struct {
	// ID is the thread id (the value the director uses as the fairness id).
	ID string
	// Outcome is matched / forked / new.
	Outcome Outcome
	// Turn is the arrival's 1-based turn number on its thread: 1 for a thread's
	// first request, and for a fork's first. Published as an attribute (the
	// arrival record the scheduler reads).
	Turn int
	// Blocks is the arrival's own full-block count.
	Blocks int
	// MatchedBlocks is the verified prefix length of the matched tip (0 for a
	// new thread); for a fork it is the parent tip's.
	MatchedBlocks int
	// StickyBusy is set when the H1 guard overrode the busy flag.
	StickyBusy bool
	// Retry is set when S2 treated a duplicate of a still-queued tip as a
	// client retry (a continuation, not a sibling).
	Retry bool
	// Elapsed is the time the resolve spent inside the table lock.
	Elapsed time.Duration
}

// threadIndex is the thread table: thread id → record, plus the tip index
// (prefix key → candidate threads). Safe for concurrent use.
type threadIndex struct {
	mu sync.Mutex

	blockSize  int
	markerTTL  time.Duration
	maxThreads int

	threads map[string]*threadRecord
	// byPrefix indexes continuable tips by their verified-prefix hash. The
	// prefix LENGTH is carried on the record, not in the key: a chained hash
	// commits to its whole prefix, so two lengths sharing one key can only be a
	// collision, which the length check rejects.
	byPrefix map[prefixhash.BlockHash][]*threadRecord
}

// newThreadIndex builds the derivation index for one process: the block size, the marker TTL and the
// bound on tracked threads.
func newThreadIndex(blockSize int, markerTTL time.Duration, maxThreads int) *threadIndex {
	return &threadIndex{
		blockSize:  blockSize,
		markerTTL:  markerTTL,
		maxThreads: maxThreads,
		threads:    make(map[string]*threadRecord),
		byPrefix:   make(map[prefixhash.BlockHash][]*threadRecord),
	}
}

// Resolve identifies the thread for an arriving request and records the arrival.
//
// The decision, in order:
//  1. the tip with the longest verified prefix, in the same scope.
//  2. tip not outstanding → continue (a repeat of a finished turn).
//  3. H1 (when enabled): the first message past the tip is an assistant message,
//     so the request contains the tip's reply → continue.
//  4. S2 (when enabled): the request is identical to the tip and the tip has
//     nothing IN FLIGHT (only arrival markers) → continue (a client retry).
//  5. tip outstanding and unproven → fork (unless forkOnBusy is off).
//  6. no matching tip → new thread.
func (ix *threadIndex) Resolve(now time.Time, a Arrival, forkOnBusy, stickyGuard, retryRule bool) ResolveResult {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	start := time.Now()

	rec, matched := ix.matchLocked(now, a.Chain, a.Scope)
	if rec == nil {
		fresh := ix.mintLocked(now, nil, a)
		return ResolveResult{ID: fresh.id, Outcome: OutcomeNew, Turn: 1, Blocks: a.Chain.fullBlocks, Elapsed: time.Since(start)}
	}

	// S2's exact key: the whole rendered prompt, tail included.
	exact := rec.tokens == a.Chain.tokens && rec.lastHash == a.Chain.lastHash()
	// A continuation needs a FREE tip: a tip with a request in flight or pending
	// belongs to that request, unless content proves completion (H1) or a
	// still-queued duplicate is a retry (S2), or the fork rule is configured off.
	cont := !rec.outstanding()
	sticky, retry := false, false
	if !cont {
		switch {
		case stickyGuard && startsWithReply(rec.msgCount, a.Roles):
			sticky, cont = true, true
		case retryRule && exact && len(rec.inflight) == 0:
			// Identical to the tip with nothing actually started: a client retry
			// of a request still sitting in the admission queue. The queue key was
			// fixed at ARRIVAL, so the two merge in the flow queue and nothing
			// re-keys them later.
			retry, cont = true, true
		default:
			cont = !forkOnBusy
		}
	}

	if cont {
		ix.advanceLocked(now, rec, a)
		return ResolveResult{
			ID: rec.id, Outcome: OutcomeMatched, Turn: rec.turns, Blocks: a.Chain.fullBlocks,
			MatchedBlocks: matched, StickyBusy: sticky, Retry: retry, Elapsed: time.Since(start),
		}
	}

	fork := ix.mintLocked(now, rec, a)
	rec.forks++
	return ResolveResult{
		ID: fork.id, Outcome: OutcomeForked, Turn: 1, Blocks: a.Chain.fullBlocks, MatchedBlocks: matched,
		Elapsed: time.Since(start),
	}
}

// matchLocked returns the thread whose verified prefix is the longest prefix of
// the arrival's chain, and that prefix's length in blocks. Scoped tips only
// match their own session; unscoped tips match any (the floor the design accepts
// when the client sends no session header).
//
// The probe DESCENDS over the arrival's own block hashes because the tip index is
// NOT prefix-closed: a tip may exist at k=7 with none at k=5, so an early stop at
// the first miss would lose it.
func (ix *threadIndex) matchLocked(now time.Time, c chain, scope string) (*threadRecord, int) {
	for k := min(c.fullBlocks, len(c.blocks)); k >= 1; k-- {
		var best *threadRecord
		bestOutstanding := true
		for _, cand := range ix.byPrefix[c.blocks[k-1]] {
			if cand.prefixLen != k || !sameScope(cand.scope, scope) {
				continue
			}
			cand.pruneMarkers(now, ix.markerTTL)
			out := cand.outstanding()
			switch {
			case best == nil:
				best, bestOutstanding = cand, out
			case bestOutstanding && !out:
				// A free tip is the better continuation target than a busy one of
				// identical content (the busy one has a sibling in flight).
				best, bestOutstanding = cand, out
			case bestOutstanding == out && betterTip(cand, best):
				best = cand
			}
		}
		if best != nil {
			return best, k
		}
	}
	return nil, 0
}

// betterTip orders two equally verifiable candidates: the LONGER tip first (it
// carries more evidence, and v1 matched the longest tip), then the most recently
// seen, then the id — the last term keeps the choice deterministic when a
// replay's clock ties, which the stage-0c gate requires.
func betterTip(cand, best *threadRecord) bool {
	if cand.fullBlocks != best.fullBlocks {
		return cand.fullBlocks > best.fullBlocks
	}
	if !cand.lastSeen.Equal(best.lastSeen) {
		return cand.lastSeen.After(best.lastSeen)
	}
	return cand.id < best.id
}

// advanceLocked moves the thread's tip to the arriving request and records the
// arrival. The tip moves at ARRIVAL: the request is the thread's newest context
// whether or not it has dispatched yet, and a later request extending it must see
// it.
func (ix *threadIndex) advanceLocked(now time.Time, rec *threadRecord, a Arrival) {
	ix.setTipLocked(rec, a.Chain)
	rec.msgCount = len(a.Roles)
	rec.lastSeen = now
	rec.turns++
	rec.markPending(a.Marker, now)
}

// setTipLocked re-points the record at a new tip chain, keeping the tip index in
// sync.
//
// THE TOLERANCE: the verified prefix is the tip's chain EXCEPT its last full
// block. A tip with two or more full blocks therefore leaves its partial block
// plus one whole block unverified — between B and 2B-1 tokens — and a request
// that differs from the tip only there still continues it. That absorbs
// end-of-tip rendering variance which the history does not reproduce: the
// generation prompt, a template's empty-think stub, a BPE merge at the boundary.
// One block of slack, rather than the partial block alone, is what makes the
// guarantee hold wherever the prompt ends: with the partial block alone the slack
// can be ZERO, and an end variance that crosses the block edge re-keys the thread.
//
// GUARDS: a tip with exactly ONE full block gets no tolerance — its prefix would
// be empty and match every request — so that one block must match and only its
// partial block is unverified; and a tip shorter than one block is not continuable
// at all (it has no block hash to index, and would otherwise match on an empty
// prefix). Both err toward NEW, which is the unsafe direction, and both are
// counted by the equivalence gate as declared classes (iii) and (v).
func (ix *threadIndex) setTipLocked(rec *threadRecord, c chain) {
	ix.unindexLocked(rec)
	rec.fullBlocks, rec.tokens, rec.lastHash = c.fullBlocks, c.tokens, c.lastHash()
	rec.prefixLen, rec.prefixKey = 0, 0
	if rec.fullBlocks > 0 {
		rec.prefixLen = max(rec.fullBlocks-1, 1)
		rec.prefixKey = c.blocks[rec.prefixLen-1]
	}
	ix.indexLocked(rec)
}

// indexLocked inserts the record under its prefix key.
func (ix *threadIndex) indexLocked(rec *threadRecord) {
	if rec.prefixLen == 0 {
		return
	}
	ix.byPrefix[rec.prefixKey] = append(ix.byPrefix[rec.prefixKey], rec)
}

// unindexLocked removes the record from the tip index (cap eviction, re-tip).
func (ix *threadIndex) unindexLocked(rec *threadRecord) {
	if rec.prefixLen == 0 {
		return
	}
	cands := ix.byPrefix[rec.prefixKey]
	for i, c := range cands {
		if c == rec {
			cands = append(cands[:i], cands[i+1:]...)
			break
		}
	}
	if len(cands) == 0 {
		delete(ix.byPrefix, rec.prefixKey)
		return
	}
	ix.byPrefix[rec.prefixKey] = cands
}

// mintLocked creates a thread. Its id is derived from the creating request's tip
// commitment, the fork ordinal and the scope: the same thread yields the same id
// on any instance without a registry, and a thread keeps its id as it continues.
//
// An id the table already holds belongs to a LIVE thread and is never
// overwritten — overwriting leaves the old record GHOSTED (still indexed, no
// longer in the table: it keeps matching, is invisible to the cap, and the index
// grows without bound). What looks like the same thread being minted twice (a
// client re-sending an earlier turn after its thread advanced, a look-alike fork
// with the same ordinal) bumps the ordinal and gets a distinct thread: the id is
// the queue key, and two different moments in a conversation are two different
// queue states.
func (ix *threadIndex) mintLocked(now time.Time, parent *threadRecord, a Arrival) *threadRecord {
	tip := a.Chain.lastHash()
	ordinal := 0
	if parent != nil {
		ordinal = parent.forks + 1
	}
	seed := mintSeed(tip, ordinal, a.Scope)
	id := threadID(seed)
	for attempt := 0; ; attempt++ {
		if _, clash := ix.threads[id]; !clash {
			break
		}
		if attempt < maxMintAttempts {
			ordinal++
			seed = mintSeed(tip, ordinal, a.Scope)
		} else {
			// Every ordinal in the window is taken by a live id (astronomically
			// unlikely): salt with a process-unique nonce rather than reuse.
			seed = mintSeed(tip, ordinal, a.Scope) ^ prefixhash.BlockHash(mintFallbackSeq.Add(1))
		}
		id = threadID(seed)
	}

	rec := &threadRecord{
		id:       id,
		scope:    a.Scope,
		msgCount: len(a.Roles),
		created:  now,
		lastSeen: now,
		turns:    1,
	}
	if parent != nil && rec.scope == "" {
		// A fork inherits its parent's scope when the request carried none:
		// siblings belong to the same session.
		rec.scope = parent.scope
	}
	ix.threads[id] = rec
	ix.setTipLocked(rec, a.Chain)
	rec.markPending(a.Marker, now)
	return rec
}

// Dispatch marks one request as dispatched (in flight) for the thread, moving its
// arrival marker out of the pending set. An unknown id is ignored (the thread may
// have been evicted while the request was queued).
func (ix *threadIndex) Dispatch(id, marker string, now time.Time) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	rec, ok := ix.threads[id]
	if !ok || rec == nil {
		return false
	}
	rec.pruneMarkers(now, ix.markerTTL)
	rec.markDispatched(marker, now)
	rec.lastSeen = now
	return true
}

// Settle ends one dispatched request on the thread (response complete or
// aborted). The tip stays where the request left it; only liveness changes.
func (ix *threadIndex) Settle(id, marker string, now time.Time) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	rec, ok := ix.threads[id]
	if !ok || rec == nil {
		return false
	}
	rec.markSettled(marker)
	rec.lastSeen = now
	return true
}

// EnforceCap bounds the table when it grows past maxThreads, evicting the
// IDLEST non-outstanding threads first. There is deliberately NO silence GC
// (see the package doc): the count cap is the only eviction path, and it never
// removes a thread with anything outstanding.
func (ix *threadIndex) EnforceCap(now time.Time) int {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	for _, rec := range ix.threads {
		rec.pruneMarkers(now, ix.markerTTL)
	}
	if ix.maxThreads <= 0 || len(ix.threads) <= ix.maxThreads {
		return 0
	}
	victims := make([]*threadRecord, 0, len(ix.threads)-ix.maxThreads)
	for _, rec := range ix.threads {
		if !rec.outstanding() {
			victims = append(victims, rec)
		}
	}
	// A TOTAL order (idlest first, then the id): the victims come from a map, so a
	// lastSeen-only comparison leaves ties to map order and two replays of one scenario can
	// drop DIFFERENT threads for good.
	sort.Slice(victims, func(i, j int) bool {
		if !victims[i].lastSeen.Equal(victims[j].lastSeen) {
			return victims[i].lastSeen.Before(victims[j].lastSeen)
		}
		return victims[i].id < victims[j].id
	})
	removed := 0
	for i := 0; i < len(victims) && len(ix.threads) > ix.maxThreads; i++ {
		ix.unindexLocked(victims[i])
		delete(ix.threads, victims[i].id)
		removed++
	}
	return removed
}

// Len is the live thread count (the records gauge).
func (ix *threadIndex) Len() int {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	return len(ix.threads)
}

// checkIndexInvariants reports every record reachable from the tip index that is
// not the live record for its id. A violation means a record was overwritten
// (ghosted) or unindexed inconsistently; the concurrency test asserts it is
// empty.
func (ix *threadIndex) checkIndexInvariants() []string {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	var problems []string
	seen := make(map[*threadRecord]int, len(ix.threads))
	for key, cands := range ix.byPrefix {
		for _, cand := range cands {
			seen[cand]++
			if ix.threads[cand.id] != cand {
				problems = append(problems, "indexed record not live: "+cand.id)
			}
			if cand.prefixKey != key || cand.prefixLen == 0 {
				problems = append(problems, "record under the wrong prefix key: "+cand.id)
			}
		}
	}
	for _, rec := range ix.threads {
		want := 0
		if rec.prefixLen > 0 {
			want = 1
		}
		if seen[rec] != want {
			problems = append(problems, "live record indexed "+itoa(seen[rec])+" times: "+rec.id)
		}
	}
	return problems
}

// itoa is a tiny helper so the invariant messages do not pull in strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [8]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// threadTable is the StateDumper payload.
type threadTable struct {
	Threads   int         `json:"threads"`
	Busy      int         `json:"busy"`
	Pending   int         `json:"pending"`
	InFlight  int         `json:"inFlight"`
	BlockSize int         `json:"blockSizeTokens"`
	Rows      []threadRow `json:"rows"`
}

// threadRow is one tracked thread as the index holds it (its id, scope, spans and outcome).
type threadRow struct {
	ID          string  `json:"id"`
	Scope       string  `json:"scope,omitempty"`
	Blocks      int     `json:"blocks"`
	Tokens      int     `json:"tokens"`
	MsgCount    int     `json:"msgCount"`
	InFlight    int     `json:"inflight"`
	Pending     int     `json:"pending"`
	Forks       int     `json:"forks"`
	Turns       int     `json:"turns"`
	IdleSeconds float64 `json:"idleSeconds"`
	AgeSeconds  float64 `json:"ageSeconds"`
}

// Snapshot returns the thread table (bounded, newest first) for DumpState.
func (ix *threadIndex) Snapshot(now time.Time) threadTable {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	snap := threadTable{Threads: len(ix.threads), BlockSize: ix.blockSize}
	rows := make([]threadRow, 0, len(ix.threads))
	for _, rec := range ix.threads {
		rec.pruneMarkers(now, ix.markerTTL)
		if rec.outstanding() {
			snap.Busy++
		}
		snap.InFlight += len(rec.inflight)
		snap.Pending += len(rec.pending)
		rows = append(rows, threadRow{
			ID:          rec.id,
			Scope:       rec.scope,
			Blocks:      rec.fullBlocks,
			Tokens:      rec.tokens,
			MsgCount:    rec.msgCount,
			InFlight:    len(rec.inflight),
			Pending:     len(rec.pending),
			Forks:       rec.forks,
			Turns:       rec.turns,
			IdleSeconds: rec.idle(now).Seconds(),
			AgeSeconds:  now.Sub(rec.created).Seconds(),
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].IdleSeconds != rows[j].IdleSeconds {
			return rows[i].IdleSeconds < rows[j].IdleSeconds
		}
		return rows[i].ID < rows[j].ID
	})
	if len(rows) > maxSnapshotRows {
		rows = rows[:maxSnapshotRows]
	}
	snap.Rows = rows
	return snap
}

// startsWithReply is H1's content test: the message at the tip's message count is
// an assistant message, i.e. the request carries the reply to the tip's request.
// A request still in flight cannot have its reply in anyone's history, so the
// content proves the tip completed even when the busy flag is stale.
//
// The index is the tip's message count, which is exactly "the first message past
// the tip" whenever the match verifies the tip's whole prompt. When the tail
// tolerance absorbed a divergence instead, the request is a near-duplicate of the
// tip and the index can be off by the messages inside that window; the error then
// errs toward continuing — the safe direction (see the package doc).
//
// (Measured caveat, design H2: a framework that spawns a child after the parent's
// assistant tool-call turn puts that message into the child's history too, where
// it looks like a continuation.)
func startsWithReply(tipMsgCount int, roles []string) bool {
	if tipMsgCount <= 0 || tipMsgCount >= len(roles) {
		return false
	}
	return roles[tipMsgCount] == "assistant"
}

// sameScope reports whether a tip is visible to a request in scope. An absent
// scope on either side matches (no session header → no scoping); two different
// sessions never match.
func sameScope(threadScope, requestScope string) bool {
	return threadScope == "" || requestScope == "" || threadScope == requestScope
}

// threadID mints the opaque thread id: the first 10 hex chars of a hash over the
// mint seed.
func threadID(seed prefixhash.BlockHash) string {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(seed))
	sum := sha256.Sum256(b[:])
	return hex.EncodeToString(sum[:])[:idHexLen]
}

// mintSeed hashes the thread id's inputs: the tip commitment, the fork ordinal
// (siblings of the same content get distinct ids) and the session scope.
func mintSeed(tip prefixhash.BlockHash, ordinal int, scope string) prefixhash.BlockHash {
	h := sha256.New()
	var b [16]byte
	binary.LittleEndian.PutUint64(b[:8], uint64(tip))
	binary.LittleEndian.PutUint64(b[8:], uint64(ordinal)) //#nosec G115 -- ordinal is a bounded counter
	h.Write(b[:])
	h.Write([]byte(scope))
	return prefixhash.BlockHash(binary.LittleEndian.Uint64(h.Sum(nil)[:8]))
}
