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

// Package workload generates deterministic KV-block index operation streams
// for tests, benchmarks, and the parity harness. It is test support with no
// production behavior, in a regular package so consumers in pkg/kvcache and
// pkg/kvevents can import it.
package workload

import (
	"fmt"
	"math/rand"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// OpKind labels one operation in a replay stream.
type OpKind int

const (
	// AddOp stores engine and request keys with pod entries.
	AddOp OpKind = iota
	// EvictOp removes entries for engine or request keys.
	EvictOp
	// ClearOp removes every entry of one pod.
	ClearOp
	// MatchOp carries a query whose result is compared across backends or
	// read paths; it does not mutate the index.
	MatchOp
)

// Op is one operation in a deterministic replay stream.
type Op struct {
	Kind        OpKind
	EngineKeys  []kvblock.BlockHash
	RequestKeys []kvblock.BlockHash
	Entries     []kvblock.PodEntry
	// KeyType selects how EvictOp keys resolve.
	KeyType kvblock.KeyType
	// Pod names the pod a ClearOp clears.
	Pod string
	// Query is the MatchOp key sequence; Filter optionally restricts pods.
	Query  []kvblock.BlockHash
	Filter []string
}

// Profile parameterizes a generated workload. The zero value is not usable;
// Agentic and Interactive provide the two design workloads with knobs.
type Profile struct {
	Name string
	// Pods is the number of distinct pod identifiers.
	Pods int
	// EntriesPerKey is the number of pods holding each block (capped at Pods).
	EntriesPerKey int
	// Depth is the starting prefix length in blocks; each turn grows it.
	Depth int
	// Turns is the conversation turns before moving to the next conversation.
	Turns int
	// Conversations is the number of sequential conversations.
	Conversations int
	// Grow is the blocks appended per turn.
	Grow int
	// EngineRatio is engine keys per request key: 1 is the canonical
	// geometry, above 1 is sub-canonical (many engine blocks per canonical
	// block).
	EngineRatio int
	// WriteRatio is the per-turn chance of a wire Evict of an earlier block.
	WriteRatio float64
	// SpecChance is the per-turn chance of a speculative tail Add.
	SpecChance float64
	// ClearChance is the per-conversation chance of a pod rollover
	// (AllBlocksCleared, after which the pod keeps serving and re-Adds).
	ClearChance float64
	// GroupChance is the per-entry chance of a group tag.
	GroupChance float64
	// Tiers is the device-tier mix entries draw from.
	Tiers []string
	// Seed drives every random choice; the same seed replays identically.
	Seed int64
}

// Agentic is the deep-prefix, read-dominated design workload: long
// conversations, few writes, mostly gpu with some cpu offload.
func Agentic(pods, entriesPerKey, depth int) Profile {
	return Profile{
		Name:          "agentic",
		Pods:          pods,
		EntriesPerKey: entriesPerKey,
		Depth:         depth,
		Turns:         10,
		Conversations: 8,
		Grow:          1,
		EngineRatio:   1,
		WriteRatio:    0.1,
		SpecChance:    0.2,
		ClearChance:   0.05,
		GroupChance:   0.05,
		Tiers:         []string{"gpu", "gpu", "gpu", "cpu"},
		Seed:          1,
	}
}

// Interactive is the shallow, write-heavy design workload: short
// conversations, frequent evictions and re-writes, single tier.
func Interactive(pods, entriesPerKey, depth int) Profile {
	return Profile{
		Name:          "interactive",
		Pods:          pods,
		EntriesPerKey: entriesPerKey,
		Depth:         depth,
		Turns:         2,
		Conversations: 64,
		Grow:          1,
		EngineRatio:   1,
		WriteRatio:    0.5,
		SpecChance:    0.1,
		ClearChance:   0.1,
		GroupChance:   0.02,
		Tiers:         []string{"gpu"},
		Seed:          1,
	}
}

// addLogEntry records one Add so wire evicts can reference keys and entries
// that exist in the index.
type addLogEntry struct {
	engine  kvblock.BlockHash // EmptyBlockHash when the add carried none
	request kvblock.BlockHash
	entries []kvblock.PodEntry
}

// Generator produces a Profile's operation stream deterministically. Each
// turn emits, in order: the turn's Add, an optional speculative tail Add,
// an optional wire Evict of an earlier block, and a Match over the whole
// prefix. A conversation that opens with a rollover emits a Clear first,
// after which the pod keeps serving and the turn's Add re-covers the whole
// prefix.
type Generator struct {
	p      Profile
	rng    *rand.Rand
	keys   []kvblock.BlockHash // current conversation prefix
	addLog []addLogEntry
	conv   uint64
	turn   int
	step   int // position within the turn's op sequence
	rolled bool
	// open marks a conversation whose starting prefix the next Add must
	// establish.
	open bool
}

// addLogCap bounds the eviction reference log; older adds fall out of scope.
const addLogCap = 1024

// New returns a generator over p with its knobs normalized.
func New(p Profile) *Generator {
	if p.Pods <= 0 {
		p.Pods = 1
	}
	if p.EntriesPerKey <= 0 {
		p.EntriesPerKey = 1
	}
	if p.EntriesPerKey > p.Pods {
		p.EntriesPerKey = p.Pods
	}
	if p.Depth <= 0 {
		p.Depth = 1
	}
	if p.Turns <= 0 {
		p.Turns = 1
	}
	if p.Conversations <= 0 {
		p.Conversations = 1
	}
	if p.Grow <= 0 {
		p.Grow = 1
	}
	if p.EngineRatio <= 0 {
		p.EngineRatio = 1
	}
	if len(p.Tiers) == 0 {
		p.Tiers = []string{"gpu"}
	}
	return &Generator{p: p, rng: rand.New(rand.NewSource(p.Seed))}
}

// PodName returns the i-th pod identifier, shared by every consumer so
// filter sets name the same pods the entries do.
func PodName(i int) string {
	return fmt.Sprintf("10.0.%d.%d:8000", i/256, i%256)
}

// blockKey derives a non-zero block hash from the seed, conversation, and
// position, so identical inputs replay identically.
func blockKey(seed int64, conv, idx uint64) kvblock.BlockHash {
	x := uint64(seed)*0x9e3779b97f4a7c15 ^ (conv+1)*0xbf58476d1ce4e5b9 ^ (idx+1)*0x94d049bb133111eb
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return kvblock.BlockHash(x | 1) // never EmptyBlockHash
}

// engineKey derives the j-th engine key covering request-key position idx.
func engineKey(seed int64, conv, idx, j uint64) kvblock.BlockHash {
	return blockKey(seed^0x5bf03635c52811b7, conv, idx<<16+j)
}

// Next returns the next operation. The second return is false once the
// profile's conversations are exhausted.
func (g *Generator) Next() (Op, bool) {
	p := g.p
	if g.keys == nil {
		if g.conv >= uint64(p.Conversations) {
			return Op{}, false
		}
		g.keys = make([]kvblock.BlockHash, 0, p.Depth+p.Turns*p.Grow)
		g.turn, g.step, g.addLog, g.open = 0, 0, g.addLog[:0], true
		if g.conv > 0 && g.rng.Float64() < p.ClearChance {
			g.rolled = true
			return Op{Kind: ClearOp, Pod: PodName(g.rng.Intn(p.Pods))}, true
		}
	}

	for g.step < 4 {
		s := g.step
		g.step++
		switch s {
		case 0: // the turn's Add
			g.turn++
			grow := p.Grow
			if g.open {
				// The conversation's first Add establishes the starting prefix.
				grow += p.Depth
				g.open = false
			}
			from := len(g.keys)
			g.keys = append(g.keys, g.prefix(grow)...)
			// After a rollover Clear the whole prefix re-Adds, not just the tail.
			start := from
			if g.rolled {
				start, g.rolled = 0, false
			}
			return g.addOp(start, len(g.keys)), true
		case 1: // speculative tail: the next block, no engine keys
			if g.p.SpecChance > 0 && g.rng.Float64() < g.p.SpecChance {
				key := blockKey(p.Seed, g.conv, uint64(len(g.keys)))
				return Op{
					Kind:        AddOp,
					RequestKeys: []kvblock.BlockHash{key},
					Entries:     g.entries(len(g.keys)),
				}, true
			}
		case 2: // wire evict of an earlier block
			if len(g.addLog) > 0 && g.rng.Float64() < g.p.WriteRatio {
				rec := g.addLog[g.rng.Intn(len(g.addLog))]
				if g.rng.Intn(2) == 0 && rec.engine != kvblock.EmptyBlockHash {
					return Op{
						Kind:       EvictOp,
						KeyType:    kvblock.EngineKey,
						EngineKeys: []kvblock.BlockHash{rec.engine},
						Entries:    rec.entries,
					}, true
				}
				return Op{
					Kind:        EvictOp,
					KeyType:     kvblock.RequestKey,
					RequestKeys: []kvblock.BlockHash{rec.request},
					Entries:     rec.entries,
				}, true
			}
		case 3: // match over the whole prefix
			query := append([]kvblock.BlockHash(nil), g.keys...)
			if g.turn >= p.Turns {
				// Conversation over; the next call opens a fresh one.
				g.conv++
				g.keys = nil
			} else {
				g.step = 0
			}
			return Op{Kind: MatchOp, Query: query}, true
		}
	}
	return Op{}, false
}

// addOp builds an Add over the request keys [start, end) of the prefix.
func (g *Generator) addOp(start, end int) Op {
	p := g.p
	req := append([]kvblock.BlockHash(nil), g.keys[start:end]...)
	eng := req
	if p.EngineRatio > 1 {
		eng = make([]kvblock.BlockHash, len(req)*p.EngineRatio)
		for i := range req {
			for j := 0; j < p.EngineRatio; j++ {
				eng[i*p.EngineRatio+j] = engineKey(p.Seed, g.conv, uint64(start+i), uint64(j))
			}
		}
	}
	op := Op{Kind: AddOp, EngineKeys: eng, RequestKeys: req, Entries: g.entries(start)}
	for i, rk := range req {
		var ek kvblock.BlockHash
		if p.EngineRatio > 1 {
			ek = eng[i*p.EngineRatio]
		} else {
			ek = rk
		}
		g.addLog = append(g.addLog, addLogEntry{engine: ek, request: rk, entries: op.Entries})
		if len(g.addLog) > addLogCap {
			g.addLog = g.addLog[1:]
		}
	}
	return op
}

// prefix derives count fresh keys continuing the current prefix.
func (g *Generator) prefix(count int) []kvblock.BlockHash {
	out := make([]kvblock.BlockHash, count)
	for i := range out {
		out[i] = blockKey(g.p.Seed, g.conv, uint64(len(g.keys)+i))
	}
	return out
}

// entries picks the pods holding the block at position from.
func (g *Generator) entries(from int) []kvblock.PodEntry {
	p := g.p
	entries := make([]kvblock.PodEntry, p.EntriesPerKey)
	for i := range entries {
		// Rotate the pod set per position so prefixes overlap across pods
		// without every key holding an identical set.
		pod := (uint64(from)*31 + uint64(i) + g.conv*7) % uint64(p.Pods)
		e := kvblock.PodEntry{
			PodIdentifier: PodName(int(pod)),
			DeviceTier:    p.Tiers[g.rng.Intn(len(p.Tiers))],
		}
		if p.GroupChance > 0 && g.rng.Float64() < p.GroupChance {
			e.HasGroup = true
			e.GroupIdx = kvblock.GroupID(1 + g.rng.Intn(3))
		}
		entries[i] = e
	}
	return entries
}
