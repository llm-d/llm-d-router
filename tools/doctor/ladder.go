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

package main

import (
	"cmp"
	"container/list"
	"errors"
	"fmt"
	"io"

	"github.com/llm-d/llm-d-router/pkg/epp/scheduling/capture"
)

// rung is a prefix-cache hit count over the prompt blocks it was measured on.
type rung struct {
	hits, blocks int
}

func (r *rung) add(hits, blocks int) {
	r.hits += hits
	r.blocks += blocks
}

func (r rung) String() string {
	if r.blocks == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%%", 100*float64(r.hits)/float64(r.blocks))
}

// ladder holds each rung of the capture's hit-rate ladder. Records that
// failed scheduling or carry no block hashes are counted and left out.
type ladder struct {
	records, failed, noHashes, endpoints int

	ideal    rung // unbounded cache shared by the fleet
	capacity rung // one LRU cache with the fleet's capacity
	perfect  rung // per-endpoint LRU caches, each request on the candidate holding its longest prefix
	routed   rung // per-endpoint LRU caches, each request on the router's pick

	predicted        rung // the router's predicted hit on its pick
	predictedRecords int
	engine           rung // the engine's reported cached tokens
	engineRecords    int
}

// computeLadder replays records, in time order, through cache models with
// fleetBlocks KV blocks in total, split evenly across the endpoints seen in
// the capture.
func computeLadder(records []capture.Record, fleetBlocks int) (ladder, error) {
	var l ladder
	if err := checkComparable(records); err != nil {
		return l, err
	}
	endpoints := map[string]bool{}
	for _, r := range records {
		for _, e := range r.Candidates {
			endpoints[e] = true
		}
		if r.Endpoint != "" {
			endpoints[r.Endpoint] = true
		}
	}
	l.endpoints = len(endpoints)
	if fleetBlocks < 1 || (l.endpoints > 0 && fleetBlocks/l.endpoints < 1) {
		return l, fmt.Errorf("--fleet-kv-blocks must give each of the %d endpoints at least one block", l.endpoints)
	}

	seen := map[uint64]bool{}
	shared := newLRU(fleetBlocks)
	perfect, routed := map[string]*lru{}, map[string]*lru{}
	for e := range endpoints {
		perfect[e] = newLRU(fleetBlocks / l.endpoints)
		routed[e] = newLRU(fleetBlocks / l.endpoints)
	}

	for _, r := range records {
		switch {
		case r.Endpoint == "":
			l.failed++
			continue
		case len(r.BlockHashes) == 0:
			l.noHashes++
			continue
		}
		l.records++
		blocks := r.BlockHashes
		n := len(blocks)

		hits := 0
		for hits < n && seen[blocks[hits]] {
			hits++
		}
		l.ideal.add(hits, n)
		for _, h := range blocks {
			seen[h] = true
		}

		l.capacity.add(shared.prefix(blocks), n)
		shared.store(blocks)

		candidates := r.Candidates
		if len(candidates) == 0 {
			candidates = []string{r.Endpoint}
		}
		best, bestHits := perfectPick(perfect, candidates, blocks)
		l.perfect.add(bestHits, n)
		perfect[best].store(blocks)

		l.routed.add(routed[r.Endpoint].prefix(blocks), n)
		routed[r.Endpoint].store(blocks)

		if p := r.PredictedCachedBlocks; p != nil {
			l.predicted.add(min(max(*p, 0), n), n)
			l.predictedRecords++
		}
		if t := r.EngineCachedTokens; t != nil {
			l.engine.add(min(max(*t, 0)/r.BlockSize, n), n)
			l.engineRecords++
		}
	}
	if l.records == 0 {
		return l, errors.New("no record with block hashes and an endpoint")
	}
	return l, nil
}

// perfectPick returns the candidate whose cache holds the longest prefix of
// blocks. Ties go to the cache holding fewer blocks, then to the lower name.
func perfectPick(caches map[string]*lru, candidates []string, blocks []uint64) (string, int) {
	best, bestHits := "", 0
	for _, e := range candidates {
		h := caches[e].prefix(blocks)
		if best == "" || cmp.Or(
			cmp.Compare(bestHits, h),
			cmp.Compare(caches[e].order.Len(), caches[best].order.Len()),
			cmp.Compare(e, best),
		) < 0 {
			best, bestHits = e, h
		}
	}
	return best, bestHits
}

// checkComparable rejects records of another schema version, and replayed
// records whose hashes can't be compared with the first replayed record's.
func checkComparable(records []capture.Record) error {
	var first *capture.Record
	for i := range records {
		r := &records[i]
		if r.SchemaVersion != capture.SchemaVersion {
			return fmt.Errorf("request %s: schema version %d, this tool reads %d", r.RequestID, r.SchemaVersion, capture.SchemaVersion)
		}
		if r.Endpoint == "" || len(r.BlockHashes) == 0 {
			continue // skipped by computeLadder, so its hash metadata may be unset
		}
		if first == nil {
			first = r
		}
		if r.HashScheme != first.HashScheme {
			return fmt.Errorf("request %s: hash scheme %q differs from %q", r.RequestID, r.HashScheme, first.HashScheme)
		}
		if r.BlockSize != first.BlockSize {
			return fmt.Errorf("request %s: block size %d differs from %d", r.RequestID, r.BlockSize, first.BlockSize)
		}
		if r.BlockSize < 1 {
			return fmt.Errorf("request %s: block size %d", r.RequestID, r.BlockSize)
		}
	}
	return nil
}

func (l ladder) write(w io.Writer) error {
	coverage := func(n int) string {
		if n == 0 || n == l.records {
			return ""
		}
		return fmt.Sprintf(" (%d of %d records)", n, l.records)
	}
	rows := []struct{ name, model, rate string }{
		{"Ideal", "unbounded shared cache", l.ideal.String()},
		{"Capacity bound", "fleet KV capacity, one shared cache", l.capacity.String()},
		{"Perfect routing", "same capacity split per endpoint", l.perfect.String()},
		{"Routed", "the router's picks, same caches", l.routed.String()},
		{"Observed (router)", "router-predicted cached blocks", l.predicted.String() + coverage(l.predictedRecords)},
		{"Observed (engine)", "engine-reported cached tokens", l.engine.String() + coverage(l.engineRecords)},
	}
	if _, err := fmt.Fprintln(w, "LADDER (prefix-cache hit rate, in blocks)"); err != nil {
		return err
	}
	for _, row := range rows {
		if _, err := fmt.Fprintf(w, "  %-17s  %-36s %6s\n", row.name, row.model, row.rate); err != nil {
			return err
		}
	}
	_, err := fmt.Fprintf(w, "records: %d (%d failed, %d without block hashes), endpoints: %d, blocks: %d\n",
		l.records, l.failed, l.noHashes, l.endpoints, l.ideal.blocks)
	return err
}

// lru is a block cache that evicts the least recently stored block.
type lru struct {
	capacity int
	order    *list.List // front is the most recent
	blocks   map[uint64]*list.Element
}

func newLRU(capacity int) *lru {
	return &lru{capacity: capacity, order: list.New(), blocks: map[uint64]*list.Element{}}
}

// prefix returns how many leading blocks are cached.
func (c *lru) prefix(blocks []uint64) int {
	n := 0
	for n < len(blocks) && c.blocks[blocks[n]] != nil {
		n++
	}
	return n
}

// store caches blocks after a request ran. They are stored last to first so
// that a request's trailing blocks are evicted before its leading blocks, in
// the order the engine frees a finished request's blocks.
func (c *lru) store(blocks []uint64) {
	for i := len(blocks) - 1; i >= 0; i-- {
		h := blocks[i]
		if e := c.blocks[h]; e != nil {
			c.order.MoveToFront(e)
			continue
		}
		c.blocks[h] = c.order.PushFront(h)
		if c.order.Len() > c.capacity {
			oldest := c.order.Back()
			c.order.Remove(oldest)
			delete(c.blocks, oldest.Value.(uint64))
		}
	}
}
