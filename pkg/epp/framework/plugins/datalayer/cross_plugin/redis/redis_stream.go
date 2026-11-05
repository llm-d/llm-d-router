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

package redis

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	goredis "github.com/redis/go-redis/v9"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/pkg/common/clamp"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

const (
	windowReplicationFull        = "full"
	windowReplicationIncremental = "incremental"
	defaultMaxBytesPerSecond     = 8 << 20
	// markerPrefix names the hash field that holds a replica's stream marker.
	// Replica IDs start with a hostname, which cannot contain a colon.
	markerPrefix       = "stream:"
	streamElementField = "e"
	// maxStreamElementBytes bounds one decoded peer element.
	maxStreamElementBytes = 1 << 20
)

// streamCursor is what this replica's stream for one hash key holds: the
// elements at generations [start, tip) of its window, under epoch.
type streamCursor struct {
	epoch, start, tip uint64
	// size is the encoded size of the elements in [start, tip).
	size uint64
	// sizes[i] is the encoded size of generation start+i. It keeps size
	// updates O(trimmed+appended) when the window slides.
	sizes []uint32
	// reset makes the next publish delete the stream and start a new epoch.
	reset bool
	// unknown is the state a write tried to store when its outcome is
	// unknown. The next publish reads the stream length to decide.
	unknown *streamCursor
}

// streamMarker is the hash field value a stream publisher writes on every tick.
type streamMarker struct {
	epoch, base, tip, size uint64
}

func formatMarker(c streamCursor) string {
	return fmt.Sprintf("%d %d %d %d", c.epoch, c.start, c.tip, c.size)
}

func parseMarker(raw string) (streamMarker, error) {
	var m streamMarker
	parts := strings.Fields(raw)
	if len(parts) != 4 {
		return m, fmt.Errorf("stream marker %q has %d fields, want 4", raw, len(parts))
	}
	values := make([]uint64, len(parts))
	for i, part := range parts {
		v, err := strconv.ParseUint(part, 10, 64)
		if err != nil {
			return m, fmt.Errorf("stream marker %q: %w", raw, err)
		}
		values[i] = v
	}
	m = streamMarker{epoch: values[0], base: values[1], tip: values[2], size: values[3]}
	if m.tip < m.base {
		return m, fmt.Errorf("stream marker %q has tip below base", raw)
	}
	return m, nil
}

// streamPeer is the window rebuilt from one peer's stream.
type streamPeer struct {
	epoch uint64
	// tip is the generation of the next element to read.
	tip   uint64
	value fwkdl.SequencedValue
	// stale makes the next read start again from the peer's base.
	stale bool
}

type streamPublication struct {
	prev, next streamCursor
	xlen       *goredis.IntCmd
	// resetting means the publication deleted the stream.
	resetting bool
	// resolving means the publication only reads the stream length to settle
	// an unknown outcome.
	resolving bool
}

func markerField(replicaID string) string { return markerPrefix + replicaID }

// streamKey shares the hash tag of hashKey so one transaction stays in one
// cluster slot.
func (s *RedisSyncer) streamKey(hashKey, replicaID string) string {
	return "{" + hashKey + "}:stream:" + replicaID
}

// streamID maps generation gen to a stream entry ID. IDs must be above 0-0.
func streamID(gen uint64) string { return "0-" + strconv.FormatUint(gen+1, 10) }

func parseStreamID(id string) (uint64, error) {
	seq, ok := strings.CutPrefix(id, "0-")
	if !ok {
		return 0, fmt.Errorf("stream ID %q is not a generation", id)
	}
	n, err := strconv.ParseUint(seq, 10, 64)
	if err != nil || n == 0 {
		return 0, fmt.Errorf("stream ID %q is not a generation", id)
	}
	return n - 1, nil
}

func (s *RedisSyncer) loadCursor(hashKey string) (streamCursor, bool) {
	v, ok := s.cursors.Load(hashKey)
	if !ok {
		return streamCursor{}, false
	}
	return *v.(*streamCursor), true
}

func (s *RedisSyncer) storeCursor(hashKey string, c streamCursor) {
	s.cursors.Store(hashKey, &c)
}

// queueStream appends the elements of v this replica has not stored yet, up to
// the publish budget, trims elements v no longer holds, and rewrites the
// marker. TTLs are refreshed even when the budget is spent.
func (s *RedisSyncer) queueStream(ctx context.Context, pipe goredis.Pipeliner, key fwkdl.StateKey, hashKey string, v fwkdl.SequencedValue) (*streamPublication, []goredis.Cmder) {
	skey := s.streamKey(hashKey, s.replicaID)
	field := markerField(s.replicaID)
	base, tip := v.SequenceBase(), v.SequenceTip()
	cur, ok := s.loadCursor(hashKey)
	pub := &streamPublication{prev: cur}
	var cmds []goredis.Cmder
	ttls := func() {
		cmds = append(cmds,
			pipe.HPExpire(ctx, hashKey, s.stateTTL, field),
			pipe.PExpire(ctx, skey, s.stateTTL),
			pipe.PExpire(ctx, hashKey, s.stateTTL))
		pub.xlen = pipe.XLen(ctx, skey)
		cmds = append(cmds, pub.xlen)
	}
	if ok && cur.unknown != nil && !cur.reset {
		pub.resolving = true
		pub.next = *cur.unknown
		ttls()
		return pub, cmds
	}
	if !ok || cur.reset || base < cur.start || tip < cur.tip {
		if ok {
			streamResets.WithLabelValues(string(key)).Inc()
		}
		cur = streamCursor{epoch: s.epochs.Add(1), start: base, tip: base}
		pub.prev = cur
		pub.resetting = true
		cmds = append(cmds, pipe.Del(ctx, skey))
	}

	next := streamCursor{epoch: cur.epoch, start: base, tip: max(cur.tip, base)}
	switch {
	case base >= cur.tip:
		// Trimmed past everything published; size stays 0 until appends below.
	case base > cur.start:
		drop := clamp.Int(base - cur.start)
		next.size = cur.size
		for _, n := range cur.sizes[:drop] {
			next.size -= uint64(n)
		}
		next.sizes = append([]uint32(nil), cur.sizes[drop:]...)
	default:
		next.size = cur.size
		next.sizes = append([]uint32(nil), cur.sizes...)
	}
	sent := 0
	for gen := next.tip; gen < tip; gen++ {
		n := v.ElementSize(gen)
		if !s.publishBudget.take(n) {
			streamBudgetExhausted.WithLabelValues(string(key), directionPublish).Inc()
			break
		}
		cmds = append(cmds, pipe.XAdd(ctx, &goredis.XAddArgs{
			Stream: skey,
			ID:     streamID(gen),
			Values: []any{streamElementField, v.AppendElement(make([]byte, 0, n), gen)},
		}))
		next.tip = gen + 1
		next.sizes = append(next.sizes, clamp.Uint32(n))
		next.size += clamp.Uint64(n)
		sent += n
	}
	if sent > 0 {
		streamBytes.WithLabelValues(string(key), directionPublish).Add(float64(sent))
	}
	if base > cur.start && cur.tip > cur.start {
		cmds = append(cmds, pipe.XTrimMinID(ctx, skey, streamID(base)))
	}
	pub.next = next
	// A reset deletes the stream, so the marker must be rewritten even when
	// the window is empty. An unchanged window only refreshes TTLs.
	if pub.resetting || next.epoch != cur.epoch || next.start != cur.start || next.tip != cur.tip || next.size != cur.size {
		cmds = append(cmds, pipe.HSet(ctx, hashKey, field, formatMarker(next)))
	}
	ttls()
	return pub, cmds
}

// finishStream records what the transaction stored. err is the first command
// error for this endpoint.
func (s *RedisSyncer) finishStream(ctx context.Context, key fwkdl.StateKey, pub *publication, err error) {
	sp := pub.stream
	length := func(c streamCursor) int64 { return clamp.Int64(c.tip - c.start) }
	if sp.resolving {
		if err != nil {
			return
		}
		settled := sp.prev
		settled.unknown = nil
		n := sp.xlen.Val()
		switch {
		case length(sp.next) == length(sp.prev) && (sp.next.start != sp.prev.start || sp.next.tip != sp.prev.tip):
			settled.reset = true
		case n == length(sp.next):
			settled = sp.next
		case n == length(sp.prev):
		default:
			settled.reset = true
		}
		s.storeCursor(pub.hashKey, settled)
		return
	}
	if err != nil {
		failed := sp.prev
		failed.reset = true
		s.storeCursor(pub.hashKey, failed)
		return
	}
	if n := sp.xlen.Val(); n != length(sp.next) {
		if v := ctrl.LoggerFrom(ctx).V(logutil.DEBUG); v.Enabled() {
			v.Info("redis-syncer: stream length mismatch, starting a new epoch", "key", string(key), "endpoint", pub.endpointID, "length", n, "want", length(sp.next))
		}
		lost := sp.next
		lost.reset = true
		s.storeCursor(pub.hashKey, lost)
		return
	}
	s.storeCursor(pub.hashKey, sp.next)
}

// unknownStreamOutcome handles a transaction whose result is unknown.
func (s *RedisSyncer) unknownStreamOutcome(pub *publication) {
	sp := pub.stream
	if sp.resolving {
		return
	}
	c := sp.prev
	if sp.resetting {
		c.reset = true
	} else {
		next := sp.next
		c.unknown = &next
	}
	s.storeCursor(pub.hashKey, c)
}

// readsStreams reports whether the peer aggregate for pub comes from
// readStreams: a sequenced value in incremental mode, or one whose hash holds a
// peer marker or whose cached aggregate has stream peers.
func (s *RedisSyncer) readsStreams(pub *publication) bool {
	if _, ok := pub.value.(fwkdl.SequencedValue); !ok {
		return false
	}
	if s.incremental {
		return true
	}
	for field := range pub.result.Val() {
		if strings.HasPrefix(field, markerPrefix) {
			return true
		}
	}
	if v, ok := s.cache.Load(pub.hashKey); ok {
		return len(v.(*aggregateCacheEntry).streams) > 0
	}
	return false
}

type peerFetch struct {
	id       string
	marker   streamMarker
	from     uint64
	old      *streamPeer
	prev     fwkdl.SequencedValue
	reserved int
	cmd      *goredis.XMessageSliceCmd
}

type streamRead struct {
	pub     *publication
	legacy  []any
	next    map[string]*streamPeer
	fetches []*peerFetch
	changed bool
}

// readStreams rebuilds the peer aggregate for each publication from the
// markers and whole values in its hash, reading the stream elements each
// peer added since the last read in one pipeline for the chunk.
func (s *RedisSyncer) readStreams(ctx context.Context, key fwkdl.StateKey, aggregate func([]any) any, pubs []*publication) {
	logger := ctrl.LoggerFrom(ctx)
	pipe := s.client.Pipeline()
	reads := make([]*streamRead, 0, len(pubs))
	for _, pub := range pubs {
		reads = append(reads, s.planRead(ctx, pipe, key, pub))
	}
	if pipe.Len() > 0 {
		// Per-command errors are handled per fetch.
		_, _ = pipe.Exec(ctx)
	}
	for _, read := range reads {
		for _, fetch := range read.fetches {
			if s.applyFetch(ctx, key, read, fetch) {
				read.changed = true
			}
		}
		peers := s.storeStreamAggregate(read, aggregate)
		if v := logger.V(logutil.DEBUG); v.Enabled() {
			v.Info("redis-syncer: Set", "key", string(key), "endpoint", read.pub.endpointID, "replica", s.replicaID, "numReplicas", peers+1)
		}
	}
}

func (s *RedisSyncer) planRead(ctx context.Context, pipe goredis.Pipeliner, key fwkdl.StateKey, pub *publication) *streamRead {
	logger := ctrl.LoggerFrom(ctx)
	proto := pub.value.(fwkdl.SequencedValue)
	read := &streamRead{pub: pub, next: map[string]*streamPeer{}}
	var cached map[string]*streamPeer
	if v, ok := s.cache.Load(pub.hashKey); ok {
		cached = v.(*aggregateCacheEntry).streams
	}
	raw := pub.result.Val()
	markers := map[string]streamMarker{}
	for _, field := range slices.Sorted(maps.Keys(raw)) {
		if field == s.replicaID || field == markerField(s.replicaID) {
			continue
		}
		id, isMarker := strings.CutPrefix(field, markerPrefix)
		if !isMarker {
			value, err := gobDecode([]byte(raw[field]), pub.value)
			if err != nil {
				if v := logger.V(logutil.DEBUG); v.Enabled() {
					v.Info("redis-syncer: decode error", "field", field, "error", err)
				}
				continue
			}
			read.legacy = append(read.legacy, value)
			continue
		}
		m, err := parseMarker(raw[field])
		if err != nil {
			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("redis-syncer: marker parse error", "field", field, "error", err)
			}
			if old := cached[id]; old != nil {
				read.next[id] = old
			}
			continue
		}
		markers[id] = m
	}
	if len(read.legacy) > 0 {
		read.changed = true
	}
	for id := range cached {
		if _, kept := read.next[id]; !kept {
			if _, ok := markers[id]; !ok {
				read.changed = true
			}
		}
	}

	for _, id := range slices.Sorted(maps.Keys(markers)) {
		m := markers[id]
		old := cached[id]
		from, prev := m.base, fwkdl.SequencedValue(nil)
		if old != nil && old.epoch == m.epoch && !old.stale {
			from, prev = max(old.tip, m.base), old.value
		}
		if from >= m.tip {
			switch {
			case prev == nil:
				empty, _ := proto.NewSequence(m.base, nil)
				read.next[id] = &streamPeer{epoch: m.epoch, tip: m.base, value: empty}
				read.changed = true
			case m.base > prev.SequenceBase():
				advanced, _ := prev.Advance(m.base, nil)
				read.next[id] = &streamPeer{epoch: m.epoch, tip: from, value: advanced}
				read.changed = true
			default:
				read.next[id] = old
			}
			continue
		}
		if old != nil {
			read.next[id] = old
		}
		count, reserved, ok := s.reserveRead(m, from)
		if !ok {
			streamBudgetExhausted.WithLabelValues(string(key), directionRead).Inc()
			continue
		}
		read.fetches = append(read.fetches, &peerFetch{
			id: id, marker: m, from: from, old: old, prev: prev, reserved: reserved,
			cmd: pipe.XRangeN(ctx, s.streamKey(pub.hashKey, id), streamID(from), streamID(m.tip-1), count),
		})
	}
	return read
}

// reserveRead returns how many elements to request from a peer starting at
// from, sized by the marker's average element size, and the bytes reserved.
func (s *RedisSyncer) reserveRead(m streamMarker, from uint64) (int64, int, bool) {
	want := clamp.Int64(m.tip - from)
	if s.readBudget == nil {
		return want, 0, true
	}
	avg := max(clamp.Int64(m.size/(m.tip-m.base)), 1)
	avail := s.readBudget.available()
	if avail <= 0 {
		return 0, 0, false
	}
	count := min(max(int64(avail)/avg, 1), want)
	reserved := int(count * avg)
	s.readBudget.charge(reserved)
	return count, reserved, true
}

// applyFetch folds one peer read into read.next and reports whether the peer's
// window changed. A failed or malformed read keeps the previous window.
func (s *RedisSyncer) applyFetch(ctx context.Context, key fwkdl.StateKey, read *streamRead, fetch *peerFetch) bool {
	msgs, err := fetch.cmd.Result()
	received := 0
	for _, msg := range msgs {
		if elem, ok := msg.Values[streamElementField].(string); ok {
			received += len(elem)
		}
	}
	s.readBudget.charge(received - fetch.reserved)
	if received > 0 {
		streamBytes.WithLabelValues(string(key), directionRead).Add(float64(received))
	}
	if err != nil || len(msgs) == 0 {
		return false
	}
	first, elems, err := decodeEntries(msgs, fetch.from, fetch.marker.tip)
	var value fwkdl.SequencedValue
	if err == nil {
		proto := read.pub.value.(fwkdl.SequencedValue)
		switch {
		case fetch.prev == nil:
			value, err = proto.NewSequence(first, elems)
		case first > fetch.from:
			value, err = fetch.prev.Advance(first, elems)
		default:
			value, err = fetch.prev.Advance(fetch.marker.base, elems)
		}
	}
	if err != nil {
		if v := ctrl.LoggerFrom(ctx).V(logutil.DEBUG); v.Enabled() {
			v.Info("redis-syncer: peer stream rejected, rereading from its base", "key", string(key), "endpoint", read.pub.endpointID, "peer", fetch.id, "error", err)
		}
		if fetch.old != nil {
			stale := *fetch.old
			stale.stale = true
			read.next[fetch.id] = &stale
		}
		return false
	}
	tip := first + uint64(len(elems))
	streamPeerLag.WithLabelValues(string(key)).Observe(float64(fetch.marker.tip - tip))
	read.next[fetch.id] = &streamPeer{epoch: fetch.marker.epoch, tip: tip, value: value}
	return true
}

// decodeEntries checks that msgs are consecutive generations starting at or
// after from and below tip, and returns the first generation and the element
// bytes. A first generation above from means the peer trimmed those elements.
func decodeEntries(msgs []goredis.XMessage, from, tip uint64) (uint64, [][]byte, error) {
	first, err := parseStreamID(msgs[0].ID)
	if err != nil {
		return 0, nil, err
	}
	if first < from {
		return 0, nil, fmt.Errorf("stream entry %d is before %d", first, from)
	}
	elems := make([][]byte, len(msgs))
	for i, msg := range msgs {
		gen, err := parseStreamID(msg.ID)
		if err != nil {
			return 0, nil, err
		}
		if gen != first+uint64(i) || gen >= tip {
			return 0, nil, fmt.Errorf("stream entry %d out of sequence", gen)
		}
		elem, ok := msg.Values[streamElementField].(string)
		if !ok || len(elem) > maxStreamElementBytes {
			return 0, nil, fmt.Errorf("stream entry %d is missing or larger than %d bytes", gen, maxStreamElementBytes)
		}
		elems[i] = []byte(elem)
	}
	return first, elems, nil
}

// storeStreamAggregate caches the aggregate of read's peers and returns how
// many peers it holds. An unchanged peer set keeps the cached aggregate.
func (s *RedisSyncer) storeStreamAggregate(read *streamRead, aggregate func([]any) any) int {
	hashKey := read.pub.hashKey
	peers := len(read.next) + len(read.legacy)
	if peers == 0 {
		s.cache.Delete(hashKey)
		return 0
	}
	var value any
	if prev, ok := s.cache.Load(hashKey); ok && !read.changed {
		value = prev.(*aggregateCacheEntry).value
	} else {
		values := make([]any, 0, peers)
		for _, id := range slices.Sorted(maps.Keys(read.next)) {
			values = append(values, read.next[id].value)
		}
		values = append(values, read.legacy...)
		// Get aggregates the cached value with the local one, so one peer needs
		// no union of its own.
		if len(values) == 1 {
			value = values[0]
		} else {
			value = aggregate(values)
		}
	}
	s.cache.Store(hashKey, &aggregateCacheEntry{value: value, streams: read.next})
	return peers
}
