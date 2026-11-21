// Copyright 2026 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sync"
)

// Wire-event recording and replay. The recorder taps the pool's receive
// path (the subscriber's decoded-batch handoff), appending every event
// message to a length-delimited file sink; the replayer feeds a recording
// back through the pool, so dedup and parent resolution run exactly as
// they did live. A recording holds wire events only: subscriber-inferred
// control messages (restart resets, retires) are transport state, not
// events, and are neither recorded nor reconstructed on replay.
//
// File format: a sequence of records, each a 4-byte big-endian length
// followed by that many bytes of
//
//	uvarint(len(topic)) topic
//	uint64 big-endian   sequence
//	uvarint(len(source)) source
//	payload             (the remainder)
//
// A truncated final record (a crash mid-write) is an error; the records
// before it replay normally.

// maxRecordSize bounds one record's declared length before allocation, so
// a corrupt length cannot request an unbounded read. Event batches are at
// most tens of kilobytes; the bound leaves orders of magnitude of slack.
const maxRecordSize = 1 << 26

// eventRecorder appends wire events to a file sink. Once a write fails,
// recording stops and the error is surfaced by Close; a sink that cannot
// be written must not take the event path down with it.
type eventRecorder struct {
	mu     sync.Mutex
	f      *os.File
	w      *bufio.Writer
	failed error
}

func newEventRecorder(path string) (*eventRecorder, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	return &eventRecorder{f: f, w: bufio.NewWriter(f)}, nil
}

// record appends one message. Control messages are not wire events.
func (r *eventRecorder) record(msg *RawMessage) {
	if msg.reset {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.failed != nil {
		return
	}
	var head [binary.MaxVarintLen64]byte

	var buf []byte
	n := binary.PutUvarint(head[:], uint64(len(msg.Topic)))
	buf = append(buf, head[:n]...)
	buf = append(buf, msg.Topic...)
	var seq [8]byte
	binary.BigEndian.PutUint64(seq[:], msg.Sequence)
	buf = append(buf, seq[:]...)
	n = binary.PutUvarint(head[:], uint64(len(msg.SourceEndpoint)))
	buf = append(buf, head[:n]...)
	buf = append(buf, msg.SourceEndpoint...)
	buf = append(buf, msg.Payload...)

	var lenPrefix [4]byte
	binary.BigEndian.PutUint32(lenPrefix[:], uint32(len(buf))) //#nosec G115 -- a message this large cannot fit in memory
	if _, err := r.w.Write(lenPrefix[:]); err != nil {
		r.failed = err
		return
	}
	if _, err := r.w.Write(buf); err != nil {
		r.failed = err
	}
}

// Close flushes the sink and reports the first write failure, if any.
func (r *eventRecorder) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.failed != nil {
		_ = r.f.Close()
		return r.failed
	}
	if err := r.w.Flush(); err != nil {
		r.failed = err
		_ = r.f.Close()
		return err
	}
	return r.f.Close()
}

// forEachRecord decodes a recording and calls fn for each message in
// record order. A partial final record is an error.
func forEachRecord(r io.Reader, fn func(*RawMessage) error) error {
	br := bufio.NewReader(r)
	for {
		var lenPrefix [4]byte
		if _, err := io.ReadFull(br, lenPrefix[:]); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		length := binary.BigEndian.Uint32(lenPrefix[:])
		if length > maxRecordSize {
			return fmt.Errorf("record length %d exceeds %d", length, maxRecordSize)
		}
		buf := make([]byte, length)
		if _, err := io.ReadFull(br, buf); err != nil {
			return fmt.Errorf("truncated record of %d bytes: %w", length, err)
		}

		msg := &RawMessage{}
		rest := buf
		topicLen, n := binary.Uvarint(rest)
		if n <= 0 || uint64(len(rest)-n) < topicLen {
			return fmt.Errorf("malformed record topic")
		}
		rest = rest[n:]
		msg.Topic = string(rest[:topicLen])
		rest = rest[topicLen:]
		if len(rest) < 8 {
			return fmt.Errorf("malformed record sequence")
		}
		msg.Sequence = binary.BigEndian.Uint64(rest[:8])
		rest = rest[8:]
		srcLen, n := binary.Uvarint(rest)
		if n <= 0 || uint64(len(rest)-n) < srcLen {
			return fmt.Errorf("malformed record source")
		}
		rest = rest[n:]
		msg.SourceEndpoint = string(rest[:srcLen])
		rest = rest[srcLen:]
		msg.Payload = rest

		if err := fn(msg); err != nil {
			return err
		}
	}
}

// ReplayFile feeds a recording's wire events back through the pool in
// record order. The pool's per-source sharding preserves each source's
// ordering; workers must be running (Start) for the events to process.
// Replayed messages are not re-recorded.
func (p *Pool) ReplayFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return forEachRecord(f, func(msg *RawMessage) error {
		p.dispatch(msg)
		return nil
	})
}
