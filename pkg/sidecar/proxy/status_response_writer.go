/*
Copyright 2025 The llm-d Authors.

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

package proxy

import (
	"bytes"
	"net/http"
	"sync"
	"sync/atomic"
)

const sseEventDelimiter = "\n\n"

// bufferedResponseWriter receives responses from prefillers
type bufferedResponseWriter struct {
	headers    http.Header
	buffer     bytes.Buffer
	statusCode int
}

func (w *bufferedResponseWriter) Header() http.Header {
	if w.headers == nil {
		w.headers = make(http.Header)
	}
	return w.headers
}

func (w *bufferedResponseWriter) Write(b []byte) (int, error) {
	if w.statusCode == 0 {
		w.statusCode = http.StatusOK
	}
	return w.buffer.Write(b)
}

func (w *bufferedResponseWriter) WriteHeader(statusCode int) {
	w.statusCode = statusCode
}

// bodyBytes returns the buffered body without copying.
func (w *bufferedResponseWriter) bodyBytes() []byte {
	return w.buffer.Bytes()
}

// deferredCommitWriter wraps a client http.ResponseWriter and holds all writes
// until the caller decides the outcome (the "commit point"). It is used by the
// MoRI-IO parallel WRITE dispatch so decode can run concurrently with prefill
// yet never surface a response to the client before prefill has succeeded:
//
//   - commit(): relay the buffered status/headers/body to the client and switch
//     to pass-through mode, so any further decode writes stream straight to the
//     client (SSE/streaming is preserved after the commit point).
//   - abort():  discard the buffered output and drop every subsequent write, so
//     a failed prefill can never let decode emit a (possibly 200) response; the
//     caller then writes the prefill error to the client directly.
//
// It is safe for concurrent use by the decode goroutine (Write/WriteHeader/
// Flush) and the coordinating goroutine (commit/abort).
type deferredCommitWriter struct {
	dst http.ResponseWriter

	mu sync.Mutex
	// committed: caller decided prefill succeeded, decode's response may reach
	// the client. aborted: caller decided prefill failed, decode is discarded.
	committed bool
	aborted   bool
	// wroteHeader records that decode has produced its status/headers.
	wroteHeader bool
	// headerFlushed records that decode's status/headers have been relayed to
	// dst; after this, writes stream straight through.
	headerFlushed bool
	statusCode    int
	header        http.Header
	buffer        bytes.Buffer
}

func newDeferredCommitWriter(dst http.ResponseWriter) *deferredCommitWriter {
	return &deferredCommitWriter{
		dst:    dst,
		header: make(http.Header),
	}
}

func (w *deferredCommitWriter) Header() http.Header {
	w.mu.Lock()
	defer w.mu.Unlock()
	// Once decode's headers have been relayed, hand back the real writer's
	// header map. Until then (buffering, or committed but decode hasn't emitted
	// its header yet) collect into our own map so we can relay it verbatim.
	if w.headerFlushed {
		return w.dst.Header()
	}
	return w.header
}

func (w *deferredCommitWriter) WriteHeader(statusCode int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.aborted {
		return
	}
	if !w.wroteHeader {
		w.wroteHeader = true
		w.statusCode = statusCode
	}
	// If we've already committed, decode's header is now known -> relay it.
	if w.committed && !w.headerFlushed {
		w.flushCommitLocked()
	}
}

func (w *deferredCommitWriter) Write(b []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.aborted {
		// Response was discarded (prefill failed); silently drop decode output.
		return len(b), nil
	}
	if !w.wroteHeader {
		w.wroteHeader = true
		if w.statusCode == 0 {
			w.statusCode = http.StatusOK
		}
	}
	if w.committed {
		if !w.headerFlushed {
			w.flushCommitLocked()
		}
		return w.dst.Write(b)
	}
	return w.buffer.Write(b)
}

// Flush only propagates once decode's response has been relayed to the client;
// while buffering it is a no-op so nothing reaches the client before the commit
// point.
func (w *deferredCommitWriter) Flush() {
	w.mu.Lock()
	streaming := w.headerFlushed && !w.aborted
	w.mu.Unlock()
	if !streaming {
		return
	}
	if f, ok := w.dst.(http.Flusher); ok {
		f.Flush()
	}
}

// flushCommitLocked relays decode's buffered status/headers/body to dst and
// marks the response as flushed so subsequent writes stream through. Caller must
// hold w.mu and must have set committed.
func (w *deferredCommitWriter) flushCommitLocked() {
	if w.headerFlushed {
		return
	}
	dstHeader := w.dst.Header()
	for key, values := range w.header {
		dstHeader[key] = values
	}
	status := w.statusCode
	if status == 0 {
		status = http.StatusOK
	}
	w.headerFlushed = true
	w.dst.WriteHeader(status)
	if w.buffer.Len() > 0 {
		_, _ = w.dst.Write(w.buffer.Bytes())
		w.buffer.Reset()
	}
	if f, ok := w.dst.(http.Flusher); ok {
		f.Flush()
	}
}

// commit allows decode's response to reach the client, preserving decode's own
// status, headers, and body. If decode has already produced output it is flushed
// immediately; otherwise the flush is deferred until decode emits its header, so
// a fast-succeeding prefill never clobbers decode's status/headers with a
// synthesized default. Returns false if the writer had already been aborted.
func (w *deferredCommitWriter) commit() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.aborted {
		return false
	}
	if w.committed {
		return true
	}
	w.committed = true
	// Only flush now if decode already emitted its header/body; otherwise wait
	// for decode's WriteHeader/Write so we relay its real status and headers.
	if w.wroteHeader || w.buffer.Len() > 0 {
		w.flushCommitLocked()
	}
	return true
}

// abort discards buffered output and drops all future writes. It is a no-op if
// decode's response has already been relayed to the client (which cannot happen
// in the current flow, since the caller only aborts before committing).
func (w *deferredCommitWriter) abort() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.headerFlushed {
		return
	}
	w.aborted = true
	w.buffer.Reset()
}

type flushableResponseWriter interface {
	http.ResponseWriter
	http.Flusher
}

// responseWriterWithBuffer wraps an http.ResponseWriter to buffer initial writes.
// Start in buffer mode to inspect the first chunk, then call flushBufferAndGoDirect()
// to write buffered content and switch to direct pass-through mode.
type responseWriterWithBuffer struct {
	writerFlusher flushableResponseWriter

	// buffering is checked atomically to allow lock-free fast paths
	// in direct mode (Write and Flush).
	buffering atomic.Bool

	// mu protects buffer, statusCode, and wroteHeader during buffering mode
	// and during the transition to direct mode.
	mu          sync.Mutex
	buffer      bytes.Buffer
	statusCode  int
	wroteHeader bool

	// ready receives an error (or nil) when the first Write happens,
	// signaling that there's data available for inspection or an error occurred.
	ready     chan struct{}
	readyOnce sync.Once
}

// newResponseWriterWithBuffer creates a new writer starting in buffer mode.
func newResponseWriterWithBuffer(w flushableResponseWriter) *responseWriterWithBuffer {
	rw := &responseWriterWithBuffer{
		writerFlusher: w,
		ready:         make(chan struct{}, 1), // buffered to avoid blocking sender
	}
	rw.buffering.Store(true)
	return rw
}

func (w *responseWriterWithBuffer) Header() http.Header {
	return w.writerFlusher.Header()
}

func (w *responseWriterWithBuffer) Write(b []byte) (int, error) {
	if !w.buffering.Load() {
		return w.writerFlusher.Write(b)
	}

	// Buffering mode, need lock to protect buffer
	w.mu.Lock()
	defer w.mu.Unlock()

	// Double-check after acquiring lock (may have transitioned to direct mode)
	if !w.buffering.Load() {
		return w.writerFlusher.Write(b)
	}

	if w.statusCode == 0 {
		w.statusCode = http.StatusOK
	}

	// Write() always returns a nil error
	n, _ := w.buffer.Write(b)

	// Signal ready when buffer contains at least 2 complete SSE events.
	// For SSE streaming, the first chunk is just the role announcement with
	// finish_reason:null. We need the second chunk to see if cache_threshold
	// was returned (early abort) or if decode is proceeding normally.
	if shouldSignalBytes(w.buffer.Bytes()) {
		w.signalReady()
	}

	return n, nil
}

func (w *responseWriterWithBuffer) WriteHeader(statusCode int) {
	if !w.buffering.Load() {
		w.writerFlusher.WriteHeader(statusCode)
		return
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.buffering.Load() {
		w.writerFlusher.WriteHeader(statusCode)
		return
	}

	if w.statusCode == 0 {
		w.statusCode = statusCode
	}
}

func (w *responseWriterWithBuffer) Flush() {
	if w.buffering.Load() {
		// Apply same logic as Write(): only signal when we have at least 2 SSE events.
		w.mu.Lock()
		shouldSignal := shouldSignalBytes(w.buffer.Bytes())
		w.mu.Unlock()
		if shouldSignal {
			w.signalReady()
		}
		return
	}
	w.writerFlusher.Flush()
}

// firstChunkReady returns a channel that receives nil when the first complete
// chunk of body data is available in the buffer. For SSE streaming responses,
// this is signaled when the buffer contains "\n\n" (complete SSE event).
// As a fallback, Flush() also signals readiness.
func (w *responseWriterWithBuffer) firstChunkReady() <-chan struct{} {
	return w.ready
}

func (w *responseWriterWithBuffer) signalReady() {
	w.readyOnce.Do(func() {
		w.ready <- struct{}{}
		close(w.ready)
	})
}

func (w *responseWriterWithBuffer) writeHeaderOnce() {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	if w.statusCode != 0 {
		w.writerFlusher.WriteHeader(w.statusCode)
	}
}

// buffered returns the currently buffered content for inspection.
func (w *responseWriterWithBuffer) buffered() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buffer.String()
}

// getStatusCode returns the status code that was set (0 if not set).
func (w *responseWriterWithBuffer) getStatusCode() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.statusCode
}

// flushBufferAndGoDirect writes any buffered content to the underlying writer
// and switches to direct mode for all subsequent writes.
func (w *responseWriterWithBuffer) flushBufferAndGoDirect() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.buffering.Load() {
		return nil // already in direct mode
	}

	w.writeHeaderOnce()

	// Write buffered content to underlying writer
	if w.buffer.Len() > 0 {
		_, err := w.writerFlusher.Write(w.buffer.Bytes())
		if err != nil {
			return err
		}
	}

	// Flush BEFORE switching to direct mode.
	// This prevents concurrent Flush() calls on the underlying writer,
	// since the proxy goroutine's Flush() will no-op while buffering=true.
	w.writerFlusher.Flush()

	// Switch to direct mode. After this, the proxy goroutine handles
	// all writes and flushes directly (no concurrency with us).
	w.buffering.Store(false)

	return nil
}

var sseEventDelimiterBytes = []byte(sseEventDelimiter)

func shouldSignalBytes(data []byte) bool {
	return bytes.Count(data, sseEventDelimiterBytes) >= 2
}
