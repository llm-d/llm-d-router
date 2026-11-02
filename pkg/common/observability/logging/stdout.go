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

package logging

import (
	"encoding/json"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

const (
	// stdoutQueueBytes bounds the log bytes held while stdout does not accept writes.
	stdoutQueueBytes = 4 << 20
	// stdoutQueueRecords bounds the number of queued records independently of their size.
	stdoutQueueRecords = 8192
	// flushTimeout bounds how long Flush and Sync wait for a stdout that is not draining.
	flushTimeout = 2 * time.Second
	// droppedRecordsBody is the body of the record that reports dropped records.
	droppedRecordsBody = "Dropped log records because stdout did not accept writes"
)

var stdoutSink = sync.OnceValue(func() *nonBlockingWriter {
	return newNonBlockingWriter(os.Stdout, stdoutQueueBytes, stdoutQueueRecords)
})

// NonBlockingStdout writes log records to stdout from a bounded queue.
// A logging call never waits for stdout, and records that do not fit in the queue are dropped and counted.
// The drop report carries the same service.name as the records of serviceName.
// A process that uses it calls Flush before it exits.
func NonBlockingStdout(serviceName string) zap.Opts {
	sink := stdoutSink()
	name := ServiceName(serviceName)
	sink.serviceName.Store(&name)
	return zap.WriteTo(sink)
}

// Flush writes the log records queued for stdout, waiting at most two seconds.
func Flush() {
	stdoutSink().flush(flushTimeout)
}

// nonBlockingWriter hands each record to a single goroutine that writes it to out.
// A write to a full pipe blocks, and os.File lets one writer in at a time, so a direct write stalls every goroutine that logs, request handlers included.
// When the queue is full the record is dropped and counted.
// The count is written as a WARN record once out accepts writes again.
type nonBlockingWriter struct {
	out       io.Writer
	queue     chan queuedRecord
	queued    atomic.Int64
	maxQueued int64
	dropped   atomic.Uint64
	// serviceName is the service.name of the drop report, omitted while unset.
	serviceName atomic.Pointer[string]
}

// queuedRecord carries a log record, or a flush marker when done is set.
type queuedRecord struct {
	data []byte
	done chan struct{}
}

func newNonBlockingWriter(out io.Writer, maxBytes int64, maxRecords int) *nonBlockingWriter {
	w := &nonBlockingWriter{
		out:       out,
		queue:     make(chan queuedRecord, maxRecords),
		maxQueued: maxBytes,
	}
	go w.run()
	return w
}

// Write queues a copy of p and never blocks.
// It reports success when p is dropped so zap does not log a write error.
func (w *nonBlockingWriter) Write(p []byte) (int, error) {
	if w.queued.Add(int64(len(p))) > w.maxQueued {
		w.queued.Add(-int64(len(p)))
		w.dropped.Add(1)
		return len(p), nil
	}
	// zap reuses p after Write returns.
	data := append([]byte(nil), p...)
	select {
	case w.queue <- queuedRecord{data: data}:
	default:
		w.queued.Add(-int64(len(p)))
		w.dropped.Add(1)
	}
	return len(p), nil
}

// Sync is called by zap after records above error level, before a panic or exit.
func (w *nonBlockingWriter) Sync() error {
	w.flush(flushTimeout)
	return nil
}

func (w *nonBlockingWriter) flush(timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	done := make(chan struct{})
	select {
	case w.queue <- queuedRecord{done: done}:
	case <-timer.C:
		return
	}
	select {
	case <-done:
	case <-timer.C:
	}
}

func (w *nonBlockingWriter) run() {
	for rec := range w.queue {
		w.reportDropped()
		if rec.done != nil {
			close(rec.done)
			continue
		}
		// A record that out failed to take is lost, so it is counted like a dropped one.
		if _, err := w.out.Write(rec.data); err != nil {
			w.dropped.Add(1)
		}
		w.queued.Add(-int64(len(rec.data)))
	}
}

func (w *nonBlockingWriter) reportDropped() {
	n := w.dropped.Swap(0)
	if n == 0 {
		return
	}
	rec := map[string]any{
		otelTimestampKey:      time.Now().UTC().Format(time.RFC3339Nano),
		otelSeverityTextKey:   severityWarn,
		otelSeverityNumberKey: severityNumberWarn,
		otelBodyKey:           droppedRecordsBody,
		"dropped_records":     n,
	}
	if name := w.serviceName.Load(); name != nil {
		rec[otelServiceNameKey] = *name
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return
	}
	// The count is kept for the next report when out fails to take this one.
	if _, err := w.out.Write(append(line, '\n')); err != nil {
		w.dropped.Add(n)
	}
}
