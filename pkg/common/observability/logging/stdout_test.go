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
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	ctrl "sigs.k8s.io/controller-runtime"
)

// stalledWriter blocks every Write until release is closed, like a pipe nobody reads.
type stalledWriter struct {
	release chan struct{}
	mu      sync.Mutex
	buf     bytes.Buffer
}

func (w *stalledWriter) Write(p []byte) (int, error) {
	<-w.release
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

func (w *stalledWriter) lines() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return strings.Split(strings.TrimSuffix(w.buf.String(), "\n"), "\n")
}

func TestNonBlockingWriterDropsAndCountsWhileOutputStalls(t *testing.T) {
	const records = 1000
	out := &stalledWriter{release: make(chan struct{})}
	w := newNonBlockingWriter(out, 1<<20, 8)
	name := "drop-test"
	w.serviceName.Store(&name)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := range records {
			if _, err := fmt.Fprintf(w, "record %d\n", i); err != nil {
				t.Errorf("Write: %v", err)
			}
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Write waited for a stalled output")
	}

	close(out.release)
	w.flush(5 * time.Second)

	lines := out.lines()
	var kept, dropped int
	next := 0
	for _, line := range lines {
		var i int
		if _, err := fmt.Sscanf(line, "record %d", &i); err == nil {
			if i < next {
				t.Fatalf("record %d written after record %d", i, next-1)
			}
			next = i + 1
			kept++
			continue
		}
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("unexpected line %q", line)
		}
		if rec[otelSeverityTextKey] != severityWarn {
			t.Errorf("drop report severity = %v, want %s", rec[otelSeverityTextKey], severityWarn)
		}
		if rec[otelServiceNameKey] != name {
			t.Errorf("drop report service.name = %v, want %s", rec[otelServiceNameKey], name)
		}
		dropped += int(rec["dropped_records"].(float64))
	}
	if dropped == 0 {
		t.Fatalf("no drop reported for %d records through a queue of 8", records)
	}
	if kept+dropped != records {
		t.Errorf("kept %d + dropped %d = %d, want %d", kept, dropped, kept+dropped, records)
	}
}

func TestNonBlockingWriterBoundsQueuedBytes(t *testing.T) {
	out := &stalledWriter{release: make(chan struct{})}
	w := newNonBlockingWriter(out, 100, 1000)
	record := strings.Repeat("x", 40) + "\n"
	for range 10 {
		if _, err := w.Write([]byte(record)); err != nil {
			t.Fatal(err)
		}
	}
	if got := w.queued.Load(); got > 100 {
		t.Errorf("queued %d bytes, want at most 100", got)
	}
	close(out.release)
	w.flush(5 * time.Second)

	// The worker resets the drop counter when it reports, so the count is read from the output.
	var kept, dropped int
	for _, line := range out.lines() {
		if line == strings.TrimSuffix(record, "\n") {
			kept++
			continue
		}
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("unexpected line %q", line)
		}
		dropped += int(rec["dropped_records"].(float64))
	}
	if kept > 100/len(record) {
		t.Errorf("kept %d records of %d bytes past a 100-byte bound", kept, len(record))
	}
	if dropped == 0 || kept+dropped != 10 {
		t.Errorf("kept %d, dropped %d, want some dropped and 10 in total", kept, dropped)
	}
}

// failingWriter fails every Write while failing is set, like a stdout on a full disk.
type failingWriter struct {
	mu      sync.Mutex
	failing bool
	buf     bytes.Buffer
}

func (w *failingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.failing {
		return 0, errors.New("no space left on device")
	}
	return w.buf.Write(p)
}

func TestNonBlockingWriterCountsRecordsOutputFailedToTake(t *testing.T) {
	out := &failingWriter{failing: true}
	w := newNonBlockingWriter(out, 1<<20, 128)
	for i := range 5 {
		fmt.Fprintf(w, "lost %d\n", i)
	}
	w.flush(5 * time.Second)

	out.mu.Lock()
	out.failing = false
	out.mu.Unlock()
	fmt.Fprintln(w, "kept")
	w.flush(5 * time.Second)

	out.mu.Lock()
	lines := strings.Split(strings.TrimSuffix(out.buf.String(), "\n"), "\n")
	out.mu.Unlock()
	if len(lines) != 2 || lines[1] != "kept" {
		t.Fatalf("got lines %q, want a drop report and kept", lines)
	}
	var rec map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
		t.Fatalf("first line is not a drop report: %q", lines[0])
	}
	if got := rec["dropped_records"]; got != float64(5) {
		t.Errorf("dropped_records = %v, want 5", got)
	}
}

func TestNonBlockingWriterFlushWritesQueuedRecordsInOrder(t *testing.T) {
	out := &stalledWriter{release: make(chan struct{})}
	close(out.release)
	w := newNonBlockingWriter(out, 1<<20, 128)
	for i := range 100 {
		fmt.Fprintf(w, "record %d\n", i)
	}
	w.flush(5 * time.Second)

	lines := out.lines()
	if len(lines) != 100 {
		t.Fatalf("got %d lines after flush, want 100", len(lines))
	}
	for i, line := range lines {
		if want := fmt.Sprintf("record %d", i); line != want {
			t.Fatalf("line %d = %q, want %q", i, line, want)
		}
	}
}

func TestNonBlockingWriterFlushGivesUpOnStalledOutput(t *testing.T) {
	out := &stalledWriter{release: make(chan struct{})}
	defer close(out.release)
	w := newNonBlockingWriter(out, 1<<20, 8)
	fmt.Fprintln(w, "stuck")

	start := time.Now()
	w.flush(100 * time.Millisecond)
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("flush took %s with a 100ms timeout", elapsed)
	}
}

const stalledStdoutChildEnv = "LOGGING_TEST_STALLED_STDOUT_CHILD"

// TestNonBlockingStdoutWithStalledPipe runs a child process whose stdout is a pipe nobody reads.
// The child logs far more than a pipe buffer holds and must still exit.
func TestNonBlockingStdoutWithStalledPipe(t *testing.T) {
	if os.Getenv(stalledStdoutChildEnv) == "1" {
		InitSetupLogging("stalled-stdout-test", NonBlockingStdout("stalled-stdout-test"))
		logger := ctrl.Log.WithName("child")
		payload := strings.Repeat("x", 200)
		for i := range 20000 {
			logger.Info("request received", "i", i, "payload", payload)
		}
		Flush()
		fmt.Fprintln(os.Stderr, "child done")
		os.Exit(0)
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var stderr bytes.Buffer
	cmd := exec.Command(os.Args[0], "-test.run=^TestNonBlockingStdoutWithStalledPipe$")
	cmd.Env = append(os.Environ(), stalledStdoutChildEnv+"=1")
	cmd.Stdout = w
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	w.Close()

	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	select {
	case err := <-exited:
		if err != nil {
			t.Fatalf("child failed: %v\nstderr:\n%s", err, stderr.String())
		}
	case <-time.After(20 * time.Second):
		_ = cmd.Process.Kill()
		<-exited
		t.Fatalf("child logging to a stalled stdout did not finish\nstderr:\n%s", stderr.String())
	}

	// The first records reached the pipe before it filled.
	// A drop report comes first when the queue filled before the worker started.
	reader := bufio.NewReader(r)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("stdout line is not a JSON record: %q", line)
		}
		if rec[otelBodyKey] == droppedRecordsBody {
			continue
		}
		if rec[otelBodyKey] != "request received" {
			t.Errorf("first application record body = %v, want request received", rec[otelBodyKey])
		}
		return
	}
}
