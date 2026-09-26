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

package server

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCountingResponseWriter_CountsPartialWrites(t *testing.T) {
	rec := httptest.NewRecorder()
	cw := newCountingResponseWriter(rec)
	n, err := cw.Write([]byte("hello"))
	if err != nil || n != 5 {
		t.Fatalf("Write = %d, %v", n, err)
	}
	n, err = cw.Write([]byte("!"))
	if err != nil || n != 1 {
		t.Fatalf("Write = %d, %v", n, err)
	}
	if cw.BytesWritten() != 6 {
		t.Fatalf("BytesWritten = %d, want 6", cw.BytesWritten())
	}
}

type failingWriter struct {
	http.ResponseWriter
	failAfter int
	wrote     int
}

func (w *failingWriter) Write(p []byte) (int, error) {
	remain := w.failAfter - w.wrote
	if remain <= 0 {
		return 0, errors.New("client disconnected")
	}
	if len(p) > remain {
		n, _ := w.ResponseWriter.Write(p[:remain])
		w.wrote += n
		return n, errors.New("client disconnected")
	}
	n, err := w.ResponseWriter.Write(p)
	w.wrote += n
	return n, err
}

func TestCountingResponseWriter_CountsPartialWriteOnError(t *testing.T) {
	inner := &failingWriter{ResponseWriter: httptest.NewRecorder(), failAfter: 3}
	cw := newCountingResponseWriter(inner)
	n, err := cw.Write([]byte("abcdef"))
	if err == nil {
		t.Fatal("expected write error")
	}
	if n != 3 {
		t.Fatalf("partial n = %d, want 3", n)
	}
	if cw.BytesWritten() != 3 {
		t.Fatalf("BytesWritten = %d, want 3", cw.BytesWritten())
	}
}

// readerOnly hides any WriteTo on the wrapped source so io.Copy takes the
// destination's ReadFrom path, matching real reverse-proxy response bodies.
type readerOnly struct{ io.Reader }

// readerFromSpy records whether its ReadFrom was reached, standing in for the
// net/http response writer's sendfile-style optimized copy path.
type readerFromSpy struct {
	http.ResponseWriter
	readFromUsed bool
}

func (w *readerFromSpy) ReadFrom(r io.Reader) (int64, error) {
	w.readFromUsed = true
	return io.Copy(w.ResponseWriter, r)
}

func TestCountingResponseWriter_ReadFrom_DelegatesToInnerReaderFrom(t *testing.T) {
	inner := &readerFromSpy{ResponseWriter: httptest.NewRecorder()}
	cw := newCountingResponseWriter(inner)
	n, err := io.Copy(cw, readerOnly{strings.NewReader("hello world")})
	if err != nil || n != 11 {
		t.Fatalf("io.Copy = %d, %v", n, err)
	}
	if !inner.readFromUsed {
		t.Fatal("expected the inner writer's ReadFrom to be used")
	}
	if cw.BytesWritten() != 11 {
		t.Fatalf("BytesWritten = %d, want 11", cw.BytesWritten())
	}
}

func TestCountingResponseWriter_ReadFrom_FallbackWithoutInnerReaderFrom(t *testing.T) {
	// failingWriter does not implement io.ReaderFrom, so ReadFrom falls back
	// to a buffered copy through it. failAfter is high enough to succeed.
	inner := &failingWriter{ResponseWriter: httptest.NewRecorder(), failAfter: 1 << 20}
	cw := newCountingResponseWriter(inner)
	n, err := io.Copy(cw, readerOnly{strings.NewReader("fallback copy path")})
	if err != nil || n != int64(len("fallback copy path")) {
		t.Fatalf("io.Copy = %d, %v", n, err)
	}
	if cw.BytesWritten() != int(n) {
		t.Fatalf("BytesWritten = %d, want %d", cw.BytesWritten(), n)
	}
}

func TestCountingResponseWriter_ReadFrom_CountsPartialCopyOnError(t *testing.T) {
	inner := &failingWriter{ResponseWriter: httptest.NewRecorder(), failAfter: 3}
	cw := newCountingResponseWriter(inner)
	n, err := io.Copy(cw, readerOnly{strings.NewReader("abcdef")})
	if err == nil {
		t.Fatal("expected copy error")
	}
	if n != 3 {
		t.Fatalf("partial copy n = %d, want 3", n)
	}
	if cw.BytesWritten() != 3 {
		t.Fatalf("BytesWritten = %d, want 3", cw.BytesWritten())
	}
}
