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
	"net/http"
	"net/http/httptest"
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
