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
	"io"
	"net/http"
)

// countingResponseWriter counts bytes written to the client so
// response_size_bytes can include partial writes on cancellation or disconnect.
type countingResponseWriter struct {
	http.ResponseWriter
	n int
}

func newCountingResponseWriter(w http.ResponseWriter) *countingResponseWriter {
	return &countingResponseWriter{ResponseWriter: w}
}

func (w *countingResponseWriter) Write(p []byte) (int, error) {
	n, err := w.ResponseWriter.Write(p)
	w.n += n
	return n, err
}

func (w *countingResponseWriter) BytesWritten() int { return w.n }

func (w *countingResponseWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func (w *countingResponseWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// ReadFrom implements io.ReaderFrom so io.Copy(dst, src) reaches the inner
// writer's optimized copy path (net/http's response writer supports
// sendfile-style copies once the response is streaming) instead of degrading
// to a per-chunk Write through this wrapper. Bytes copied are still counted
// so response_size_bytes stays exact, including the partial count on a
// mid-copy failure.
func (w *countingResponseWriter) ReadFrom(r io.Reader) (int64, error) {
	if rf, ok := w.ResponseWriter.(io.ReaderFrom); ok {
		n, err := rf.ReadFrom(r)
		w.n += int(n)
		return n, err
	}
	// The inner writer has no optimized path; copy straight into it and count
	// here, which still skips the per-chunk Write-wrapper indirection.
	n, err := io.Copy(w.ResponseWriter, r)
	w.n += int(n)
	return n, err
}
