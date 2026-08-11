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
	"io"
	"net/http"
	"strings"

	"github.com/felixge/httpsnoop"
)

// routedExpertsRewriter splices prefill routed_experts over the decode response.
// routed_experts only appears in non-streaming responses, so SSE passes through
// untouched while non-streaming bodies are buffered and spliced at finalize.
type routedExpertsRewriter struct {
	header    http.Header
	resolve   func() map[int]string
	streaming bool
	buf       []byte
}

// newRoutedExpertsResponseWriterWithFinalize wraps w to merge the prefill's
// routed_experts into the decode response (sequential P/D, routing already
// known). With nothing to merge it returns w unchanged and a no-op finalize.
func newRoutedExpertsResponseWriterWithFinalize(
	w http.ResponseWriter,
	prefillByIdx map[int]string,
) (http.ResponseWriter, func() error) {
	if len(prefillByIdx) == 0 {
		return w, func() error { return nil }
	}
	return newDeferredRoutedExpertsResponseWriter(w, func() map[int]string { return prefillByIdx })
}

// newDeferredRoutedExpertsResponseWriter resolves the prefill routing lazily at
// finalize, for concurrent P/D paths (e.g. Mooncake) where it isn't known when
// the decode response starts. Non-streaming bodies are always buffered; if
// resolve returns no routing the buffer is forwarded unchanged.
func newDeferredRoutedExpertsResponseWriter(
	w http.ResponseWriter,
	resolve func() map[int]string,
) (http.ResponseWriter, func() error) {
	r := &routedExpertsRewriter{header: w.Header(), resolve: resolve}
	writer := httpsnoop.Wrap(w, httpsnoop.Hooks{
		WriteHeader: func(next httpsnoop.WriteHeaderFunc) httpsnoop.WriteHeaderFunc {
			return func(statusCode int) { r.writeHeader(next, statusCode) }
		},
		Write: func(next httpsnoop.WriteFunc) httpsnoop.WriteFunc {
			return func(body []byte) (int, error) { return r.write(next, body) }
		},
		ReadFrom: func(_ httpsnoop.ReadFromFunc) httpsnoop.ReadFromFunc {
			return func(src io.Reader) (int64, error) { return r.readFrom(w.Write, src) }
		},
	})
	return writer, func() error { return r.flush(w.Write) }
}

func (r *routedExpertsRewriter) writeHeader(next httpsnoop.WriteHeaderFunc, statusCode int) {
	// Re-marshaling changes the body size, so any upstream Content-Length is stale.
	r.header.Del("Content-Length")
	r.isSSE(nil)
	next(statusCode)
}

func (r *routedExpertsRewriter) write(next httpsnoop.WriteFunc, body []byte) (int, error) {
	if r.isSSE(body) {
		return next(body)
	}
	// Buffer non-streaming body; the full document is spliced at finalize.
	r.buf = append(r.buf, body...)
	return len(body), nil
}

func (r *routedExpertsRewriter) readFrom(next httpsnoop.WriteFunc, src io.Reader) (int64, error) {
	if r.isSSE(nil) {
		n, err := io.Copy(routedExpertsStreamWriter{forward: next}, src)
		return n, err
	}
	body, err := io.ReadAll(src)
	if err != nil {
		return 0, err
	}
	r.buf = append(r.buf, body...)
	return int64(len(body)), nil
}

func (r *routedExpertsRewriter) flush(next httpsnoop.WriteFunc) error {
	if r.streaming || len(r.buf) == 0 {
		return nil
	}
	body := r.buf
	r.buf = nil
	if prefillByIdx := r.resolve(); len(prefillByIdx) > 0 {
		body, _ = mergeRoutedExpertsJSON(prefillByIdx, body)
	}
	_, err := next(body)
	return err
}

func (r *routedExpertsRewriter) isSSE(body []byte) bool {
	if r.streaming {
		return true
	}
	contentType := r.header.Get("Content-Type")
	if strings.Contains(contentType, "text/event-stream") || bytes.HasPrefix(body, []byte("data:")) {
		r.streaming = true
		return true
	}
	return false
}

type routedExpertsStreamWriter struct {
	forward httpsnoop.WriteFunc
}

func (w routedExpertsStreamWriter) Write(body []byte) (int, error) {
	return w.forward(body)
}
