/*
Copyright 2026 The Kubernetes Authors.

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
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
)

// kvTransferParamsCaptureWriter wraps a ResponseWriter to intercept and extract
// kv_transfer_params from the decode response body for caching.
type kvTransferParamsCaptureWriter struct {
	http.ResponseWriter
	buf    *bytes.Buffer
	writer *bufio.Writer
}

// newKVTransferParamsCaptureWriter creates a ResponseWriter that captures
// kv_transfer_params from the JSON response body. Returns the wrapped writer
// and a finalize function that parses the captured body and extracts the params.
func newKVTransferParamsCaptureWriter(w http.ResponseWriter) (http.ResponseWriter, func() map[string]any) {
	buf := &bytes.Buffer{}
	writer := bufio.NewWriter(io.MultiWriter(w, buf))

	wrapped := &kvTransferParamsCaptureWriter{
		ResponseWriter: w,
		buf:            buf,
		writer:         writer,
	}

	finalize := func() map[string]any {
		if err := wrapped.writer.Flush(); err != nil {
			return nil
		}
		var response map[string]any
		if err := json.Unmarshal(wrapped.buf.Bytes(), &response); err != nil {
			return nil
		}
		if params, ok := response["kv_transfer_params"].(map[string]any); ok {
			return params
		}
		return nil
	}

	return wrapped, finalize
}

func (w *kvTransferParamsCaptureWriter) Write(b []byte) (int, error) {
	return w.writer.Write(b)
}

func (w *kvTransferParamsCaptureWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hj, ok := w.ResponseWriter.(http.Hijacker); ok {
		return hj.Hijack()
	}
	return nil, nil, http.ErrNotSupported
}

func (w *kvTransferParamsCaptureWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
