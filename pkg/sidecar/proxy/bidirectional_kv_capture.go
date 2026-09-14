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
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/felixge/httpsnoop"
)

// kvTransferParamsCapture intercepts decode responses to extract kv_transfer_params
// without modifying the response body. Supports both non-streaming and SSE modes.
type kvTransferParamsCapture struct {
	header         http.Header
	wroteHeader    bool
	streaming      bool
	streamBuffer   []byte
	capturedParams map[string]any
	bodyBuffer     []byte
}

const kvTransferParamsField = "kv_transfer_params"

// newKVTransferParamsCaptureWriter wraps the http.ResponseWriter to intercept
// kv_transfer_params from the decode response. Returns the wrapped writer and
// a finalize function that returns the captured params (or nil if none found).
func newKVTransferParamsCaptureWriter(w http.ResponseWriter) (http.ResponseWriter, func() map[string]any) {
	capture := &kvTransferParamsCapture{
		header: w.Header(),
	}

	writer := httpsnoop.Wrap(w, httpsnoop.Hooks{
		WriteHeader: func(next httpsnoop.WriteHeaderFunc) httpsnoop.WriteHeaderFunc {
			return func(statusCode int) {
				capture.writeHeader(next, statusCode)
			}
		},
		Write: func(next httpsnoop.WriteFunc) httpsnoop.WriteFunc {
			return func(body []byte) (int, error) {
				return capture.write(next, body)
			}
		},
		ReadFrom: func(_ httpsnoop.ReadFromFunc) httpsnoop.ReadFromFunc {
			return func(src io.Reader) (int64, error) {
				return capture.readFrom(w.Write, src)
			}
		},
	})

	finalize := func() map[string]any {
		capture.flushSSEBuffer()
		if len(capture.bodyBuffer) > 0 {
			capture.extractFromBody(capture.bodyBuffer)
		}
		return capture.capturedParams
	}

	return writer, finalize
}

func (c *kvTransferParamsCapture) writeHeader(next httpsnoop.WriteHeaderFunc, statusCode int) {
	c.wroteHeader = true
	next(statusCode)
}

func (c *kvTransferParamsCapture) write(next httpsnoop.WriteFunc, body []byte) (int, error) {
	c.interceptBody(body)
	return next(body)
}

func (c *kvTransferParamsCapture) readFrom(next httpsnoop.WriteFunc, src io.Reader) (int64, error) {
	if c.isSSE(nil) {
		// SSE: stream through while intercepting complete lines
		n, err := io.Copy(kvTransferParamsCaptureStreamWriter{
			capture: c,
			forward: next,
		}, src)
		if err != nil {
			return n, err
		}
		return n, nil
	}

	// Non-streaming: buffer full body for extraction
	body, err := io.ReadAll(src)
	if err != nil {
		return 0, err
	}
	c.bodyBuffer = append(c.bodyBuffer, body...)
	n, err := next(body)
	return int64(n), err
}

func (c *kvTransferParamsCapture) interceptBody(body []byte) {
	if c.isSSE(body) {
		c.interceptSSEChunk(body)
	} else {
		c.bodyBuffer = append(c.bodyBuffer, body...)
	}
}

func (c *kvTransferParamsCapture) isSSE(body []byte) bool {
	if c.streaming {
		return true
	}
	contentType := c.header.Get("Content-Type")
	if strings.Contains(contentType, "text/event-stream") || bytes.HasPrefix(body, []byte("data:")) {
		c.streaming = true
		return true
	}
	return false
}

func (c *kvTransferParamsCapture) interceptSSEChunk(body []byte) {
	c.streamBuffer = append(c.streamBuffer, body...)
	for {
		lineEnd := bytes.IndexByte(c.streamBuffer, '\n')
		if lineEnd < 0 {
			break
		}
		line := c.streamBuffer[:lineEnd+1]
		c.extractFromSSELine(line)
		c.streamBuffer = c.streamBuffer[lineEnd+1:]
	}
}

func (c *kvTransferParamsCapture) flushSSEBuffer() {
	if len(c.streamBuffer) == 0 {
		return
	}
	c.extractFromSSELine(c.streamBuffer)
	c.streamBuffer = nil
}

func (c *kvTransferParamsCapture) extractFromSSELine(line []byte) {
	trimmedLine := bytes.TrimRight(line, "\r\n")
	data, ok := bytes.CutPrefix(trimmedLine, []byte("data: "))
	if !ok {
		return
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
		return
	}

	var response map[string]any
	if err := json.Unmarshal(data, &response); err != nil {
		return
	}

	if params, ok := response[kvTransferParamsField].(map[string]any); ok {
		c.capturedParams = params
	}
}

func (c *kvTransferParamsCapture) extractFromBody(body []byte) {
	if len(bytes.TrimSpace(body)) == 0 {
		return
	}

	var response map[string]any
	if err := json.Unmarshal(body, &response); err != nil {
		return
	}

	if params, ok := response[kvTransferParamsField].(map[string]any); ok {
		c.capturedParams = params
	}
}

type kvTransferParamsCaptureStreamWriter struct {
	capture *kvTransferParamsCapture
	forward httpsnoop.WriteFunc
}

func (w kvTransferParamsCaptureStreamWriter) Write(body []byte) (int, error) {
	w.capture.interceptBody(body)
	return w.forward(body)
}
