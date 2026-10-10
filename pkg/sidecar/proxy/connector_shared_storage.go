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
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

const finishReasonCacheThreshold = "cache_threshold"

func (s *Server) handleSharedStorage(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, apiType reqcommon.APIType) {
	s.logger.V(logging.DEBUG).Info("running Shared Storage protocol", "url", prefillPodHostPort)

	original, body, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	// If "cache_hit_threshold" is present in the request, we try to decode first. The decode node must meet the cache hit threshold in order to execute.
	// If the decode node is below the threshold, it won't process the request and return a "cache_threshold" finish reason. In that case,
	// we fall back to P/D disaggregation: perform prefill and then decode.
	// For more information refer to the RFC https://github.com/vllm-project/vllm/issues/24256
	if cacheHitThreshold, hasCacheHitThreshold := body[reqcommon.FieldCacheHitThreshold]; hasCacheHitThreshold {
		s.logger.V(logging.DEBUG).Info("cache_hit_threshold field found in the request, trying to decode first", reqcommon.FieldCacheHitThreshold, cacheHitThreshold)
		decodeReq := cloneRequestWithBody(r.Context(), r, original)
		attemptStart := time.Now()
		attemptWriter, attemptStatus := captureResponseStatus(w)
		attemptReturned := false
		defer recordDecodeAbort(&attemptReturned, attemptStart)
		needsPrefill, err := s.tryDecode(attemptWriter, decodeReq, body)
		attemptReturned = true
		// An attempt that falls back to prefill is not sampled; the decode after
		// prefill is this request's decode stage.
		if !needsPrefill {
			metrics.RecordDecodeDuration(time.Since(attemptStart))
			if err != nil || attemptStatus.failed() {
				metrics.RecordError(metrics.StageDecode)
			}
		}
		if err != nil {
			return
		}
		if !needsPrefill {
			s.logger.V(logging.DEBUG).Info("decode succeeded without prefill")
			return
		}
		s.logger.V(logging.DEBUG).Info("decode failed due to failing to meet the cache hit threshold", reqcommon.FieldCacheHitThreshold, cacheHitThreshold)
	}

	// we clone the completion request to avoid modifying the original request
	prefillRequest := maps.Clone(body)
	if err := s.prefill(w, r, prefillPodHostPort, prefillRequest, apiType); err != nil {
		s.logger.Error(err, "prefill failed")
		return
	}

	s.logger.V(logging.DEBUG).Info("forwarding to decoder after prefill")
	body[reqcommon.FieldCacheHitThreshold] = 0
	decodeRequestBody, err := json.Marshal(body)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send Invalid JSON error response to client")
		}
		return
	}

	decodeReq := cloneRequestWithBody(r.Context(), r, decodeRequestBody)
	decodeStart := time.Now()
	decodeWriter, decodeStatus := captureResponseStatus(w)
	decodeReturned := false
	defer recordDecodeAbort(&decodeReturned, decodeStart)
	s.decoderProxy.ServeHTTP(decodeWriter, decodeReq)
	decodeReturned = true
	metrics.RecordDecodeDuration(time.Since(decodeStart))
	if decodeStatus.failed() {
		metrics.RecordError(metrics.StageDecode)
	}
}

// tryDecode attempts to decode and returns whether prefill is needed.
func (s *Server) tryDecode(w http.ResponseWriter, r *http.Request, body map[string]any) (bool, error) {
	if isStreaming, _ := body[reqcommon.FieldStream].(bool); isStreaming {
		if flusher, ok := w.(flushableResponseWriter); ok {
			bw := newResponseWriterWithBuffer(flusher)
			return s.tryDecodeStreaming(bw, r)
		}
	}
	return s.tryDecodeBuffered(w, r)
}

// tryDecodeBuffered handles non-streaming decode attempts.
// It buffers the entire response before inspecting it.
func (s *Server) tryDecodeBuffered(w http.ResponseWriter, r *http.Request) (bool, error) {
	dw := &bufferedResponseWriter{}
	s.decoderProxy.ServeHTTP(dw, r)

	if isHTTPError(dw.statusCode) {

		w.WriteHeader(dw.statusCode)
		if dw.buffer.Len() > 0 {
			WriteAll(w, dw.buffer.Bytes())
		}

		err := errors.New("decode request failed")
		s.logger.Error(err, "unexpected status code", "code", dw.statusCode)

		return false, err
	}

	// Parse response to check finish_reason
	var response map[string]any
	if err := json.Unmarshal(dw.buffer.Bytes(), &response); err != nil {
		s.logger.Error(err, "failed to unmarshal decode response", "response", dw.buffer.String())

		if err := errorInternalServerError(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return false, err
	}

	// Check for cache_threshold finish reason
	if s.hasCacheThresholdFinishReason(response) {
		return true, nil
	}

	// Decode succeeded, write response to client
	maps.Copy(w.Header(), dw.headers)
	WriteAll(w, dw.buffer.Bytes())

	return false, nil
}

// tryDecodeStreaming handles streaming decode attempts.
// It buffers the initial response to check for cache_threshold, then switches
// to direct streaming mode if decode succeeds.
func (s *Server) tryDecodeStreaming(w *responseWriterWithBuffer, r *http.Request) (bool, error) {
	// Run ServeHTTP in a goroutine so we can inspect the initial choice to determine if we need to prefill.
	done := make(chan struct{})
	// Written by the decode goroutine, read after done is closed.
	var aborted bool
	go func() {
		defer close(done)
		// net/http only recovers http.ErrAbortHandler on the request goroutine.
		defer func() {
			if rec := recover(); rec != nil {
				if rec != http.ErrAbortHandler {
					panic(rec)
				}
				aborted = true
			}
		}()
		s.decoderProxy.ServeHTTP(w, r)
	}()

	// Wait for either:
	// - firstChunkReady(): first body data is available in buffer
	// - done: request completed (possibly with no body, e.g., error response)
	select {
	case <-w.firstChunkReady():
	case <-done:
		s.logger.V(logging.DEBUG).Info("request completed without body data")
	}

	statusCode := w.getStatusCode()
	if isHTTPError(statusCode) {
		// A status of 0 means the decoder never called WriteHeader or Write.
		// Flushing that would commit an empty 200. Leave it unwritten so the
		// abort below drops the connection instead.
		var flushErr error
		if statusCode != 0 {
			flushErr = w.flushBufferAndGoDirect()
			if flushErr != nil {
				s.logger.Error(flushErr, "failed to flush buffer to client")
			}
		}
		// The decode goroutine may still be writing. Wait for it, then replay
		// the abort here. net/http only recovers http.ErrAbortHandler on the
		// request goroutine.
		<-done
		if aborted {
			panic(http.ErrAbortHandler)
		}
		if flushErr != nil {
			return false, flushErr
		}
		return false, fmt.Errorf("decode request failed with status code: %d", statusCode)
	}

	// Check buffered SSE content for cache_threshold finish reason.
	if s.checkBufferedResponseForCacheThreshold(w.buffered()) {
		s.logger.V(logging.DEBUG).Info("finish reason cache_threshold detected, needs prefill")
		return true, nil
	}

	// No cache_threshold finish reason found, flush buffer and switch to direct mode
	// to let the rest of the response stream through.
	s.logger.V(logging.DEBUG).Info("first response for request shows success without cache_threshold finish reason")
	if err := w.flushBufferAndGoDirect(); err != nil {
		s.logger.Error(err, "failed to flush buffer to client and switch to direct mode")
		return false, err
	}
	<-done
	if aborted {
		// Replay the abort on the request goroutine, where net/http recovers it
		// and drops the connection.
		panic(http.ErrAbortHandler)
	}
	return false, nil
}

// hasCacheThresholdFinishReason checks if a parsed response contains cache_threshold finish reason.
func (s *Server) hasCacheThresholdFinishReason(response map[string]any) bool {
	choices, ok := response[responseFieldChoices].([]any)
	if !ok || len(choices) == 0 {
		return false
	}

	for _, raw := range choices {
		choice, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		finishReason, ok := choice[responseFieldFinishReason].(string)
		if ok && finishReason == finishReasonCacheThreshold {
			return true
		}
	}
	return false
}

// checkBufferedResponseForCacheThreshold checks the buffered SSE response for cache_threshold finish reason.
// This is only called for streaming responses, so data is always in SSE format.
func (s *Server) checkBufferedResponseForCacheThreshold(data string) bool {
	// Parse SSE format: "data: {...json...}\n\ndata: {...json...}\n\n"
	for _, line := range strings.Split(data, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line == reqcommon.SSEDone || !strings.HasPrefix(line, reqcommon.SSEDataPrefix) {
			continue
		}

		jsonData := strings.TrimPrefix(line, reqcommon.SSEDataPrefix)
		var response map[string]any
		if err := json.Unmarshal([]byte(jsonData), &response); err != nil {
			s.logger.V(logging.DEBUG).Info("skipping malformed SSE chunk", "chunk", jsonData)
			continue
		}

		if s.hasCacheThresholdFinishReason(response) {
			return true
		}
	}
	return false
}

// prefill routes a request to a prefill node
func (s *Server) prefill(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, body map[string]any, apiType reqcommon.APIType) error {
	// Prepare prefill request
	reqcommon.CapSingleToken(body, apiType)
	body[reqcommon.FieldCacheHitThreshold] = 0

	pbody, err := json.Marshal(body)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send Invalid JSON error response to client")
		}
		return err
	}
	preq := cloneRequestWithBody(r.Context(), r, pbody)

	prefillHandler, err := s.prefillerProxyHandler(prefillPodHostPort)
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send Bad Gateway error response to client")
		}
		return err
	}

	// send prefill request
	s.logger.V(logging.DEBUG).Info("sending prefill request", "to", prefillPodHostPort)
	pw := &bufferedResponseWriter{}
	prefillStart := time.Now()
	prefillHandler.ServeHTTP(pw, preq)
	metrics.RecordPrefillDuration(time.Since(prefillStart))

	if isHTTPError(pw.statusCode) {
		metrics.RecordError(metrics.StagePrefill)
		s.logger.Error(nil, "prefill request failed", "code", pw.statusCode)
		w.WriteHeader(pw.statusCode)
		if pw.buffer.Len() > 0 {
			WriteAll(w, pw.buffer.Bytes())
		}
		return fmt.Errorf("prefill request failed with status code: %d", pw.statusCode)
	}

	s.logger.V(logging.DEBUG).Info("prefill completed successfully")
	return nil
}
