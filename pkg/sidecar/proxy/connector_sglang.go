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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

var (
	sglangBootstrapPort int
	// The prefill leg must finish before buffered decode output can be committed.
	sglangPrefillWaitTimeout = 5 * time.Minute
)

func init() {
	// Default SGLang bootstrap port
	sglangBootstrapPort = 8998

	// Override from environment variable if set
	if portStr := os.Getenv("SGLANG_BOOTSTRAP_PORT"); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			sglangBootstrapPort = port
		}
	}
}

func (s *Server) handleSGLang(w http.ResponseWriter, r *http.Request, prefillPodHostPort string) {
	s.logger.V(logging.DEBUG).Info("running SGLang protocol", "url", prefillPodHostPort)

	// Make Request
	requestData, err := s.parseSGLangRequest(r)

	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	roomID := s.generateSGLangRoomID()

	// Inject bootstrap info for both prefill and decode
	bootstrapInfo := s.addSGLangBootstrapInfo(requestData, prefillPodHostPort, roomID)

	body, err := json.Marshal(bootstrapInfo)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Send concurrent prefill and decode requests
	s.handleSGLangConcurrentRequests(w, r, body, prefillPodHostPort)
}

func (s *Server) handleSGLangConcurrentRequests(w http.ResponseWriter, r *http.Request, body []byte, prefillHost string) {
	tracer := tracing.Tracer(tracerScope)
	parentCtx := r.Context()
	dispatchCtx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	prefillCtx, prefillSpan := tracer.Start(dispatchCtx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.prefill_target", prefillHost),
		attribute.String("llm_d.pd_proxy.connector", KVConnectorSGLang),
		attribute.Bool("llm_d.pd_proxy.prefill.async", true),
	)
	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		prefillSpan.SetStatus(codes.Error, "failed to create prefill handler")
		prefillSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	prefillReq := cloneRequestWithBody(prefillCtx, r, body)
	decodeCtx, decodeSpan := tracer.Start(dispatchCtx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.connector", KVConnectorSGLang),
		attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host),
		attribute.Bool("llm_d.pd_proxy.decode.concurrent_with_prefill", true),
	)
	decodeReq := cloneRequestWithBody(decodeCtx, r, body)

	var prefillResponse *bufferedResponseWriter
	prefillDone := make(chan struct{})
	prefillStart := time.Now()
	go func() {
		defer close(prefillDone)
		defer prefillSpan.End()
		defer func() {
			if rec := recover(); rec != nil && rec != http.ErrAbortHandler {
				s.logger.Error(fmt.Errorf("panic: %v", rec), "panic in prefill request")
				cancel()
			}
		}()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, prefillReq)
		prefillResponse = pw
		prefillDuration := time.Since(prefillStart)
		prefillSpan.SetAttributes(
			attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
			attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(prefillDuration.Milliseconds())),
		)
		if isHTTPError(pw.statusCode) {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
		}
		s.logger.V(logging.TRACE).Info("prefill request completed", "status", pw.statusCode)
	}()

	decodeWriter := newDeferredCommitWriter(w)
	decodeDone := make(chan struct{})
	var decodePanic any
	decodeStart := time.Now()
	go func() {
		defer close(decodeDone)
		defer decodeSpan.End()
		defer func() {
			decodePanic = recover()
			decodeSpan.SetAttributes(
				attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(time.Since(decodeStart).Milliseconds())),
			)
			if decodePanic != nil {
				decodeSpan.SetStatus(codes.Error, "decode request aborted")
			}
		}()
		s.decoderProxy.ServeHTTP(decodeWriter, decodeReq)
	}()

	timer := time.NewTimer(sglangPrefillWaitTimeout)
	defer timer.Stop()
	prefillSucceeded := false
	prefillTimedOut := false

	select {
	case <-prefillDone:
		prefillSucceeded = prefillResponse != nil && !isHTTPError(prefillResponse.statusCode)
		if prefillSucceeded {
			decodeWriter.commit()
		} else {
			decodeWriter.abort()
			cancel()
		}
	case <-timer.C:
		prefillTimedOut = true
		decodeWriter.abort()
		cancel()
	case <-parentCtx.Done():
		decodeWriter.abort()
		cancel()
		<-decodeDone
		return
	}

	<-decodeDone
	decodeDuration := time.Since(decodeStart)

	switch {
	case prefillSucceeded:
		if decodePanic != nil {
			panic(decodePanic)
		}
	case prefillTimedOut:
		w.WriteHeader(http.StatusGatewayTimeout)
		if _, err := w.Write([]byte(`{"error":"SGLang prefill did not complete before the wait timeout"}`)); err != nil {
			s.logger.Error(err, "failed to send SGLang prefill timeout to client")
		}
	default:
		status := http.StatusBadGateway
		if prefillResponse != nil {
			status = prefillResponse.statusCode
			for key, values := range prefillResponse.Header() {
				w.Header()[key] = append([]string(nil), values...)
			}
		}
		w.WriteHeader(status)
		if prefillResponse != nil {
			if _, err := w.Write(prefillResponse.bodyBytes()); err != nil {
				s.logger.Error(err, "failed to send SGLang prefill error to client")
			}
		}
	}

	// Calculate end-to-end P/D timing metrics for concurrent P/D.
	if currentSpan := trace.SpanFromContext(parentCtx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		var trueTTFT time.Duration
		if requestStartValue := parentCtx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
				trueTTFT = decodeStart.Sub(requestStart)
			}
		}

		currentSpan.SetAttributes(
			attribute.Float64("llm_d.pd_proxy.total_duration_ms", float64(totalDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.true_ttft_ms", float64(trueTTFT.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.decode_duration_ms", float64(decodeDuration.Milliseconds())),
			attribute.Bool("llm_d.pd_proxy.concurrent_pd", true),
		)
	}
}

func (s *Server) addSGLangBootstrapInfo(requestData map[string]interface{}, prefillHostPort string, roomID int64) map[string]interface{} {
	modifiedRequest := make(map[string]interface{})
	for k, v := range requestData {
		modifiedRequest[k] = v
	}

	// Generate bootstrap host from prefill host
	bootstrapHost := extractHost(prefillHostPort)

	prefillRank, prefillDPSize, hasPrefillRank := s.sglangPrefillRank(prefillHostPort)
	if hasPrefillRank {
		roomID = alignSGLangRoom(roomID, prefillRank, prefillDPSize)
	}

	// Add bootstrap information
	modifiedRequest[requestFieldBootstrapHost] = bootstrapHost
	modifiedRequest[requestFieldBootstrapPort] = sglangBootstrapPort
	modifiedRequest[requestFieldBootstrapRoom] = roomID

	s.logger.V(logging.TRACE).Info("bootstrap info added",
		"bootstrap_host", bootstrapHost,
		"bootstrap_port", sglangBootstrapPort,
		"bootstrap_room", roomID)

	return modifiedRequest
}

func (s *Server) sglangPrefillRank(prefillHostPort string) (int, int, bool) {
	dpSize := s.config.DataParallelSize
	if dpSize <= 1 {
		return 0, dpSize, false
	}

	prefillHostPort, _ = strings.CutPrefix(prefillHostPort, "http://")
	_, portString, err := net.SplitHostPort(prefillHostPort)
	if err != nil {
		return 0, dpSize, false
	}
	prefillPort, err := strconv.Atoi(portString)
	if err != nil {
		return 0, dpSize, false
	}
	rank := prefillPort - sglangBootstrapPort
	if rank < 0 || rank >= dpSize {
		return 0, dpSize, false
	}
	return rank, dpSize, true
}

func alignSGLangRoom(roomID int64, rank, dpSize int) int64 {
	size := int64(dpSize)
	base := roomID - roomID%size
	rankOffset := int64(rank)
	if base > math.MaxInt64-rankOffset {
		base -= size
	}
	return base + rankOffset
}

func (s *Server) parseSGLangRequest(r *http.Request) (map[string]interface{}, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}

	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	return requestData, nil
}

func (s *Server) generateSGLangRoomID() int64 {
	return rand.Int64()
}
