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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/telemetry"
)

var (
	sglangBootstrapPort int
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

func (s *Server) runSGLangProtocol(w http.ResponseWriter, r *http.Request, prefillPodHostPort string) {
	s.logger.V(4).Info("running SGLang protocol", "url", prefillPodHostPort)

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
	s.sendSGLangConcurrentRequests(w, r, body, prefillPodHostPort)
}

func (s *Server) sendSGLangConcurrentRequests(w http.ResponseWriter, r *http.Request, body []byte, prefillHost string) {
	tracer := telemetry.Tracer()
	ctx := r.Context()

	// Prefill Stage - async
	ctx, prefillSpan := tracer.Start(ctx, "llm_d.pd_proxy.prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.prefill_target", prefillHost),
		attribute.String("llm_d.pd_proxy.connector", "sglang"),
		attribute.Bool("llm_d.pd_proxy.prefill.async", true),
	)
	prefillStart := time.Now()

	// Create separate requests for prefill and decode
	// Use context.WithoutCancel for prefillReq to prevent it from being aborted
	// if the main HTTP handler (which serves decodeReq) finishes first.
	prefillReq := cloneWithJSONBody(context.WithoutCancel(r.Context()), r, body)
	decodeReq := cloneWithJSONBody(r.Context(), r, body)

	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		prefillSpan.SetStatus(codes.Error, "failed to create prefill handler")
		prefillSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Send prefill request asynchronously
	go func() {
		defer prefillSpan.End()
		defer func() {
			if rec := recover(); rec != nil && rec != http.ErrAbortHandler {
				s.logger.Error(fmt.Errorf("panic: %v", rec), "panic in prefill request")
			}
		}()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, prefillReq)
		prefillDuration := time.Since(prefillStart)
		prefillSpan.SetAttributes(
			attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
			attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(prefillDuration.Milliseconds())),
		)
		if pw.statusCode < 200 || pw.statusCode >= 300 {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
		}
		s.logger.V(5).Info("prefill request completed", "status", pw.statusCode)
	}()

	// Decode Stage - sync
	ctx, decodeSpan := tracer.Start(ctx, "llm_d.pd_proxy.decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.connector", "sglang"),
		attribute.Bool("llm_d.pd_proxy.decode.concurrent_with_prefill", true),
	)
	decodeStart := time.Now()

	// Send decode request synchronously
	decodeReq = decodeReq.WithContext(ctx)
	s.decoderProxy.ServeHTTP(w, decodeReq)

	decodeDuration := time.Since(decodeStart)
	decodeSpan.SetAttributes(
		attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(decodeDuration.Milliseconds())),
		attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host),
	)

	// Calculate end-to-end P/D timing metrics for concurrent P/D.
	// True TTFT captures time from gateway request start to decode start.
	// In SGLang's concurrent mode, prefill duration is tracked in the async prefill span.
	if currentSpan := trace.SpanFromContext(ctx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		var trueTTFT time.Duration
		if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
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

func cloneWithJSONBody(ctx context.Context, r *http.Request, body []byte) *http.Request {
	req := r.Clone(ctx)
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	return req
}

func (s *Server) addSGLangBootstrapInfo(requestData map[string]interface{}, prefillHostPort string, roomID int64) map[string]interface{} {
	modifiedRequest := make(map[string]interface{})
	for k, v := range requestData {
		modifiedRequest[k] = v
	}

	// Generate bootstrap host from prefill host
	bootstrapHost := s.getBootstrapHost(prefillHostPort)

	// Add bootstrap information
	modifiedRequest[requestFieldBootstrapHost] = bootstrapHost
	modifiedRequest[requestFieldBootstrapPort] = sglangBootstrapPort
	modifiedRequest[requestFieldBootstrapRoom] = roomID

	s.logger.V(5).Info("bootstrap info added",
		"bootstrap_host", bootstrapHost,
		"bootstrap_port", sglangBootstrapPort,
		"bootstrap_room", roomID)

	return modifiedRequest
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
	return time.Now().UnixNano() + int64(rand.IntN(1000))
}

func (s *Server) getBootstrapHost(prefillHostPort string) string {
	// Extract hostname from prefill host
	parts := strings.Split(prefillHostPort, ":")
	return parts[0]
}
