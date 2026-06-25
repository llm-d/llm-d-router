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
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

const mooncakeBootstrapTimeout = 5 * time.Second // set to same value as the other timeout on vllm

// Safety bound on waiting for the concurrent prefill's routed_experts at decode
// finalize; prefill is normally done well before decode finishes.
const mooncakeRoutedExpertsTimeout = 30 * time.Second

const mooncakeDataParallelRankHeader = "X-data-parallel-rank" // to send rank id in header to prefill

func (s *Server) handleMooncake(w http.ResponseWriter, r *http.Request, prefillPodHostPort string) {
	s.logger.V(4).Info("running Mooncake protocol", "url", prefillPodHostPort)

	body, err := io.ReadAll(r.Body)
	if err != nil {
		if err := errorJSONInvalid(fmt.Errorf("failed to read request body: %w", err), w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	var requestData map[string]any
	if err := json.Unmarshal(body, &requestData); err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	bootstrapAddr := fmt.Sprintf("http://%s:%d", extractHost(prefillPodHostPort), s.config.MooncakeBootstrapPort)

	engineMap, err := s.getMooncakeEngineMap(r.Context(), prefillPodHostPort, bootstrapAddr)
	if err != nil {
		s.logger.Error(err, "failed to query mooncake engine ID", "bootstrap_addr", bootstrapAddr)
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	// golang map randomnize key(rankid) order to spread prefill load for multi-ranks.
	var dpRank, engineID string
	for dpRank, engineID = range engineMap {
		break
	}

	transferID := "xfer-" + newUUID()
	s.logger.V(5).Info("mooncake protocol info",
		"transfer_id", transferID,
		"bootstrap_addr", bootstrapAddr,
		"dp_rank", dpRank,
		"engine_id", engineID)

	// Build prefill request body
	prefillData := make(map[string]any)
	for k, v := range requestData {
		prefillData[k] = v
	}
	prefillData[requestFieldKVTransferParams] = map[string]any{
		requestFieldDoRemotePrefill: false,
		requestFieldDoRemoteDecode:  true,
		requestFieldTransferID:      transferID,
	}
	// update fields from original body
	prefillData[requestFieldStream] = false // return asap.
	delete(prefillData, requestFieldStreamOptions)
	prefillData[requestFieldMaxTokens] = 1
	if _, ok := prefillData[requestFieldMaxCompletionTokens]; ok {
		prefillData[requestFieldMaxCompletionTokens] = 1
	}

	prefillBody, err := json.Marshal(prefillData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	s.logger.V(5).Info("Prefill request", "body", string(prefillBody))

	// Build decode request body
	decodeData := make(map[string]any)
	for k, v := range requestData {
		decodeData[k] = v
	}
	decodeData[requestFieldKVTransferParams] = map[string]any{
		requestFieldDoRemotePrefill:     true,
		requestFieldDoRemoteDecode:      false,
		requestFieldTransferID:          transferID,
		requestFieldRemoteBootstrapAddr: bootstrapAddr,
		requestFieldRemoteEngineID:      engineID,
	}

	decodeBody, err := json.Marshal(decodeData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	s.logger.V(5).Info("Decode request", "body", string(decodeBody))

	s.handleMooncakeConcurrentRequests(w, r, prefillBody, decodeBody, prefillPodHostPort, dpRank)
}

// getMooncakeEngineMap returns the dp_rank -> engine_id mapping for the given prefill, querying the bootstrap server on first use and caching it.
func (s *Server) getMooncakeEngineMap(ctx context.Context, prefillHostPort, bootstrapAddr string) (map[string]string, error) {
	if m, ok := s.mooncakeEngineIDs.Get(prefillHostPort); ok {
		return m, nil
	}

	ctx, cancel := context.WithTimeout(ctx, mooncakeBootstrapTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, bootstrapAddr+"/query", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get mooncake bootstrap request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to query bootstrap endpoint: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if isHTTPError(resp.StatusCode) {
		return nil, fmt.Errorf("bootstrap endpoint returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read bootstrap response: %w", err)
	}

	// response format: {"0": {"engine_id": "...", ...}, "1": {...}}
	// key is dp_ranks; each rank has its own engine_id.
	var data map[string]map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("failed to parse bootstrap response: %w", err)
	}

	engineMap := make(map[string]string, len(data))
	for dpRank, entry := range data {
		if id, ok := entry["engine_id"].(string); ok {
			engineMap[dpRank] = id
		}
	}
	if len(engineMap) == 0 {
		return nil, errors.New("engine_id not found in bootstrap response")
	}
	// add the full map into LRU
	s.mooncakeEngineIDs.Add(prefillHostPort, engineMap)
	return engineMap, nil
}

func (s *Server) handleMooncakeConcurrentRequests(w http.ResponseWriter, r *http.Request, prefillBody, decodeBody []byte, prefillHost, dpRank string) {
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	// WithoutCancel for prefill so it isn't aborted when the decode response finishes first
	prefillReq := cloneRequestWithBody(context.WithoutCancel(ctx), r, prefillBody)
	decodeReq := cloneRequestWithBody(ctx, r, decodeBody)

	// Route prefill to the same DP rank whose engine_id was given to decode, so the
	// KV it produces lands on the engine decode pulls from. No-op for a single rank.
	prefillReq.Header.Set(mooncakeDataParallelRankHeader, dpRank)

	// Prefill runs in a goroutine: only populates KV cache, response is discarded.
	// Decode runs on the main thread: writes the actual response back to the client via w.
	ctx, prefillSpan := tracer.Start(ctx, "llm_d.pd_proxy.prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.prefill_target", prefillHost),
		attribute.String("llm_d.pd_proxy.connector", KVConnectorMooncake),
		attribute.Bool("llm_d.pd_proxy.prefill.async", true),
	)
	prefillStart := time.Now()

	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		prefillSpan.SetStatus(codes.Error, "failed to create prefill handler")
		prefillSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Carries the prefill's routed_experts to the decode finalizer to splice over
	// the decode response's invalid prompt-region rows. Buffered (cap 1) and always
	// sent once, so the finalizer never blocks on a non-capturing/failed prefill.
	prefillRoutedExperts := make(chan map[int]string, 1)
	go func() {
		defer prefillSpan.End()
		defer func() {
			if rec := recover(); rec != nil && rec != http.ErrAbortHandler {
				s.logger.Error(fmt.Errorf("panic: %v", rec), "panic in prefill request")
			}
		}()
		// buffered writer captures response for status check only, not sent to client
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, prefillReq)
		prefillDuration := time.Since(prefillStart)
		prefillSpan.SetAttributes(
			attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
			attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(prefillDuration.Milliseconds())),
		)
		var routedExperts map[int]string
		if isHTTPError(pw.statusCode) {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
		} else {
			var prefillerResponse map[string]any
			if err := json.Unmarshal(pw.bodyBytes(), &prefillerResponse); err == nil {
				routedExperts = extractRoutedExperts(prefillerResponse)
			}
		}
		prefillRoutedExperts <- routedExperts
		s.logger.V(5).Info("mooncake prefill request completed", "status", pw.statusCode)
	}()

	// Decode Stage
	ctx, decodeSpan := tracer.Start(ctx, "llm_d.pd_proxy.decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.connector", KVConnectorMooncake),
		attribute.Bool("llm_d.pd_proxy.decode.concurrent_with_prefill", true),
	)
	decodeStart := time.Now()

	decodeReq = decodeReq.WithContext(ctx)
	// Resolve the prefill routing at finalize (it is ready by then, since decode
	// pulled its KV); the timeout just forwards unspliced rather than blocking.
	decodeWriter, finalizeRoutedExperts := newDeferredRoutedExpertsResponseWriter(w, func() map[int]string {
		select {
		case re := <-prefillRoutedExperts:
			return re
		case <-time.After(mooncakeRoutedExpertsTimeout):
			s.logger.Info("timed out waiting for prefill routed_experts; forwarding decode response unspliced")
			return nil
		}
	})
	s.decoderProxy.ServeHTTP(decodeWriter, decodeReq)
	if err := finalizeRoutedExperts(); err != nil {
		s.logger.Error(err, "failed to flush routed experts response writer")
	}

	decodeDuration := time.Since(decodeStart)
	decodeSpan.SetAttributes(
		attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(decodeDuration.Milliseconds())),
		attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host),
	)

	// Calculate end-to-end P/D timing metrics for concurrent P/D.
	// True TTFT captures time from gateway request start to decode start.
	// In Mooncake's concurrent mode, prefill duration is tracked in the async prefill span.
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
