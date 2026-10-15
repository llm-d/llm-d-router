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

package proxy

import (
	"encoding/json"
	"fmt"
	"maps"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// p2pPeer is a pod's OffloadingConnector P2P tier address.
type p2pPeer struct {
	host string
	port int
}

func (p *p2pPeer) String() string {
	return net.JoinHostPort(p.host, strconv.Itoa(p.port))
}

// p2pRouting carries the OffloadingConnector P2P tier inputs EPP computed for
// a request. prefill is the prefill pod's tier address from
// routing.KVCachePrefillHeader, nil when the header is absent or malformed; a
// prefill request without it fails. source is the request body's
// kv_transfer_params.remote_kv_source block, nil when absent or rejected. body
// is the parsed request body the source was taken from, without the source;
// nil when the dispatcher left the body unread.
type p2pRouting struct {
	prefill *p2pPeer
	source  *kvSource
	body    map[string]any
}

// kvSource is a validated kv_transfer_params.remote_kv_source block: the peer
// a pod pulls cached prefix blocks from over the P2P tier.
type kvSource struct {
	kvRequestID string
	p2pPeer
}

// params returns the block in its kv_transfer_params wire shape.
func (k *kvSource) params() map[string]any {
	return reqcommon.P2PPeerParams(k.kvRequestID, k.host, k.port)
}

// requestBody returns the parsed request body: the one the dispatcher read
// for the pull source, or r's body read here when the dispatcher left it
// unread.
func (s *Server) requestBody(r *http.Request, w http.ResponseWriter, p2p p2pRouting) (map[string]any, bool) {
	if p2p.body != nil {
		return p2p.body, true
	}
	_, body, ok := s.readJSONBody(r, w)
	return body, ok
}

// handleP2P implements the vLLM OffloadingConnector P2P orchestration contract. The
// prefiller stores KV under a kv_request_id with no peer address; the decoder
// pulls it using the prefiller's OffloadingConnector P2P tier host/port. The
// prefill request runs to completion before decode is dispatched, so the decoder's
// fetch finds the blocks already stored, matching the NIXL path.
func (s *Server) handleP2P(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, p2p p2pRouting, apiType reqcommon.APIType) {
	if p2p.prefill == nil {
		err := fmt.Errorf("request header %s is absent or not a host:port with a TCP port; the EPP must run p2p-source-producer", routing.KVCachePrefillHeader)
		s.logger.Error(err, "cannot build the decode request", "prefill_host", extractHost(prefillPodHostPort))
		if err := errorInternalServerError(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	requestData, ok := s.requestBody(r, w, p2p)
	if !ok {
		return
	}

	kvRequestID := newUUID()
	kv := make([]any, 0, 8)
	kv = append(kv,
		"prefill_host", extractHost(prefillPodHostPort),
		"kv_request_id", kvRequestID,
		"p2p_prefill", p2p.prefill.String())
	if p2p.source != nil {
		kv = append(kv, "p2p_source", p2p.source.String())
	}
	s.logger.Info("running P2P protocol", kv...)

	// Prefill request: store KV under kv_request_id, no peer address. Capped to a
	// single output token so the prefiller returns as soon as KV is stored.
	prefillData := maps.Clone(requestData)
	prefillKVParams := map[string]any{
		reqcommon.FieldRemoteDecoder: map[string]any{
			reqcommon.FieldKVRequestID: kvRequestID,
		},
	}
	addP2PPullToPrefill(prefillKVParams, p2p.source)
	prefillData[reqcommon.FieldKVTransferParams] = prefillKVParams
	reqcommon.CapSingleToken(prefillData, apiType)

	prefillBody, err := json.Marshal(prefillData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("prefill request body", logging.HTTPBodyKey, string(prefillBody))
	}

	// Decode request: pull KV from the prefiller's OffloadingConnector P2P tier. Original body
	// (streaming, token limits) is preserved.
	decodeData := maps.Clone(requestData)
	decodeData[reqcommon.FieldKVTransferParams] = map[string]any{
		reqcommon.FieldRemotePrefiller: reqcommon.P2PPeerParams(kvRequestID, p2p.prefill.host, p2p.prefill.port),
	}

	decodeBody, err := json.Marshal(decodeData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("decode request body", logging.HTTPBodyKey, string(decodeBody))
	}

	s.handleP2PSequentialRequests(w, r, prefillBody, decodeBody, prefillPodHostPort)
}

// handleP2PSequentialRequests runs the prefill request to completion, then
// dispatches decode.
//
// The decoder's fetch has to find the prefiller's blocks already stored. A
// fetch that arrives first parks as unsatisfied demand on the prefiller's
// session and burns the OffloadingConnector's fixed load deadline waiting for
// KV that has not been produced yet; on expiry the decoder aborts the load and
// recomputes the prompt locally, which is the work disaggregation exists to
// avoid. Sequencing the requests also matches the NIXL connector, which forwards to
// the prefiller and waits for it to return before dispatching decode.
func (s *Server) handleP2PSequentialRequests(w http.ResponseWriter, r *http.Request, prefillBody, decodeBody []byte, prefillHost string) {
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// ---------- Prefill request: store KV, response discarded ----------
	prefillCtx, prefillSpan := tracer.Start(ctx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillTarget(prefillHost),
		semconv.LLMDPDProxyConnector(constants.KVConnectorOffloading),
		semconv.LLMDPDProxyPrefillAsync(false),
	)
	prefillStart := time.Now()

	pw := &bufferedResponseWriter{}
	prefillHandler.ServeHTTP(pw, cloneRequestWithBody(prefillCtx, r, prefillBody))
	prefillDuration := time.Since(prefillStart)
	metrics.RecordPrefillDuration(prefillDuration)

	prefillFailed := isHTTPError(pw.statusCode)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
		semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
	)
	if prefillFailed {
		metrics.RecordError(metrics.StagePrefill)
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
	}
	prefillSpan.End()
	s.logger.V(logging.DEBUG).Info("PD Multi Tier prefill request completed", "status", pw.statusCode)

	if prefillFailed {
		// Return the prefill error verbatim; decode is never dispatched, so it
		// cannot fetch KV that was never stored.
		for key, values := range pw.Header() {
			for _, v := range values {
				w.Header().Add(key, v)
			}
		}
		status := pw.statusCode
		if status < http.StatusContinue {
			// The prefiller never wrote a status (transport failure before the
			// proxy's error handler ran). WriteHeader panics below 100.
			status = http.StatusBadGateway
		}
		w.WriteHeader(status)
		if _, writeErr := w.Write(pw.bodyBytes()); writeErr != nil {
			s.logger.Error(writeErr, "failed to send prefill error to client")
		}
		return
	}

	// ---------- Decode request: pulls the stored KV and streams the response ----------
	decodeCtx, decodeSpan := tracer.Start(ctx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyConnector(constants.KVConnectorOffloading),
		semconv.LLMDPDProxyDecodeConcurrentWithPrefill(false),
	)
	decodeStart := time.Now()

	decodeWriter, decodeStatus := captureResponseStatus(w)
	decodeReturned := false
	defer recordDecodeAbort(&decodeReturned, decodeStart)
	s.decoderProxy.ServeHTTP(decodeWriter, cloneRequestWithBody(decodeCtx, r, decodeBody))
	decodeReturned = true

	decodeDuration := time.Since(decodeStart)
	metrics.RecordDecodeDuration(decodeDuration)
	if decodeStatus.failed() {
		metrics.RecordError(metrics.StageDecode)
		decodeSpan.SetStatus(codes.Error, "decode request failed")
	}
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyDecodeDurationMs(float64(decodeDuration.Milliseconds())),
		semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host),
	)

	// End-to-end P/D timing. True TTFT captures time from gateway request start
	// to decode start; prefill duration is tracked in the prefill span.
	var totalDuration, trueTTFT time.Duration
	if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
		if requestStart, ok := requestStartValue.(time.Time); ok {
			totalDuration = time.Since(requestStart)
			trueTTFT = decodeStart.Sub(requestStart)
		}
	}
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyTotalDurationMs(float64(totalDuration.Milliseconds())),
		semconv.LLMDPDProxyTrueTTFTMs(float64(trueTTFT.Milliseconds())),
		semconv.LLMDPDProxyDecodeDurationMsSummary(float64(decodeDuration.Milliseconds())),
		semconv.LLMDPDProxyConcurrentPD(false),
	)
}

// p2pPullAvailable reports whether this deployment can pull cached prefix over
// the OffloadingConnector P2P tier. That tier is the PD connector itself when
// KVConnector is offloading, or is composed alongside NIXL via MultiConnector
// (declared with --enable-p2p-pull) when the PD connector is NIXLv2. On any
// other connector --enable-p2p-pull has no effect, since no MultiConnector
// routes the remote_kv_source params to an OffloadingConnector.
func (s *Server) p2pPullAvailable() bool {
	return s.config.KVConnector == constants.KVConnectorOffloading ||
		(s.config.EnableP2PPull && s.config.KVConnector == constants.KVConnectorNIXLV2)
}

// addP2PPullToPrefill adds the OffloadingConnector P2P pull block to a prefill
// request's kv_transfer_params so the prefiller pulls cached prefix from
// source while keeping its own computed blocks available for the decoder.
// No-op when there is no source. The remote_kv_source key composes with NIXL
// params: vLLM's MultiConnector routes it to the OffloadingConnector and the
// NIXL fields to the NixlConnector.
func addP2PPullToPrefill(prefillKVParams map[string]any, source *kvSource) {
	if source != nil {
		prefillKVParams[reqcommon.FieldRemoteKVSource] = source.params()
	}
}

// kvSourceFromBody removes kv_transfer_params.remote_kv_source from the parsed
// body, and kv_transfer_params itself when that was its only key, and reports
// whether the source was present. It returns the block when it is
// well formed and names an allowed peer: a string kv_request_id, a string
// remote_host and an integer remote_port in the TCP range, with the host on
// the SSRF allowlist. EPP writes the block, so the sidecar copies it and
// derives nothing. A rejected block is logged and dropped, so the request
// proceeds without a pull.
func (s *Server) kvSourceFromBody(logger logr.Logger, clientIP string, requestData map[string]any) (source *kvSource, present bool) {
	kvParams, _ := requestData[reqcommon.FieldKVTransferParams].(map[string]any)
	raw, ok := kvParams[reqcommon.FieldRemoteKVSource]
	if !ok {
		// EPP writes the header and the block in one step, so a header alone
		// is a client or an EPP that does not match this sidecar.
		logger.Info("warning: KV cache source header without a remote_kv_source block in the body; proceeding without a pull",
			"clientIP", clientIP)
		return nil, false
	}
	delete(kvParams, reqcommon.FieldRemoteKVSource)
	if len(kvParams) == 0 {
		delete(requestData, reqcommon.FieldKVTransferParams)
	}

	block, _ := raw.(map[string]any)
	kvRequestID, _ := block[reqcommon.FieldKVRequestID].(string)
	host, _ := block[reqcommon.FieldRemoteHost].(string)
	port, portOK := toWholeInt(block[reqcommon.FieldRemotePort])
	switch {
	case kvRequestID == "" || host == "" || !portOK || !routing.IsTCPPort(port):
		logger.Info("ignoring malformed remote_kv_source in request body", "value", raw)
		return nil, true
	case !s.allowlistValidator.IsAllowedHost(host):
		logger.Info("SSRF protection: KV cache source not in allowlist, ignoring",
			"target", host, "clientIP", clientIP)
		return nil, true
	}
	return &kvSource{kvRequestID: kvRequestID, p2pPeer: p2pPeer{host: host, port: port}}, true
}

// parsePrefillP2P returns the prefill pod's P2P tier address carried by
// routing.KVCachePrefillHeader, or nil when the header is absent or not a
// host:port with a TCP port.
func parsePrefillP2P(logger logr.Logger, value string) *p2pPeer {
	if value == "" {
		return nil
	}
	host, portStr, err := net.SplitHostPort(value)
	port, convErr := strconv.Atoi(portStr)
	if err != nil || host == "" || convErr != nil || !routing.IsTCPPort(port) {
		logger.Info("ignoring malformed prefill P2P address header", "value", value)
		return nil
	}
	return &p2pPeer{host: host, port: port}
}

// decodeWithP2PSource serves a decoder-only request through the local vLLM
// with kv_transfer_params rebuilt as {remote_kv_source: source}, so the engine
// looks up and pulls cached prefix blocks from the peer instead of recomputing
// them. The sidecar owns the field, so client-supplied keys are dropped. It
// routes through dispatchDecode so chunked decode still applies.
func (s *Server) decodeWithP2PSource(w http.ResponseWriter, r *http.Request, requestData map[string]any, source *kvSource) {
	requestData[reqcommon.FieldKVTransferParams] = map[string]any{reqcommon.FieldRemoteKVSource: source.params()}

	s.logger.Info("running P2P source protocol",
		"kv_request_id", source.kvRequestID,
		"p2p_source", source.String())

	newBody, err := json.Marshal(requestData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("decoder request body with p2p source", logging.HTTPBodyKey, string(newBody))
	}

	s.dispatchDecode(w, cloneRequestWithBody(r.Context(), r, newBody), requestData)
}
