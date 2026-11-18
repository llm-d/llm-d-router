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
	"context"
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
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// p2pPeer is a pod's OffloadingConnector P2P tier address.
type p2pPeer struct {
	host string
	port int
}

// String formats the address as host:port, or "" for an absent peer.
func (p *p2pPeer) String() string {
	if p == nil {
		return ""
	}
	return net.JoinHostPort(p.host, strconv.Itoa(p.port))
}

// p2pRouting carries the OffloadingConnector P2P tier inputs EPP computed for
// a request. prefill is the prefill pod's tier address from
// routing.KVCachePrefillHeader, nil when the header is absent or malformed; a
// prefill request without it fails. source is the request body's
// kv_transfer_params.remote_kv_source block, nil when absent or rejected. body
// is the parsed request body the source was taken from, without the source;
// nil when the dispatcher left the body unread. raw is body in its JSON wire
// form, for callers that need bytes rather than the parsed map.
type p2pRouting struct {
	prefill   *p2pPeer
	source    *kvSource
	body      map[string]any
	raw       []byte
	grpcReq   *vllmwire.Request
	ecParams  map[string]any
	encodeRan bool
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

// pullSource returns the source a prefill request pulls from: p.source, or nil
// when it names the prefill pod's own P2P tier. EPP never names the prefill
// endpoint as the source, so a match here is an EPP without that check. On
// vLLM v0.32.0 the prefill pod would dial its own tier and wait out the NIXL
// handshake timeout. Another rank of the prefill pod differs in port and is a
// valid peer.
func (p p2pRouting) pullSource(logger logr.Logger) *kvSource {
	if p.source != nil && p.prefill != nil && p.source.p2pPeer == *p.prefill {
		logger.Info("warning: KV cache source is the prefill pod's own P2P tier; proceeding without a pull",
			"p2p_source", p.source.String())
		return nil
	}
	return p.source
}

// requestBody returns the request body as raw bytes and as the parsed map:
// the ones the dispatcher read for the pull source, or r's body read here
// when the dispatcher left it unread.
func (s *Server) requestBody(r *http.Request, w http.ResponseWriter, p2p p2pRouting) ([]byte, map[string]any, bool) {
	if p2p.body != nil {
		return p2p.raw, p2p.body, true
	}
	return s.readJSONBody(r, w)
}

// handleP2P implements the vLLM OffloadingConnector P2P orchestration contract. The
// prefiller stores KV under a kv_request_id with no peer address; the decoder
// pulls it using the prefiller's OffloadingConnector P2P tier host/port. The
// prefill request runs to completion before decode is dispatched, so the decoder's
// fetch finds the blocks already stored, matching the NIXL path.
func (s *Server) handleP2P(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, p2p p2pRouting, apiType reqcommon.APIType) {
	logger := log.FromContext(r.Context())
	if p2p.prefill == nil {
		err := fmt.Errorf("request header %s is absent or not a host:port with a TCP port; the EPP must run p2p-source-producer with prefillProfileName matching the disaggregation prefill profile", routing.KVCachePrefillHeader)
		logger.Error(err, "cannot build the decode request", "prefill_host", extractHost(prefillPodHostPort))
		if err := errorInternalServerError(w, r, err); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}
	original, requestData, ok := s.requestBody(r, w, p2p)
	if !ok {
		return
	}
	toolFields, _ := captureInferenceToolCallingRequest(r, apiType, original)

	source := p2p.pullSource(logger)

	kvRequestID := newUUID()
	kv := make([]any, 0, 8)
	kv = append(kv,
		"prefill_host", extractHost(prefillPodHostPort),
		"kv_request_id", kvRequestID,
		"p2p_prefill", p2p.prefill.String())
	if source != nil {
		kv = append(kv, "p2p_source", source.String())
	}
	logger.Info("running P2P protocol", kv...)

	// Prefill request: store KV under kv_request_id, no peer address. Capped to a
	// single output token so the prefiller returns as soon as KV is stored.
	prefillData := maps.Clone(requestData)
	prefillKVParams := reqcommon.RemoteDecoderParams(kvRequestID)
	promptTokenIDs := s.clientPromptTokenIDs(requestData)
	addP2PPullToPrefill(prefillKVParams, source)
	setPromptTokenIDs(prefillKVParams, promptTokenIDs)
	prefillData[reqcommon.FieldKVTransferParams] = prefillKVParams
	reqcommon.CapSingleToken(prefillData, apiType)

	prefillBody, err := json.Marshal(prefillData)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := logger.V(logging.TRACE); v.Enabled() {
		v.Info("prefill request body", logBodyBytesKey, len(prefillBody))
	}

	// Decode request: pull KV from the prefiller's OffloadingConnector P2P tier. Original body
	// (streaming, token limits) is preserved.
	decodeData := maps.Clone(requestData)
	decodeData[reqcommon.FieldKVTransferParams] = reqcommon.RemotePrefillerParams(kvRequestID, p2p.prefill.host, p2p.prefill.port)
	setPromptTokenIDs(decodeData[reqcommon.FieldKVTransferParams].(map[string]any), promptTokenIDs)

	decodeBody, err := json.Marshal(decodeData)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := logger.V(logging.TRACE); v.Enabled() {
		v.Info("decode request body", logBodyBytesKey, len(decodeBody))
	}

	s.handleP2PSequentialRequests(w, r, prefillBody, decodeBody, prefillPodHostPort, toolFields)
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
// sequentialPDLeg adapts one wire format (JSON or vLLM gRPC) to sequential P/D.
type sequentialPDLeg interface {
	report(ctx context.Context)
	runPrefill(ctx context.Context, prefillHandler http.Handler, pw *bufferedResponseWriter, r *http.Request, prefillSpan trace.Span) error
	completePrefill(ctx context.Context, w http.ResponseWriter, r *http.Request, pw *bufferedResponseWriter, prefillErr error, prefillSpan trace.Span) bool
	serveDecode(ctx context.Context, decodeWriter http.ResponseWriter, r *http.Request, decodeSpan trace.Span)
}

type httpSequentialPDLeg struct {
	s           *Server
	prefillBody []byte
	decodeBody  []byte
	toolFields  *toolCallingRequest
}

func (l *httpSequentialPDLeg) report(ctx context.Context) {
	l.toolFields.report(ctx)
}

func (l *httpSequentialPDLeg) runPrefill(prefillCtx context.Context, prefillHandler http.Handler, pw *bufferedResponseWriter, r *http.Request, prefillSpan trace.Span) error {
	l.toolFields.compareDispatchedForSpan(l.prefillBody, prefillSpan)
	prefillHandler.ServeHTTP(pw, cloneRequestWithBody(prefillCtx, r, l.prefillBody))
	return nil
}

func (l *httpSequentialPDLeg) completePrefill(_ context.Context, w http.ResponseWriter, r *http.Request, pw *bufferedResponseWriter, _ error, prefillSpan trace.Span) bool {
	logger := log.FromContext(r.Context())
	prefillFailed := isHTTPError(pw.statusCode)
	if prefillFailed {
		observeStageError(r.Context(), routing.StagePrefill)
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
	}
	prefillSpan.End()
	logger.V(logging.DEBUG).Info("PD Multi Tier prefill request completed", "status", pw.statusCode)
	if !prefillFailed {
		return true
	}
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
		logger.Error(writeErr, "failed to send prefill error to client")
	}
	return false
}

func (l *httpSequentialPDLeg) serveDecode(decodeCtx context.Context, decodeWriter http.ResponseWriter, r *http.Request, decodeSpan trace.Span) {
	l.toolFields.compareDispatchedForSpan(l.decodeBody, decodeSpan)
	l.s.decoderProxy.ServeHTTP(decodeWriter, cloneRequestWithBody(decodeCtx, r, l.decodeBody))
}

func (s *Server) handleP2PSequentialRequests(w http.ResponseWriter, r *http.Request, prefillBody, decodeBody []byte, prefillHost string, toolFields *toolCallingRequest) {
	s.runSequentialPD(w, r, prefillHost, constants.KVConnectorOffloading, &httpSequentialPDLeg{
		s:           s,
		prefillBody: prefillBody,
		decodeBody:  decodeBody,
		toolFields:  toolFields,
	})
}

func (s *Server) runSequentialPD(w http.ResponseWriter, r *http.Request, prefillHost, connector string, leg sequentialPDLeg) {
	defer leg.report(r.Context())
	logger := log.FromContext(r.Context())
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		logger.Error(err, "failed to create prefill handler")
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}

	prefillCtx, prefillSpan := tracer.Start(ctx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillTarget(prefillHost),
		semconv.LLMDPDProxyConnector(connector),
		semconv.LLMDPDProxyPrefillAsync(false),
	)
	prefillStart := time.Now()

	pw := &bufferedResponseWriter{}
	prefillErr := leg.runPrefill(prefillCtx, prefillHandler, pw, r, prefillSpan)
	prefillDuration := time.Since(prefillStart)
	observeStageDuration(r.Context(), routing.StagePrefill, prefillDuration)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
		semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
	)
	if !leg.completePrefill(prefillCtx, w, r, pw, prefillErr, prefillSpan) {
		return
	}

	decodeCtx, decodeSpan := tracer.Start(ctx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyConnector(connector),
		semconv.LLMDPDProxyDecodeConcurrentWithPrefill(false),
	)
	decodeStart := time.Now()

	decodeWriter, decodeStatus := captureResponseStatus(w)
	decodeReturned := false
	defer recordDecodeAbort(r.Context(), &decodeReturned, decodeStart)
	leg.serveDecode(decodeCtx, decodeWriter, r, decodeSpan)
	decodeReturned = true

	decodeDuration := time.Since(decodeStart)
	observeStageDuration(r.Context(), routing.StageDecode, decodeDuration)
	if decodeStatus.failed() {
		observeStageError(r.Context(), routing.StageDecode)
		decodeSpan.SetStatus(codes.Error, "decode request failed")
	}
	decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDurationMs(float64(decodeDuration.Milliseconds())))
	if s.config.DecoderURL != nil {
		decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host))
	}
	recordSequentialPDTimingAttrs(ctx, decodeSpan, prefillStart, prefillDuration, decodeStart, decodeDuration)
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
// whether the source was present. It returns the source when the block is
// well formed and names an allowed peer: a string kv_request_id, a string
// remote_host and an integer remote_port in the TCP range, with the host on
// the SSRF allowlist. Only those three fields are kept; any other key in the
// block is dropped. A rejected block is logged and dropped, so the request
// proceeds without a pull.
func (s *Server) kvSourceFromBody(logger logr.Logger, clientIP string, requestData map[string]any) (source *kvSource, present bool) {
	// kv_transfer_params stays as json.RawMessage in the decoded body (it is
	// deliberately not in inspectedRequestFields), so parse the raw form.
	rawValue, exists := requestData[reqcommon.FieldKVTransferParams]
	if !exists || rawValue == nil {
		logger.Info("warning: KV cache source header without a remote_kv_source block in the body; proceeding without a pull",
			"clientIP", clientIP)
		return nil, false
	}
	// Decode only the remote_kv_source key, leaving prompt_token_ids and any
	// other fields as raw bytes: materializing a 1M-token ID array costs
	// ~286 ms and 128 MB (Go 1.27 benchmark) versus ~125 ms and 15 MB when
	// kept raw, and the sidecar re-encodes the array on each hop.
	var rawParams map[string]json.RawMessage
	switch v := rawValue.(type) {
	case map[string]any:
		// Already decoded (unit tests build maps directly).
		rawParams = make(map[string]json.RawMessage, len(v))
		for k, val := range v {
			if encoded, err := json.Marshal(val); err == nil {
				rawParams[k] = encoded
			}
		}
	case json.RawMessage:
		if err := json.Unmarshal(v, &rawParams); err != nil || rawParams == nil {
			logger.Info("warning: KV cache source header without a remote_kv_source block in the body; proceeding without a pull",
				"clientIP", clientIP)
			return nil, false
		}
	default:
		logger.Info("warning: KV cache source header without a remote_kv_source block in the body; proceeding without a pull",
			"clientIP", clientIP)
		return nil, false
	}
	sourceRaw, ok := rawParams[reqcommon.FieldRemoteKVSource]
	if !ok {
		// EPP writes the header and the block in one step, so a header alone
		// is a client or an EPP that does not match this sidecar.
		logger.Info("warning: KV cache source header without a remote_kv_source block in the body; proceeding without a pull",
			"clientIP", clientIP)
		return nil, false
	}
	// Remove the source key, leaving other keys (including prompt_token_ids)
	// as untouched raw bytes.
	delete(rawParams, reqcommon.FieldRemoteKVSource)
	if len(rawParams) == 0 {
		delete(requestData, reqcommon.FieldKVTransferParams)
	} else if cleaned, err := json.Marshal(rawParams); err == nil {
		requestData[reqcommon.FieldKVTransferParams] = json.RawMessage(cleaned)
	}
	var block map[string]any
	if err := json.Unmarshal(sourceRaw, &block); err != nil {
		logger.Info("ignoring malformed remote_kv_source in request body", "value", string(sourceRaw))
		return nil, true
	}
	return s.validateKVSourceBlock(logger, clientIP, block), true
}

// validateKVSourceBlock returns the source when block is a well formed
// remote_kv_source that names an allowed peer, or nil.
func (s *Server) validateKVSourceBlock(logger logr.Logger, clientIP string, raw any) *kvSource {
	block, _ := raw.(map[string]any)
	kvRequestID, _ := block[reqcommon.FieldKVRequestID].(string)
	host, _ := block[reqcommon.FieldRemoteHost].(string)
	port, portOK := toWholeInt(block[reqcommon.FieldRemotePort])
	switch {
	case kvRequestID == "" || host == "" || !portOK || !routing.IsTCPPort(port):
		logger.Info("ignoring malformed remote_kv_source in request body", "value", raw)
		return nil
	case !s.allowlistValidator.IsAllowedHost(host):
		logger.Info("SSRF protection: KV cache source not in allowlist, ignoring",
			"target", host, "clientIP", clientIP)
		return nil
	}
	return &kvSource{kvRequestID: kvRequestID, p2pPeer: p2pPeer{host: host, port: port}}
}

// parsePrefillP2P returns the prefill pod's P2P tier address carried by
// routing.KVCachePrefillHeader, or nil when the header is absent or not a
// host:port with a TCP port.
func parsePrefillP2P(logger logr.Logger, value string) *p2pPeer {
	if value == "" {
		return nil
	}
	host, port, ok := routing.ParseHostPort(value)
	if !ok {
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
func (s *Server) decodeWithP2PSource(w http.ResponseWriter, r *http.Request, p2p p2pRouting) {
	if isVLLMGRPCRequest(r) {
		s.decodeWithP2PSourceVLLMGRPC(w, r, p2p)
		return
	}
	logger := log.FromContext(r.Context())
	raw, requestData, ok := s.requestBody(r, w, p2p)
	if !ok {
		return
	}
	apiType := reqcommon.DetectAPIType(r.URL.Path)
	toolFields, _ := captureInferenceToolCallingRequest(r, apiType, raw)
	defer toolFields.report(r.Context())

	source := p2p.source
	// Rebuild kv_transfer_params from scratch: the sidecar owns this field, so
	// client-supplied keys are dropped rather than forwarded to vLLM, except
	// prompt_token_ids, which only lets the engine skip re-tokenizing.
	sourceKVParams := map[string]any{reqcommon.FieldRemoteKVSource: source.params()}
	setPromptTokenIDs(sourceKVParams, s.clientPromptTokenIDs(requestData))
	requestData[reqcommon.FieldKVTransferParams] = sourceKVParams

	logger.Info("running P2P source protocol",
		"kv_request_id", source.kvRequestID,
		"p2p_source", source.String())

	newBody, err := json.Marshal(requestData)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := logger.V(logging.TRACE); v.Enabled() {
		v.Info("decoder request body with p2p source", logBodyBytesKey, len(newBody))
	}

	s.dispatchDecode(w, cloneRequestWithBody(r.Context(), r, newBody), requestData, newBody, toolFields)
}
