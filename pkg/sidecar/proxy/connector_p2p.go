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
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"

	logging "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// handleP2P implements the vLLM OffloadingConnector P2P orchestration contract. The
// prefiller stores KV under a kv_request_id with no peer address; the decoder
// pulls it using the prefiller's OffloadingConnector P2P tier host/port. Both legs are
// dispatched concurrently: the connector parks any KV blocks stored before the
// decoder's fetch binds the session, so ordering between the legs is safe.
func (s *Server) handleP2P(w http.ResponseWriter, r *http.Request, prefillPodHostPort, kvCacheSource string) {
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

	kvRequestID := newUUID()
	prefillP2PPort := s.p2pPortFor(prefillPodHostPort)
	s.logger.Info("running P2P protocol",
		"prefill_host", extractHost(prefillPodHostPort),
		"kv_request_id", kvRequestID,
		"p2p_connector_port", prefillP2PPort)

	// Prefill leg: store KV under kv_request_id, no peer address. Capped to a
	// single output token so the prefiller returns as soon as KV is stored.
	prefillData := make(map[string]any, len(requestData)+1)
	for k, v := range requestData {
		prefillData[k] = v
	}
	prefillKVParams := map[string]any{
		requestFieldRemoteDecoder: map[string]any{
			requestFieldKVRequestID: kvRequestID,
		},
	}
	s.addP2PPullToPrefill(prefillKVParams, kvCacheSource, prefillPodHostPort)
	prefillData[requestFieldKVTransferParams] = prefillKVParams
	reqcommon.PrimeSingleTokenRequest(prefillData, requestData)

	prefillBody, err := json.Marshal(prefillData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("prefill request body", "body", string(prefillBody))
	}

	// Decode leg: pull KV from the prefiller's OffloadingConnector P2P tier. Original body
	// (streaming, token limits) is preserved.
	decodeData := make(map[string]any, len(requestData)+1)
	for k, v := range requestData {
		decodeData[k] = v
	}
	decodeData[requestFieldKVTransferParams] = map[string]any{
		requestFieldRemotePrefiller: map[string]any{
			requestFieldKVRequestID: kvRequestID,
			requestFieldRemoteHost:  extractHost(prefillPodHostPort),
			requestFieldRemotePort:  prefillP2PPort,
		},
	}

	decodeBody, err := json.Marshal(decodeData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("decode request body", "body", string(decodeBody))
	}

	s.runConcurrentPD(w, r, prefillBody, decodeBody, prefillPodHostPort, KVConnectorOffloading, nil)
}

// p2pPullAvailable reports whether this deployment can pull cached prefix over
// the OffloadingConnector P2P tier. That tier is the PD connector itself when
// KVConnector is offloading, or is composed alongside NIXL via MultiConnector
// (declared with --enable-p2p-pull) when the PD connector is NIXLv2. On any
// other connector --enable-p2p-pull has no effect, since no MultiConnector
// routes the remote_kv_source params to an OffloadingConnector.
func (s *Server) p2pPullAvailable() bool {
	return s.config.KVConnector == KVConnectorOffloading ||
		(s.config.EnableP2PPull && s.config.KVConnector == KVConnectorNIXLV2)
}

// addP2PPullToPrefill adds the OffloadingConnector P2P pull block to a prefill
// leg's kv_transfer_params so the prefiller pulls cached prefix from
// kvCacheSource while keeping its own computed blocks available for the
// decoder. It is a no-op when no source is set or the source is the selected
// prefill endpoint, since there is nothing to pull from oneself. The
// remote_kv_source key composes with NIXL params: vLLM's MultiConnector
// routes it to the OffloadingConnector and the NIXL fields to the
// NixlConnector.
func (s *Server) addP2PPullToPrefill(prefillKVParams map[string]any, kvCacheSource, prefillPodHostPort string) {
	source := normalizeEndpoint(kvCacheSource)
	prefiller := normalizeEndpoint(prefillPodHostPort)
	if source != "" && source != prefiller {
		prefillKVParams[requestFieldRemoteKVSource] = s.p2pSourceParams(source)
	}
}

// normalizeEndpoint canonicalizes an endpoint for equality comparison.
// Scheme-qualified values (any scheme, not only http) are parsed as URLs
// and reduced to their host:port; bare values are treated as host:port
// directly. Both forms are re-joined through the net package so IPv6
// literals compare consistently; values without a parseable port are
// returned scheme-stripped, matching extractHost's fallback.
func normalizeEndpoint(s string) string {
	hostport := s
	if scheme, rest, ok := strings.Cut(s, "://"); ok && scheme != "" {
		if u, err := url.Parse(s); err == nil && u.Host != "" {
			hostport = u.Host
		} else {
			hostport = rest
		}
	}
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		return hostport
	}
	return net.JoinHostPort(host, port)
}

// p2pSourceParams builds the kv_transfer_params.remote_kv_source block for a
// pull from sourceHostPort's OffloadingConnector P2P tier. The kv_request_id is its
// own fresh UUID: in P2P mode it is consumer-side only.
func (s *Server) p2pSourceParams(sourceHostPort string) map[string]any {
	return map[string]any{
		requestFieldKVRequestID: newUUID(),
		requestFieldRemoteHost:  extractHost(sourceHostPort),
		requestFieldRemotePort:  s.p2pPortFor(sourceHostPort),
	}
}

// p2pPortFor resolves the P2P tier control port on the target endpoint. The
// sidecar serves rank r on <base port>+r (data_parallel.go), so the routed
// endpoint's port encodes the pod-local rank. vLLM binds the tier at the
// engine's configured P2P base + the global data_parallel_index, so the
// mapping is correct when each pod is its own DP group (local rank ==
// global index). Multi-pod DP groups (e.g. LWS wide-EP) preserve it by
// compensating the engine's per-pod base (P2P_BASE = <p2p-connector-port>
// - START_RANK; see docs/disaggregation.md), so every pod binds local rank
// r at <p2p-connector-port>+r and this pod-local derivation is the
// contract there too. A port outside the rank range (or unparsable) falls
// back to the base P2P port, which is rank 0's tier.
func (s *Server) p2pPortFor(targetHostPort string) int {
	base := s.config.P2PConnectorPort
	if s.config.DataParallelSize <= 1 || s.dpBasePort == 0 {
		return base
	}
	// Backward compatible behavior: trim `http:` prefix (see createProxyHandler).
	targetHostPort, _ = strings.CutPrefix(targetHostPort, "http://")
	_, portStr, err := net.SplitHostPort(targetHostPort)
	if err != nil {
		s.logger.V(logging.DEBUG).Info("P2P target has no parsable port, using base P2P port",
			"target", targetHostPort, "error", err)
		return base
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		s.logger.V(logging.DEBUG).Info("P2P target port is not numeric, using base P2P port",
			"target", targetHostPort, "error", err)
		return base
	}
	rank := port - s.dpBasePort
	if rank < 0 || rank >= s.config.DataParallelSize {
		s.logger.V(logging.DEBUG).Info("P2P target port outside the DP rank range, using base P2P port",
			"target", targetHostPort, "dpBasePort", s.dpBasePort,
			"dataParallelSize", s.config.DataParallelSize)
		return base
	}
	return base + rank
}

// decodeWithP2PSource serves a decoder-only request through the local vLLM
// with kv_transfer_params.remote_kv_source injected, so the engine looks up and pulls
// cached prefix blocks from the peer at sourceHostPort instead of recomputing
// them. It replaces any client-supplied kv_transfer_params (the sidecar owns
// that field) and routes through dispatchDecode so chunked decode still
// applies. When sourceHostPort resolves to this rank's own serving endpoint,
// injecting would tell the engine to pull the prefix it is already
// computing, so it decodes normally; another rank on the same pod is a
// valid peer and does get the injection.
func (s *Server) decodeWithP2PSource(w http.ResponseWriter, r *http.Request, sourceHostPort string) {
	raw, requestData, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	source := normalizeEndpoint(sourceHostPort)
	self := normalizeEndpoint(net.JoinHostPort(os.Getenv("POD_IP"), s.config.Port))
	if source == self {
		s.logger.V(logging.DEBUG).Info("KV cache source is this rank's endpoint, skipping p2p injection",
			"source", sourceHostPort, "self", self)
		s.dispatchDecode(w, cloneRequestWithBody(r.Context(), r, raw), requestData)
		return
	}

	p2pParams := s.p2pSourceParams(source)
	// Rebuild kv_transfer_params from scratch: the sidecar owns this field, so
	// client-supplied keys are dropped rather than forwarded to vLLM.
	requestData[requestFieldKVTransferParams] = map[string]any{requestFieldRemoteKVSource: p2pParams}

	s.logger.Info("running P2P source protocol",
		"source_host", extractHost(source),
		"kv_request_id", p2pParams[requestFieldKVRequestID],
		"p2p_connector_port", p2pParams[requestFieldRemotePort])

	newBody, err := json.Marshal(requestData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("decoder request body with p2p source", "body", string(newBody))
	}

	s.dispatchDecode(w, cloneRequestWithBody(r.Context(), r, newBody), requestData)
}
