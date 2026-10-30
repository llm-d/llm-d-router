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
	"net/http"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// contextKey is a custom type for context keys to avoid collisions
type contextKey string

const requestStartTimeKey contextKey = "request_start_time"

func openAIAPIAttr(apiType reqcommon.APIType) attribute.KeyValue {
	return semconv.LLMDOpenAIAPI(apiType.String())
}

// disaggregatedPrefillHandler routes OpenAI-style requests: optional encoder (EPD) stage,
// optional P/D prefill when the prefill header is set, otherwise decoder (or data-parallel).
func (s *Server) disaggregatedPrefillHandler(apiType reqcommon.APIType) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		requestStart := time.Now()
		metrics.RecordRequest(apiType.String())
		tracer := tracing.Tracer(tracerScope)
		ctx, span := tracer.Start(r.Context(), "forward_request",
			trace.WithSpanKind(trace.SpanKindServer),
		)
		defer span.End()
		ctx = context.WithValue(ctx, toolCallingRequestSpanKey, span)
		ctx = context.WithValue(ctx, toolCallingTelemetryEnabledKey{}, s.config.EnableToolCallingTelemetry)

		// Tag this handler's log lines with the trace they belong to. The proxy
		// logger is process-scoped, so seed it into the context first. Connectors
		// reached from here still log through s.logger and stay uncorrelated;
		// the request context carries the correlated logger for them to adopt.
		ctx = tracing.LoggerWithSpanContext(log.IntoContext(ctx, s.logger), span)
		logger := log.FromContext(ctx)

		ctx = context.WithValue(ctx, requestStartTimeKey, requestStart)
		r = r.WithContext(ctx)

		requestPath := ""
		if r.URL != nil {
			requestPath = r.URL.Path
		}
		span.SetAttributes(
			semconv.LLMDPDProxyConnector(s.config.KVConnector),
			semconv.LLMDPDProxyKVConnector(s.config.KVConnector),
			semconv.LLMDPDProxyECConnector(s.config.ECConnector),
			semconv.LLMDPDProxyRequestPath(requestPath),
			openAIAPIAttr(apiType),
		)

		prefillHostPorts := routing.TakeRoutingHeaderValues(r.Header, routing.PrefillEndpointHeader)

		if len(prefillHostPorts) == 1 {
			prefillHostPorts = strings.Split(prefillHostPorts[0], ",")
		}

		numHosts := len(prefillHostPorts)
		var prefillHostPort string
		if numHosts > 0 {
			if s.config.EnablePrefillerSampling {
				prefillHostPort = strings.TrimSpace(prefillHostPorts[s.prefillSamplerFn(numHosts)])
			} else {
				prefillHostPort = strings.TrimSpace(prefillHostPorts[0])
			}
		}

		if len(prefillHostPort) == 0 {
			logger.V(logging.DEBUG).Info("skip disaggregated prefill", "api", apiType.String())
			span.SetAttributes(
				semconv.LLMDPDProxyDisaggregationUsed(false),
				semconv.LLMDPDProxyReason("no_prefill_header"),
			)
		} else {
			span.SetAttributes(
				semconv.LLMDPDProxyDisaggregationUsed(true),
				semconv.LLMDPDProxyPrefillTarget(prefillHostPort),
				semconv.LLMDPDProxyPrefillCandidates(numHosts),
			)
		}

		if len(prefillHostPort) > 0 {
			if !s.allowlistValidator.IsAllowed(prefillHostPort) {
				denySSRF(w, r, logger, span, "prefill target", prefillHostPort)
				return
			}
			logger.V(logging.DEBUG).Info("SSRF protection: prefill target allowed", "target", prefillHostPort)
		}

		p2p := p2pRouting{prefill: parsePrefillP2P(logger, strings.TrimSpace(routing.TakeRoutingHeaderValue(r.Header, routing.KVCachePrefillHeader)))}
		// The decode pod dials this address. EPP sets it from the prefill
		// endpoint it chose. Checked only when the host differs from the
		// prefill target, which IsAllowed already cleared.
		if p2p.prefill != nil && p2p.prefill.host != extractHost(prefillHostPort) && !s.allowlistValidator.IsAllowedHost(p2p.prefill.host) {
			denySSRF(w, r, logger, span, "prefill P2P address", p2p.prefill.String())
			return
		}
		// The source header is the signal to read the body: reading every body
		// to learn that most carry no source would buffer every request. The
		// P/D connectors and decodeWithP2PSource take the source from p2p. The
		// other paths forward r, rebuilt over the body without the source when
		// one was present, so a rejected source never reaches vLLM. The header
		// value duplicates the body block, which is what is validated.
		kvCacheSource := strings.TrimSpace(routing.TakeRoutingHeaderValue(r.Header, routing.KVCacheSourceHeader))
		switch {
		case kvCacheSource == "":
		case !s.p2pPullAvailable():
			// Once per request that carries a source: the plugin is enabled on a
			// pool whose sidecar cannot pull, so the feature is off here.
			logger.Info("warning: ignoring KV cache source: connector does not support P2P pulls",
				"connector", s.config.KVConnector, "source", kvCacheSource)
		default:
			raw, parsed, ok := s.readJSONBody(r, w)
			if !ok {
				return
			}
			p2p.body = parsed
			source, present := s.kvSourceFromBody(logger, r.RemoteAddr, parsed)
			if present {
				var err error
				if raw, err = json.Marshal(parsed); err != nil {
					if err := errorJSONInvalid(err, w); err != nil {
						logger.Error(err, "failed to send error response to client")
					}
					return
				}
			}
			p2p.source = source
			p2p.raw = raw
			r = cloneRequestWithBody(r.Context(), r, raw)
		}
		if p2p.source != nil {
			span.SetAttributes(semconv.LLMDPDProxyKVCacheSource(p2p.source.String()))
		}

		encoderHostPorts := routing.TakeRoutingHeaderValues(r.Header, routing.EncoderEndpointsHeader)
		if len(encoderHostPorts) == 1 {
			encoderHostPorts = strings.Split(encoderHostPorts[0], ",")
		}

		var allowedEncoders []string
		if len(encoderHostPorts) > 0 {
			allowedEncoders = make([]string, 0, len(encoderHostPorts))
			for _, encoderHost := range encoderHostPorts {
				encoderHost = strings.TrimSpace(encoderHost)
				if s.allowlistValidator.IsAllowed(encoderHost) {
					allowedEncoders = append(allowedEncoders, encoderHost)
					logger.V(logging.DEBUG).Info("SSRF protection: encoder target allowed", "target", encoderHost)
				} else {
					logger.Info("SSRF protection: encoder target not in allowlist, removing from list",
						"target", encoderHost,
						"clientIP", r.RemoteAddr,
						"userAgent", r.Header.Get("User-Agent"),
						"requestPath", r.URL.Path)
				}
			}
		}

		if len(allowedEncoders) > 0 && s.handleECConnector != nil {
			logger.V(logging.DEBUG).Info("encoder headers detected, using EC connector",
				"encoderCount", len(allowedEncoders),
				"encoderCandidates", len(encoderHostPorts),
				"hasPrefiller", len(prefillHostPort) > 0)
			span.SetAttributes(
				semconv.LLMDECProxyEncodeDisaggregationUsed(true),
				semconv.LLMDECProxyEncoderCount(len(allowedEncoders)),
				semconv.LLMDECProxyEncoderCandidates(len(encoderHostPorts)),
			)
			if len(prefillHostPort) > 0 {
				metrics.RecordDisagg(metricsutil.DisaggPathEncodePrefillDecode)
			} else {
				metrics.RecordDisagg(metricsutil.DisaggPathEncodeDecode)
			}
			if p2p.source != nil {
				// The P/D stage rebuilds the body after the encoder stage and
				// carries no source, so EPP's pull decision is not honored here.
				logger.Info("warning: ignoring KV cache source: a cached-prefix pull is not wired through encoder disaggregation",
					"source", p2p.source.String())
			}
			s.handleECConnector(w, r, prefillHostPort, allowedEncoders, p2p.prefill, apiType)
			return
		}

		if len(encoderHostPorts) > 0 && len(allowedEncoders) == 0 {
			logger.Info("SSRF protection: all encoder targets filtered out, falling back to P/D or decoder-only")
			span.SetAttributes(
				semconv.LLMDECProxyEncodeDisaggregationUsed(false),
				semconv.LLMDECProxyEncoderCount(len(allowedEncoders)),
				semconv.LLMDECProxyEncoderCandidates(len(encoderHostPorts)),
			)
		}

		if len(prefillHostPort) > 0 {
			logger.V(logging.DEBUG).Info("using P/D protocol")
			metrics.RecordDisagg(metricsutil.DisaggPathPrefillDecode)
			s.handlePDConnector(w, r, prefillHostPort, p2p, apiType)
			return
		}

		logger.V(logging.DEBUG).Info("no prefiller or encoder, using decoder only")
		// dataParallelHandler and the decoder passthrough forward r unread, so
		// the guard in readJSONBody would never run on those branches.
		// readJSONBody consumes r.Body, so r is rebuilt over the same bytes.
		if apiType == reqcommon.APITypeResponses && p2p.body == nil {
			raw, _, ok := s.readJSONBody(r, w)
			if !ok {
				return
			}
			r = cloneRequestWithBody(r.Context(), r, raw)
		}
		if !s.forwardDataParallel || !s.dataParallelHandler(w, r, func() {
			observeToolCallingPassthrough(r, apiType)
		}) {
			if p2p.source != nil {
				s.decodeWithP2PSource(w, r, p2p)
				return
			}
			if s.config.DecodeChunkSize > 0 && r.URL.Path == reqcommon.PathChatCompletions {
				s.runChunkedDecode(w, r)
				return
			}
			observeToolCallingPassthrough(r, apiType)
			s.decoderProxy.ServeHTTP(w, r)
		}
	}
}

// denySSRF refuses the request: the client named a target outside the pool.
func denySSRF(w http.ResponseWriter, r *http.Request, logger logr.Logger, span trace.Span, what, target string) {
	logger.Error(nil, "SSRF protection: "+what+" not in allowlist",
		"target", target,
		"clientIP", r.RemoteAddr,
		"userAgent", r.Header.Get("User-Agent"),
		"requestPath", r.URL.Path)
	span.SetAttributes(
		semconv.LLMDPDProxyError("ssrf_protection_denied"),
		semconv.LLMDPDProxyDeniedTarget(target),
	)
	span.SetStatus(codes.Error, "SSRF protection: "+what+" not in allowlist")
	http.Error(w, "Forbidden: "+what+" not allowed by SSRF protection", http.StatusForbidden)
}
