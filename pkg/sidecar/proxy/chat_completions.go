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
	"net/http"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/routing"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/telemetry"
)

// contextKey is a custom type for context keys to avoid collisions
type contextKey string

const requestStartTimeKey contextKey = "request_start_time"

var (
	// ChatCompletionsPath is the OpenAI chat completions path
	ChatCompletionsPath = "/v1/chat/completions"

	// CompletionsPath is the legacy completions path
	CompletionsPath = "/v1/completions"

	// ResponsesPath is the OpenAI Responses API path
	ResponsesPath = "/v1/responses"
)

func openAIAPIAttr(apiType APIType) attribute.KeyValue {
	return attribute.String("llm_d.openai.api", apiType.String())
}

// disaggregatedPrefillHandler routes OpenAI-style requests: optional encoder (EPD) stage,
// optional P/D prefill when the prefill header is set, otherwise decoder (or data-parallel).
func (s *Server) disaggregatedPrefillHandler(apiType APIType) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		requestStart := time.Now()
		tracer := telemetry.Tracer()
		ctx, span := tracer.Start(r.Context(), "llm_d.pd_proxy.request",
			trace.WithSpanKind(trace.SpanKindServer),
		)
		defer span.End()

		ctx = context.WithValue(ctx, requestStartTimeKey, requestStart)
		r = r.WithContext(ctx)

		requestPath := ""
		if r.URL != nil {
			requestPath = r.URL.Path
		}
		span.SetAttributes(
			attribute.String("llm_d.pd_proxy.connector", s.config.KVConnector),
			attribute.String("llm_d.pd_proxy.kv_connector", s.config.KVConnector),
			attribute.String("llm_d.pd_proxy.ec_connector", s.config.ECConnector),
			attribute.String("llm_d.pd_proxy.request_path", requestPath),
			openAIAPIAttr(apiType),
		)

		prefillHostPorts := r.Header.Values(routing.PrefillEndpointHeader)
		r.Header.Del(routing.PrefillEndpointHeader)

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
			s.logger.V(4).Info("skip disaggregated prefill", "api", apiType.String())
			span.SetAttributes(
				attribute.Bool("llm_d.pd_proxy.disaggregation_used", false),
				attribute.String("llm_d.pd_proxy.reason", "no_prefill_header"),
			)
		} else {
			span.SetAttributes(
				attribute.Bool("llm_d.pd_proxy.disaggregation_used", true),
				attribute.String("llm_d.pd_proxy.prefill_target", prefillHostPort),
				attribute.Int("llm_d.pd_proxy.prefill_candidates", numHosts),
			)
		}

		if len(prefillHostPort) > 0 {
			if !s.allowlistValidator.IsAllowed(prefillHostPort) {
				s.logger.Error(nil, "SSRF protection: prefill target not in allowlist",
					"target", prefillHostPort,
					"clientIP", r.RemoteAddr,
					"userAgent", r.Header.Get("User-Agent"),
					"requestPath", r.URL.Path)
				span.SetAttributes(
					attribute.String("llm_d.pd_proxy.error", "ssrf_protection_denied"),
					attribute.String("llm_d.pd_proxy.denied_target", prefillHostPort),
				)
				span.SetStatus(codes.Error, "SSRF protection: prefill target not in allowlist")
				http.Error(w, "Forbidden: prefill target not allowed by SSRF protection", http.StatusForbidden)
				return
			}
			s.logger.V(4).Info("SSRF protection: prefill target allowed", "target", prefillHostPort)
		}

		encoderHostPorts := r.Header.Values(routing.EncoderEndpointsHeader)
		r.Header.Del(routing.EncoderEndpointsHeader)
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
					s.logger.V(4).Info("SSRF protection: encoder target allowed", "target", encoderHost)
				} else {
					s.logger.Info("SSRF protection: encoder target not in allowlist, removing from list",
						"target", encoderHost,
						"clientIP", r.RemoteAddr,
						"userAgent", r.Header.Get("User-Agent"),
						"requestPath", r.URL.Path)
				}
			}
		}

		if len(allowedEncoders) > 0 && s.runEPDConnectorProtocol != nil {
			s.logger.V(4).Info("encoder headers detected, using EPD protocol",
				"encoderCount", len(allowedEncoders),
				"encoderCandidates", len(encoderHostPorts),
				"hasPrefiller", len(prefillHostPort) > 0)
			span.SetAttributes(
				attribute.Bool("llm_d.epd_proxy.encode_disaggregation_used", true),
				attribute.Int("llm_d.epd_proxy.encoder_count", len(allowedEncoders)),
				attribute.Int("llm_d.epd_proxy.encoder_candidates", len(encoderHostPorts)),
			)
			s.runEPDConnectorProtocol(w, r, prefillHostPort, allowedEncoders)
			return
		}

		if len(encoderHostPorts) > 0 && len(allowedEncoders) == 0 {
			s.logger.Info("SSRF protection: all encoder targets filtered out, falling back to P/D or decoder-only")
			span.SetAttributes(
				attribute.Bool("llm_d.epd_proxy.encode_disaggregation_used", false),
				attribute.Int("llm_d.epd_proxy.encoder_allowed", len(allowedEncoders)),
				attribute.Int("llm_d.epd_proxy.encoder_candidates", len(encoderHostPorts)),
			)
		}

		if len(prefillHostPort) > 0 {
			s.logger.V(4).Info("using P/D protocol")
			s.runPDConnectorProtocol(w, r, prefillHostPort, apiType)
			return
		}

		s.logger.V(4).Info("no prefiller or encoder, using decoder only")
		if !s.forwardDataParallel || !s.dataParallelHandler(w, r) {
			if s.config.DecodeChunkSize > 0 {
				s.runChunkedDecode(w, r)
				return
			}
			s.decoderProxy.ServeHTTP(w, r)
		}
	}
}
