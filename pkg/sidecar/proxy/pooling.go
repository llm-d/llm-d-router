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

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	PreprocessPath  = "/pooling/preprocess"
	InferencePath   = "/pooling/inference"
	PostprocessPath = "/pooling/postprocess"

	// PoolingPath is vLLM's pooling endpoint
	PoolingPath = "/pooling"
)

// disaggregatedIOProcessorHandler routes IO Processor requests
func (s *Server) disaggregatedIOProcessorHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		requestStart := time.Now()
		tracer := tracing.Tracer(tracerScope)
		ctx, span := tracer.Start(r.Context(), "forward_request",
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
			attribute.String("llm_d.io_disagg_proxy.io_connector", s.config.IOConnector),
			attribute.String("llm_d.io_disagg_proxy.request_path", requestPath),
		)

		// Set preprocesssor host and port
		preprocessHostPorts := r.Header.Values(routing.PreprocessEndpointHeader)
		r.Header.Del(routing.PreprocessEndpointHeader)

		if len(preprocessHostPorts) == 1 {
			preprocessHostPorts = strings.Split(preprocessHostPorts[0], ",")
		}

		numPreprocessHosts := len(preprocessHostPorts)
		var preprocessHostPort string
		if numPreprocessHosts > 0 {
			preprocessHostPort = strings.TrimSpace(preprocessHostPorts[0])
		}

		if len(preprocessHostPort) == 0 {
			s.logger.Error(nil, "No preprocess target was provided")
			span.SetAttributes(
				attribute.Bool("llm_d.io_disagg_proxy.preprocess_disaggregation_used", false),
				attribute.String("llm_d.io_disagg_proxy.reason", "no_preprocess_header"),
				attribute.String("llm_d.io_disagg_proxy.error", "no_preprocess_header"),
			)
			span.SetStatus(codes.Error, "No preprocess target was provided")
			http.Error(w, "Bad Request: No preprocess target was provided", http.StatusBadRequest)
			return
		}
		span.SetAttributes(
			attribute.Bool("llm_d.io_disagg_proxy.preprocess_disaggregation_used", true),
			attribute.String("llm_d.io_disagg_proxy.preprocess_target", preprocessHostPort),
			attribute.Int("llm_d.io_disagg_proxy.preprocess_candidates", numPreprocessHosts),
		)

		if len(preprocessHostPort) > 0 {
			if !s.allowlistValidator.IsAllowed(preprocessHostPort) {
				s.logger.Error(nil, "SSRF protection: preprocess target not in allowlist",
					"target", preprocessHostPort,
					"clientIP", r.RemoteAddr,
					"userAgent", r.Header.Get("User-Agent"),
					"requestPath", r.URL.Path)
				span.SetAttributes(
					attribute.String("llm_d.io_disagg_proxy.error", "ssrf_protection_denied"),
					attribute.String("llm_d.io_disagg_proxy.denied_target", preprocessHostPort),
				)
				span.SetStatus(codes.Error, "SSRF protection: preprocess target not in allowlist")
				http.Error(w, "Forbidden: preprocess target not allowed by SSRF protection", http.StatusForbidden)
				return
			}
			s.logger.V(4).Info("SSRF protection: preprocess target allowed", "target", preprocessHostPort)
		}

		// Set inference host ports
		inferenceHostPorts := r.Header.Values(routing.InfereceEndpointsHeader)
		r.Header.Del(routing.InfereceEndpointsHeader)
		if len(inferenceHostPorts) == 1 {
			inferenceHostPorts = strings.Split(inferenceHostPorts[0], ",")
		}

		var allowedInferenceInstances []string
		if len(inferenceHostPorts) > 0 {
			allowedInferenceInstances = make([]string, 0, len(inferenceHostPorts))
			for _, inferenceHost := range inferenceHostPorts {
				inferenceHost = strings.TrimSpace(inferenceHost)
				if s.allowlistValidator.IsAllowed(inferenceHost) {
					allowedInferenceInstances = append(allowedInferenceInstances, inferenceHost)
					s.logger.V(4).Info("SSRF protection: inference target allowed", "target", inferenceHost)
				} else {
					s.logger.Info("SSRF protection: inference target not in allowlist, removing from list",
						"target", inferenceHost,
						"clientIP", r.RemoteAddr,
						"userAgent", r.Header.Get("User-Agent"),
						"requestPath", r.URL.Path)
				}
			}
		}

		if len(allowedInferenceInstances) > 0 && s.handleIOConnector != nil {
			s.logger.V(4).Info("inference headers detected, using IO connector",
				"inferenceInstanceCount", len(allowedInferenceInstances),
				"inferenceCandidates", len(inferenceHostPorts),
				"hasPreprocessor", len(preprocessHostPort) > 0,
				"hasPostprocessor", true)
			span.SetAttributes(
				attribute.Bool("llm_d.io_disagg_proxy.inference_disaggregation_used", true),
				attribute.Int("llm_d.io_disagg_proxy.inference_count", len(allowedInferenceInstances)),
				attribute.Int("llm_d.io_disagg_proxy.inference_candidates", len(inferenceHostPorts)),
			)
			s.handleIOConnector(w, r, preprocessHostPort, allowedInferenceInstances)

			return
		}

		s.logger.Error(nil, "No inference target was provided")
		span.SetAttributes(
			attribute.Bool("llm_d.io_disagg_proxy.inference_disaggregation_used", false),
			attribute.String("llm_d.io_disagg_proxy.reason", "no_inference_header"),
			attribute.String("llm_d.io_disagg_proxy.error", "no_inference_header"),
		)
		span.SetStatus(codes.Error, "No inference target was provided")
		http.Error(w, "Bad Request: No inference target was provided", http.StatusBadRequest)
	}
}
