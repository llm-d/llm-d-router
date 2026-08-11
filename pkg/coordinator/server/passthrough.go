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

package server

import (
	"errors"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
)

// handlePassthrough is the chi NotFound catch-all: any path the coordinator
// does not register (e.g. /v1/models, /v1/messages, /v1/responses, /v1/embeddings)
// is reverse-proxied to the gateway with EPP-Profile: decode, so EPP dispatches
// it to a decode pod. Method, body, query, and forwarded headers are preserved;
// X-Request-Id is validated and replaced with a UUID if malformed, matching
// handleInference's sanitization.
func (s *Server) handlePassthrough(w http.ResponseWriter, r *http.Request) {
	requestID := r.Header.Get(reqcommon.RequestIDHeaderKey)
	if !validRequestID.MatchString(requestID) {
		requestID = uuid.New().String()
	}

	logger := ctrl.Log.WithName("passthrough").WithValues(reqcommon.RequestIDHeaderKey, requestID)
	logger.V(logutil.DEFAULT).Info("received request", "method", r.Method, "path", r.URL.Path)

	// A passthrough response may stream (e.g. /v1/messages with stream:true).
	// Clear the write deadline like handleInference does for streaming so a
	// long response is not cut by WriteTimeout mid-stream.
	if err := http.NewResponseController(w).SetWriteDeadline(time.Time{}); err != nil && !errors.Is(err, http.ErrNotSupported) {
		logger.V(logutil.DEFAULT).Info("could not clear write deadline", "error", err)
	}

	proxy := newPassthroughProxy(logger, s.gatewayURL, s.gwClient.Transport(), requestID)
	proxy.ServeHTTP(w, r)
}

// newPassthroughProxy builds the reverse proxy that streams to the gateway.
// The director rewrites the outbound scheme/host to the gateway and stamps the
// decode profile and sanitized request id. Transport errors return 502; a
// failure after the upstream response has started can only surface through
// ErrorLog, so it is wired to the request-scoped logger.
func newPassthroughProxy(logger logr.Logger, gatewayURL *url.URL, transport http.RoundTripper, requestID string) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			r.URL.Scheme = gatewayURL.Scheme
			r.URL.Host = gatewayURL.Host
			r.Host = gatewayURL.Host
			r.Header.Set(reqcommon.RequestIDHeaderKey, requestID)
			r.Header.Set(gateway.EPPProfileHeader, gateway.PhaseDecode)
		},
		FlushInterval: -1,
		Transport:     transport,
		ErrorLog:      log.New(&passthroughErrorLogWriter{logger: logger}, "", 0),
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			logger.Error(err, "passthrough proxy error")
			w.WriteHeader(http.StatusBadGateway)
		},
	}
}

// passthroughErrorLogWriter adapts httputil.ReverseProxy's *log.Logger sink to
// logr. The stdlib proxy writes here when a read fails after the response has
// started, which is the only signal that the client received a truncated body.
type passthroughErrorLogWriter struct {
	logger logr.Logger
}

func (w *passthroughErrorLogWriter) Write(p []byte) (int, error) {
	w.logger.Error(errors.New(strings.TrimSpace(string(p))), "passthrough proxy streaming error: client received a partial response")
	return len(p), nil
}
