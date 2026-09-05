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
	"fmt"
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

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
)

// passthroughHandler is the chi NotFound catch-all: any path the coordinator
// does not register (e.g. /v1/models, /v1/messages, /v1/responses, /v1/embeddings)
// is reverse-proxied to the gateway with EPP-Profile: decode, so EPP dispatches
// it to a decode pod. Method, body, query, and forwarded headers are preserved;
// X-Request-Id is validated and replaced with a UUID if malformed, matching
// handleInference's sanitization.
type passthroughHandler struct {
	gatewayURL         *url.URL
	transport          http.RoundTripper
	maxRequestBodySize int64
}

// newPassthroughHandler resolves and validates the gateway target once, at
// server construction, so a misconfigured gateway URL fails startup rather
// than the first passthrough request. maxRequestBodySize is the same MB
// bound New() applies to /v1/chat/completions et al; the passthrough enforces
// it via http.MaxBytesReader so unregistered paths cannot bypass the cap.
func newPassthroughHandler(gwClient *gateway.Client, maxRequestBodySize int64) (*passthroughHandler, error) {
	if gwClient == nil {
		return nil, errors.New("server: gateway client is required")
	}
	gatewayURL, err := url.Parse(gwClient.BaseURL())
	if err != nil {
		return nil, fmt.Errorf("server: parse gateway URL %q: %w", gwClient.BaseURL(), err)
	}
	if gatewayURL.Host == "" {
		return nil, fmt.Errorf("server: gateway URL %q is missing a host", gwClient.BaseURL())
	}
	return &passthroughHandler{
		gatewayURL:         gatewayURL,
		transport:          gwClient.Transport(),
		maxRequestBodySize: maxRequestBodySize,
	}, nil
}

func (h *passthroughHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	requestID := r.Header.Get(reqcommon.RequestIDHeaderKey)
	clientRequestID := requestID
	requestIDReplaced := !validRequestID.MatchString(requestID)
	if requestIDReplaced {
		requestID = uuid.New().String()
	}

	logger := ctrl.Log.WithName("passthrough").WithValues(reqcommon.RequestIDHeaderKey, requestID)
	if requestIDReplaced && clientRequestID != "" {
		// Log the rejected length, never the raw value, to avoid reflecting
		// attacker-controlled content into the log.
		logger.V(logutil.DEFAULT).Info("replaced invalid client request ID", "rejectedLength", len(clientRequestID))
	}
	logger.V(logutil.DEFAULT).Info("received request", "method", r.Method, "path", r.URL.Path)

	// The passthrough proxies request bodies verbatim and never parses them,
	// so it cannot detect stream:true without defeating that design. Clear
	// the write deadline unconditionally: a streaming response (the stated
	// motivator, e.g. /v1/messages with stream:true) would otherwise be cut
	// by WriteTimeout mid-stream. The trade-off is that non-streaming
	// passthrough responses lose the slow-client guard; the gateway and
	// server IdleTimeout still bound stalled connections.
	if err := http.NewResponseController(w).SetWriteDeadline(time.Time{}); err != nil && !errors.Is(err, http.ErrNotSupported) {
		logger.V(logutil.DEFAULT).Info("could not clear write deadline", "error", err)
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxRequestBodySize*config.BytesPerMB)

	proxy := newPassthroughProxy(logger, h.gatewayURL, h.transport, requestID)
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
		ErrorHandler: func(w http.ResponseWriter, req *http.Request, err error) {
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				logger.V(logutil.DEFAULT).Info("passthrough proxy: request body exceeds cap", "capBytes", maxBytesErr.Limit)
				w.WriteHeader(http.StatusRequestEntityTooLarge)
				return
			}
			// A request whose context already ended (client disconnected, or a
			// client-set deadline expired) is a routine lifecycle event, not a
			// backend fault. Check req.Context().Err() rather than err's identity:
			// Transport.RoundTrip's ResponseHeaderTimeout returns net/http's
			// errTimeout, which satisfies errors.Is(err, context.DeadlineExceeded)
			// by stdlib design and would otherwise misclassify a hung backend as
			// a client cancellation.
			if ctxErr := req.Context().Err(); ctxErr != nil {
				logger.V(logutil.VERBOSE).Info("passthrough proxy: client cancelled", "error", ctxErr)
			} else {
				logger.Error(err, "passthrough proxy error")
			}
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
