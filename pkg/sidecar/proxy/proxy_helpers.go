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
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"

	tlsutil "github.com/llm-d/llm-d-router/internal/tls"
	"github.com/llm-d/llm-d-router/pkg/common"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// startHTTP starts the HTTP reverse proxy.
func (s *Server) startHTTP(ctx context.Context) error {
	// Start SSRF protection validator
	if err := s.allowlistValidator.Start(ctx); err != nil {
		s.logger.Error(err, "Failed to start allowlist validator")
		return err
	}

	ln := s.HTTPListener
	var err error
	if ln == nil {
		ln, err = net.Listen("tcp", ":"+s.config.Port)
		if err != nil {
			s.logger.Error(err, "Failed to start")
			return err
		}
	}
	s.addr = ln.Addr()
	close(s.readyCh)

	// Wrap handler with OpenTelemetry middleware to extract trace context from incoming requests
	handler := otelhttp.NewHandler(s.handler, "llm-d-pd-proxy",
		otelhttp.WithSpanNameFormatter(func(_ string, r *http.Request) string {
			path := ""
			if r.URL != nil {
				path = r.URL.Path
			}
			return r.Method + " " + path
		}),
	)

	server := &http.Server{
		Handler: handler,
		// No ReadTimeout/WriteTimeout for LLM inference - can take hours for large contexts
		IdleTimeout:       300 * time.Second, // 5 minutes for keep-alive connections
		ReadHeaderTimeout: 30 * time.Second,  // Reasonable for headers only
		MaxHeaderBytes:    1 << 20,           // 1 MB for headers is sufficient
	}

	var cert *tls.Certificate
	if s.config.SecureServing {
		var tempCert tls.Certificate
		if s.config.CertPath != "" {
			certFile := filepath.Join(s.config.CertPath, "tls.crt")
			keyFile := filepath.Join(s.config.CertPath, "tls.key")
			tempCert, err = tls.LoadX509KeyPair(certFile, keyFile)
			if err != nil {
				return fmt.Errorf("load key pair from cert %q and key %q: %w", certFile, keyFile, err)
			}
		} else {
			tempCert, err = tlsutil.CreateSelfSignedTLSCertificate(s.logger)
			if err != nil {
				return fmt.Errorf("create self-signed certificate: %w", err)
			}
		}
		cert = &tempCert
	}

	if cert != nil {
		getCertificate := func(info *tls.ClientHelloInfo) (*tls.Certificate, error) {
			return cert, nil
		}
		if s.config.CertPath != "" {
			reloader, err := common.NewCertReloader(ctx, s.config.CertPath, cert)
			if err != nil {
				return fmt.Errorf("start certificate reloader: %w", err)
			}
			getCertificate = func(info *tls.ClientHelloInfo) (*tls.Certificate, error) {
				return reloader.Get(), nil
			}
		}

		// MinVersion is a literal so gosec/CodeQL can resolve it statically;
		// Options.Complete already rejects a configured version below TLS 1.2.
		// An empty suite list leaves CipherSuites nil, which selects the
		// crypto/tls default, matching the coordinator and EPP.
		minVersion := uint16(tls.VersionTLS12)
		if s.config.TLSMinVersion > tls.VersionTLS12 {
			minVersion = s.config.TLSMinVersion
		}
		server.TLSConfig = &tls.Config{
			MinVersion:     minVersion,
			CipherSuites:   s.config.TLSCipherSuites,
			GetCertificate: getCertificate,
		}
	}
	s.logger.Info("server TLS", "tls", s.config.SecureServing, "cert_path", s.config.CertPath)

	// Setup graceful termination (not strictly needed for sidecars)
	go func() {
		<-ctx.Done()
		s.logger.Info("shutting down")

		// Stop allowlist validator
		s.allowlistValidator.Stop()

		ctx, cancelFn := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancelFn()
		if err := server.Shutdown(ctx); err != nil {
			s.logger.Error(err, "failed to gracefully shutdown")
		}
	}()

	s.logger.Info("starting", "addr", s.addr.String())
	if cert != nil {
		if err := server.ServeTLS(ln, "", ""); err != nil && err != http.ErrServerClosed {
			s.logger.Error(err, "failed to start")
			return err
		}
	} else {
		if err := server.Serve(ln); err != nil && err != http.ErrServerClosed {
			s.logger.Error(err, "failed to start")
			return err
		}
	}

	return nil
}

// Passthrough decoder handler
func (s *Server) createDecoderProxyHandler(decoderURL *url.URL, decoderInsecureSkipVerify bool) *httputil.ReverseProxy {
	decoderProxy := httputil.NewSingleHostReverseProxy(decoderURL)
	decoderProxy.Transport = s.newProxyTransport(decoderURL.Scheme, decoderInsecureSkipVerify)
	decoderProxy.ErrorHandler = func(res http.ResponseWriter, _ *http.Request, err error) {

		// Log errors from the decoder proxy
		var writeError error
		switch {
		case errors.Is(err, syscall.ECONNREFUSED):
			s.logger.Error(err, "failed to connect to vLLM decoder",
				"decoderURL", s.config.DecoderURL.String())
			res.WriteHeader(http.StatusServiceUnavailable)
			_, writeError = res.Write(decoderServiceUnavailableResponseJSON)

		default:
			s.logger.Error(err, "http: proxy error",
				"decoderURL", s.config.DecoderURL.String())
			writeError = errorBadGateway(err, res)
		}
		if writeError != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
	}
	return decoderProxy
}

func bodyAsJSON(r *http.Request) ([]byte, map[string]any, error) {
	defer r.Body.Close()
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read request body: %w", err)
	}
	parsed, err := decodeRequestBody(raw)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", errInvalidJSON, err)
	}
	return raw, parsed, nil
}

// inspectedRequestFields lists the top-level request fields the sidecar reads
// as Go values. decodeRequestBody decodes only these; every other field stays
// a json.RawMessage so its bytes are forwarded unchanged. encoding/json sorts
// map keys at every depth on Marshal, which would reorder free-form content
// such as tools[].function.parameters that chat templates render into the
// prompt verbatim.
var inspectedRequestFields = map[string]struct{}{
	requestFieldKVTransferParams:     {},
	requestFieldECTransferParams:     {},
	requestFieldMaxTokens:            {},
	requestFieldMaxCompletionTokens:  {},
	requestFieldMaxOutputTokens:      {},
	requestFieldMinTokens:            {},
	requestFieldSamplingParams:       {},
	requestFieldStream:               {},
	requestFieldStreamOptions:        {},
	requestFieldCacheHitThreshold:    {},
	requestFieldContinueFinalMessage: {},
	requestFieldAddGenerationPrompt:  {},
}

// requestMessages returns the request's messages, decoding the array on first
// use and keeping each element as raw bytes so that re-marshaling the request
// preserves the key order inside every message. An absent field yields a nil
// slice and no error.
func requestMessages(req map[string]any) ([]json.RawMessage, error) {
	switch v := req[requestFieldMessages].(type) {
	case nil:
		return nil, nil
	case []json.RawMessage:
		return v, nil
	case json.RawMessage:
		var messages []json.RawMessage
		if err := json.Unmarshal(v, &messages); err != nil {
			return nil, err
		}
		return messages, nil
	default:
		return nil, fmt.Errorf("messages is %T, want a JSON array", v)
	}
}

// decodeRequestBody parses a JSON object body, applying inspectedRequestFields.
func decodeRequestBody(raw []byte) (map[string]any, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	if fields == nil {
		return nil, errors.New("request body is not a JSON object")
	}
	parsed := make(map[string]any, len(fields))
	for k, v := range fields {
		if _, ok := inspectedRequestFields[k]; !ok {
			parsed[k] = v
			continue
		}
		var decoded any
		if err := json.Unmarshal(v, &decoded); err != nil {
			return nil, err
		}
		parsed[k] = decoded
	}
	return parsed, nil
}

func (s *Server) readJSONBody(r *http.Request, w http.ResponseWriter) ([]byte, map[string]any, bool) {
	raw, parsed, err := bodyAsJSON(r)
	if err != nil {
		s.logger.V(logging.DEBUG).Info("invalid request body", "error", err)
		if writeErr := errorJSONInvalid(err, w); writeErr != nil {
			s.logger.Error(writeErr, "failed to send error response to client")
		}
		return nil, nil, false
	}
	return raw, parsed, true
}

func cloneRequestWithBody(ctx context.Context, r *http.Request, body []byte) *http.Request {
	cloned := r.Clone(ctx)
	cloned.Body = io.NopCloser(bytes.NewReader(body))
	cloned.ContentLength = int64(len(body))
	return cloned
}

// extractHost returns the host part of a host:port string. A `http://`
// prefix is trimmed first, matching createProxyHandler's tolerance for
// routed endpoint values. If parsing fails (e.g. no port), the input is
// returned as-is.
func extractHost(hostWithPort string) string {
	hostWithPort, _ = strings.CutPrefix(hostWithPort, "http://")
	host, _, err := net.SplitHostPort(hostWithPort)
	if err != nil {
		return hostWithPort
	}
	return host
}

// isHostPort reports whether s parses as host:port with a non-empty host.
// net.SplitHostPort accepts ":8000" (empty host), which is not a usable target.
func isHostPort(s string) bool {
	host, _, err := net.SplitHostPort(s)
	return err == nil && host != ""
}

func newUUID() string {
	return uuid.New().String()
}

// isHTTPError returns true if the status code indicates an error (not in the 2xx range).
func isHTTPError(statusCode int) bool {
	return statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices
}

// isRetryableStatus returns true for transient 5xx errors where retrying
// the same host is likely to succeed (e.g. TCP connection reset, overloaded
// accept queue). Non-transient errors like 500/501 indicate bugs or
// unsupported operations and should fail fast.
func isRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusBadGateway ||
		statusCode == http.StatusServiceUnavailable ||
		statusCode == http.StatusGatewayTimeout
}

// WriteAll writes b to w, discarding the error. The caller has already sent
// headers and status, so there is no recovery action for a write failure.
func WriteAll(w io.Writer, b []byte) {
	_, _ = w.Write(b)
}
