package proxy

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"

	"github.com/llm-d/llm-d-router/pkg/common"
)

// startHTTP starts the HTTP reverse proxy.
func (s *Server) startHTTP(ctx context.Context) error {
	// Start SSRF protection validator
	if err := s.allowlistValidator.Start(ctx); err != nil {
		s.logger.Error(err, "Failed to start allowlist validator")
		return err
	}

	ln, err := net.Listen("tcp", ":"+s.config.Port)
	if err != nil {
		s.logger.Error(err, "Failed to start")
		return err
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
			certFile := s.config.CertPath + "/tls.crt"
			keyFile := s.config.CertPath + "/tls.key"
			tempCert, err = tls.LoadX509KeyPair(certFile, keyFile)
			if err != nil {
				return fmt.Errorf("failed to load TLS key pair from cert %q and key %q: %w", certFile, keyFile, err)
			}
		} else {
			tempCert, err = CreateSelfSignedTLSCertificate()
			if err != nil {
				return fmt.Errorf("failed to generate self-signed TLS certificate: %w", err)
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
				return fmt.Errorf("failed to start reloader: %w", err)
			}
			getCertificate = func(info *tls.ClientHelloInfo) (*tls.Certificate, error) {
				return reloader.Get(), nil
			}
		}

		server.TLSConfig = &tls.Config{
			MinVersion: tls.VersionTLS12,
			CipherSuites: []uint16{
				tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
				tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
			},
			GetCertificate: getCertificate,
		}
		s.logger.Info("server TLS configured")
	}

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

func bodyAsJSON(r *http.Request) ([]byte, error) {
	defer func() { _ = r.Body.Close() }()
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	if err := validateRequestBody(raw); err != nil {
		return nil, fmt.Errorf("%w: %w", errInvalidJSON, err)
	}
	return raw, nil
}

// validateRequestBody reports an error when raw is not a JSON object.
func validateRequestBody(raw []byte) error {
	if !gjson.ValidBytes(raw) {
		return errors.New("request body is not valid JSON")
	}
	if !gjson.ParseBytes(raw).IsObject() {
		return errors.New("request body is not a JSON object")
	}
	return nil
}

// requestBody edits a JSON object body in place. Only the fields the sidecar
// sets are rewritten; every other byte is forwarded unchanged, so nested key
// order (for example tools[].function.parameters, which chat templates render
// into the prompt verbatim) survives proxying. Each edit copies the body once;
// the first error is kept and returned by bytes.
type requestBody struct {
	raw []byte
	err error
}

func editRequestBody(raw []byte) *requestBody {
	return &requestBody{raw: raw}
}

func (b *requestBody) get(path string) gjson.Result {
	return gjson.GetBytes(b.raw, path)
}

func (b *requestBody) set(path string, value any) *requestBody {
	if b.err == nil {
		b.raw, b.err = sjson.SetBytes(b.raw, path, value)
	}
	return b
}

func (b *requestBody) setRaw(path string, value []byte) *requestBody {
	if b.err == nil {
		b.raw, b.err = sjson.SetRawBytes(b.raw, path, value)
	}
	return b
}

func (b *requestBody) del(path string) *requestBody {
	if b.err == nil {
		b.raw, b.err = sjson.DeleteBytes(b.raw, path)
	}
	return b
}

// singleToken turns the body into a non-streaming, single-output-token request
// for legs that only need to run prefill or encoding. max_completion_tokens is
// capped only when the client sent it.
func (b *requestBody) singleToken() *requestBody {
	b.set(requestFieldMaxTokens, 1)
	if b.get(requestFieldMaxCompletionTokens).Exists() {
		b.set(requestFieldMaxCompletionTokens, 1)
	}
	return b.set(requestFieldStream, false).del(requestFieldStreamOptions)
}

func (b *requestBody) bytes() ([]byte, error) {
	return b.raw, b.err
}

func (s *Server) readJSONBody(r *http.Request, w http.ResponseWriter) ([]byte, bool) {
	raw, err := bodyAsJSON(r)
	if err != nil {
		if !errors.Is(err, errInvalidJSON) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(err.Error()))
		} else if writeErr := errorJSONInvalid(err, w); writeErr != nil {
			s.logger.Error(writeErr, "failed to send error response to client")
		}
		return nil, false
	}
	return raw, true
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
