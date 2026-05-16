package proxy

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"syscall"
	"time"

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
			return "llm_d.pd_proxy." + r.Method + " " + path
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

// isHTTPError returns true if the status code indicates an error (not in the 2xx range).
func isHTTPError(statusCode int) bool {
	return statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices
}

// shouldFallbackToDecode returns false for client error 4xx status codes (400–451). For all other status codes, it returns true.
func shouldFallbackToDecode(pw *bufferedResponseWriter) bool {
	if pw.statusCode >= http.StatusBadRequest && pw.statusCode <= http.StatusUnavailableForLegalReasons {
		return false
	}
	return true
}
