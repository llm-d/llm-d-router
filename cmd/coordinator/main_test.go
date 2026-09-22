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

package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	tlsutil "github.com/llm-d/llm-d-router/internal/tls"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/server"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

func newTestServer(t *testing.T, listenAddr string) *server.Server {
	t.Helper()
	srv, err := server.New(config.ServerConfig{
		ListenAddr:      listenAddr,
		ShutdownTimeout: time.Second,
		ReadTimeout:     time.Second,
		WriteTimeout:    time.Second,
	}, pipeline.New(nil), gateway.NewWithTransport(nil, "http://gateway-stub.invalid"))
	require.NoError(t, err)
	return srv
}

// waitForHealthz polls /healthz. A successful TCP dial precedes
// http.Server.Serve dispatching, and srv.Shutdown on an unstarted
// http.Server returns nil, so the socket alone is not a readiness signal.
func waitForHealthz(t *testing.T, addr string, timeout time.Duration) {
	t.Helper()
	client := &http.Client{Timeout: 200 * time.Millisecond}
	require.Eventually(t, func() bool {
		resp, err := client.Get("http://" + addr + "/healthz")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, timeout, 20*time.Millisecond, "coordinator /healthz on %s never returned 200", addr)
}

func writeMetricsCertificate(t *testing.T, dir string) {
	t.Helper()

	cert, err := tlsutil.CreateSelfSignedTLSCertificate(logr.Discard())
	require.NoError(t, err)

	keyDER, err := x509.MarshalPKCS8PrivateKey(cert.PrivateKey)
	require.NoError(t, err)

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.crt"), certPEM, 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.key"), keyPEM, 0o600))
}

func TestServeMetricsHTTP(t *testing.T) {
	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	port := lis.Addr().(*net.TCPAddr).Port

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- serveMetrics(ctx, port, "", lis) }()

	client := &http.Client{Timeout: 2 * time.Second}
	require.Eventually(t, func() bool {
		resp, err := client.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/metrics")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 2*time.Second, 20*time.Millisecond)

	cancel()
	require.NoError(t, <-errCh)
}

// TestServeMetricsHTTP_NilListener covers the lis == nil branch that main()
// always takes in production, where serveMetrics binds port itself instead
// of serving on a caller-supplied listener.
func TestServeMetricsHTTP_NilListener(t *testing.T) {
	port, err := fwknet.GetFreePort()
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- serveMetrics(ctx, port, "", nil) }()

	client := &http.Client{Timeout: 2 * time.Second}
	require.Eventually(t, func() bool {
		resp, err := client.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/metrics")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 2*time.Second, 20*time.Millisecond)

	cancel()
	require.NoError(t, <-errCh)
}

func TestServeMetricsHTTPS(t *testing.T) {
	certDir := t.TempDir()
	writeMetricsCertificate(t, certDir)
	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	port := lis.Addr().(*net.TCPAddr).Port

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- serveMetrics(ctx, port, certDir, lis) }()

	client := &http.Client{
		Timeout:   2 * time.Second,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}, //nolint:gosec // test certificate
	}
	require.Eventually(t, func() bool {
		resp, err := client.Get("https://127.0.0.1:" + strconv.Itoa(port) + "/metrics")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 2*time.Second, 20*time.Millisecond)

	resp, err := (&http.Client{Timeout: 500 * time.Millisecond}).Get("http://127.0.0.1:" + strconv.Itoa(port) + "/metrics")
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	require.Equal(t, http.StatusBadRequest, resp.StatusCode)

	cancel()
	require.NoError(t, <-errCh)
}

func TestServeMetricsInvalidTLSFiles(t *testing.T) {
	tests := []struct {
		name  string
		write bool
	}{
		{name: "missing"},
		{name: "invalid", write: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			certDir := t.TempDir()
			if tt.write {
				require.NoError(t, os.WriteFile(filepath.Join(certDir, "tls.crt"), []byte("invalid"), 0o600))
				require.NoError(t, os.WriteFile(filepath.Join(certDir, "tls.key"), []byte("invalid"), 0o600))
			}
			// TLS cert/key loading fails before any bind is attempted, so a bare
			// port number carries no bind race here.
			port, err := fwknet.GetFreePort()
			require.NoError(t, err)

			err = serveMetrics(context.Background(), port, certDir, nil)
			require.ErrorIs(t, err, errMetricsTLS)
		})
	}
}

// With MetricsPort <= 0 no metrics goroutine joins the errgroup, so the only
// exit path is context cancellation. run must return nil once the
// coordinator server drains.
func TestRun_MetricsDisabled_DrainsCleanlyOnCancel(t *testing.T) {
	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	listenAddr := lis.Addr().String()

	cfg := config.ServerConfig{
		ListenAddr:      listenAddr,
		ShutdownTimeout: time.Second,
		ReadTimeout:     time.Second,
		WriteTimeout:    time.Second,
		MetricsPort:     0,
	}
	srv := newTestServer(t, listenAddr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- run(ctx, srv, cfg, lis) }()

	waitForHealthz(t, listenAddr, 2*time.Second)
	cancel()

	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("run did not return within 5s after cancel")
	}
}

// TestRun_NilListener_DrainsCleanlyOnCancel covers the lis == nil branch that
// main() always takes in production, where run binds cfg.ListenAddr itself
// instead of serving on a caller-supplied listener.
func TestRun_NilListener_DrainsCleanlyOnCancel(t *testing.T) {
	port, err := fwknet.GetFreePort()
	require.NoError(t, err)
	listenAddr := "127.0.0.1:" + strconv.Itoa(port)

	cfg := config.ServerConfig{
		ListenAddr:      listenAddr,
		ShutdownTimeout: time.Second,
		ReadTimeout:     time.Second,
		WriteTimeout:    time.Second,
		MetricsPort:     0,
	}
	srv := newTestServer(t, listenAddr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- run(ctx, srv, cfg, nil) }()

	waitForHealthz(t, listenAddr, 2*time.Second)
	cancel()

	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("run did not return within 5s after cancel")
	}
}

// A bind failure in the metrics goroutine must fault the errgroup and drain
// the coordinator server; run's returned error surfaces the metrics-server
// failure, and the coordinator socket must no longer accept connections.
func TestRun_MetricsPortCollision_DrainsCoordinatorServer(t *testing.T) {
	// Bind the wildcard the same way serveMetrics does so the collision is
	// guaranteed on macOS as well as Linux. fwknet.ReserveListener binds only
	// 127.0.0.1, which does not shadow [::]:<port> on macOS.
	blocker, err := net.Listen("tcp", ":0")
	require.NoError(t, err)
	t.Cleanup(func() { _ = blocker.Close() })
	blockedPort := blocker.Addr().(*net.TCPAddr).Port

	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	listenAddr := lis.Addr().String()

	cfg := config.ServerConfig{
		ListenAddr:      listenAddr,
		ShutdownTimeout: time.Second,
		ReadTimeout:     time.Second,
		WriteTimeout:    time.Second,
		MetricsPort:     blockedPort,
	}
	srv := newTestServer(t, listenAddr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- run(ctx, srv, cfg, lis) }()

	select {
	case err := <-done:
		require.Error(t, err)
		require.Contains(t, err.Error(), "metrics server:")
	case <-time.After(5 * time.Second):
		t.Fatal("run did not return within 5s")
	}

	conn, dialErr := net.DialTimeout("tcp", listenAddr, 100*time.Millisecond)
	if dialErr == nil {
		_ = conn.Close()
		t.Fatalf("coordinator server at %s still accepts connections after run returned", listenAddr)
	}
}

func TestRun_InvalidMetricsTLSDrainsCoordinatorServer(t *testing.T) {
	// The metrics server fails loading its TLS cert/key before ever binding
	// metricsPort, so a bare port number carries no bind race for it.
	metricsPort, err := fwknet.GetFreePort()
	require.NoError(t, err)
	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	listenAddr := lis.Addr().String()

	cfg := config.ServerConfig{
		ListenAddr:      listenAddr,
		ShutdownTimeout: time.Second,
		ReadTimeout:     time.Second,
		WriteTimeout:    time.Second,
		MetricsPort:     metricsPort,
		MetricsCertDir:  t.TempDir(),
	}
	srv := newTestServer(t, listenAddr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- run(ctx, srv, cfg, lis) }()

	select {
	case err := <-done:
		require.ErrorIs(t, err, errMetricsTLS)
	case <-time.After(5 * time.Second):
		t.Fatal("run did not return within 5s")
	}

	conn, dialErr := net.DialTimeout("tcp", listenAddr, 100*time.Millisecond)
	if dialErr == nil {
		_ = conn.Close()
		t.Fatalf("coordinator server at %s still accepts connections after run returned", listenAddr)
	}
}
