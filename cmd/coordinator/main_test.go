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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

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

func waitForDial(t *testing.T, addr string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("no listener came up on %s within %s", addr, timeout)
}

func writeMetricsCertificate(t *testing.T, dir string) {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "localhost"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	require.NoError(t, err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	require.NoError(t, err)

	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER}), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.key"), pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600))
}

func TestServeMetricsHTTP(t *testing.T) {
	port, err := fwknet.GetFreePort()
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- serveMetrics(ctx, port, "") }()

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
	port, err := fwknet.GetFreePort()
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- serveMetrics(ctx, port, certDir) }()

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
			port, err := fwknet.GetFreePort()
			require.NoError(t, err)

			err = serveMetrics(context.Background(), port, certDir)
			require.ErrorIs(t, err, errMetricsTLS)
		})
	}
}

// With MetricsPort <= 0 no metrics goroutine joins the errgroup, so the only
// exit path is context cancellation. run must return nil once the
// coordinator server drains.
func TestRun_MetricsDisabled_DrainsCleanlyOnCancel(t *testing.T) {
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
	go func() { done <- run(ctx, srv, cfg) }()

	waitForDial(t, listenAddr, 2*time.Second)
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

	inferencePort, err := fwknet.GetFreePort()
	require.NoError(t, err)
	listenAddr := "127.0.0.1:" + strconv.Itoa(inferencePort)

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
	go func() { done <- run(ctx, srv, cfg) }()

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
	metricsPort, err := fwknet.GetFreePort()
	require.NoError(t, err)
	inferencePort, err := fwknet.GetFreePort()
	require.NoError(t, err)
	listenAddr := "127.0.0.1:" + strconv.Itoa(inferencePort)

	cfg := config.ServerConfig{
		ListenAddr:      listenAddr,
		ShutdownTimeout: time.Second,
		ReadTimeout:     time.Second,
		WriteTimeout:    time.Second,
		MetricsPort:     metricsPort,
		MetricsCertPath: t.TempDir(),
	}
	srv := newTestServer(t, listenAddr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- run(ctx, srv, cfg) }()

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
