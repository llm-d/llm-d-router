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

package proxy

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"golang.org/x/sync/errgroup"
)

func writeSelfSignedCert(t *testing.T, dir string) {
	t.Helper()

	cert, err := CreateSelfSignedTLSCertificate()
	require.NoError(t, err)

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})
	keyBytes, err := x509.MarshalPKCS8PrivateKey(cert.PrivateKey)
	require.NoError(t, err)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes})

	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.crt"), certPEM, 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.key"), keyPEM, 0o600))
}

func freeAddr(t *testing.T) string {
	t.Helper()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := ln.Addr().String()
	require.NoError(t, ln.Close())
	return addr
}

func TestServeMetrics_PlainHTTP(t *testing.T) {
	s := &Server{logger: logr.Discard()}
	addr := freeAddr(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- s.serveMetrics(ctx, addr) }()

	client := &http.Client{Timeout: 2 * time.Second}
	require.Eventually(t, func() bool {
		resp, err := client.Get("http://" + addr + "/metrics")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 2*time.Second, 20*time.Millisecond, "expected plaintext /metrics to become reachable")

	cancel()
	require.NoError(t, <-errCh)
}

func TestServeMetrics_TLS(t *testing.T) {
	certDir := t.TempDir()
	writeSelfSignedCert(t, certDir)

	s := &Server{
		logger: logr.Discard(),
		config: Config{MetricsCertDir: certDir},
	}
	addr := freeAddr(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- s.serveMetrics(ctx, addr) }()

	client := &http.Client{
		Timeout:   2 * time.Second,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}, //nolint:gosec // self-signed test cert
	}
	require.Eventually(t, func() bool {
		resp, err := client.Get("https://" + addr + "/metrics")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 2*time.Second, 20*time.Millisecond, "expected TLS /metrics to become reachable")

	// A plaintext request to the same address must not be served as /metrics
	plainClient := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := plainClient.Get("http://" + addr + "/metrics")
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusBadRequest, resp.StatusCode)

	cancel()
	require.NoError(t, <-errCh)
}

// TestServeMetrics_TLSMissingCert checks a --metrics-cert-dir missing tls.crt,
// tls.key, or both. The metrics server rejects each case and the sidecar fails
// to start.
func TestServeMetrics_TLSMissingCert(t *testing.T) {
	tests := []struct {
		name      string
		writeCert bool
		writeKey  bool
	}{
		{name: "both missing"},
		{name: "key missing", writeCert: true},
		{name: "cert missing", writeKey: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			certDir := t.TempDir()
			if tt.writeCert || tt.writeKey {
				cert, err := CreateSelfSignedTLSCertificate()
				require.NoError(t, err)
				if tt.writeCert {
					certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})
					require.NoError(t, os.WriteFile(filepath.Join(certDir, "tls.crt"), certPEM, 0o600))
				}
				if tt.writeKey {
					keyBytes, err := x509.MarshalPKCS8PrivateKey(cert.PrivateKey)
					require.NoError(t, err)
					keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes})
					require.NoError(t, os.WriteFile(filepath.Join(certDir, "tls.key"), keyPEM, 0o600))
				}
			}

			s := &Server{
				logger: logr.Discard(),
				config: Config{MetricsPort: mustFreePort(t), MetricsCertDir: certDir},
			}

			// serveMetrics itself must detect this specific broken input.
			err := s.serveMetrics(context.Background(), freeAddr(t))
			require.Error(t, err)

			// The same error must reach the caller, so the sidecar
			// fails at startup instead of running without metrics.
			grp, ctx := errgroup.WithContext(context.Background())
			s.maybeStartMetrics(ctx, grp)
			require.Error(t, grp.Wait())
		})
	}
}

// mustFreePort returns an ephemeral port number for Config.MetricsPort.
func mustFreePort(t *testing.T) int {
	t.Helper()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := ln.Addr().(*net.TCPAddr).Port
	require.NoError(t, ln.Close())
	return port
}

// TestStart_MetricsTLSFailureStopsDataPlane runs the full Start path with a
// --metrics-cert-dir missing tls.crt and tls.key: Start returns the error and
// the data-plane listener is closed.
func TestStart_MetricsTLSFailureStopsDataPlane(t *testing.T) {
	decoderURL, err := url.Parse("http://decoder.invalid:8000")
	require.NoError(t, err)

	s := NewProxy(Config{
		Port:           strconv.Itoa(mustFreePort(t)),
		DecoderURL:     decoderURL,
		MetricsPort:    mustFreePort(t),
		MetricsCertDir: t.TempDir(), // no tls.crt or tls.key
	})
	s.allowlistValidator = &AllowlistValidator{enabled: false}

	errCh := make(chan error, 1)
	go func() { errCh <- s.Start(context.Background()) }()

	// The data plane comes up before the metrics failure tears it down.
	select {
	case <-s.readyCh:
	case err := <-errCh:
		t.Fatalf("Start returned before the data plane was listening: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("data-plane listener never came up")
	}
	dataPlaneAddr := s.addr.String()

	select {
	case err := <-errCh:
		require.Error(t, err, "expected the metrics TLS failure to fail Start")
	case <-time.After(5 * time.Second):
		t.Fatal("Start kept running after the metrics server failed")
	}

	// Start only returns once startHTTP has returned, so the data-plane
	// listener is closed by then and the port accepts nothing.
	conn, err := net.DialTimeout("tcp", dataPlaneAddr, time.Second)
	if err == nil {
		_ = conn.Close()
		t.Fatalf("data-plane listener on %s still accepts connections", dataPlaneAddr)
	}
}
