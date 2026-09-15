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
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	tlsutil "github.com/llm-d/llm-d-router/internal/tls"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// writeSelfSignedCert writes a tls.crt and tls.key pair into dir and returns
// the DER bytes of the certificate.
func writeSelfSignedCert(t *testing.T, dir string) []byte {
	t.Helper()

	cert, err := tlsutil.CreateSelfSignedTLSCertificate(serverLog)
	require.NoError(t, err)

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})
	keyBytes, err := x509.MarshalPKCS8PrivateKey(cert.PrivateKey)
	require.NoError(t, err)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes})

	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.crt"), certPEM, 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.key"), keyPEM, 0o600))
	return cert.Certificate[0]
}

// serve starts a server for cfg on a free port and returns its address.
func serve(t *testing.T, cfg config.ServerConfig) string {
	t.Helper()

	port, err := fwknet.GetFreePort()
	require.NoError(t, err)
	cfg.ListenAddr = fmt.Sprintf("127.0.0.1:%d", port)

	srv, err := New(cfg, pipeline.New(nil), gateway.NewWithTransport(nil, "http://gateway-stub.invalid"))
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe(ctx) }()

	t.Cleanup(func() {
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		_ = srv.Shutdown(shutdownCtx)
		require.ErrorIs(t, <-errCh, http.ErrServerClosed)
	})

	require.Eventually(t, func() bool {
		conn, err := net.DialTimeout("tcp", cfg.ListenAddr, 100*time.Millisecond)
		if err != nil {
			return false
		}
		_ = conn.Close()
		return true
	}, 5*time.Second, 20*time.Millisecond, "listener never came up on %s", cfg.ListenAddr)

	return cfg.ListenAddr
}

func TestServe_PlainHTTPWhenNotSecure(t *testing.T) {
	addr := serve(t, config.ServerConfig{})

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://" + addr + "/healthz")
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestServe_SelfSignedTLSWithoutCertPath(t *testing.T) {
	addr := serve(t, config.ServerConfig{SecureCoordinator: true})

	client := &http.Client{
		Timeout:   5 * time.Second,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}, //nolint:gosec // self-signed cert under test
	}
	resp, err := client.Get("https://" + addr + "/healthz")
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	// A plaintext request reaches net/http's TLS guard, which answers 400
	// without routing to a handler.
	plain := &http.Client{Timeout: 5 * time.Second}
	plainResp, err := plain.Get("http://" + addr + "/healthz")
	require.NoError(t, err)
	defer plainResp.Body.Close()
	require.Equal(t, http.StatusBadRequest, plainResp.StatusCode)
}

func TestServe_TLSServesCertFromCertPath(t *testing.T) {
	certDir := t.TempDir()
	want := writeSelfSignedCert(t, certDir)

	addr := serve(t, config.ServerConfig{SecureCoordinator: true, CertPath: certDir})

	require.Equal(t, want, servedCert(t, addr), "listener served a certificate other than the one in cert-path")
}

func TestServe_TLSReloadsRotatedCert(t *testing.T) {
	certDir := t.TempDir()
	first := writeSelfSignedCert(t, certDir)

	addr := serve(t, config.ServerConfig{SecureCoordinator: true, CertPath: certDir})
	require.Equal(t, first, servedCert(t, addr))

	second := writeSelfSignedCert(t, certDir)
	require.NotEqual(t, first, second, "rotation test needs a distinct second certificate")

	require.Eventually(t, func() bool {
		return bytes.Equal(second, servedCert(t, addr))
	}, 10*time.Second, 50*time.Millisecond, "rotated certificate was not picked up without a restart")
}

// servedCert returns the DER bytes of the leaf certificate addr presents.
func servedCert(t *testing.T, addr string) []byte {
	t.Helper()

	conn, err := tls.Dial("tcp", addr, &tls.Config{InsecureSkipVerify: true}) //nolint:gosec // self-signed cert under test
	require.NoError(t, err)
	defer conn.Close()

	peers := conn.ConnectionState().PeerCertificates
	require.NotEmpty(t, peers)
	return peers[0].Raw
}

func TestServe_TLSMinVersionRejectsOlderClient(t *testing.T) {
	addr := serve(t, config.ServerConfig{SecureCoordinator: true, TLSMinVersion: "VersionTLS13"})

	_, err := tls.Dial("tcp", addr, &tls.Config{
		InsecureSkipVerify: true, //nolint:gosec // self-signed cert under test
		MaxVersion:         tls.VersionTLS12,
	})
	require.Error(t, err, "TLS 1.2 client must be rejected when the floor is TLS 1.3")
}

func TestNew_RejectsInvalidTLSProfile(t *testing.T) {
	tests := []struct {
		name string
		cfg  config.ServerConfig
	}{
		{
			name: "unknown TLS version",
			cfg:  config.ServerConfig{SecureCoordinator: true, TLSMinVersion: "VersionTLS99"},
		},
		{
			name: "unknown cipher suite",
			cfg:  config.ServerConfig{SecureCoordinator: true, TLSCipherSuites: []string{"TLS_NOT_A_CIPHER"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.cfg, pipeline.New(nil), gateway.NewWithTransport(nil, "http://gateway-stub.invalid"))
			require.Error(t, err)
		})
	}
}

func TestParseTLSProfile_EmptyKeepsCryptoDefaults(t *testing.T) {
	profile, err := parseTLSProfile("", nil)
	require.NoError(t, err)
	require.Zero(t, profile.minVersion, "empty tls_min_version must leave the crypto/tls default")
	require.Empty(t, profile.cipherSuites, "empty tls_cipher_suites must leave the crypto/tls default")
}
