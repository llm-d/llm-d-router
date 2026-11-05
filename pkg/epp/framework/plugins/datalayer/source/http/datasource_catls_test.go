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

package http

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// writeCACert writes a self-signed CA PEM and returns its path.
func writeCACert(t *testing.T) string {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "test-ca"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "ca.pem")
	require.NoError(t, os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	return path
}

func TestCACertPool(t *testing.T) {
	badPEM := filepath.Join(t.TempDir(), "bad.pem")
	require.NoError(t, os.WriteFile(badPEM, []byte("not a certificate"), 0o600))

	tests := []struct {
		name    string
		path    string
		wantErr error
	}{
		{name: "valid CA bundle", path: writeCACert(t)},
		{name: "missing file", path: filepath.Join(t.TempDir(), "nope.pem"), wantErr: ErrReadCACert},
		{name: "invalid PEM", path: badPEM, wantErr: ErrNoValidCACerts},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, err := caCertPool(tt.path)
			if tt.wantErr != nil {
				assert.ErrorIs(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.NotNil(t, pool)
		})
	}
}

func TestNewHTTPDataSource_CACertPath(t *testing.T) {
	parser := func(r io.Reader) (any, error) { return struct{}{}, nil }

	tests := []struct {
		name       string
		caCertPath string
		wantErr    error
		wantRootCA bool
	}{
		{name: "valid CA sets RootCAs", caCertPath: writeCACert(t), wantRootCA: true},
		{name: "empty CA uses system pool", caCertPath: "", wantRootCA: false},
		{name: "bad CA path errors", caCertPath: "/nonexistent/ca.pem", wantErr: ErrReadCACert},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds, err := NewHTTPDataSource[any]("https", "/metrics", TLSOptions{CACertPath: tt.caCertPath}, "test-type", "ds", parser)
			if tt.wantErr != nil {
				assert.ErrorIs(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			cl, ok := ds.client.(*client)
			require.True(t, ok)
			tr, ok := cl.Transport.(*http.Transport)
			require.True(t, ok)
			assert.False(t, tr.TLSClientConfig.InsecureSkipVerify)
			if tt.wantRootCA {
				assert.NotNil(t, tr.TLSClientConfig.RootCAs)
			} else {
				assert.Nil(t, tr.TLSClientConfig.RootCAs)
			}
		})
	}
}

// writeCertKey writes a self-signed cert and its key as separate PEMs; returns their paths.
func writeCertKey(t *testing.T) (certPath, keyPath string) {
	t.Helper()
	dir := t.TempDir()
	certPath = filepath.Join(dir, "cert.pem")
	keyPath = filepath.Join(dir, "key.pem")
	writeCertKeyTo(t, certPath, keyPath, "test-client")
	return certPath, keyPath
}

// writeCertKeyTo writes a self-signed client cert with the given common name and its key.
func writeCertKeyTo(t *testing.T, certPath, keyPath, commonName string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: commonName},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	require.NoError(t, err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600))
	require.NoError(t, os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
}

func TestNewHTTPDataSource_ClientCert(t *testing.T) {
	parser := func(r io.Reader) (any, error) { return struct{}{}, nil }
	certPath, keyPath := writeCertKey(t)

	tests := []struct {
		name           string
		opts           TLSOptions
		wantErr        error
		wantClientCert bool
	}{
		{name: "valid client cert sets GetClientCertificate", opts: TLSOptions{ClientCertPath: certPath, ClientKeyPath: keyPath}, wantClientCert: true},
		{name: "no client cert", opts: TLSOptions{}, wantClientCert: false},
		{name: "bad client cert path errors", opts: TLSOptions{ClientCertPath: "/nonexistent/cert.pem", ClientKeyPath: "/nonexistent/key.pem"}, wantErr: ErrLoadClientCert},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds, err := NewHTTPDataSource[any]("https", "/metrics", tt.opts, "test-type", "ds", parser)
			if tt.wantErr != nil {
				assert.ErrorIs(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			cl, ok := ds.client.(*client)
			require.True(t, ok)
			if !tt.wantClientCert {
				tr, ok := cl.Transport.(*http.Transport)
				require.True(t, ok)
				assert.Nil(t, tr.TLSClientConfig.GetClientCertificate)
				assert.Nil(t, ds.certWatcher)
				return
			}
			rt, ok := cl.Transport.(*rotatingTransport)
			require.True(t, ok)
			require.NotNil(t, rt.current.Load().TLSClientConfig.GetClientCertificate)
			cert, err := rt.current.Load().TLSClientConfig.GetClientCertificate(&tls.CertificateRequestInfo{})
			require.NoError(t, err)
			assert.NotEmpty(t, cert.Certificate)
			assert.NotNil(t, ds.certWatcher)
		})
	}
}

// TestHTTPDataSource_ClientCertRotation verifies that a client cert rotated on disk
// is presented on later scrapes, including over previously warm keep-alive connections.
func TestHTTPDataSource_ClientCertRotation(t *testing.T) {
	dir := t.TempDir()
	certPath, keyPath := filepath.Join(dir, "cert.pem"), filepath.Join(dir, "key.pem")
	writeCertKeyTo(t, certPath, keyPath, "client-v1")

	var mu sync.Mutex
	var seen string
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = r.TLS.PeerCertificates[0].Subject.CommonName
		mu.Unlock()
	}))
	srv.TLS = &tls.Config{ClientAuth: tls.RequireAnyClientCert}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	require.NoError(t, err)

	parser := func(r io.Reader) (any, error) { return struct{}{}, nil }
	ds, err := NewHTTPDataSource[any]("https", "/metrics",
		TLSOptions{SkipVerify: true, ClientCertPath: certPath, ClientKeyPath: keyPath}, "test-type", "ds", parser)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	require.NoError(t, ds.Start(ctx))

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{MetricsHost: u.Host}, nil)
	scrapedAs := func() string {
		_, err := ds.Poll(ctx, ep)
		require.NoError(t, err)
		mu.Lock()
		defer mu.Unlock()
		return seen
	}
	require.Equal(t, "client-v1", scrapedAs())

	// Wait for the watcher to be watching the files before rotating them.
	time.Sleep(100 * time.Millisecond)
	writeCertKeyTo(t, certPath, keyPath, "client-v2")
	require.Eventually(t, func() bool { return scrapedAs() == "client-v2" }, 15*time.Second, 50*time.Millisecond)
}
