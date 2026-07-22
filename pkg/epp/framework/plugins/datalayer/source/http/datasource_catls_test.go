/*
Copyright 2026 The Kubernetes Authors.

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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
			tlsCfg := tlsConfigOf(t, ds.client)
			assert.False(t, tlsCfg.InsecureSkipVerify)
			if tt.wantRootCA {
				assert.NotNil(t, tlsCfg.RootCAs)
			} else {
				assert.Nil(t, tlsCfg.RootCAs)
			}
		})
	}
}

// writeCertKey writes a self-signed cert and its key as separate PEMs; returns their paths.
func writeCertKey(t *testing.T) (certPath, keyPath string) {
	t.Helper()
	dir := t.TempDir()
	return writeCertKeyAt(t, dir, 2), filepath.Join(dir, "key.pem")
}

// writeCertKeyAt writes cert.pem and key.pem into dir, vary serial to rotate in place.
func writeCertKeyAt(t *testing.T, dir string, serial int64) (certPath string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(serial),
		Subject:      pkix.Name{CommonName: "test-client"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	require.NoError(t, err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	require.NoError(t, err)
	certPath = filepath.Join(dir, "cert.pem")
	keyPath := filepath.Join(dir, "key.pem")
	require.NoError(t, os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	require.NoError(t, os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600))
	return certPath
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
		{name: "valid client cert is served by the watcher", opts: TLSOptions{ClientCertPath: certPath, ClientKeyPath: keyPath}, wantClientCert: true},
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
			tlsCfg := tlsConfigOf(t, ds.client)
			// The cert is served by the watcher, not pinned into Certificates.
			assert.Empty(t, tlsCfg.Certificates)
			if !tt.wantClientCert {
				assert.Nil(t, tlsCfg.GetClientCertificate)
				return
			}
			require.NotNil(t, tlsCfg.GetClientCertificate)
			cert, err := tlsCfg.GetClientCertificate(nil)
			require.NoError(t, err)
			require.NotNil(t, cert)
			assert.NotEmpty(t, cert.Certificate)
		})
	}
}
