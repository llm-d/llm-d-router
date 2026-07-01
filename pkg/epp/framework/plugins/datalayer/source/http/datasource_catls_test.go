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
	"net/http"
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
		wantErr string
	}{
		{name: "valid CA bundle", path: writeCACert(t)},
		{name: "missing file", path: filepath.Join(t.TempDir(), "nope.pem"), wantErr: "reading CA cert"},
		{name: "invalid PEM", path: badPEM, wantErr: "no valid CA certs"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, err := caCertPool(tt.path)
			if tt.wantErr != "" {
				assert.ErrorContains(t, err, tt.wantErr)
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
		wantErr    string
		wantRootCA bool
	}{
		{name: "valid CA sets RootCAs", caCertPath: writeCACert(t), wantRootCA: true},
		{name: "empty CA uses system pool", caCertPath: "", wantRootCA: false},
		{name: "bad CA path errors", caCertPath: "/nonexistent/ca.pem", wantErr: "reading CA cert"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds, err := NewHTTPDataSource[any]("https", "/metrics", false, tt.caCertPath, "test-type", "ds", parser)
			if tt.wantErr != "" {
				assert.ErrorContains(t, err, tt.wantErr)
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
