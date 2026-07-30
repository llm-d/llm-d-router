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
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func makeCA(t *testing.T) (*x509.Certificate, *ecdsa.PrivateKey) {
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
	cert, err := x509.ParseCertificate(der)
	require.NoError(t, err)
	return cert, key
}

// makeServerCert returns a leaf signed by ca, valid for 127.0.0.1.
func makeServerCert(t *testing.T, ca *x509.Certificate, caKey *ecdsa.PrivateKey) tls.Certificate {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: "127.0.0.1"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca, &key.PublicKey, caKey)
	require.NoError(t, err)
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

func writeCertPEM(t *testing.T, path string, cert *x509.Certificate) {
	t.Helper()
	require.NoError(t, os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw}), 0o600))
}

// TestTLSReloader_CAReload covers CA-bundle rotation on the scrape client: a
// rotated-out bundle must reject the old server (verification follows the file),
// and a bad bundle must keep the last-good one (fail-closed).
func TestTLSReloader_CAReload(t *testing.T) {
	cases := []struct {
		name      string
		rotate    func(t *testing.T, caPath string)
		wantError bool
	}{
		{
			name:      "rotated-out CA rejects the old server",
			rotate:    func(t *testing.T, caPath string) { ca, _ := makeCA(t); writeCertPEM(t, caPath, ca) },
			wantError: true,
		},
		{
			name: "bad CA keeps the last-good bundle",
			rotate: func(t *testing.T, caPath string) {
				require.NoError(t, os.WriteFile(caPath, []byte("not a valid pem"), 0o600))
			},
			wantError: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			caPath := filepath.Join(dir, "ca.pem")
			ca, caKey := makeCA(t)
			writeCertPEM(t, caPath, ca)
			serverCert := makeServerCert(t, ca, caKey)

			srv := httptest.NewUnstartedServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
			srv.TLS = &tls.Config{Certificates: []tls.Certificate{serverCert}}
			srv.StartTLS()
			defer srv.Close()

			r, err := newTLSReloader(TLSOptions{CACertPath: caPath})
			require.NoError(t, err)
			cl := &http.Client{Transport: r}

			resp, err := cl.Get(srv.URL)
			require.NoError(t, err, "server signed by the trusted CA should be accepted")
			require.NoError(t, resp.Body.Close())

			tc.rotate(t, caPath)
			r.nextCheck.Store(0) // force the reload on the next request

			resp, err = cl.Get(srv.URL)
			if tc.wantError {
				require.Error(t, err, "rotated-out CA must reject the old server")
				return
			}
			require.NoError(t, err, "a bad CA reload must keep the last-good bundle")
			require.NoError(t, resp.Body.Close())
		})
	}
}
