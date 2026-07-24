package server

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

// writeCAFile writes a self-signed CA cert to a temp file and returns its path.
func writeCAFile(t *testing.T) string {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "test-ca"},
		IsCA:                  true,
		BasicConstraintsValid: true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "ca.crt")
	require.NoError(t, os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	return path
}

func TestConfigureMetricsTLS(t *testing.T) {
	validCA := writeCAFile(t)
	badPEM := filepath.Join(t.TempDir(), "bad.crt")
	require.NoError(t, os.WriteFile(badPEM, []byte("not a pem"), 0o600))

	tests := []struct {
		name           string
		certDir        string
		clientCA       string
		wantErr        error
		wantSecure     bool
		wantClientAuth bool
	}{
		{name: "neither"},
		{name: "cert dir only", certDir: "/etc/tls", wantSecure: true},
		{name: "mTLS", certDir: "/etc/tls", clientCA: validCA, wantSecure: true, wantClientAuth: true},
		{name: "missing CA file", clientCA: filepath.Join(t.TempDir(), "nope"), wantErr: errReadMetricsClientCA},
		{name: "invalid CA PEM", clientCA: badPEM, wantErr: errNoValidMetricsCA},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := &Options{MetricsCertDir: tc.certDir, MetricsClientCAFile: tc.clientCA}
			mo := &metricsserver.Options{}
			err := ConfigureMetricsTLS(opts, mo)
			if tc.wantErr != nil {
				require.ErrorIs(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.wantSecure, mo.SecureServing)
			if tc.certDir != "" {
				require.Equal(t, tc.certDir, mo.CertDir)
			}
			if !tc.wantSecure {
				require.Empty(t, mo.TLSOpts)
				return
			}
			require.Len(t, mo.TLSOpts, 1)
			cfg := &tls.Config{}
			mo.TLSOpts[0](cfg)
			require.Equal(t, uint16(tls.VersionTLS12), cfg.MinVersion)
			if tc.wantClientAuth {
				require.Equal(t, tls.RequireAndVerifyClientCert, cfg.ClientAuth)
				require.NotNil(t, cfg.ClientCAs)
			} else {
				require.Equal(t, tls.NoClientCert, cfg.ClientAuth)
				require.Nil(t, cfg.ClientCAs)
			}
		})
	}
}
