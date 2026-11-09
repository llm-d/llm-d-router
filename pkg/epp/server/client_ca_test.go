/*
Copyright 2025 The llm-d Authors.

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
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"math/big"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	healthgrpc "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
)

// testClientCA is a CA and one client certificate it signed.
type testClientCA struct {
	pool   *x509.CertPool
	client tls.Certificate
}

func newTestClientCA(t *testing.T) testClientCA {
	t.Helper()
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	caTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "test-ca"},
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTmpl, caTmpl, &caKey.PublicKey, caKey)
	require.NoError(t, err)
	caCert, err := x509.ParseCertificate(caDER)
	require.NoError(t, err)

	clientKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	clientTmpl := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: "gateway"},
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	clientDER, err := x509.CreateCertificate(rand.Reader, clientTmpl, caCert, &clientKey.PublicKey, caKey)
	require.NoError(t, err)

	pool := x509.NewCertPool()
	pool.AddCert(caCert)
	return testClientCA{
		pool:   pool,
		client: tls.Certificate{Certificate: [][]byte{clientDER}, PrivateKey: clientKey},
	}
}

// startRunnerWithClientCA serves the ext_proc runner over TLS with a
// self-signed server certificate and the given client CA, on a loopback port.
func startRunnerWithClientCA(t *testing.T, pool *x509.CertPool) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	r := NewDefaultExtProcServerRunner()
	r.SecureServing = true
	r.ClientCAs = pool
	r.GrpcListener = lis
	r.Datastore = datastore.NewDatastore(t.Context(), nil)
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	go func() { _ = r.AsRunnable(logutil.NewTestLogger()).Start(ctx) }()
	return lis.Addr().String()
}

// healthCheck dials addr with the given client TLS config and returns the
// status code of one health RPC. The server certificate is self-signed, so the
// client skips its verification; the test is about the client certificate.
func healthCheck(t *testing.T, addr string, clientTLS *tls.Config) codes.Code {
	t.Helper()
	clientTLS.InsecureSkipVerify = true //nolint:gosec // self-signed server certificate in a test
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(credentials.NewTLS(clientTLS)))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	_, err = healthgrpc.NewHealthClient(conn).Check(ctx, &healthgrpc.HealthCheckRequest{})
	return status.Code(err)
}

// A client CA on the runner must close the listener to peers without a
// certificate from that CA, and keep it open to peers with one.
func TestExtProcServerRequiresClientCertificateWhenClientCASet(t *testing.T) {
	ca := newTestClientCA(t)
	addr := startRunnerWithClientCA(t, ca.pool)

	code := healthCheck(t, addr, &tls.Config{})
	require.Equal(t, codes.Unavailable, code, "a client without a certificate must fail the handshake")

	other := newTestClientCA(t)
	code = healthCheck(t, addr, &tls.Config{Certificates: []tls.Certificate{other.client}})
	require.Equal(t, codes.Unavailable, code, "a certificate from another CA must fail the handshake")

	code = healthCheck(t, addr, &tls.Config{Certificates: []tls.Certificate{ca.client}})
	require.Contains(t, []codes.Code{codes.OK, codes.Unimplemented}, code,
		"a certificate from the configured CA must complete the handshake and the RPC")
}

// Without a client CA the listener keeps accepting any TLS client, which is
// the default and what existing gateways rely on.
func TestExtProcServerAcceptsAnyClientWithoutClientCA(t *testing.T) {
	addr := startRunnerWithClientCA(t, nil)
	code := healthCheck(t, addr, &tls.Config{})
	require.Contains(t, []codes.Code{codes.OK, codes.Unimplemented}, code)
}
