//go:build unix

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

package runner

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"

	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// gatedCertDir returns a cert directory whose tls.crt is a named pipe, so the
// ext_proc server blocks in TLS setup until release writes the certificate.
func gatedCertDir(t *testing.T) (dir string, release func()) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "ext-proc"},
		NotAfter:     time.Now().Add(time.Hour),
	}
	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	require.NoError(t, err)
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	require.NoError(t, err)
	dir = t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tls.key"),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}), 0o600))
	certPath := filepath.Join(dir, "tls.crt")
	require.NoError(t, syscall.Mkfifo(certPath, 0o600))
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	// If the test stops before release, opening both ends unblocks a server
	// waiting to read and a writer waiting for a reader.
	t.Cleanup(func() {
		if f, err := os.OpenFile(certPath, os.O_RDWR|syscall.O_NONBLOCK, 0); err == nil {
			_ = f.Close()
		}
	})
	return dir, func() {
		// Opening a named pipe for writing blocks until the server opens it to read.
		go func() { _ = os.WriteFile(certPath, certPEM, 0o600) }()
	}
}

// TestRunWithFileDiscovery_ReadinessWaitsForExtProc is a regression test for
// https://github.com/llm-d/llm-d-router/issues/4645: the health server could
// report SERVING while the ext_proc server was still building its TLS
// credentials. The ext_proc server is held in TLS setup here, so the health
// server is up and every other readiness condition holds.
func TestRunWithFileDiscovery_ReadinessWaitsForExtProc(t *testing.T) {
	dir := t.TempDir()
	endpointsPath := filepath.Join(dir, "endpoints.yaml")
	require.NoError(t, os.WriteFile(endpointsPath, []byte("endpoints: []\n"), 0o644))
	configText := fmt.Sprintf(`apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - name: file-discovery
    type: file-discovery
    parameters:
      path: %q
      watchFile: false
  - name: random-picker
    type: random-picker
  - name: single-profile-handler
    type: single-profile-handler
  - name: metrics-source
    type: metrics-data-source
  - name: metrics-extractor
    type: core-metrics-extractor
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: random-picker
dataLayer:
  injectDefaults: false
  discovery:
    pluginRef: file-discovery
  sources:
    - pluginRef: metrics-source
      extractors:
        - pluginRef: metrics-extractor
`, endpointsPath)

	certDir, releaseCert := gatedCertDir(t)
	grpcListener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	defer grpcListener.Close()
	healthListener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	defer healthListener.Close()

	opts := runserver.NewOptions()
	opts.MetricsPort = 0
	opts.SecureServing = true
	opts.CertPath = certDir
	opts.HealthChecking = true
	opts.PoolName = testPoolName
	opts.PoolNamespace = "test-ns"
	opts.ConfigText = configText

	r := NewRunner()
	r.grpcListener = grpcListener
	r.healthListener = healthListener
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)

	runErr := make(chan error, 1)
	go func() { runErr <- r.runWithFileDiscovery(ctx, opts, rawConfig) }()

	healthAddr := healthListener.Addr().String()
	// Liveness answering means the health server is up.
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		got, err := healthStatus(ctx, healthAddr, LivenessCheckService)
		assert.NoError(c, err)
		assert.Equal(c, healthPb.HealthCheckResponse_SERVING, got)
	}, 10*time.Second, 20*time.Millisecond, "health server never came up")
	for _, service := range []string{ReadinessCheckService, "", extProcPb.ExternalProcessor_ServiceDesc.ServiceName} {
		got, err := healthStatus(ctx, healthAddr, service)
		require.NoError(t, err)
		require.Equal(t, healthPb.HealthCheckResponse_NOT_SERVING, got, "service %q while ext_proc is in TLS setup", service)
	}

	releaseCert()
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		got, err := healthStatus(ctx, healthAddr, ReadinessCheckService)
		assert.NoError(c, err)
		assert.Equal(c, healthPb.HealthCheckResponse_SERVING, got)
	}, 10*time.Second, 20*time.Millisecond, "readiness never became SERVING once ext_proc served")

	cancel()
	select {
	case err := <-runErr:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("runWithFileDiscovery returned unexpected error: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("runWithFileDiscovery did not return after context cancel")
	}
}
