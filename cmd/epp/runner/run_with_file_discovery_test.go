/*
Copyright 2025 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"

	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestRunWithFileDiscovery_Smoke is a wiring test for the file-discovery path.
// It does not exercise ext_proc routing; that lives in the integration test.
// The asserts here guard against regressions in: (a) phaseTwo + resolveDiscovery
// agreeing on a single plugin instance (the bug elevran caught), (b) the
// RunnableGroup's Ready() gate firing once the plugin loads its initial file,
// and (c) the health and ext_proc gRPC servers binding their ports.
func TestRunWithFileDiscovery_Smoke(t *testing.T) {
	dir := t.TempDir()
	endpointsPath := filepath.Join(dir, "endpoints.yaml")
	require.NoError(t, os.WriteFile(endpointsPath, []byte(
		"endpoints:\n"+
			"  - name: stub\n"+
			"    address: 127.0.0.1\n"+
			"    port: \"19999\"\n"), 0o644))

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

	grpcPort := freeTCPPort(t)
	healthPort := freeTCPPort(t)
	metricsPort := freeTCPPort(t)

	opts := runserver.NewOptions()
	opts.GRPCPort = grpcPort
	opts.GRPCHealthPort = healthPort
	opts.MetricsPort = metricsPort
	opts.SecureServing = false
	opts.HealthChecking = true
	opts.EnablePprof = false
	opts.PoolName = "test-pool"
	opts.PoolNamespace = "test-ns"
	opts.ConfigText = configText

	r := NewRunner()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	require.NotNil(t, rawConfig.DataLayer)
	require.NotNil(t, rawConfig.DataLayer.Discovery)

	runErr := make(chan error, 1)
	go func() { runErr <- r.runWithFileDiscovery(ctx, opts, rawConfig) }()

	// Health gRPC is gated on the discovery plugin's Ready() channel. Reaching
	// SERVING proves: the plugin loaded ./endpoints.yaml, fired Ready, and the
	// runnable group started both the health server and (separately) the ext_proc
	// server. If phaseTwo and resolveDiscovery disagreed on the plugin instance,
	// Start() would never run and this would time out.
	healthAddr := fmt.Sprintf("127.0.0.1:%d", healthPort)
	deadline := time.After(10 * time.Second)
	for {
		select {
		case err := <-runErr:
			t.Fatalf("runWithFileDiscovery exited before health came up: %v", err)
		case <-deadline:
			t.Fatal("timeout waiting for health gRPC to reach SERVING")
		default:
		}
		if checkHealthServing(healthAddr) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	// ext_proc port is bound (also gated on Ready).
	extProcConn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", grpcPort), time.Second)
	require.NoError(t, err, "ext_proc port should accept TCP connections")
	_ = extProcConn.Close()

	// Confirm the resolved discovery plugin matches the one the loader registered.
	disc, err := r.resolveDiscovery(rawConfig)
	require.NoError(t, err)
	assert.Equal(t, "file-discovery", disc.TypedName().Type)
	assert.Equal(t, "file-discovery", disc.TypedName().Name)

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

func freeTCPPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := l.Addr().(*net.TCPAddr).Port
	require.NoError(t, l.Close())
	return port
}

func checkHealthServing(addr string) bool {
	cc, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return false
	}
	defer cc.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	resp, err := healthPb.NewHealthClient(cc).Check(ctx, &healthPb.HealthCheckRequest{})
	return err == nil && resp.GetStatus() == healthPb.HealthCheckResponse_SERVING
}
