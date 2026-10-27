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
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	fwknet "github.com/llm-d/llm-d-router/test/framework/net"

	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

const (
	// bucketStartupSubprocessEnv marks the re-executed copy of this test.
	bucketStartupSubprocessEnv = "LLMD_METRICS_BUCKETS_STARTUP_SUBPROCESS"
	// bucketStartupDoneMarker is printed once the subprocess has gathered the
	// registered histogram, so a subprocess that exits zero without reaching the
	// assertions cannot pass for a successful run.
	bucketStartupDoneMarker = "metrics-buckets-startup-check-done"
)

// configuredTPOTEdges and configuredITLEdges share no value with either default
// ladder, so a gathered histogram carrying them cannot have come from the
// built-in edges.
var (
	configuredTPOTEdges = []float64{0.004, 0.009, 0.019}
	configuredITLEdges  = []float64{0.003, 0.007}
)

// TestFileDiscoveryStartupUsesConfiguredBuckets drives runWithFileDiscovery,
// the startup path that needs no Kubernetes client, with a metrics section in
// its configuration file and then gathers the process registry.
//
// The collectors are built inside a sync.Once during registration, and other
// tests in this package register metrics, so the check has to own a fresh
// process: the test re-executes the test binary with only this test selected.
// Resetting the Once from a test would change production code to suit the test
// and would race every other test holding a reference to a collector.
func TestFileDiscoveryStartupUsesConfiguredBuckets(t *testing.T) {
	if os.Getenv(bucketStartupSubprocessEnv) == "" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestFileDiscoveryStartupUsesConfiguredBuckets$", "-test.v", "-test.timeout=4m")
		cmd.Env = append(os.Environ(), bucketStartupSubprocessEnv+"=1")
		out, err := cmd.CombinedOutput()
		require.NoError(t, err, "subprocess output:\n%s", out)
		require.Contains(t, string(out), bucketStartupDoneMarker, "subprocess output:\n%s", out)
		return
	}

	dir := t.TempDir()
	endpointsPath := filepath.Join(dir, "endpoints.yaml")
	require.NoError(t, os.WriteFile(endpointsPath, []byte("endpoints: []\n"), 0o644))

	configPath := filepath.Join(dir, "epp-config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
metrics:
  requestTPOTBuckets: [0.004, 0.009, 0.019]
  requestITLBuckets: [0.003, 0.007]
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
    endpoints:
      pluginRef: file-discovery
  sources:
    - pluginRef: metrics-source
      extractors:
        - pluginRef: metrics-extractor
`, endpointsPath)), 0o644))

	grpcListener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = grpcListener.Close() })

	healthListener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = healthListener.Close() })
	healthPort := uint16(healthListener.Addr().(*net.TCPAddr).Port) //#nosec -- port is an OS-assigned ephemeral TCP port, always <= 65535

	opts := runserver.NewOptions()
	opts.GRPCPort = uint16(grpcListener.Addr().(*net.TCPAddr).Port) //#nosec -- port is an OS-assigned ephemeral TCP port, always <= 65535
	opts.GRPCHealthPort = healthPort
	opts.MetricsPort = 0
	opts.SecureServing = false
	opts.HealthChecking = true
	opts.EnablePprof = false
	opts.PoolName = testPoolName
	opts.PoolNamespace = "test-ns"
	opts.ConfigFile = configPath

	r := NewRunner()
	r.grpcListener = grpcListener
	r.healthListener = healthListener

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	require.NotNil(t, rawConfig.Metrics, "the metrics section has to survive parsing to reach registration")

	runErr := make(chan error, 1)
	go func() { runErr <- r.runWithFileDiscovery(ctx, opts, rawConfig) }()

	healthAddr := fmt.Sprintf("127.0.0.1:%d", healthPort)
	deadline := time.After(60 * time.Second)
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

	// A histogram with no children is absent from a gather, so each one needs an
	// observation before its buckets can be read.
	require.True(t, eppmetrics.RecordRequestTPOT(ctx, "m", "tm", "f", "p", true,
		time.Now().Add(-2*time.Second), time.Now().Add(-time.Second), time.Now(), 4))
	require.True(t, eppmetrics.RecordInterTokenLatency(ctx, "m", "tm", "f", "p", 0.005))

	assert.Equal(t, configuredTPOTEdges, gatheredBucketEdges(t, "llm_d_epp_request_streaming_tpot_seconds"))
	assert.Equal(t, configuredITLEdges, gatheredBucketEdges(t, "llm_d_epp_request_streaming_itl_seconds"))
	assert.NotEqual(t, eppmetrics.DefaultRequestTPOTBuckets, gatheredBucketEdges(t, "llm_d_epp_request_streaming_tpot_seconds"))

	cancel()
	select {
	case err := <-runErr:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("runWithFileDiscovery returned unexpected error: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("runWithFileDiscovery did not return after context cancel")
	}

	fmt.Println(bucketStartupDoneMarker)
}

// gatheredBucketEdges reads the upper bounds of name from the registry the
// runner registers into.
func gatheredBucketEdges(t *testing.T, name string) []float64 {
	t.Helper()

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)

	var family *dto.MetricFamily
	for _, candidate := range families {
		if candidate.GetName() == name {
			family = candidate
			break
		}
	}
	require.NotNil(t, family, "%s is absent from the gathered registry", name)
	require.NotEmpty(t, family.GetMetric())

	histogram := family.GetMetric()[0].GetHistogram()
	require.NotNil(t, histogram)

	edges := make([]float64, 0, len(histogram.GetBucket()))
	for _, bucket := range histogram.GetBucket() {
		edges = append(edges, bucket.GetUpperBound())
	}
	return edges
}
