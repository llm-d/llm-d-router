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

package epp

import (
	"context"
	_ "embed"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace/noop"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	metricsutils "k8s.io/component-base/metrics/testutil"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	eppRunner "github.com/llm-d/llm-d-router/cmd/epp/runner"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	dlmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	eppServer "github.com/llm-d/llm-d-router/pkg/epp/server"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
	integration "github.com/llm-d/llm-d-router/test/integration"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// Global State (Initialized in TestMain)
var (
	k8sClient     client.Client
	testEnv       *envtest.Environment
	testScheme    = runtime.NewScheme()
	logger        = zap.New(zap.UseDevMode(true), zap.Level(zapcore.Level(logutil.DEFAULT)))
	baseResources []*unstructured.Unstructured
)

const (
	testPoolName = "vllm-qwen3-32b-pool"

	// mockDataSourceType is the plugin type name used for the mock data source in integration tests.
	mockDataSourceType = "mock-metrics-source"

	// portBindMaxAttempts bounds how many times the harness re-rolls free ports and
	// restarts the manager when a port is stolen before the server binds it (issue
	// #1066). portBindRetryDelay paces the retries so a transient port grab can clear.
	portBindMaxAttempts = 5
	portBindRetryDelay  = 500 * time.Millisecond
)

//go:embed testdata/datalayer-config.yaml
var testDLConfig string

type runMode string
type standaloneStrategy string

const (
	modeStandard    runMode            = "standard"
	modeStandalone  runMode            = "standalone"
	strategyNoCRD   standaloneStrategy = "no_crd"   // Pure standalone
	strategyWithCRD standaloneStrategy = "with_crd" // Standalone but watching CRDs
)

// HarnessConfig holds configuration options for the TestHarness.
type HarnessConfig struct {
	// runMode is the master switch. It tells you explicitly what the config is for.
	runMode runMode

	// standaloneStrategy settings are used when runMode == modeStandalone.
	standaloneStrategy standaloneStrategy

	// configText overrides the default testConfig if provided. A nil value means use default.
	configText *string

	// Tracing indicates if tracing should be enabled for this test.
	Tracing bool
}

// HarnessOption is a functional option for configuring the TestHarness.
type HarnessOption func(*HarnessConfig)

// WithStandaloneMode configures the harness to run in standalone runMode
func WithStandaloneMode(standaloneStrategy standaloneStrategy) HarnessOption {
	return func(c *HarnessConfig) {
		c.runMode = modeStandalone
		c.standaloneStrategy = standaloneStrategy
	}
}

// WithStandardMode configures the harness to run in standard runMode
func WithStandardMode() HarnessOption {
	return func(c *HarnessConfig) {
		c.runMode = modeStandard
	}
}

// WithConfigText overrides the default EPP configuration text.
func WithConfigText(text string) HarnessOption {
	return func(c *HarnessConfig) {
		c.configText = &text
	}
}

// WithTracing enables tracing for the test harness.
func WithTracing() HarnessOption {
	return func(c *HarnessConfig) {
		c.Tracing = true
	}
}

// metricsBackend abstracts how pod metrics are injected into the test environment.
type metricsBackend interface {
	SetPodMetrics(m map[types.NamespacedName]*fwkdl.Metrics)
}

// mockDataSourceBackend wraps the mock DataSource to implement metricsBackend.
type mockDataSourceBackend struct {
	mockDataSource *dlmocks.MetricsDataSource
}

func (b *mockDataSourceBackend) SetPodMetrics(m map[types.NamespacedName]*fwkdl.Metrics) {
	b.mockDataSource.SetMetrics(m)
}

// TestHarness encapsulates the environment for a single isolated EPP test run.
// It manages the lifecycle of the controller manager, the EPP server, and the K8s namespace.
type TestHarness struct {
	t         *testing.T
	ctx       context.Context
	Namespace string

	// --- Config State ---
	runMode            runMode
	standaloneStrategy standaloneStrategy
	Tracing            bool

	Client    extProcPb.ExternalProcessor_ProcessClient
	Datastore datastore.Datastore

	// --- Tracing State ---
	Exporter *tracetest.InMemoryExporter
	tp       *sdktrace.TracerProvider

	// Internal handles for cleanup
	grpcConn       *grpc.ClientConn
	metricsBackend metricsBackend

	Runner *eppRunner.Runner
}

// hasCRDs returns true when the harness is running in a mode that has CRD support.
func (h *TestHarness) hasCRDs() bool {
	return h.runMode != modeStandalone || h.standaloneStrategy != strategyNoCRD
}

// NewTestHarness boots up a fully isolated test environment.
// It creates a unique Namespace, scopes the Manager to that Namespace, and starts the components.
// Note: EPP tests must run serially because they rely on the global Prometheus registry.
func NewTestHarness(ctx context.Context, t *testing.T, opts ...HarnessOption) *TestHarness {
	t.Helper()

	config := &HarnessConfig{}
	for _, opt := range opts {
		opt(config)
	}

	// Determine config text and namespace prefix.
	configText := testDLConfig
	if config.configText != nil {
		configText = *config.configText
	}

	// Create dedicated namespace for the whole test.
	uid := uuid.New().String()[:8]
	testNamespaceName := "epp-test-" + uid
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespaceName}}
	require.NoError(t, k8sClient.Create(ctx, ns), "failed to create test namespace")

	// Tracing Setup (InMemory). Global and port-independent, so it is configured once
	// rather than per port-bind attempt below.
	var exporter *tracetest.InMemoryExporter
	var tp *sdktrace.TracerProvider
	if config.Tracing {
		exporter = tracetest.NewInMemoryExporter()
		tp = sdktrace.NewTracerProvider(
			sdktrace.WithSyncer(exporter),
		)
		otel.SetTracerProvider(tp)
	}

	// Reserve the ext_proc port once, up front, and keep the listener bound for the
	// lifetime of the test: the server serves on this listener instead of binding a
	// port number chosen earlier, so no other process can take it in between (issue
	// #1066). The reservation is outside the retry closure because a held port cannot
	// be stolen and so never needs re-rolling.
	lis, err := testutils.ReserveListener()
	require.NoError(t, err, "failed to reserve ext_proc port")
	t.Cleanup(func() { _ = lis.Close() })
	grpcPort := lis.Addr().(*net.TCPAddr).Port

	var (
		eppOptions *eppServer.Options
		runner     *eppRunner.Runner
		dataStore  datastore.Datastore
		backend    metricsBackend
		mgrCtx     context.Context
		mgrCancel  context.CancelFunc
		mgrDone    chan struct{}
	)
	attempt := func() error {
		eppOptions = defaultEppServerOptions(t, testNamespaceName, configText)
		if config.runMode == modeStandalone && config.standaloneStrategy == strategyNoCRD {
			// Only standalone EPP without crd need to set the EndpointSelector.
			eppOptions.EndpointSelector = labels.SelectorFromSet(labels.Set{"app": testPoolName})
		}

		// Shorten the Prometheus refresh interval so WaitForReadyPodsMetric (10s timeout)
		// has many opportunities to observe the metric update instead of only ~2.
		eppOptions.RefreshPrometheusMetricsInterval = 500 * time.Millisecond

		mockDataSource := dlmocks.NewDataSource(plugin.TypedName{
			Type: mockDataSourceType,
			Name: mockDataSourceType,
		})
		r, mgr, ds, err := eppRunner.NewTestRunnerSetup(ctx, testEnv.Config, eppOptions, mockDataSource, lis)
		if err != nil {
			// Not a port-bind race; surfaced as-is (RetryOnAddrInUse will not retry it).
			return fmt.Errorf("failed to create manager: %w", err)
		}
		runner, dataStore = r, ds
		backend = metricsBackend(&mockDataSourceBackend{mockDataSource: mockDataSource})

		// Fresh cancellable context per attempt, derived from the stable parent ctx; the
		// previous attempt's context is cancelled before retrying, so it does not nest
		// (fatcontext flags the reassignment of the outer mgrCtx, which is intentional here).
		mgrCtx, mgrCancel = context.WithCancel(ctx) //nolint:fatcontext
		mgrDone = make(chan struct{})
		mgrErr := make(chan error, 1)
		go func() {
			defer close(mgrDone)
			err := mgr.Start(mgrCtx)
			mgrErr <- err
			// Context cancellation is expected during teardown.
			if err != nil && !strings.Contains(err.Error(), "context canceled") {
				logger.Error(err, "manager stopped unexpectedly")
			}
		}()

		// Wait until the ext-proc server starts serving, or the manager reports an early
		// failure. On failure, stop this attempt's manager and wait for it to release
		// its listeners before retrying.
		if err := integration.WaitExtProcReady(grpcPort, mgrErr); err != nil {
			mgrCancel()
			<-mgrDone
			return err
		}
		return nil
	}
	require.NoError(t,
		integration.RetryOnAddrInUse(portBindMaxAttempts, portBindRetryDelay, attempt, metrics.Reset),
		"EPP manager failed to bind its ports",
	)

	extProcClient, conn := integration.ExtProcServerClient(
		mgrCtx,
		t,
		grpcPort,
		logger,
	)

	h := &TestHarness{
		t:                  t,
		ctx:                mgrCtx,
		Namespace:          eppOptions.PoolNamespace,
		runMode:            config.runMode,
		standaloneStrategy: config.standaloneStrategy,
		Tracing:            config.Tracing,
		Client:             extProcClient,
		Datastore:          dataStore,
		Exporter:           exporter,
		tp:                 tp,
		grpcConn:           conn,
		metricsBackend:     backend,
		Runner:             runner,
	}

	// 8. Register Cleanup

	t.Cleanup(func() {
		mgrCancel()
		// Wait for the manager (and its gRPC server) to fully stop before returning.
		// Without this, the gRPC server may still hold its port when the next test
		// calls GetFreePort(), causing a bind: address already in use race.
		<-mgrDone
		if config.Tracing {
			_ = tp.Shutdown(ctx)
			// Reset to no-op to avoid pollution between tests.
			otel.SetTracerProvider(noop.NewTracerProvider())
		}
		_ = h.grpcConn.Close()
		// Deleting the Namespace cascades to all contained resources.
		_ = k8sClient.Delete(context.Background(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: eppOptions.PoolNamespace}})
		// Crucial: Reset global metrics registry to prevent pollution between serial tests.
		metrics.Reset()
	})

	return h
}

func defaultEppServerOptions(t *testing.T, namespace, configText string) *eppServer.Options {
	t.Helper()

	eppOptions := eppServer.NewOptions()
	eppOptions.PoolName = testPoolName
	eppOptions.PoolNamespace = namespace
	eppOptions.ConfigText = configText

	// GRPCPort is left at its default: the harness hands the ext_proc server a
	// pre-bound listener, which takes precedence over the port.

	// No test dials the health server, so let the kernel assign the port: a
	// port 0 bind cannot lose a race to another listener.
	eppOptions.GRPCHealthPort = 0
	eppOptions.EndpointTargetPorts = []int{8000}
	eppOptions.SecureServing = false
	return eppOptions
}

// GetSpans returns the currently recorded spans from the in-memory exporter.
func (h *TestHarness) GetSpans() tracetest.SpanStubs {
	return h.Exporter.GetSpans()
}

// --- Fluent Builder API ---

// WithBaseResources injects the standard pool and objective definitions into the test namespace.
// These resources are pre-parsed in TestMain to avoid I/O overhead in the loop.
func (h *TestHarness) WithBaseResources() *TestHarness {
	h.t.Helper()
	for _, obj := range baseResources {
		copy := obj.DeepCopy()
		copy.SetNamespace(h.Namespace)
		require.NoError(h.t, k8sClient.Create(h.ctx, copy), "failed to create base resource: %s", obj.GetKind())
	}
	return h
}

// WithPods creates pod objects in the API server and configures the metrics backend.
func (h *TestHarness) WithPods(pods []PodState) *TestHarness {
	h.t.Helper()
	metricsMap := make(map[types.NamespacedName]*fwkdl.Metrics)

	// Build metrics map.
	for _, p := range pods {
		metricsKeyName := fmt.Sprintf("pod-%d-rank-0", p.index)
		activeModelsMap := make(map[string]int)
		for _, m := range p.activeModels {
			activeModelsMap[m] = 1
		}

		metricsMap[types.NamespacedName{Namespace: h.Namespace, Name: metricsKeyName}] = &fwkdl.Metrics{
			WaitingQueueSize:    p.queueSize,
			KVCacheUsagePercent: p.kvCacheUsage,
			ActiveModels:        activeModelsMap,
			WaitingModels:       make(map[string]int),
		}
	}
	h.metricsBackend.SetPodMetrics(metricsMap)

	// Create K8s Objects.
	for _, p := range pods {
		name := fmt.Sprintf("pod-%d", p.index)

		pod := testutil.MakePod(name).
			Namespace(h.Namespace).
			ReadyCondition(). // Sets Status.Conditions.
			Labels(map[string]string{"app": testPoolName}).
			IP(fmt.Sprintf("192.168.1.%d", p.index+1)).
			Complete().
			ObjRef()

		// Snapshot the status (Create wipes it).
		intendedStatus := pod.Status

		// Create the resource.
		require.NoError(h.t, k8sClient.Create(h.ctx, pod), "failed to create pod %s", name)

		// Restore Status on the created K8s object which now has the correct ResourceVersion/UID.
		pod.Status = intendedStatus

		// Update Status subresource.
		require.NoError(h.t, k8sClient.Status().Update(h.ctx, pod), "failed to update status for pod %s", name)
	}
	return h
}

// WaitForReadyPodsMetric blocks until the prometheus metric 'inference_pool_ready_pods' matches the expected count.
func (h *TestHarness) WaitForReadyPodsMetric(expectedCount int) {
	h.t.Helper()

	expected := cleanMetric(metricReadyPods(expectedCount))
	require.Eventually(h.t, func() bool {
		err := metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(expected),
			"inference_pool_ready_pods")
		return err == nil
	}, 10*time.Second, 50*time.Millisecond, "Timed out waiting for inference_pool_ready_pods metric to settle")
}

// WaitForSync blocks until the EPP Datastore has synced the expected number of pods.
func (h *TestHarness) WaitForSync(expectedPods int, checkModelObjective string) *TestHarness {
	h.t.Helper()

	var lastPoolSynced bool
	var lastPodsFound int
	require.Eventually(h.t, func() bool {
		hasCRDs := h.hasCRDs()
		lastPoolSynced = h.Datastore.PoolHasSynced()
		lastPodsFound = len(h.Datastore.PodList(datastore.AllPodsPredicate))
		if hasCRDs && !lastPoolSynced {
			return false
		}
		if lastPodsFound != expectedPods {
			return false
		}
		if hasCRDs && checkModelObjective != "" && h.Datastore.ObjectiveGet(checkModelObjective) == nil {
			return false
		}
		return true
	}, 10*time.Second, 50*time.Millisecond,
		"Datastore sync timed out (runMode=%v standaloneStrategy=%v poolSynced=%v podsFound=%d expected=%d)",
		h.runMode,
		h.standaloneStrategy,
		lastPoolSynced,
		lastPodsFound,
		expectedPods,
	)
	return h
}

// ExpectMetrics asserts that specific metrics match the expected Prometheus output.
// It uses Eventually to allow for slight delays in metric recording (e.g. async token counting).
func (h *TestHarness) ExpectMetrics(expected map[string]string) {
	h.t.Helper()
	for name, value := range expected {
		var err error
		assert.Eventually(h.t, func() bool {
			err = metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(value), name)
			return err == nil
		}, 2*time.Second, 50*time.Millisecond, "Timed out waiting for metric %s to match: %v", name)
		if err != nil {
			h.t.Errorf("Metric mismatch for %s: %v", name, err)
		}
	}
}
