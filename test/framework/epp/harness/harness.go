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

// Package harness boots a shared envtest environment and per-test EPP servers for
// integration suites.
//
// It sits at the top of the framework layering and may import anything, including
// pkg/epp/server and cmd/epp/runner; only integration and e2e suites import it.
package harness

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
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
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	metricsutils "k8s.io/component-base/metrics/testutil"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	eppRunner "github.com/llm-d/llm-d-router/cmd/epp/runner"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	dlmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	eppServer "github.com/llm-d/llm-d-router/pkg/epp/server"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	fwkk8s "github.com/llm-d/llm-d-router/test/framework/k8s"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// Global State (Initialized in Run)
var (
	k8sClient     client.Client
	testEnv       *envtest.Environment
	testScheme    = runtime.NewScheme()
	logger        = zap.New(zap.UseDevMode(true), zap.Level(zapcore.Level(logutil.DEFAULT)))
	baseResources []*unstructured.Unstructured
)

// repoRootPath is the on-disk path to this repository. Hermetic tests use local
// llm-d CRDs and fixtures so API group migrations are exercised before CI.
var repoRootPath string

// Run builds the shared envtest environment, runs the package's tests and tears the
// environment down again. TestMain delegates to it.
func Run(m *testing.M) int {
	ctrl.SetLogger(logger)

	repoRootPath = moduleDir("github.com/llm-d/llm-d-router")
	gaieModulePath := moduleDir("sigs.k8s.io/gateway-api-inference-extension")
	crdPaths := []string{
		filepath.Join(gaieModulePath, "config", "crd", "bases"),
		filepath.Join(repoRootPath, "config", "crd", "bases"),
	}

	// 1. EnvTest Setup (API Server + Etcd)
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     crdPaths,
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := testEnv.Start()
	if err != nil {
		panic(fmt.Sprintf("failed to start test environment: %v", err))
	}

	// 2. Client & Scheme Registration
	utilruntime.Must(clientgoscheme.AddToScheme(testScheme))
	utilruntime.Must(v1alpha2.Install(testScheme))
	utilruntime.Must(v1.Install(testScheme))
	k8sClient, err = client.New(cfg, client.Options{Scheme: testScheme})
	if err != nil {
		panic(err)
	}

	// 3. Global Metric Registration
	// Necessary because we cannot parallelize tests using the global registry.
	metrics.Register()

	// 4. Pre-parse Base Resources
	// We load the YAML once here to avoid unnecessary I/O in every test case.
	baseResources = loadBaseResources()

	code := m.Run()

	_ = testEnv.Stop()
	return code
}

// moduleDir returns the on-disk directory of a module in the build list.
func moduleDir(path string) string {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", path).Output()
	if err != nil {
		// go list reports the actual reason on stderr, which ExitError carries.
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			err = fmt.Errorf("%w: %s", err, exitErr.Stderr)
		}
		panic(fmt.Sprintf("failed to locate module %s: %v", path, err))
	}
	dir := strings.TrimSpace(string(out))
	if dir == "" {
		// An empty dir turns the CRD paths relative, which envtest reports as a
		// missing path that does not look wrong.
		panic("go list returned no directory for module " + path)
	}
	return dir
}

// loadBaseResources parses the YAML manifest once at startup.
func loadBaseResources() []*unstructured.Unstructured {
	path := filepath.Join(repoRootPath, "test", "testdata", "inferencepool-with-model-hermetic.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		panic(fmt.Sprintf("failed to read manifest %s: %v", path, err))
	}

	var objs []*unstructured.Unstructured
	decoder := yaml.NewYAMLOrJSONDecoder(strings.NewReader(string(data)), 4096)
	for {
		u := &unstructured.Unstructured{}
		if err := decoder.Decode(u); err != nil {
			if err.Error() == "EOF" {
				break
			}
			panic(fmt.Sprintf("failed to decode YAML: %v", err))
		}
		objs = append(objs, u)
	}
	return objs
}

// K8sClient returns the client for the shared envtest API server. It is named for the
// global it exposes: TestHarness.Client is the ext_proc stream client, not this one.
func K8sClient() client.Client {
	return k8sClient
}

// Config returns the rest config of the shared envtest API server.
func Config() *rest.Config {
	return testEnv.Config
}

// Logger returns the package-wide test logger.
func Logger() logr.Logger {
	return logger
}

// Scheme returns the scheme the shared client is built with.
func Scheme() *runtime.Scheme {
	return testScheme
}

const (
	// TestPoolName is the InferencePool name every harness-created pod is labelled with.
	TestPoolName = "vllm-qwen3-32b-pool"

	// mockDataSourceType is the plugin type name used for the mock data source in integration tests.
	mockDataSourceType = "mock-metrics-source"
)

//go:embed testdata/datalayer-config.yaml
var testDLConfig string

// RunMode selects which EPP deployment shape the harness boots.
type RunMode string

// StandaloneStrategy selects whether a standalone EPP watches CRDs.
type StandaloneStrategy string

const (
	// ModeStandard runs EPP against the CRD-backed control plane.
	ModeStandard RunMode = "standard"
	// ModeStandalone runs EPP without a control plane.
	ModeStandalone RunMode = "standalone"
	// StrategyNoCRD is pure standalone.
	StrategyNoCRD StandaloneStrategy = "no_crd"
	// StrategyWithCRD is standalone but watching CRDs.
	StrategyWithCRD StandaloneStrategy = "with_crd"
)

// HarnessConfig holds configuration options for the TestHarness.
type HarnessConfig struct {
	// mode is the master switch. It tells you explicitly what the config is for.
	mode RunMode

	// strategy settings are used when mode == ModeStandalone.
	strategy StandaloneStrategy

	// configText overrides the default testConfig if provided. A nil value means use default.
	configText *string

	// Tracing indicates if tracing should be enabled for this test.
	Tracing bool

	// emitEndpointScores enables emitting per-endpoint scores in the request-path dynamic metadata.
	emitEndpointScores bool
}

// HarnessOption is a functional option for configuring the TestHarness.
type HarnessOption func(*HarnessConfig)

// WithStandaloneMode configures the harness to run in standalone mode
func WithStandaloneMode(strategy StandaloneStrategy) HarnessOption {
	return func(c *HarnessConfig) {
		c.mode = ModeStandalone
		c.strategy = strategy
	}
}

// WithStandardMode configures the harness to run in standard mode
func WithStandardMode() HarnessOption {
	return func(c *HarnessConfig) {
		c.mode = ModeStandard
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

// WithEmitEndpointScores starts the EPP with --emit-endpoint-scores enabled.
func WithEmitEndpointScores() HarnessOption {
	return func(c *HarnessConfig) {
		c.emitEndpointScores = true
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
	mode     RunMode
	strategy StandaloneStrategy
	Tracing  bool

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
	return h.mode != ModeStandalone || h.strategy != StrategyNoCRD
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

	// Tracing Setup (InMemory).
	var exporter *tracetest.InMemoryExporter
	var tp *sdktrace.TracerProvider
	if config.Tracing {
		exporter = tracetest.NewInMemoryExporter()
		tp = sdktrace.NewTracerProvider(
			sdktrace.WithSyncer(exporter),
		)
		otel.SetTracerProvider(tp)
	}

	// Reserve the ext_proc port once, up front: the server serves on this listener, so
	// the port stays bound for the lifetime of the test and no other process can take
	// it (issue #1066).
	lis, err := fwknet.ReserveListener()
	require.NoError(t, err, "failed to reserve ext_proc port")
	t.Cleanup(func() { _ = lis.Close() })
	grpcPort := lis.Addr().(*net.TCPAddr).Port

	eppOptions := defaultEppServerOptions(t, testNamespaceName, configText)
	eppOptions.EmitEndpointScores = config.emitEndpointScores
	if config.mode == ModeStandalone && config.strategy == StrategyNoCRD {
		// Only standalone EPP without crd need to set the EndpointSelector.
		eppOptions.EndpointSelector = labels.SelectorFromSet(labels.Set{"app": TestPoolName})
	}

	// Shorten the Prometheus refresh interval so WaitForReadyPodsMetric (10s timeout)
	// has many opportunities to observe the metric update instead of only ~2.
	eppOptions.RefreshPrometheusMetricsInterval = 500 * time.Millisecond

	mockDataSource := dlmocks.NewDataSource(plugin.TypedName{
		Type: mockDataSourceType,
		Name: mockDataSourceType,
	})
	runner, mgr, dataStore, err := eppRunner.NewTestRunnerSetup(ctx, testEnv.Config, eppOptions, mockDataSource, lis)
	require.NoError(t, err, "failed to create manager")
	backend := metricsBackend(&mockDataSourceBackend{mockDataSource: mockDataSource})

	mgrCtx, mgrCancel := context.WithCancel(ctx)
	mgrDone := make(chan struct{})
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

	// Cleanups run LIFO, so this teardown runs after the manager-stop cleanup below.
	t.Cleanup(func() {
		if config.Tracing {
			_ = tp.Shutdown(ctx)
			// Reset to no-op to avoid pollution between tests.
			otel.SetTracerProvider(noop.NewTracerProvider())
		}
		// Deleting the Namespace cascades to all contained resources.
		_ = k8sClient.Delete(context.Background(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: eppOptions.PoolNamespace}})
		// Crucial: Reset global metrics registry to prevent pollution between serial tests.
		metrics.Reset()
	})
	// Registered before the readiness wait below can fail the test, so the manager
	// goroutine cannot outlive it. Waiting on mgrDone keeps the manager fully stopped
	// before the global metrics registry is reset and the namespace is deleted.
	t.Cleanup(func() {
		mgrCancel()
		<-mgrDone
	})

	extProcClient, conn := fwkepp.ExtProcServerClient(
		mgrCtx,
		t,
		grpcPort,
		logger,
		mgrErr,
	)

	h := &TestHarness{
		t:              t,
		ctx:            mgrCtx,
		Namespace:      eppOptions.PoolNamespace,
		mode:           config.mode,
		strategy:       config.strategy,
		Tracing:        config.Tracing,
		Client:         extProcClient,
		Datastore:      dataStore,
		Exporter:       exporter,
		tp:             tp,
		grpcConn:       conn,
		metricsBackend: backend,
		Runner:         runner,
	}

	return h
}

func defaultEppServerOptions(t *testing.T, namespace, configText string) *eppServer.Options {
	t.Helper()

	eppOptions := eppServer.NewOptions()
	eppOptions.PoolName = TestPoolName
	eppOptions.PoolNamespace = namespace
	eppOptions.ConfigText = configText

	// No test dials the health server, so let the kernel assign the port: a
	// port 0 bind cannot lose a race to another listener.
	eppOptions.GRPCHealthPort = 0
	eppOptions.EndpointTargetPorts = []int{8000}
	eppOptions.SecureServing = false
	eppOptions.AllowExperimentalPlugins = true
	return eppOptions
}

// GRPCConn returns the connection to this harness's ext_proc server.
func (h *TestHarness) GRPCConn() *grpc.ClientConn {
	return h.grpcConn
}

// SetPodMetrics injects pod metrics into the harness's metrics backend.
func (h *TestHarness) SetPodMetrics(m map[types.NamespacedName]*fwkdl.Metrics) {
	h.metricsBackend.SetPodMetrics(m)
}

// GetSpans returns the currently recorded spans from the in-memory exporter.
func (h *TestHarness) GetSpans() tracetest.SpanStubs {
	return h.Exporter.GetSpans()
}

// --- Fluent Builder API ---

// WithBaseResources injects the standard pool and objective definitions into the test namespace.
// The resources are parsed once at startup to avoid I/O overhead in the loop.
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

		pod := fwkk8s.MakePod(name).
			Namespace(h.Namespace).
			ReadyCondition(). // Sets Status.Conditions.
			Labels(map[string]string{"app": TestPoolName}).
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

	expected := CleanMetric(MetricReadyPods(expectedCount))
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
		"Datastore sync timed out (mode=%v strategy=%v poolSynced=%v podsFound=%d expected=%d)",
		h.mode,
		h.strategy,
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

// --- Data Structures & Metrics Helpers ---

type PodState struct {
	index        int
	queueSize    int
	kvCacheUsage float64
	activeModels []string
}

// P constructs a Pod State: Index, Queue, KV%, Models...
// Usage: P(0, 5, 0.2, "model-a")
func P(idx int, q int, kv float64, models ...string) PodState {
	return PodState{index: idx, queueSize: q, kvCacheUsage: kv, activeModels: models}
}

type label struct{ name, value string }

func labelsToString(labels []label) string {
	parts := make([]string, len(labels))
	for i, l := range labels {
		parts[i] = fmt.Sprintf("%s=%q", l.name, l.value)
	}
	return strings.Join(parts, ",")
}

// MetricReqTotal renders the expected inference_objective_request_total exposition text.
func MetricReqTotal(model, target string, priority int) string {
	return fmt.Sprintf(`
    # HELP inference_objective_request_total [ALPHA] [Deprecated: Use llm_d_epp_request_total] Counter of inference objective requests broken out for each model and target model.
    # TYPE inference_objective_request_total counter
    inference_objective_request_total{%s} 1
    `, labelsToString([]label{{"model_name", model}, {"priority", strconv.Itoa(priority)}, {"target_model_name", target}}))
}

// MetricReadyPods renders the expected inference_pool_ready_pods exposition text.
func MetricReadyPods(count int) string {
	return fmt.Sprintf(`
    # HELP inference_pool_ready_pods [ALPHA] [Deprecated: Use llm_d_epp_ready_endpoints] The number of ready pods in the inference server pool.
    # TYPE inference_pool_ready_pods gauge
    inference_pool_ready_pods{%s} %d
    `, labelsToString([]label{{"name", TestPoolName}}), count)
}

// CleanMetric removes indentation from multiline metric strings and ensures a trailing newline exists, which is
// required by the Prometheus text parser.
func CleanMetric(s string) string {
	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if trimmed != "" {
			cleaned = append(cleaned, trimmed)
		}
	}
	return strings.Join(cleaned, "\n") + "\n"
}
