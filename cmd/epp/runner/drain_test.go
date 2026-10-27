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
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	corePb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	localsyncer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/cross_plugin/local"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
	"github.com/llm-d/llm-d-router/test/integration"
)

// fakeServer stands in for a gRPC server runnable. Like GracefulStop, it returns
// only after its in-flight work finishes: stopped closes when its context ends,
// and the runnable returns once release is closed.
type fakeServer struct {
	stopped chan struct{}
	release chan struct{}
}

func newFakeServer() *fakeServer {
	return &fakeServer{stopped: make(chan struct{}), release: make(chan struct{})}
}

func (s *fakeServer) run(ctx context.Context) error {
	<-ctx.Done()
	close(s.stopped)
	<-s.release
	return nil
}

func closedWithin(ch <-chan struct{}, d time.Duration) bool {
	select {
	case <-ch:
		return true
	case <-time.After(d):
		return false
	}
}

func isClosed(ch <-chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

func waitErr(t *testing.T, done <-chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("serveWithDrain did not return")
		return nil
	}
}

// A leader whose lease is lost keeps serving requests already routed to it, or
// arriving before the proxy sees the new leader, for the drain window. Its
// health server keeps answering liveness until ext_proc has finished its streams.
func TestServeWithDrainAfterLeaseLoss(t *testing.T) {
	const drain = 300 * time.Millisecond
	errLost := errors.New("leader election lost")
	lose := make(chan struct{})
	startManager := func(context.Context) error {
		<-lose
		return errLost
	}
	extProc, health := newFakeServer(), newFakeServer()
	close(health.release)
	draining, elected := &atomic.Bool{}, &atomic.Bool{}
	elected.Store(true)
	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), startManager, extProc.run, health.run,
			draining, elected, drain)
	}()
	close(lose)

	deadline := time.Now().Add(100 * time.Millisecond)
	for !draining.Load() && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if !draining.Load() {
		t.Fatal("readiness did not turn NotServing after the lease was lost")
	}
	if closedWithin(extProc.stopped, drain/2) {
		t.Fatal("ext_proc stopped before the drain window elapsed")
	}
	if !closedWithin(extProc.stopped, 2*drain) {
		t.Fatal("ext_proc was not stopped after the drain window")
	}
	if closedWithin(health.stopped, 100*time.Millisecond) {
		t.Fatal("health stopped while ext_proc was still finishing streams")
	}
	close(extProc.release)
	if !closedWithin(health.stopped, time.Second) {
		t.Fatal("health was not stopped after ext_proc returned")
	}
	if err := waitErr(t, done); !errors.Is(err, errLost) {
		t.Fatalf("serveWithDrain returned %v, want %v", err, errLost)
	}
}

// A manager that fails before this instance was ever elected has no traffic to
// drain: both servers stop at once and the error is returned.
func TestServeWithDrainManagerFailsBeforeElection(t *testing.T) {
	errStart := errors.New("cache sync failed")
	extProc, health := newFakeServer(), newFakeServer()
	close(extProc.release)
	close(health.release)
	draining, elected := &atomic.Bool{}, &atomic.Bool{}
	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), func(context.Context) error { return errStart },
			extProc.run, health.run, draining, elected, time.Minute)
	}()
	if err := waitErr(t, done); !errors.Is(err, errStart) {
		t.Fatalf("serveWithDrain returned %v, want %v", err, errStart)
	}
	if !isClosed(extProc.stopped) || !isClosed(health.stopped) {
		t.Fatal("servers still running after serveWithDrain returned")
	}
	if draining.Load() {
		t.Fatal("draining set by a manager failure before election")
	}
}

// With leader election disabled there is no lease to lose, so a manager failure
// stops both servers at once and the error is returned.
func TestServeWithDrainManagerFailsWithoutLeaderElection(t *testing.T) {
	errMgr := errors.New("controller failed")
	extProc, health := newFakeServer(), newFakeServer()
	close(extProc.release)
	close(health.release)
	draining := &atomic.Bool{}
	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), func(context.Context) error { return errMgr },
			extProc.run, health.run, draining, nil, time.Minute)
	}()
	if err := waitErr(t, done); !errors.Is(err, errMgr) {
		t.Fatalf("serveWithDrain returned %v, want %v", err, errMgr)
	}
	if !isClosed(extProc.stopped) || !isClosed(health.stopped) {
		t.Fatal("servers still running after serveWithDrain returned")
	}
	if draining.Load() {
		t.Fatal("draining set by a manager failure without leader election")
	}
}

// On SIGTERM the manager returns nil; the servers drain the same way.
func TestServeWithDrainOnSIGTERM(t *testing.T) {
	const drain = 300 * time.Millisecond
	ctx, sigterm := context.WithCancel(context.Background())
	startManager := func(c context.Context) error {
		<-c.Done()
		return nil
	}
	extProc, health := newFakeServer(), newFakeServer()
	close(health.release)
	draining, elected := &atomic.Bool{}, &atomic.Bool{}
	elected.Store(true)

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(ctx, startManager, extProc.run, health.run, draining, elected, drain)
	}()
	sigterm()

	if closedWithin(extProc.stopped, drain/2) {
		t.Fatal("ext_proc stopped before the drain window elapsed")
	}
	if !draining.Load() {
		t.Fatal("readiness did not turn NotServing on SIGTERM")
	}
	if !closedWithin(extProc.stopped, 2*drain) {
		t.Fatal("ext_proc was not stopped after the drain window")
	}
	if closedWithin(health.stopped, 100*time.Millisecond) {
		t.Fatal("health stopped while ext_proc was still finishing streams")
	}
	close(extProc.release)
	if err := waitErr(t, done); err != nil {
		t.Fatalf("serveWithDrain returned %v, want nil", err)
	}
}

type drainEndpointCandidates struct {
	available atomic.Bool
}

func (c *drainEndpointCandidates) Locate(context.Context, map[string]any) []fwkdl.Endpoint {
	if !c.available.Load() {
		return nil
	}
	return []fwkdl.Endpoint{fwkdl.NewEndpoint(nil, &fwkdl.Metrics{UpdateTime: time.Now()})}
}

func TestServeWithDrainFlowControl(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mgrErr error
		drain  time.Duration
	}{
		{name: "SIGTERM", drain: 300 * time.Millisecond},
		{name: "lease loss", mgrErr: errors.New("leader election lost"), drain: 300 * time.Millisecond},
		{name: "zero drain"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, sigterm := context.WithCancel(context.Background())
			defer sigterm()
			opts := runserver.NewOptions()
			opts.PoolName = testPoolName
			opts.ConfigText = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
`
			r := NewRunner()
			r.servingCtx, r.cancelServing = context.WithCancel(context.WithoutCancel(ctx))
			t.Cleanup(r.stopServing)
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)
			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
			require.NoError(t, err)
			candidates := &drainEndpointCandidates{}
			_, admission, _, _ := r.initAdmissionControl(r.servingCtx, opts, eppConfig, candidates)

			admit := func(id string) error {
				reqCtx := &handlers.RequestContext{
					Request: &handlers.Request{},
					SchedulingRequest: &fwksched.InferenceRequest{
						RequestID:  id,
						FairnessID: "team-a",
					},
					RequestReceivedTimestamp: time.Now(),
				}
				return admission.Admit(t.Context(), reqCtx, 0)
			}
			queued := make(chan error, 1)
			go func() { queued <- admit("queued-before-shutdown") }()
			select {
			case err := <-queued:
				t.Fatalf("request did not wait for an endpoint: %v", err)
			case <-time.After(100 * time.Millisecond):
			}

			extProc, health := newFakeServer(), newFakeServer()
			close(health.release)
			t.Cleanup(func() {
				if !isClosed(extProc.release) {
					close(extProc.release)
				}
			})
			done := make(chan error, 1)
			managerStopped := make(chan struct{})
			stopManager := make(chan error, 1)
			startManager := func(context.Context) error {
				err := <-stopManager
				close(managerStopped)
				return err
			}
			elected := &atomic.Bool{}
			elected.Store(true)
			go func() {
				err := serveWithDrain(ctx, startManager, extProc.run, health.run,
					&atomic.Bool{}, elected, tc.drain)
				r.stopServing()
				done <- err
			}()
			if tc.mgrErr == nil {
				sigterm()
			}
			stopManager <- tc.mgrErr
			require.True(t, closedWithin(managerStopped, time.Second))

			select {
			case err := <-queued:
				t.Fatalf("queued request was terminated by shutdown: %v", err)
			case <-time.After(50 * time.Millisecond):
			}
			duringDrain := make(chan error, 1)
			go func() { duringDrain <- admit("arrived-during-drain") }()
			select {
			case err := <-duringDrain:
				t.Fatalf("request arriving during drain did not wait for an endpoint: %v", err)
			case <-time.After(50 * time.Millisecond):
			}
			candidates.available.Store(true)
			for _, result := range []<-chan error{queued, duringDrain} {
				select {
				case err := <-result:
					require.NoError(t, err, "requests should dispatch when an endpoint becomes available")
				case <-time.After(time.Second):
					t.Fatal("flow control did not dispatch a request during drain")
				}
			}

			require.True(t, closedWithin(extProc.stopped, time.Second))
			require.NoError(t, admit("inflight-after-drain-window"),
				"flow control should run until ext_proc finishes its streams")
			close(extProc.release)
			require.ErrorIs(t, waitErr(t, done), tc.mgrErr)

			err = admit("after-shutdown")
			var publicErr errcommon.Error
			require.ErrorAs(t, err, &publicErr)
			require.Equal(t, errcommon.ServiceUnavailable, publicErr.Code)
		})
	}
}

func TestRunnerDrainKeepsEndpointMetrics(t *testing.T) {
	for _, tc := range []struct {
		name        string
		flowControl bool
	}{
		{name: "Kubernetes drain", flowControl: true},
		{name: "Kubernetes legacy drain"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testRunnerDrain(t, tc.flowControl)
		})
	}
}

func testRunnerDrain(t *testing.T, flowControl bool) {
	t.Helper()
	var saturated atomic.Bool
	saturated.Store(flowControl)
	modelServer, scrapes := newDrainTestBackend(t, &saturated)
	opts := newDrainTestOptions(modelServer)
	opts.ConfigText = fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl=%t
plugins:
- type: random-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
`, flowControl)
	ctx, sigterm := context.WithCancel(t.Context())
	defer sigterm()
	r, endpoint := setupDrainTestRunner(ctx, t, opts)
	wantKVUsage := 0.0
	if flowControl {
		wantKVUsage = 1
	}
	require.Eventually(t, func() bool {
		metrics := endpoint.GetMetrics()
		return !metrics.UpdateTime.IsZero() && metrics.KVCacheUsagePercent == wantKVUsage
	}, time.Second, opts.RefreshMetricsInterval)

	listener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	stopping, done := startDrainTestServer(ctx, t, r, listener, 3*time.Second)
	queuedStream, queued := sendDrainTestRequest(t, listener, "queued-before-sigterm")
	if flowControl {
		select {
		case result := <-queued:
			t.Fatalf("request did not queue against saturated endpoint: %+v", result)
		case <-time.After(100 * time.Millisecond):
		}
	}
	sigterm()
	require.Eventually(t, r.draining.Load, time.Second, opts.RefreshMetricsInterval)
	time.Sleep(utilization.DefaultMetricsStalenessThreshold + 100*time.Millisecond)
	require.NoError(t, r.PluginHandle.Context().Err(), "plugin dependencies stopped during drain")
	arrivingStream, arriving := sendDrainTestRequest(t, listener, "arrived-during-drain")
	saturated.Store(false)
	requireDrainTestRoute(t, queuedStream, queued, modelServer)
	requireDrainTestRoute(t, arrivingStream, arriving, modelServer)
	require.Eventually(t, func() bool {
		return time.Since(endpoint.GetMetrics().UpdateTime) < utilization.DefaultMetricsStalenessThreshold
	}, time.Second, opts.RefreshMetricsInterval, "endpoint metrics became stale during drain")

	require.True(t, closedWithin(stopping, 5*time.Second), "ext_proc did not start GracefulStop")
	select {
	case err := <-done:
		t.Fatalf("ext_proc exited while streams were still open: %v", err)
	default:
	}
	snapshot := scrapes.Load()
	require.Eventually(t, func() bool {
		return scrapes.Load() > snapshot
	}, time.Second, opts.RefreshMetricsInterval, "metrics polling stopped before streams finished")
	for _, stream := range []extProcPb.ExternalProcessor_ProcessClient{queuedStream, arrivingStream} {
		require.NoError(t, stream.CloseSend())
		_, err := stream.Recv()
		require.ErrorIs(t, err, io.EOF)
	}
	require.NoError(t, waitErr(t, done))
	requireDrainTestMetricsStopped(t, r, scrapes, opts.RefreshMetricsInterval)
}

func TestRunnerShutdownKeepsEndpointMetrics(t *testing.T) {
	var saturated atomic.Bool
	saturated.Store(true)
	modelServer, scrapes := newDrainTestBackend(t, &saturated)
	opts := newDrainTestOptions(modelServer)
	discoveryPlugin, discoveryConfig := drainTestFileDiscoveryConfig(t, modelServer)
	opts.ConfigText = fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
%s- type: random-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
dataLayer:
%s`, discoveryPlugin, discoveryConfig)
	ctx, sigterm := context.WithCancel(t.Context())
	defer sigterm()
	listener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	r, done := startFileDiscoveryDrainTest(ctx, t, opts, listener)
	require.Eventually(t, func() bool {
		return scrapes.Load() >= 2
	}, time.Second, opts.RefreshMetricsInterval)
	stream, routed := sendDrainTestRequest(t, listener, "queued-before-sigterm")
	select {
	case result := <-routed:
		t.Fatalf("request did not queue against saturated endpoint: %+v", result)
	case <-time.After(100 * time.Millisecond):
	}
	sigterm()
	snapshot := scrapes.Load()
	time.Sleep(utilization.DefaultMetricsStalenessThreshold + 100*time.Millisecond)
	require.NoError(t, r.PluginHandle.Context().Err(), "plugin dependencies stopped before streams finished")
	require.Eventually(t, func() bool {
		return scrapes.Load() > snapshot
	}, time.Second, opts.RefreshMetricsInterval, "metrics polling stopped before streams finished")
	saturated.Store(false)
	requireDrainTestRoute(t, stream, routed, modelServer)
	require.NoError(t, stream.CloseSend())
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)
	if err := waitErr(t, done); err != nil {
		require.ErrorIs(t, err, context.Canceled)
	}
	requireDrainTestMetricsStopped(t, r, scrapes, opts.RefreshMetricsInterval)
}

func TestRunnerDrainKeepsCrossReplicaState(t *testing.T) {
	fwkplugin.Register(localsyncer.LocalSyncerType, fwkplugin.StabilityBeta, localsyncer.LocalSyncerFactory)
	for _, fileDiscovery := range []bool{false, true} {
		t.Run(fmt.Sprintf("fileDiscovery=%t", fileDiscovery), func(t *testing.T) {
			var saturated atomic.Bool
			modelServer, scrapes := newDrainTestBackend(t, &saturated)
			opts := newDrainTestOptions(modelServer)
			discoveryPlugin, discoveryConfig := "", ""
			if fileDiscovery {
				discoveryPlugin, discoveryConfig = drainTestFileDiscoveryConfig(t, modelServer)
			}
			opts.ConfigText = fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
%s- type: %s
- type: %s
- type: concurrency-detector
  parameters:
    maxConcurrency: 1
- type: random-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
dataLayer:
%s  crossReplica:
    syncerPluginRef: %s
    syncInterval: 20ms
flowControl:
  saturationDetector:
    pluginRef: concurrency-detector
`, discoveryPlugin, localsyncer.LocalSyncerType, inflightload.InFlightLoadProducerType, discoveryConfig, localsyncer.LocalSyncerType)
			ctx, sigterm := context.WithCancel(t.Context())
			defer sigterm()
			listener, err := fwknet.ReserveListener()
			require.NoError(t, err)
			t.Cleanup(func() { _ = listener.Close() })
			var r *Runner
			var done <-chan error
			if fileDiscovery {
				r, done = startFileDiscoveryDrainTest(ctx, t, opts, listener)
			} else {
				r, _ = setupDrainTestRunner(ctx, t, opts)
				_, done = startDrainTestServer(ctx, t, r, listener, 0)
			}
			require.Eventually(t, func() bool {
				return scrapes.Load() >= 2
			}, time.Second, opts.RefreshMetricsInterval)
			inFlightStream, routed := sendDrainTestRequest(t, listener, "in-flight-before-sigterm")
			requireDrainTestRoute(t, inFlightStream, routed, modelServer)
			syncer, ok := r.PluginHandle.CrossReplicaSyncer().(fwkdl.CrossReplicaSyncer)
			require.True(t, ok)
			contributor, ok := r.PluginHandle.Plugin(inflightload.InFlightLoadProducerType).(fwkdl.CrossReplicaContributor)
			require.True(t, ok)
			spec := contributor.CrossReplicaState()
			require.Eventually(t, func() bool {
				value, found, err := syncer.Get(t.Context(), spec, opts.PoolNamespace+"/model-server")
				load, ok := value.(*attrconcurrency.InFlightLoad)
				return err == nil && found && ok && load.Requests == 1
			}, 3*time.Second, opts.RefreshMetricsInterval, "in-flight load was not published before shutdown")
			queuedStream, queued := sendDrainTestRequest(t, listener, "queued-before-sigterm")
			select {
			case result := <-queued:
				t.Fatalf("request did not queue against in-flight load: %+v", result)
			case <-time.After(100 * time.Millisecond):
			}
			sigterm()
			snapshot := scrapes.Load()
			time.Sleep(utilization.DefaultMetricsStalenessThreshold + 100*time.Millisecond)
			require.NoError(t, r.PluginHandle.Context().Err(), "plugin dependencies stopped before streams finished")
			require.Eventually(t, func() bool {
				return scrapes.Load() > snapshot
			}, time.Second, opts.RefreshMetricsInterval, "metrics polling stopped before streams finished")
			require.NoError(t, inFlightStream.Send(&extProcPb.ProcessingRequest{
				Request: &extProcPb.ProcessingRequest_ResponseHeaders{
					ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &corePb.HeaderMap{Headers: []*corePb.HeaderValue{
							{Key: ":status", RawValue: []byte("200")},
							{Key: "content-type", RawValue: []byte("application/json")},
						}},
					},
				},
			}))
			response, err := inFlightStream.Recv()
			require.NoError(t, err)
			require.NotNil(t, response.GetResponseHeaders())
			require.NoError(t, inFlightStream.Send(&extProcPb.ProcessingRequest{
				Request: &extProcPb.ProcessingRequest_ResponseBody{
					ResponseBody: &extProcPb.HttpBody{
						Body:        []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]}`),
						EndOfStream: true,
					},
				},
			}))
			response, err = inFlightStream.Recv()
			require.NoError(t, err)
			require.NotNil(t, response.GetResponseBody())
			require.NoError(t, inFlightStream.CloseSend())
			_, err = inFlightStream.Recv()
			require.ErrorIs(t, err, io.EOF)
			requireDrainTestRoute(t, queuedStream, queued, modelServer)
			require.NoError(t, queuedStream.CloseSend())
			_, err = queuedStream.Recv()
			require.ErrorIs(t, err, io.EOF)
			if err := waitErr(t, done); err != nil {
				require.True(t, fileDiscovery)
				require.ErrorIs(t, err, context.Canceled)
			}
			requireDrainTestMetricsStopped(t, r, scrapes, opts.RefreshMetricsInterval)
		})
	}
}

func newDrainTestBackend(t *testing.T, saturated *atomic.Bool) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	scrapes := &atomic.Int64{}
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/metrics" {
			http.NotFound(w, req)
			return
		}
		kvUsage := 0.0
		if saturated.Load() {
			kvUsage = 1
		}
		_, err := fmt.Fprintf(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 0\n# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0\n# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc %g\n", kvUsage)
		if err != nil {
			t.Errorf("writing model server metrics: %v", err)
		}
		scrapes.Add(1)
	}))
	t.Cleanup(modelServer.Close)
	return modelServer, scrapes
}

func newDrainTestOptions(modelServer *httptest.Server) *runserver.Options {
	opts := runserver.NewOptions()
	opts.PoolName = testPoolName
	opts.PoolNamespace = "test-ns"
	opts.EndpointSelector = labels.Everything()
	opts.EndpointTargetPorts = []int{modelServer.Listener.Addr().(*net.TCPAddr).Port}
	opts.SecureServing = false
	opts.GRPCHealthPort = 0
	opts.MetricsPort = 0
	opts.RefreshMetricsInterval = 20 * time.Millisecond
	return opts
}

func drainTestFileDiscoveryConfig(t *testing.T, modelServer *httptest.Server) (string, string) {
	t.Helper()
	address := modelServer.Listener.Addr().(*net.TCPAddr)
	path := filepath.Join(t.TempDir(), "endpoints.yaml")
	require.NoError(t, os.WriteFile(path, []byte(fmt.Sprintf(`endpoints:
- name: model-server
  namespace: test-ns
  address: %s
  port: "%d"
`, address.IP.String(), address.Port)), 0o644))
	return fmt.Sprintf(`- type: file-discovery
  parameters:
    path: %q
    watchFile: false
`, path), `  discovery:
    endpoints:
      pluginRef: file-discovery
`
}

func setupDrainTestRunner(ctx context.Context, t *testing.T, opts *runserver.Options) (*Runner, fwkdl.Endpoint) {
	t.Helper()
	r := NewRunner()
	servingCtx, cancelServing := context.WithCancel(context.WithoutCancel(ctx))
	r.cancelServing = cancelServing
	t.Cleanup(r.stopServing)
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{corev1.SchemeGroupVersion})
	mapper.Add(corev1.SchemeGroupVersion.WithKind("Pod"), meta.RESTScopeNamespace)
	_, ds, err := r.setup(ctx, servingCtx, &rest.Config{Host: "http://127.0.0.1"}, opts, []func(*ctrl.Options){
		func(o *ctrl.Options) {
			skipNameValidation := true
			o.Controller.SkipNameValidation = &skipNameValidation
			o.Metrics.BindAddress = "0"
			o.MapperProvider = func(*rest.Config, *http.Client) (meta.RESTMapper, error) {
				return mapper, nil
			}
		},
	})
	require.NoError(t, err)
	require.Nil(t, r.servingCtx, "setup must not require a serving context stored on Runner")
	require.Same(t, servingCtx, r.PluginHandle.Context())
	t.Cleanup(ds.Clear)
	port := strconv.Itoa(opts.EndpointTargetPorts[0])
	ds.EndpointUpsert(ctx, &fwkdl.EndpointMetadata{
		ID:          types.NamespacedName{Namespace: opts.PoolNamespace, Name: "model-server"},
		Address:     "127.0.0.1",
		Port:        port,
		MetricsHost: net.JoinHostPort("127.0.0.1", port),
	})
	endpoints := ds.PodList(func(fwkdl.Endpoint) bool { return true })
	require.Len(t, endpoints, 1)
	return r, endpoints[0]
}

func startDrainTestServer(ctx context.Context, t *testing.T, r *Runner, listener net.Listener, drain time.Duration) (<-chan struct{}, <-chan error) {
	t.Helper()
	r.serverRunner.GrpcListener = listener
	stopping := make(chan struct{})
	finished := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		defer close(finished)
		err := serveWithDrain(ctx, func(c context.Context) error {
			<-c.Done()
			return nil
		}, func(c context.Context) error {
			context.AfterFunc(c, func() { close(stopping) })
			return r.serverRunner.AsRunnable(ctrl.Log).Start(c)
		}, func(c context.Context) error {
			<-c.Done()
			return nil
		}, r.draining, nil, drain)
		r.stopServing()
		done <- err
	}()
	t.Cleanup(func() {
		require.True(t, closedWithin(finished, 10*time.Second), "drain did not finish after streams closed")
	})
	return stopping, done
}

func startFileDiscoveryDrainTest(ctx context.Context, t *testing.T, opts *runserver.Options, listener net.Listener) (*Runner, <-chan error) {
	t.Helper()
	r := NewRunner()
	t.Cleanup(r.stopServing)
	r.grpcListener = listener
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	done := make(chan error, 1)
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		done <- r.runWithFileDiscovery(ctx, opts, rawConfig)
	}()
	t.Cleanup(func() {
		require.True(t, closedWithin(finished, 10*time.Second), "file discovery did not finish after streams closed")
	})
	return r, done
}

type drainTestRouteResult struct {
	response *extProcPb.ProcessingResponse
	err      error
}

func sendDrainTestRequest(t *testing.T, listener net.Listener, id string) (extProcPb.ExternalProcessor_ProcessClient, <-chan drainTestRouteResult) {
	t.Helper()
	conn, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	t.Cleanup(cancel)
	stream, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	require.NoError(t, err)
	for _, request := range integration.ReqRaw(map[string]string{
		":path":                      "/v1/chat/completions",
		"content-type":               "application/json",
		reqcommon.RequestIDHeaderKey: id,
	}, `{"model":"test-model","messages":[{"role":"user","content":"Hello"}]}`) {
		require.NoError(t, stream.Send(request))
	}
	result := make(chan drainTestRouteResult, 1)
	go func() {
		response, err := stream.Recv()
		result <- drainTestRouteResult{response: response, err: err}
	}()
	return stream, result
}

func requireDrainTestRoute(t *testing.T, stream extProcPb.ExternalProcessor_ProcessClient, result <-chan drainTestRouteResult, modelServer *httptest.Server) {
	t.Helper()
	select {
	case route := <-result:
		require.NoError(t, route.err)
		require.NotNil(t, route.response.GetRequestHeaders(), "expected successful routing, got %v", route.response)
		foundDestination := false
		for _, header := range route.response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
			if header.Header.Key == metadata.DestinationEndpointKey {
				require.Equal(t, modelServer.Listener.Addr().String(), string(header.Header.RawValue))
				foundDestination = true
			}
		}
		require.True(t, foundDestination)
	case <-time.After(3 * time.Second):
		t.Fatal("queued request did not dispatch after backend capacity recovered during shutdown")
	}
	response, err := stream.Recv()
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())
}

func requireDrainTestMetricsStopped(t *testing.T, r *Runner, scrapes *atomic.Int64, interval time.Duration) {
	t.Helper()
	require.ErrorIs(t, r.PluginHandle.Context().Err(), context.Canceled)
	// An in-flight scrape can finish after the serving context is cancelled.
	snapshot, unchangedSince := scrapes.Load(), time.Now()
	require.Eventually(t, func() bool {
		current := scrapes.Load()
		if current != snapshot {
			snapshot, unchangedSince = current, time.Now()
		}
		return time.Since(unchangedSince) >= 3*interval
	}, time.Second, interval, "metrics polling outlived server shutdown")
}
