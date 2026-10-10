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
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
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
	var flowStopped atomic.Bool

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), startManager, extProc.run, health.run,
			func() { flowStopped.Store(true) }, draining, elected, drain)
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
	if flowStopped.Load() {
		t.Fatal("flow control stopped while ext_proc was still finishing streams")
	}
	close(extProc.release)
	if !closedWithin(health.stopped, time.Second) {
		t.Fatal("health was not stopped after ext_proc returned")
	}
	if err := waitErr(t, done); !errors.Is(err, errLost) {
		t.Fatalf("serveWithDrain returned %v, want %v", err, errLost)
	}
	require.True(t, flowStopped.Load(), "flow control should stop after the servers return")
}

// A manager that fails before this instance was ever elected has no traffic to
// drain: both servers stop at once and the error is returned.
func TestServeWithDrainManagerFailsBeforeElection(t *testing.T) {
	errStart := errors.New("cache sync failed")
	extProc, health := newFakeServer(), newFakeServer()
	close(extProc.release)
	close(health.release)
	draining, elected := &atomic.Bool{}, &atomic.Bool{}
	var flowStopped atomic.Bool

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), func(context.Context) error { return errStart },
			extProc.run, health.run, func() { flowStopped.Store(true) }, draining, elected, time.Minute)
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
	require.True(t, flowStopped.Load(), "flow control should stop on a manager startup failure")
}

// With leader election disabled there is no lease to lose, so a manager failure
// stops both servers at once and the error is returned.
func TestServeWithDrainManagerFailsWithoutLeaderElection(t *testing.T) {
	errMgr := errors.New("controller failed")
	extProc, health := newFakeServer(), newFakeServer()
	close(extProc.release)
	close(health.release)
	draining := &atomic.Bool{}
	var flowStopped atomic.Bool

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), func(context.Context) error { return errMgr },
			extProc.run, health.run, func() { flowStopped.Store(true) }, draining, nil, time.Minute)
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
	require.True(t, flowStopped.Load(), "flow control should stop on a manager failure")
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
		done <- serveWithDrain(ctx, startManager, extProc.run, health.run, nil, draining, elected, drain)
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
	}{
		{name: "SIGTERM"},
		{name: "lease loss", mgrErr: errors.New("leader election lost")},
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
			t.Cleanup(r.stopFlowControl)
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)
			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
			require.NoError(t, err)
			candidates := &drainEndpointCandidates{}
			_, admission, _, _ := r.initAdmissionControl(ctx, opts, eppConfig, candidates)

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
				done <- serveWithDrain(ctx, startManager, extProc.run, health.run, r.stopFlowControl,
					&atomic.Bool{}, elected, 300*time.Millisecond)
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
		name          string
		drain         time.Duration
		fileDiscovery bool
	}{
		{name: "Kubernetes drain", drain: time.Second},
		{name: "Kubernetes zero drain"},
		{name: "file discovery", fileDiscovery: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var saturated atomic.Bool
			var scrapes atomic.Int64
			saturated.Store(true)
			modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.URL.Path != "/metrics" {
					http.NotFound(w, req)
					return
				}
				kvUsage := 0.0
				if saturated.Load() {
					kvUsage = 1.0
				}
				_, err := fmt.Fprintf(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 0\n# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0\n# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc %g\n", kvUsage)
				if err != nil {
					t.Errorf("writing model server metrics: %v", err)
				}
				scrapes.Add(1)
			}))
			t.Cleanup(modelServer.Close)
			modelAddress := modelServer.Listener.Addr().(*net.TCPAddr)
			address, port := modelAddress.IP.String(), modelAddress.Port

			ctx, sigterm := context.WithCancel(t.Context())
			defer sigterm()
			opts := runserver.NewOptions()
			opts.PoolName = testPoolName
			opts.PoolNamespace = "test-ns"
			opts.EndpointSelector = labels.Everything()
			opts.EndpointTargetPorts = []int{port}
			opts.SecureServing = false
			opts.GRPCHealthPort = 0
			opts.MetricsPort = 0
			opts.RefreshMetricsInterval = 20 * time.Millisecond
			discoveryPlugin, discoveryConfig := "", ""
			if tc.fileDiscovery {
				path := filepath.Join(t.TempDir(), "endpoints.yaml")
				require.NoError(t, os.WriteFile(path, []byte(fmt.Sprintf(`endpoints:
- name: model-server
  namespace: test-ns
  address: %s
  port: "%d"
`, address, port)), 0o644))
				discoveryPlugin = fmt.Sprintf(`- type: file-discovery
  parameters:
    path: %q
    watchFile: false
`, path)
				discoveryConfig = `dataLayer:
  discovery:
    endpoints:
      pluginRef: file-discovery
`
			}
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
%s`, discoveryPlugin, discoveryConfig)
			r := NewRunner()
			t.Cleanup(r.stopFlowControl)
			listener, err := fwknet.ReserveListener()
			require.NoError(t, err)
			t.Cleanup(func() { _ = listener.Close() })
			managerStopped := make(chan struct{})
			done := make(chan error, 1)
			finished := make(chan struct{})
			var endpoint fwkdl.Endpoint
			if tc.fileDiscovery {
				rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
				require.NoError(t, err)
				r.grpcListener = listener
				go func() {
					defer close(finished)
					done <- r.runWithFileDiscovery(ctx, opts, rawConfig)
				}()
			} else {
				mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{corev1.SchemeGroupVersion})
				mapper.Add(corev1.SchemeGroupVersion.WithKind("Pod"), meta.RESTScopeNamespace)
				_, ds, err := r.setup(ctx, &rest.Config{Host: "http://127.0.0.1"}, opts, []func(*ctrl.Options){
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
				t.Cleanup(ds.Clear)
				ds.EndpointUpsert(ctx, &fwkdl.EndpointMetadata{
					ID:          types.NamespacedName{Namespace: opts.PoolNamespace, Name: "model-server"},
					Address:     address,
					Port:        strconv.Itoa(port),
					MetricsHost: fmt.Sprintf("%s:%d", address, port),
				})
				endpoints := ds.PodList(func(fwkdl.Endpoint) bool { return true })
				require.Len(t, endpoints, 1)
				endpoint = endpoints[0]
				require.Eventually(t, func() bool {
					return endpoint.GetMetrics().KVCacheUsagePercent == 1
				}, time.Second, opts.RefreshMetricsInterval)
				r.serverRunner.GrpcListener = listener
				stopManager := make(chan error, 1)
				go func() {
					<-ctx.Done()
					stopManager <- nil
				}()
				go func() {
					defer close(finished)
					done <- serveWithDrain(ctx, func(context.Context) error {
						err := <-stopManager
						close(managerStopped)
						return err
					}, r.serverRunner.AsRunnable(ctrl.Log).Start, func(c context.Context) error {
						<-c.Done()
						return nil
					}, r.stopFlowControl, r.draining, nil, tc.drain)
				}()
			}
			require.Eventually(t, func() bool {
				return scrapes.Load() >= 2
			}, time.Second, opts.RefreshMetricsInterval)

			conn, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
			require.NoError(t, err)
			t.Cleanup(func() {
				sigterm()
				_ = conn.Close()
				require.True(t, closedWithin(finished, 5*time.Second), "drain did not finish after streams closed")
			})
			streamCtx, cancelStreams := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancelStreams()
			type routeResult struct {
				response *extProcPb.ProcessingResponse
				err      error
			}
			send := func(id string) (extProcPb.ExternalProcessor_ProcessClient, <-chan routeResult) {
				stream, err := extProcPb.NewExternalProcessorClient(conn).Process(streamCtx)
				require.NoError(t, err)
				for _, request := range integration.ReqRaw(map[string]string{
					":path":                      "/v1/chat/completions",
					"content-type":               "application/json",
					reqcommon.RequestIDHeaderKey: id,
				}, `{"model":"test-model","messages":[{"role":"user","content":"Hello"}]}`) {
					require.NoError(t, stream.Send(request))
				}
				result := make(chan routeResult, 1)
				go func() {
					response, err := stream.Recv()
					result <- routeResult{response: response, err: err}
				}()
				return stream, result
			}
			queuedStream, queued := send("queued-before-sigterm")
			select {
			case result := <-queued:
				t.Fatalf("request did not queue against saturated endpoint: %+v", result)
			case <-time.After(100 * time.Millisecond):
			}
			shutdownAt := time.Now()
			sigterm()
			if !tc.fileDiscovery {
				require.True(t, closedWithin(managerStopped, time.Second))
			}
			time.Sleep(utilization.DefaultMetricsStalenessThreshold + 100*time.Millisecond)

			streams := []extProcPb.ExternalProcessor_ProcessClient{queuedStream}
			results := []<-chan routeResult{queued}
			if tc.drain > 0 {
				arrivingStream, arriving := send("arrived-during-drain")
				streams = append(streams, arrivingStream)
				results = append(results, arriving)
			}
			saturated.Store(false)
			for _, result := range results {
				select {
				case route := <-result:
					require.NoError(t, route.err)
					require.NotNil(t, route.response.GetRequestHeaders(), "expected successful routing, got %v", route.response)
					foundDestination := false
					for _, header := range route.response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
						if header.Header.Key == metadata.DestinationEndpointKey {
							require.Equal(t, fmt.Sprintf("%s:%d", address, port), string(header.Header.RawValue))
							foundDestination = true
						}
					}
					require.True(t, foundDestination)
					if tc.drain > 0 {
						require.Less(t, time.Since(shutdownAt), tc.drain, "request did not dispatch within the drain window")
					}
				case <-time.After(time.Second):
					t.Fatal("queued request did not dispatch after backend capacity recovered during shutdown")
				}
			}
			if endpoint != nil {
				require.Less(t, time.Since(endpoint.GetMetrics().UpdateTime), utilization.DefaultMetricsStalenessThreshold)
			}
			for _, stream := range streams {
				response, err := stream.Recv()
				require.NoError(t, err)
				require.NotNil(t, response.GetRequestBody())
			}
			if tc.drain > 0 {
				time.Sleep(tc.drain)
			}
			require.False(t, isClosed(finished), "ext_proc exited while streams were still open")
			snapshot := scrapes.Load()
			require.Eventually(t, func() bool {
				return scrapes.Load() > snapshot
			}, time.Second, opts.RefreshMetricsInterval, "metrics polling stopped before streams finished")
			for _, stream := range streams {
				require.NoError(t, stream.CloseSend())
				_, err := stream.Recv()
				require.ErrorIs(t, err, io.EOF)
			}
			err = waitErr(t, done)
			if !tc.fileDiscovery || !errors.Is(err, context.Canceled) {
				require.NoError(t, err)
			}
			require.ErrorIs(t, r.flowControlCtx.Err(), context.Canceled)
			time.Sleep(3 * opts.RefreshMetricsInterval)
			snapshot = scrapes.Load()
			time.Sleep(3 * opts.RefreshMetricsInterval)
			require.Equal(t, snapshot, scrapes.Load(), "metrics polling outlived server shutdown")
		})
	}
}
