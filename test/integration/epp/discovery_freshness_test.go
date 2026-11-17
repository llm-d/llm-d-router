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

package epp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	epprunner "github.com/llm-d/llm-d-router/cmd/epp/runner"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
	fwkgaie "github.com/llm-d/llm-d-router/test/framework/gaie"
	fwkk8s "github.com/llm-d/llm-d-router/test/framework/k8s"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

type membershipFault struct {
	mu      sync.Mutex
	blocked bool
	watches []io.ReadCloser
}

type membershipTransport struct {
	http.RoundTripper
	fault *membershipFault
}

func (r membershipTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !strings.Contains(req.URL.Path, "/pods") && !strings.Contains(req.URL.Path, "/inferencepools") {
		return r.RoundTripper.RoundTrip(req)
	}
	r.fault.mu.Lock()
	blocked := r.fault.blocked
	r.fault.mu.Unlock()
	if blocked {
		return nil, errors.New("test discovery interruption")
	}
	resp, err := r.RoundTripper.RoundTrip(req)
	if err != nil || req.URL.Query().Get("watch") != "true" {
		return resp, err
	}
	r.fault.mu.Lock()
	if r.fault.blocked {
		_ = resp.Body.Close()
	} else {
		r.fault.watches = append(r.fault.watches, resp.Body)
	}
	r.fault.mu.Unlock()
	return resp, nil
}

func (f *membershipFault) interrupt() {
	f.mu.Lock()
	f.blocked = true
	watches := f.watches
	f.watches = nil
	f.mu.Unlock()
	for _, body := range watches {
		_ = body.Close()
	}
}

func TestDiscoveryFreshnessRejectsWhileLeaseRenewalContinues(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	api := eppharness.K8sClient()
	namespace := "discovery-freshness-" + uuid.NewString()[:8]
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
	require.NoError(t, api.Create(ctx, ns))
	t.Cleanup(func() { _ = api.Delete(context.Background(), ns); metrics.Reset() })
	pool := fwkgaie.MakeInferencePool(eppharness.TestPoolName).Namespace(namespace).
		Selector(map[string]string{"app": "model"}).TargetPorts(8000).EndpointPickerRef("epp").ObjRef()
	pool.Spec.EndpointPickerRef.Port = &v1.Port{Number: v1.PortNumber(9002)}
	require.NoError(t, api.Create(ctx, pool))
	createPod := func(name, address string) *corev1.Pod {
		pod := fwkk8s.MakePod(name).Namespace(namespace).Labels(map[string]string{"app": "model"}).
			IP(address).ReadyCondition().Complete().ObjRef()
		intendedStatus := pod.Status
		require.NoError(t, api.Create(ctx, pod))
		pod.Status = intendedStatus
		require.NoError(t, api.Status().Update(ctx, pod))
		return pod
	}
	pod := createPod("original", "192.168.1.1")
	opts := eppharness.DefaultEppServerOptions(t, namespace, `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: passthrough-parser
- type: random-picker
dataLayer:
  injectDefaults: true
`)
	opts.EnableLeaderElection = true
	opts.LeaseDuration = 4 * time.Second
	opts.RenewDeadline = 3 * time.Second
	opts.RetryPeriod = 500 * time.Millisecond
	opts.DiscoveryStalenessThreshold = 2 * time.Second
	healthPort, err := fwknet.GetFreePort()
	require.NoError(t, err)
	opts.GRPCHealthPort = uint16(healthPort) //#nosec -- OS-assigned TCP port.
	listener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	fault := &membershipFault{}
	cfg := rest.CopyConfig(eppharness.Config())
	cfg.Wrap(func(base http.RoundTripper) http.RoundTripper {
		return membershipTransport{RoundTripper: base, fault: fault}
	})
	_, mgr, ds, err := epprunner.NewTestRunnerSetup(ctx, cfg, opts, nil, listener)
	require.NoError(t, err)
	done := make(chan error, 1)
	go func() { done <- mgr.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			require.NoError(t, err)
		case <-time.After(10 * time.Second):
			t.Error("EPP manager did not stop")
		}
	})
	healthConn, err := grpc.NewClient(fmt.Sprintf("127.0.0.1:%d", healthPort), grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = healthConn.Close() })
	health := healthPb.NewHealthClient(healthConn)
	ready := func() bool {
		probeCtx, stop := context.WithTimeout(ctx, time.Second)
		defer stop()
		resp, err := health.Check(probeCtx, &healthPb.HealthCheckRequest{Service: epprunner.ReadinessCheckService})
		return err == nil && resp.Status == healthPb.HealthCheckResponse_SERVING
	}
	require.Eventually(t, ready, 15*time.Second, 50*time.Millisecond)
	_, conn := fwkepp.ExtProcServerClient(ctx, t, listener.Addr().(*net.TCPAddr).Port, eppharness.Logger(), done)
	send := func() (extProcPb.ExternalProcessor_ProcessClient, *extProcPb.ProcessingResponse) {
		stream, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
		require.NoError(t, err)
		for _, req := range fwkepp.ReqRaw(map[string]string{":path": "/v1/chat/completions", "content-type": "application/json"}, `{"model":"model","messages":[{"role":"user","content":"hello"}]}`) {
			require.NoError(t, stream.Send(req))
		}
		resp, err := stream.Recv()
		require.NoError(t, err)
		return stream, resp
	}
	existing, resp := send()
	require.Equal(t, "192.168.1.1:8000", headerValue(resp.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey))
	_, err = existing.Recv()
	require.NoError(t, err)
	pod.Labels["version"] = "v2"
	require.NoError(t, api.Update(ctx, pod))
	require.Eventually(t, func() bool {
		endpoints := ds.PodList(datastore.AllPodsPredicate)
		return len(endpoints) == 1 && endpoints[0].GetMetadata().Labels["version"] == "v2"
	}, time.Second, 5*time.Millisecond)
	labelStream, labelResponse := send()
	require.Equal(t, "192.168.1.1:8000", headerValue(labelResponse.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey), "normal routing must survive a singleton label change")
	require.NoError(t, labelStream.CloseSend())
	for _, upgrade := range []bool{false, true} {
		pod.Labels["version"] = fmt.Sprintf("v-%t", upgrade)
		require.NoError(t, api.Update(ctx, pod))
		require.Eventually(t, func() bool {
			endpoints := ds.PodList(datastore.AllPodsPredicate)
			return len(endpoints) == 1 && endpoints[0].GetMetadata().Labels["version"] == pod.Labels["version"]
		}, time.Second, 5*time.Millisecond)
		stream, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
		require.NoError(t, err)
		headers := fwkepp.ReqRaw(map[string]string{":path": "/v1/realtime", ":method": "GET", "connection": "upgrade", "upgrade": "websocket"}, "")[0]
		headers.GetRequestHeaders().EndOfStream = !upgrade
		require.NoError(t, stream.Send(headers))
		response, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, "192.168.1.1:8000", headerValue(response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey), "bodyless and WebSocket routing must survive singleton label changes")
		require.NoError(t, stream.CloseSend())
	}
	leaseKey := types.NamespacedName{Namespace: namespace, Name: fmt.Sprintf("epp-%s-%s.llm-d.ai", namespace, pool.Name)}
	lease := &coordinationv1.Lease{}
	require.NoError(t, api.Get(ctx, leaseKey, lease))
	renewedBefore := lease.Spec.RenewTime.DeepCopy()
	fault.interrupt()
	pod.Labels["app"] = "withdrawn"
	require.NoError(t, api.Update(ctx, pod))
	createPod("replacement", "192.168.1.2")
	require.Eventually(t, func() bool { return !ready() }, 8*time.Second, 50*time.Millisecond)
	require.Len(t, ds.PodList(datastore.AllPodsPredicate), 1, "the disconnected watch retains cached membership")
	rejected, resp := send()
	require.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, resp.GetImmediateResponse().GetStatus().GetCode())
	_ = rejected.CloseSend()
	for _, upgrade := range []bool{true, false} {
		stream, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
		require.NoError(t, err)
		headers := fwkepp.ReqRaw(map[string]string{":path": "/v1/realtime", ":method": "GET", "connection": "upgrade", "upgrade": "websocket"}, "")[0]
		headers.GetRequestHeaders().EndOfStream = !upgrade
		require.NoError(t, stream.Send(headers))
		response, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, response.GetImmediateResponse().GetStatus().GetCode(), "bodyless/upgrade routing must reject stale membership")
		_ = stream.CloseSend()
	}
	require.Eventually(t, func() bool {
		if api.Get(ctx, leaseKey, lease) != nil || lease.Spec.RenewTime == nil {
			return false
		}
		return lease.Spec.RenewTime.After(renewedBefore.Time)
	}, 3*time.Second, 50*time.Millisecond, "discovery failure must leave Lease renewal working")
	live, err := health.Check(ctx, &healthPb.HealthCheckRequest{Service: epprunner.LivenessCheckService})
	require.NoError(t, err)
	require.Equal(t, healthPb.HealthCheckResponse_SERVING, live.Status)
	for _, req := range ReqResponseOnly(map[string]string{":status": "200", "content-type": "application/json"}, `{"ok":true}`) {
		require.NoError(t, existing.Send(req))
	}
	for range 2 {
		_, err := existing.Recv()
		require.NoError(t, err)
	}
	require.NoError(t, existing.CloseSend())
	fault.mu.Lock()
	fault.blocked = false
	fault.mu.Unlock()
	require.Eventually(t, ready, 20*time.Second, 50*time.Millisecond)
	require.Eventually(t, func() bool {
		recovered, response := send()
		defer func() { _ = recovered.CloseSend() }()
		if immediate := response.GetImmediateResponse(); immediate != nil {
			require.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, immediate.GetStatus().GetCode())
			return false
		}
		destination := headerValue(response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey)
		require.Equal(t, "192.168.1.2:8000", destination, "recovery must never route to the withdrawn Pod")
		return destination == "192.168.1.2:8000"
	}, 20*time.Second, 50*time.Millisecond, "routing recovers when sampled membership intersects the watch-driven candidates")
}
