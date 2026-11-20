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
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/utils/ptr"

	eppRunner "github.com/llm-d/llm-d-router/cmd/epp/runner"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
	fwkk8s "github.com/llm-d/llm-d-router/test/framework/k8s"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

func modelSlice(name, namespace, service, hostname, addr string) *discoveryv1.EndpointSlice {
	return &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				discoveryv1.LabelServiceName: service,
				discoveryv1.LabelManagedBy:   "test-controller",
			},
		},
		AddressType: discoveryv1.AddressTypeIPv6,
		Ports:       []discoveryv1.EndpointPort{{Name: ptr.To("http"), Port: ptr.To[int32](9080), Protocol: ptr.To(corev1.ProtocolTCP)}},
		Endpoints: []discoveryv1.Endpoint{{
			Addresses:  []string{addr},
			Hostname:   ptr.To(hostname),
			Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
		}},
	}
}

// endpointAddresses returns the datastore endpoints as name -> address.
func endpointAddresses(ds datastore.Datastore) map[string]string {
	out := map[string]string{}
	for _, ep := range ds.PodList(datastore.AllPodsPredicate) {
		md := ep.GetMetadata()
		out[md.ID.Name] = md.Address
	}
	return out
}

// TestEndpointSliceDiscoveryRunsUnderControllerManager starts the EPP the way
// Run does when endpointslice-discovery is the configured discovery plugin:
// the full controller manager, with no InferencePool, no --pool-name and no
// Pod reconciler. The plugin's scoped EndpointSlice source is bound to that
// manager and is the only source of endpoints.
func TestEndpointSliceDiscoveryRunsUnderControllerManager(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	k8s := eppharness.K8sClient()

	suffix := uuid.New().String()[:8]
	namespace, otherNamespace := "es-discovery-"+suffix, "es-other-"+suffix
	for _, ns := range []string{namespace, otherNamespace} {
		require.NoError(t, k8s.Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}))
		t.Cleanup(func() {
			_ = k8s.Delete(context.Background(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}})
		})
	}
	// Present before the manager starts: one selected slice, one of another
	// Service in the namespace, and one with the selected labels in another
	// namespace. Only the first is in scope.
	require.NoError(t, k8s.Create(ctx, modelSlice("model-server-d0", namespace, "model-server", "d0", "2001:db8::20")))
	require.NoError(t, k8s.Create(ctx, modelSlice("other-o0", namespace, "other", "o0", "2001:db8::99")))
	require.NoError(t, k8s.Create(ctx, modelSlice("model-server-x0", otherNamespace, "model-server", "x0", "2001:db8::98")))

	// A ready Pod in the pool namespace. The Pod reconciler would add it as an
	// endpoint; with a Kubernetes-backed discovery plugin it must not run.
	pod := fwkk8s.MakePod("stray-pod").Namespace(namespace).ReadyCondition().
		Labels(map[string]string{"app": "model-server"}).IP("192.168.1.1").Complete().ObjRef()
	podStatus := pod.Status
	require.NoError(t, k8s.Create(ctx, pod))
	pod.Status = podStatus
	require.NoError(t, k8s.Status().Update(ctx, pod))

	configText := fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - name: slices
    type: endpointslice-discovery
    parameters:
      namespace: %s
      selector: %s=model-server
      portName: http
  - name: random-picker
    type: random-picker
  - name: single-profile-handler
    type: single-profile-handler
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: random-picker
dataLayer:
  injectDefaults: false
  discovery:
    endpoints:
      pluginRef: slices
`, namespace, discoveryv1.LabelServiceName)
	opts := eppharness.DefaultEppServerOptions(t, namespace, configText)
	opts.PoolName = "" // No InferencePool: the pool name defaults.

	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = lis.Close() })

	_, mgr, ds, err := eppRunner.NewTestRunnerSetup(ctx, eppharness.Config(), opts, nil, lis)
	require.NoError(t, err)

	mgrCtx, mgrCancel := context.WithCancel(ctx)
	mgrDone := make(chan struct{})
	go func() {
		defer close(mgrDone)
		_ = mgr.Start(mgrCtx)
	}()
	t.Cleanup(func() {
		mgrCancel()
		<-mgrDone
		metrics.Reset()
	})

	require.Eventually(t, func() bool {
		return endpointAddresses(ds)["model-server.d0"] == "2001:db8::20"
	}, 30*time.Second, 50*time.Millisecond)
	require.Never(t, func() bool { return len(endpointAddresses(ds)) != 1 }, 2*time.Second, 100*time.Millisecond,
		"only the selected slice may supply endpoints, not the Pod or out-of-scope slices")

	// The manager cache holds only the scoped slices.
	list := &unstructured.UnstructuredList{}
	list.SetAPIVersion("discovery.k8s.io/v1")
	list.SetKind("EndpointSliceList")
	require.NoError(t, mgr.GetCache().List(ctx, list))
	names := make([]string, 0, len(list.Items))
	for _, item := range list.Items {
		names = append(names, item.GetNamespace()+"/"+item.GetName())
	}
	sort.Strings(names)
	require.Equal(t, []string{namespace + "/model-server-d0"}, names)

	// Changes flow through the same path.
	require.NoError(t, k8s.Create(ctx, modelSlice("model-server-d1", namespace, "model-server", "d1", "2001:db8::21")))
	require.Eventually(t, func() bool {
		return endpointAddresses(ds)["model-server.d1"] == "2001:db8::21"
	}, 10*time.Second, 50*time.Millisecond)

	require.NoError(t, k8s.Delete(ctx, &discoveryv1.EndpointSlice{ObjectMeta: metav1.ObjectMeta{Name: "model-server-d0", Namespace: namespace}}))
	require.Eventually(t, func() bool {
		_, ok := endpointAddresses(ds)["model-server.d0"]
		return !ok
	}, 10*time.Second, 50*time.Millisecond)
}
