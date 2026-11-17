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
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/endpointslice"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

// sliceRecorder is a goroutine-safe DiscoveryNotifier for assertions.
type sliceRecorder struct {
	mu        sync.Mutex
	endpoints map[types.NamespacedName]*fwkdl.EndpointMetadata
}

func (r *sliceRecorder) Upsert(m *fwkdl.EndpointMetadata) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.endpoints[m.ID] = m
}

func (r *sliceRecorder) Delete(id types.NamespacedName) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.endpoints, id)
}

func (r *sliceRecorder) addresses() map[string]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]string, len(r.endpoints))
	for id, m := range r.endpoints {
		out[id.Name] = m.Address
	}
	return out
}

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

// TestEndpointSliceDiscoveryThroughNotificationRuntime runs the
// endpointslice-discovery plugin the way the discovery plugin path does: the
// plugin registers a scoped EndpointSlice source, the runtime binds it to the
// discovery-mode manager, and the plugin only maps slices to endpoints.
func TestEndpointSliceDiscoveryThroughNotificationRuntime(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
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
	// Present before the watch starts: one selected slice, one of another
	// Service in the namespace, and one with the selected labels in another
	// namespace. Only the first is in scope.
	require.NoError(t, k8s.Create(ctx, modelSlice("model-server-d0", namespace, "model-server", "d0", "2001:db8::20")))
	require.NoError(t, k8s.Create(ctx, modelSlice("other-o0", namespace, "other", "o0", "2001:db8::99")))
	require.NoError(t, k8s.Create(ctx, modelSlice("model-server-x0", otherNamespace, "model-server", "x0", "2001:db8::98")))

	params, err := json.Marshal(map[string]string{
		"namespace": namespace,
		"selector":  discoveryv1.LabelServiceName + "=model-server",
		"portName":  "http",
	})
	require.NoError(t, err)
	p, err := endpointslice.Factory("slices", json.NewDecoder(strings.NewReader(string(params))), nil)
	require.NoError(t, err)
	disc := p.(fwkdl.EndpointDiscovery)

	rt := datalayer.NewRuntime(time.Second)
	require.NoError(t, p.(fwkdl.Registrant).RegisterDependencies(rt))
	require.NoError(t, rt.Configure(&datalayer.Config{}, eppharness.Logger()))
	require.True(t, rt.HasNotificationSources())

	mgr, err := runserver.NewNotificationManager(eppharness.Config(), rt.NotificationCacheByObject())
	require.NoError(t, err)
	require.NoError(t, rt.BindNotificationSources(mgr))

	recorder := &sliceRecorder{endpoints: map[types.NamespacedName]*fwkdl.EndpointMetadata{}}
	go func() { _ = disc.Start(ctx, recorder) }()
	go func() { _ = mgr.Start(ctx) }()

	// Once the runtime reports the initial list processed, the datastore
	// already holds the in-scope endpoint: the runner gates serving on this.
	require.Eventually(t, func() bool { return rt.CheckReady() == nil }, 30*time.Second, 50*time.Millisecond)
	require.Equal(t, map[string]string{"model-server.d0": "2001:db8::20"}, recorder.addresses())

	// The manager cache holds only the scoped slices.
	list := &unstructured.UnstructuredList{}
	list.SetAPIVersion("discovery.k8s.io/v1")
	list.SetKind("EndpointSliceList")
	require.NoError(t, mgr.GetCache().List(ctx, list))
	names := make([]string, 0, len(list.Items))
	for _, item := range list.Items {
		names = append(names, item.GetNamespace()+"/"+item.GetName())
	}
	require.Equal(t, []string{namespace + "/model-server-d0"}, names)

	// Changes flow through the same path.
	require.NoError(t, k8s.Create(ctx, modelSlice("model-server-d1", namespace, "model-server", "d1", "2001:db8::21")))
	require.Eventually(t, func() bool {
		return recorder.addresses()["model-server.d1"] == "2001:db8::21"
	}, 10*time.Second, 50*time.Millisecond)

	require.NoError(t, k8s.Delete(ctx, &discoveryv1.EndpointSlice{ObjectMeta: metav1.ObjectMeta{Name: "model-server-d0", Namespace: namespace}}))
	require.Eventually(t, func() bool {
		_, ok := recorder.addresses()["model-server.d0"]
		return !ok
	}, 10*time.Second, 50*time.Millisecond)
}
