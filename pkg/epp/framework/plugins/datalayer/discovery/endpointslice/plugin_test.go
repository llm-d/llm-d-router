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

package endpointslice

import (
	"context"
	"encoding/json"
	"maps"
	"strings"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

const (
	testNamespace = "default"
	testService   = "model-server"
)

type recordingNotifier struct {
	mu      sync.Mutex
	state   map[types.NamespacedName]*fwkdl.EndpointMetadata
	upserts int
}

func newRecordingNotifier() *recordingNotifier {
	return &recordingNotifier{state: make(map[types.NamespacedName]*fwkdl.EndpointMetadata)}
}

func (r *recordingNotifier) Upsert(m *fwkdl.EndpointMetadata) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.upserts++
	r.state[m.ID] = m
}

func (r *recordingNotifier) Delete(id types.NamespacedName) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.state, id)
}

func (r *recordingNotifier) get(id types.NamespacedName) *fwkdl.EndpointMetadata {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.state[id]
}

func (r *recordingNotifier) len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.state)
}

// slice builds an EndpointSlice of a selector-less headless Service with one
// externally managed IPv6 endpoint and no targetRef.
func slice(name, hostname, addr string, ready bool, extraLabels map[string]string) *discoveryv1.EndpointSlice {
	l := map[string]string{discoveryv1.LabelServiceName: testService}
	for k, v := range extraLabels {
		l[k] = v
	}
	return &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: testNamespace,
			Labels:    l,
		},
		AddressType: discoveryv1.AddressTypeIPv6,
		Ports:       []discoveryv1.EndpointPort{{Name: ptr.To("http"), Port: ptr.To[int32](9080), Protocol: ptr.To(corev1.ProtocolTCP)}},
		Endpoints: []discoveryv1.Endpoint{{
			Addresses:  []string{addr},
			Hostname:   ptr.To(hostname),
			Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(ready), Serving: ptr.To(ready)},
		}},
	}
}

var testParams = params{
	Namespace: testNamespace,
	Selector:  discoveryv1.LabelServiceName + "=" + testService,
	PortName:  "http",
}

func endpointID(hostname string) types.NamespacedName {
	return types.NamespacedName{Namespace: testNamespace, Name: testService + "." + hostname}
}

func start(t *testing.T, client *fake.Clientset, p params) (*EndpointSliceDiscovery, *recordingNotifier) {
	t.Helper()
	d, err := newEndpointSliceDiscovery("test", p, client)
	if err != nil {
		t.Fatalf("newEndpointSliceDiscovery: %v", err)
	}
	n := newRecordingNotifier()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	errCh := make(chan error, 1)
	go func() { errCh <- d.Start(ctx, n) }()
	select {
	case <-d.Ready():
	case err := <-errCh:
		t.Fatalf("Start returned before Ready: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("plugin did not become ready")
	}
	return d, n
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestNewRequiresNamespace(t *testing.T) {
	if _, err := newEndpointSliceDiscovery("", params{}, fake.NewClientset()); err == nil {
		t.Fatal("expected error for missing namespace")
	}
}

func TestNewRejectsInvalidSelector(t *testing.T) {
	if _, err := newEndpointSliceDiscovery("", params{Namespace: testNamespace, Selector: "a in ("}, fake.NewClientset()); err == nil {
		t.Fatal("expected error for invalid selector")
	}
}

func TestInitialSync(t *testing.T) {
	other := slice("other-abc", "other", "2001:db8::99", true, nil)
	other.Labels[discoveryv1.LabelServiceName] = "other"
	client := fake.NewClientset(
		slice("model-server-p0", "p0", "2001:db8::10", true, map[string]string{"llm-d.ai/role": "prefill"}),
		slice("model-server-d0", "d0", "2001:db8::20", true, map[string]string{"llm-d.ai/role": "decode"}),
		other,
	)
	_, n := start(t, client, testParams)

	if got := n.len(); got != 2 {
		t.Fatalf("endpoints = %d, want 2", got)
	}
	d0 := n.get(endpointID("d0"))
	if d0 == nil {
		t.Fatal("d0 endpoint missing")
	}
	if d0.Address != "2001:db8::20" || d0.Port != "9080" || d0.MetricsHost != "[2001:db8::20]:9080" {
		t.Errorf("d0 = %+v", d0)
	}
	wantLabels := map[string]string{discoveryv1.LabelServiceName: testService, "llm-d.ai/role": "decode"}
	if !maps.Equal(d0.Labels, wantLabels) {
		t.Errorf("d0 labels = %v, want %v", d0.Labels, wantLabels)
	}
}

func TestReadinessAddressChangeAndDeletion(t *testing.T) {
	client := fake.NewClientset(slice("model-server-d0", "d0", "2001:db8::20", true, nil))
	_, n := start(t, client, testParams)
	ctx := context.Background()
	slices := client.DiscoveryV1().EndpointSlices(testNamespace)

	if _, err := slices.Update(ctx, slice("model-server-d0", "d0", "2001:db8::20", false, nil), metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	eventually(t, "not-ready endpoint removed", func() bool { return n.get(endpointID("d0")) == nil })

	if _, err := slices.Update(ctx, slice("model-server-d0", "d0", "2001:db8::21", true, nil), metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	eventually(t, "endpoint re-added with new address", func() bool {
		m := n.get(endpointID("d0"))
		return m != nil && m.Address == "2001:db8::21"
	})

	if err := slices.Delete(ctx, "model-server-d0", metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}
	eventually(t, "endpoint deleted with its slice", func() bool { return n.len() == 0 })
}

func TestUnsetReadyCountsAsReady(t *testing.T) {
	s := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	s.Endpoints[0].Conditions = discoveryv1.EndpointConditions{}
	_, n := start(t, fake.NewClientset(s), testParams)
	if n.get(endpointID("d0")) == nil {
		t.Fatal("endpoint with unset ready condition missing")
	}
}

func TestTerminatingSkipped(t *testing.T) {
	s := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	s.Endpoints[0].Conditions.Terminating = ptr.To(true)
	_, n := start(t, fake.NewClientset(s), testParams)
	if n.len() != 0 {
		t.Fatal("terminating endpoint reported")
	}
}

func TestEndpointInTwoSlicesReportedOnce(t *testing.T) {
	client := fake.NewClientset(
		slice("model-server-a", "d0", "2001:db8::20", true, nil),
		slice("model-server-b", "d0", "2001:db8::20", true, nil),
	)
	_, n := start(t, client, testParams)
	if got := n.len(); got != 1 {
		t.Fatalf("endpoints = %d, want 1", got)
	}
}

func TestPortSelection(t *testing.T) {
	multi := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	multi.Ports = append(multi.Ports, discoveryv1.EndpointPort{Name: ptr.To("metrics"), Port: ptr.To[int32](9090)})

	p := testParams
	p.PortName = ""
	_, n := start(t, fake.NewClientset(multi), p)
	if n.len() != 0 {
		t.Error("slice with two ports reported without portName")
	}

	p.PortName = "metrics"
	_, n = start(t, fake.NewClientset(multi), p)
	if m := n.get(endpointID("d0")); m == nil || m.Port != "9090" {
		t.Errorf("portName=metrics: endpoint = %+v", m)
	}
}

func TestUnchangedSliceNotUpsertedAgain(t *testing.T) {
	client := fake.NewClientset(slice("model-server-d0", "d0", "2001:db8::20", true, nil))
	_, n := start(t, client, testParams)
	s := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	s.ResourceVersion = "2"
	if _, err := client.DiscoveryV1().EndpointSlices(testNamespace).Update(context.Background(), s, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(300 * time.Millisecond)
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.upserts != 1 {
		t.Errorf("upserts = %d, want 1", n.upserts)
	}
}

func TestDumpStateOmitsAddresses(t *testing.T) {
	d, _ := start(t, fake.NewClientset(slice("model-server-d0", "d0", "2001:db8::20", true, nil)), testParams)
	raw, err := d.DumpState()
	if err != nil {
		t.Fatal(err)
	}
	var st discoveryState
	if err := json.Unmarshal(raw, &st); err != nil {
		t.Fatal(err)
	}
	if st.TotalEndpoints != 1 || len(st.Endpoints) != 1 || st.Endpoints[0] != endpointID("d0").String() {
		t.Errorf("state = %+v", st)
	}
	if strings.Contains(string(raw), "2001:db8::20") {
		t.Errorf("state leaks address: %s", raw)
	}
}
