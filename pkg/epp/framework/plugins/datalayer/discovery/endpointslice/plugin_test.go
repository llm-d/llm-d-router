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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/internal/debugdump"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
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

func (r *recordingNotifier) upsertCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.upserts
}

// slice builds an EndpointSlice of a selector-less headless Service with one
// externally managed IPv6 endpoint and no targetRef.
func slice(name, hostname, addr string, ready bool, extraLabels map[string]string) *discoveryv1.EndpointSlice {
	l := map[string]string{discoveryv1.LabelServiceName: testService}
	for k, v := range extraLabels {
		l[k] = v
	}
	return &discoveryv1.EndpointSlice{
		TypeMeta: metav1.TypeMeta{APIVersion: "discovery.k8s.io/v1", Kind: "EndpointSlice"},
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

var d0ID = types.NamespacedName{Namespace: testNamespace, Name: testService + ".d0"}

// harness runs a plugin's Start and feeds notifications to its extractor, as
// the datalayer runtime does.
type harness struct {
	t         *testing.T
	d         *EndpointSliceDiscovery
	extractor fwkdl.NotificationExtractor
	n         *recordingNotifier
}

func start(t *testing.T, p params) *harness {
	t.Helper()
	d, err := newEndpointSliceDiscovery("test", p)
	if err != nil {
		t.Fatalf("newEndpointSliceDiscovery: %v", err)
	}
	reg := &recordingRegistrar{}
	if err := d.RegisterDependencies(reg); err != nil {
		t.Fatalf("RegisterDependencies: %v", err)
	}
	extractor, ok := reg.registrations[0].Extractor.(fwkdl.NotificationExtractor)
	if !ok {
		t.Fatalf("registered extractor %T is not a NotificationExtractor", reg.registrations[0].Extractor)
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
	return &harness{t: t, d: d, extractor: extractor, n: n}
}

func (h *harness) notify(kind fwkdl.EventType, s *discoveryv1.EndpointSlice) {
	h.t.Helper()
	obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(s)
	if err != nil {
		h.t.Fatalf("ToUnstructured: %v", err)
	}
	if err := h.extractor.Extract(context.Background(), fwkdl.NotificationEvent{Type: kind, Object: &unstructured.Unstructured{Object: obj}}); err != nil {
		h.t.Fatalf("Extract: %v", err)
	}
}

func (h *harness) apply(slices ...*discoveryv1.EndpointSlice) {
	h.t.Helper()
	for _, s := range slices {
		h.notify(fwkdl.EventAddOrUpdate, s)
	}
}

type recordingRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(p fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, p)
	return nil
}

func TestNewRequiresNamespace(t *testing.T) {
	if _, err := newEndpointSliceDiscovery("", params{}); err == nil {
		t.Fatal("expected error for missing namespace")
	}
}

func TestNewRejectsInvalidSelector(t *testing.T) {
	if _, err := newEndpointSliceDiscovery("", params{Namespace: testNamespace, Selector: "a in ("}); err == nil {
		t.Fatal("expected error for invalid selector")
	}
}

func TestRegistersScopedEndpointSliceSource(t *testing.T) {
	d, err := newEndpointSliceDiscovery("slices", testParams)
	if err != nil {
		t.Fatal(err)
	}
	reg := &recordingRegistrar{}
	if err := d.RegisterDependencies(reg); err != nil {
		t.Fatal(err)
	}
	if len(reg.registrations) != 1 {
		t.Fatalf("registrations = %d, want 1", len(reg.registrations))
	}
	got := reg.registrations[0]
	if got.SourceType != notifications.NotificationSourceType {
		t.Errorf("SourceType = %q", got.SourceType)
	}
	src, ok := got.DefaultSource.(fwkdl.ScopedNotificationSource)
	if !ok {
		t.Fatalf("DefaultSource %T is not a ScopedNotificationSource", got.DefaultSource)
	}
	if src.GVK() != endpointSliceGVK || src.Namespace() != testNamespace ||
		src.LabelSelector().String() != testParams.Selector {
		t.Errorf("source = gvk %v namespace %q selector %q", src.GVK(), src.Namespace(), src.LabelSelector())
	}
	if ext, ok := got.Extractor.(fwkdl.NotificationExtractor); !ok || ext.GVK() != endpointSliceGVK {
		t.Errorf("extractor = %T", got.Extractor)
	}
}

func TestInitialSync(t *testing.T) {
	h := start(t, testParams)
	h.apply(
		slice("model-server-p0", "p0", "2001:db8::10", true, map[string]string{"llm-d.ai/role": "prefill"}),
		slice("model-server-d0", "d0", "2001:db8::20", true, map[string]string{"llm-d.ai/role": "decode"}),
	)

	if got := h.n.len(); got != 2 {
		t.Fatalf("endpoints = %d, want 2", got)
	}
	d0 := h.n.get(d0ID)
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

func TestIgnoresSlicesOutsideScope(t *testing.T) {
	h := start(t, testParams)
	other := slice("other-abc", "other", "2001:db8::99", true, nil)
	other.Labels[discoveryv1.LabelServiceName] = "other"
	elsewhere := slice("model-server-x0", "x0", "2001:db8::98", true, nil)
	elsewhere.Namespace = "elsewhere"
	h.apply(other, elsewhere)
	if got := h.n.len(); got != 0 {
		t.Fatalf("endpoints = %d, want 0", got)
	}
}

func TestSliceLeavingSelectorIsRemoved(t *testing.T) {
	h := start(t, testParams)
	h.apply(slice("model-server-d0", "d0", "2001:db8::20", true, nil))
	moved := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	moved.Labels[discoveryv1.LabelServiceName] = "other"
	h.apply(moved)
	if h.n.get(d0ID) != nil {
		t.Fatal("endpoint of a slice that left the selector is still reported")
	}
}

func TestReadinessAddressChangeAndDeletion(t *testing.T) {
	h := start(t, testParams)
	h.apply(slice("model-server-d0", "d0", "2001:db8::20", true, nil))

	h.apply(slice("model-server-d0", "d0", "2001:db8::20", false, nil))
	if h.n.get(d0ID) != nil {
		t.Fatal("not-ready endpoint still reported")
	}

	h.apply(slice("model-server-d0", "d0", "2001:db8::21", true, nil))
	if m := h.n.get(d0ID); m == nil || m.Address != "2001:db8::21" {
		t.Fatalf("endpoint after address change = %+v", m)
	}

	h.notify(fwkdl.EventDelete, slice("model-server-d0", "d0", "2001:db8::21", true, nil))
	if got := h.n.len(); got != 0 {
		t.Fatalf("endpoints after slice deletion = %d, want 0", got)
	}
}

func TestUnsetReadyCountsAsReady(t *testing.T) {
	h := start(t, testParams)
	s := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	s.Endpoints[0].Conditions = discoveryv1.EndpointConditions{}
	h.apply(s)
	if h.n.get(d0ID) == nil {
		t.Fatal("endpoint with unset ready condition missing")
	}
}

func TestTerminatingSkipped(t *testing.T) {
	h := start(t, testParams)
	s := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	s.Endpoints[0].Conditions.Terminating = ptr.To(true)
	h.apply(s)
	if h.n.len() != 0 {
		t.Fatal("terminating endpoint reported")
	}
}

func TestEndpointInTwoSlicesReportedOnce(t *testing.T) {
	h := start(t, testParams)
	h.apply(
		slice("model-server-a", "d0", "2001:db8::20", true, nil),
		slice("model-server-b", "d0", "2001:db8::20", true, nil),
	)
	if got := h.n.len(); got != 1 {
		t.Fatalf("endpoints = %d, want 1", got)
	}
	// The endpoint stays while one of its slices remains.
	h.notify(fwkdl.EventDelete, slice("model-server-a", "d0", "2001:db8::20", true, nil))
	if h.n.get(d0ID) == nil {
		t.Fatal("endpoint removed although another slice still lists it")
	}
}

func TestPortSelection(t *testing.T) {
	multi := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	multi.Ports = append(multi.Ports, discoveryv1.EndpointPort{Name: ptr.To("metrics"), Port: ptr.To[int32](9090)})

	p := testParams
	p.PortName = ""
	h := start(t, p)
	h.apply(multi)
	if h.n.len() != 0 {
		t.Error("slice with two ports reported without portName")
	}

	p.PortName = "metrics"
	h = start(t, p)
	h.apply(multi)
	if m := h.n.get(d0ID); m == nil || m.Port != "9090" {
		t.Errorf("portName=metrics: endpoint = %+v", m)
	}
}

func TestUnchangedSliceNotUpsertedAgain(t *testing.T) {
	h := start(t, testParams)
	h.apply(slice("model-server-d0", "d0", "2001:db8::20", true, nil))
	s := slice("model-server-d0", "d0", "2001:db8::20", true, nil)
	s.ResourceVersion = "2"
	// Extract returns after Start applied the event, so no wait is needed.
	h.apply(s)
	if got := h.n.upsertCount(); got != 1 {
		t.Errorf("upserts = %d, want 1", got)
	}
}

func TestEndpointIDFallsBackToAddress(t *testing.T) {
	h := start(t, testParams)
	s := slice("model-server-d0", "", "2001:db8::20", true, nil)
	s.Endpoints[0].Hostname = nil
	h.apply(s)
	id := types.NamespacedName{Namespace: testNamespace, Name: testService + ".2001:db8::20"}
	m := h.n.get(id)
	if m == nil {
		t.Fatalf("endpoint %s missing", id)
	}
	if m.Address != "2001:db8::20" || m.Port != "9080" {
		t.Errorf("endpoint = %+v", m)
	}
}

func TestExtractFailsAfterStop(t *testing.T) {
	d, err := newEndpointSliceDiscovery("test", testParams)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- d.Start(ctx, newRecordingNotifier()) }()
	<-d.Ready()
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("Start: %v", err)
	}
	ext := &sliceExtractor{plugin: d}
	obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(slice("model-server-d0", "d0", "2001:db8::20", true, nil))
	if err != nil {
		t.Fatal(err)
	}
	err = ext.Extract(context.Background(), fwkdl.NotificationEvent{Type: fwkdl.EventAddOrUpdate, Object: &unstructured.Unstructured{Object: obj}})
	if err == nil {
		t.Fatal("Extract after Start returned: got nil error, want an error")
	}
}

func TestDumpStateOmitsAddresses(t *testing.T) {
	h := start(t, testParams)
	h.apply(slice("model-server-d0", "d0", "2001:db8::20", true, nil))
	raw, err := h.d.DumpState()
	if err != nil {
		t.Fatal(err)
	}
	var st debugdump.State
	if err := json.Unmarshal(raw, &st); err != nil {
		t.Fatal(err)
	}
	if st.TotalEndpoints != 1 || len(st.Endpoints) != 1 || st.Endpoints[0] != d0ID.String() {
		t.Errorf("state = %+v", st)
	}
	if strings.Contains(string(raw), "2001:db8::20") {
		t.Errorf("state leaks address: %s", raw)
	}
}
