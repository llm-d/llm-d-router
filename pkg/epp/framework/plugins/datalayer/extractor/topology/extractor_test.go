/*
Copyright 2026 The Kubernetes Authors.

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

package topology

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
)

// captureRegistrar collects PendingRegistrations for test inspection.
type captureRegistrar struct {
	regs []fwkdl.PendingRegistration
}

func (r *captureRegistrar) Register(reg fwkdl.PendingRegistration) error {
	r.regs = append(r.regs, reg)
	return nil
}

func (r *captureRegistrar) epHandler() fwkdl.EndpointExtractor {
	for _, reg := range r.regs {
		if ext, ok := reg.Extractor.(fwkdl.EndpointExtractor); ok {
			return ext
		}
	}
	return nil
}

func (r *captureRegistrar) podHandler() fwkdl.NotificationExtractor {
	for _, reg := range r.regs {
		if ext, ok := reg.Extractor.(fwkdl.NotificationExtractor); ok {
			return ext
		}
	}
	return nil
}

func makeDecoder(v any) *json.Decoder {
	b, _ := json.Marshal(v)
	return json.NewDecoder(bytes.NewReader(b))
}

func readTopology(ep fwkdl.Endpoint, name string) (*attrtopology.Topology, bool) {
	dk := attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(name).String()
	raw, ok := ep.GetAttributes().Get(dk)
	if !ok {
		return nil, false
	}
	t, ok := raw.(*attrtopology.Topology)
	return t, ok
}

func newEndpoint(podName string, labels map[string]string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: podName, Namespace: "default"},
		PodName:        podName,
		Labels:         labels,
	}, nil)
}

func makePod(name, namespace, hostname string) *unstructured.Unstructured {
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(podGVK)
	u.SetName(name)
	u.SetNamespace(namespace)
	if hostname != "" {
		_ = unstructured.SetNestedField(u.Object, hostname, "spec", "hostname")
	}
	return u
}

// getHandlers creates a TopologyExtractor with the given params and returns
// its endpoint and pod handlers via a captureRegistrar.
func getHandlers(t *testing.T, pluginParams *params) (*TopologyExtractor, fwkdl.EndpointExtractor, fwkdl.NotificationExtractor) {
	t.Helper()
	var dec *json.Decoder
	if pluginParams != nil {
		dec = makeDecoder(pluginParams)
	}
	plugin, err := Factory("test", dec, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	ext := plugin.(*TopologyExtractor)
	var reg captureRegistrar
	if err := ext.RegisterDependencies(&reg); err != nil {
		t.Fatalf("RegisterDependencies: %v", err)
	}
	epH := reg.epHandler()
	podH := reg.podHandler()
	if epH == nil {
		t.Fatal("no EndpointExtractor registered")
	}
	if podH == nil {
		t.Fatal("no NotificationExtractor registered")
	}
	return ext, epH, podH
}

// --- label-configured path ---

func TestEndpointHandler_LabelPresent(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{HostnameLabel: "topology.hostname"})

	ep := newEndpoint("worker-1", map[string]string{"topology.hostname": "rack-42"})
	if err := epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok := readTopology(ep, "test")
	if !ok {
		t.Fatal("expected Topology attribute")
	}
	if got.Hostname != "rack-42" {
		t.Errorf("hostname = %q, want %q", got.Hostname, "rack-42")
	}
}

func TestEndpointHandler_LabelMissing(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{HostnameLabel: "topology.hostname"})

	ep := newEndpoint("worker-2", map[string]string{"other": "value"})
	if err := epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute when label is absent")
	}
}

func TestEndpointHandler_NoLabelMap_WithLabelConfig(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{HostnameLabel: "topology.hostname"})

	ep := newEndpoint("worker-3", nil)
	if err := epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute when pod has no labels")
	}
}

func TestEndpointHandler_Delete_WithLabelConfig(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{HostnameLabel: "topology.hostname"})

	ep := newEndpoint("worker-4", map[string]string{"topology.hostname": "rack-1"})
	if err := epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventDelete, Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute for delete event")
	}
}

func TestPodHandler_WithLabel_IsNoop(t *testing.T) {
	_, _, podH := getHandlers(t, &params{HostnameLabel: "topology.hostname"})

	// Pod notification should do nothing when label is configured.
	pod := makePod("worker-1", "default", "actual-hostname")
	if err := podH.Extract(context.Background(), fwkdl.NotificationEvent{
		Type: fwkdl.EventAddOrUpdate, Object: pod,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	// No endpoint to check — the important thing is no panic and no side-effects.
}

// --- no-label path ---

func TestEndpointHandler_NoLabel_StoresEndpoint(t *testing.T) {
	ext, epH, _ := getHandlers(t, nil)

	ep := newEndpoint("worker-5", nil)
	if err := epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	ext.mu.RLock()
	_, stored := ext.endpoints[types.NamespacedName{Name: "worker-5", Namespace: "default"}]
	ext.mu.RUnlock()
	if !stored {
		t.Error("expected endpoint to be stored in map")
	}
}

func TestEndpointHandler_NoLabel_DoesNotSetAttribute(t *testing.T) {
	_, epH, _ := getHandlers(t, nil)

	ep := newEndpoint("worker-5", nil)
	if err := epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute set by endpoint handler in no-label mode")
	}
}

func TestEndpointHandler_NoLabel_Delete_RemovesFromMap(t *testing.T) {
	ext, epH, _ := getHandlers(t, nil)

	ep := newEndpoint("worker-6", nil)
	key := types.NamespacedName{Name: "worker-6", Namespace: "default"}

	// Store the endpoint first.
	_ = epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	})

	// Now delete it.
	if err := epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventDelete, Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	ext.mu.RLock()
	_, still := ext.endpoints[key]
	ext.mu.RUnlock()
	if still {
		t.Error("expected endpoint removed from map after delete")
	}
}

func TestPodHandler_NoLabel_SetsHostname(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	ep := newEndpoint("worker-7", nil)
	// Store endpoint via endpoint handler.
	_ = epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	})

	pod := makePod("worker-7", "default", "node-hostname")
	if err := podH.Extract(context.Background(), fwkdl.NotificationEvent{
		Type: fwkdl.EventAddOrUpdate, Object: pod,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok := readTopology(ep, "test")
	if !ok {
		t.Fatal("expected Topology attribute after pod notification")
	}
	if got.Hostname != "node-hostname" {
		t.Errorf("hostname = %q, want %q", got.Hostname, "node-hostname")
	}
}

func TestPodHandler_NoLabel_NoEndpointInMap(t *testing.T) {
	_, _, podH := getHandlers(t, nil)

	// Pod notification fires but no endpoint was stored — should be a no-op.
	pod := makePod("unknown-pod", "default", "some-hostname")
	if err := podH.Extract(context.Background(), fwkdl.NotificationEvent{
		Type: fwkdl.EventAddOrUpdate, Object: pod,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}
}

func TestPodHandler_NoLabel_EmptyHostname(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	ep := newEndpoint("worker-8", nil)
	_ = epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	})

	// Pod has no spec.hostname set.
	pod := makePod("worker-8", "default", "")
	if err := podH.Extract(context.Background(), fwkdl.NotificationEvent{
		Type: fwkdl.EventAddOrUpdate, Object: pod,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute when spec.hostname is empty")
	}
}

func TestPodHandler_Delete_IsNoop(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	ep := newEndpoint("worker-9", nil)
	_ = epH.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	})

	pod := makePod("worker-9", "default", "node-hostname")
	if err := podH.Extract(context.Background(), fwkdl.NotificationEvent{
		Type: fwkdl.EventDelete, Object: pod,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute for pod delete event")
	}
}

// --- registration ---

func TestRegisterDependencies_RegistersBothHandlers(t *testing.T) {
	plugin, err := Factory("test", nil, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	var reg captureRegistrar
	if err := plugin.(*TopologyExtractor).RegisterDependencies(&reg); err != nil {
		t.Fatalf("RegisterDependencies: %v", err)
	}
	if len(reg.regs) != 2 {
		t.Fatalf("expected 2 registrations, got %d", len(reg.regs))
	}
	if reg.epHandler() == nil {
		t.Error("no EndpointExtractor registered")
	}
	if reg.podHandler() == nil {
		t.Error("no NotificationExtractor registered")
	}
}

func TestPodHandler_GVK(t *testing.T) {
	_, _, podH := getHandlers(t, nil)
	gvk := podH.GVK()
	if gvk.Version != "v1" || gvk.Kind != "Pod" || gvk.Group != "" {
		t.Errorf("unexpected GVK: %v", gvk)
	}
}

func TestTypedName(t *testing.T) {
	plugin, err := Factory("my-extractor", nil, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	tn := plugin.TypedName()
	if tn.Type != attrtopology.TopologyExtractorType {
		t.Errorf("type = %q, want %q", tn.Type, attrtopology.TopologyExtractorType)
	}
	if tn.Name != "my-extractor" {
		t.Errorf("name = %q, want %q", tn.Name, "my-extractor")
	}
}
