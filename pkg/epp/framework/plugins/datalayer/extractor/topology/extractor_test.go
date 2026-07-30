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
	"fmt"
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

const (
	testPluginName   = "test"
	testNamespace    = "default"
	testNodeHostname = "node-hostname"
)

func readTopology(ep fwkdl.Endpoint) (*attrtopology.Topology, bool) {
	dk := attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(testPluginName).String()
	raw, ok := ep.GetAttributes().Get(dk)
	if !ok {
		return nil, false
	}
	t, ok := raw.(*attrtopology.Topology)
	return t, ok
}

// newRankEndpoint creates an endpoint whose NamespacedName uses the rank suffix
// matching the real datastore convention (pod.Name + "-rank-" + idx).
func newRankEndpoint(podName string, rank int, labels map[string]string) fwkdl.Endpoint {
	epName := fmt.Sprintf("%s-rank-%d", podName, rank)
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: epName, Namespace: testNamespace},
		PodName:        podName,
		Labels:         labels,
	}, nil)
}

// makePod builds an unstructured Pod with optional spec.hostname and readiness.
func makePod(name, hostname string, ready bool) *unstructured.Unstructured {
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(podGVK)
	u.SetName(name)
	u.SetNamespace(testNamespace)
	if hostname != "" {
		_ = unstructured.SetNestedField(u.Object, hostname, "spec", "hostname")
	}
	status := "False"
	if ready {
		status = "True"
	}
	_ = unstructured.SetNestedSlice(u.Object, []any{
		map[string]any{"type": "Ready", "status": status},
	}, "status", "conditions")
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

func epEvent(t fwkdl.EventType, ep fwkdl.Endpoint) fwkdl.EndpointEvent {
	return fwkdl.EndpointEvent{Type: t, Endpoint: ep}
}

func podEvent(t fwkdl.EventType, pod *unstructured.Unstructured) fwkdl.NotificationEvent {
	return fwkdl.NotificationEvent{Type: t, Object: pod}
}

// --- label-configured path ---

func TestLabel_EndpointHandler_LabelPresent(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{Hostname: "topology.hostname"})

	ep := newRankEndpoint("worker-1", 0, map[string]string{"topology.hostname": "rack-42"})
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok := readTopology(ep)
	if !ok {
		t.Fatal("expected Topology attribute")
	}
	if got.Hostname != "rack-42" {
		t.Errorf("hostname = %q, want %q", got.Hostname, "rack-42")
	}
}

func TestLabel_EndpointHandler_LabelMissing(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{Hostname: "topology.hostname"})

	ep := newRankEndpoint("worker-2", 0, map[string]string{"other": "value"})
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if _, ok := readTopology(ep); ok {
		t.Error("expected no Topology attribute when label is absent")
	}
}

func TestLabel_EndpointHandler_NilLabels(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{Hostname: "topology.hostname"})

	ep := newRankEndpoint("worker-3", 0, nil)
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if _, ok := readTopology(ep); ok {
		t.Error("expected no Topology attribute when pod has no labels")
	}
}

func TestLabel_EndpointHandler_DeleteIsNoop(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{Hostname: "topology.hostname"})

	ep := newRankEndpoint("worker-4", 0, map[string]string{"topology.hostname": "rack-1"})
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventDelete, ep)); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if _, ok := readTopology(ep); ok {
		t.Error("expected no Topology attribute for delete event")
	}
}

func TestLabel_PodHandler_IsNoop(t *testing.T) {
	_, _, podH := getHandlers(t, &params{Hostname: "topology.hostname"})

	pod := makePod("worker-1", "actual-hostname", true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("Extract: %v", err)
	}
}

// --- no-label path: pod notification fires first (normal production order) ---

func TestNoLabel_PodFirst_SingleEndpoint(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	pod := makePod("worker-5", testNodeHostname, true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	ep := newRankEndpoint("worker-5", 0, nil)
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("epH.Extract: %v", err)
	}

	got, ok := readTopology(ep)
	if !ok {
		t.Fatal("expected Topology attribute")
	}
	if got.Hostname != testNodeHostname {
		t.Errorf("hostname = %q, want %q", got.Hostname, testNodeHostname)
	}
}

func TestNoLabel_PodFirst_MultiPort(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	// Pod notification fires once.
	pod := makePod("worker-6", testNodeHostname, true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	// Two rank endpoints arrive.
	ep0 := newRankEndpoint("worker-6", 0, nil)
	ep1 := newRankEndpoint("worker-6", 1, nil)
	for i, ep := range []fwkdl.Endpoint{ep0, ep1} {
		if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
			t.Fatalf("epH.Extract rank %d: %v", i, err)
		}
	}

	for i, ep := range []fwkdl.Endpoint{ep0, ep1} {
		got, ok := readTopology(ep)
		if !ok {
			t.Fatalf("rank %d: expected Topology attribute", i)
		}
		if got.Hostname != testNodeHostname {
			t.Errorf("rank %d: hostname = %q, want %q", i, got.Hostname, testNodeHostname)
		}
	}
}

// --- no-label path: endpoint notification fires first (race edge case) ---

func TestNoLabel_EndpointFirst_SingleEndpoint(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	ep := newRankEndpoint("worker-7", 0, nil)
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("epH.Extract: %v", err)
	}

	pod := makePod("worker-7", testNodeHostname, true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	got, ok := readTopology(ep)
	if !ok {
		t.Fatal("expected Topology attribute")
	}
	if got.Hostname != testNodeHostname {
		t.Errorf("hostname = %q, want %q", got.Hostname, testNodeHostname)
	}
}

func TestNoLabel_EndpointFirst_MultiPort(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	ep0 := newRankEndpoint("worker-8", 0, nil)
	ep1 := newRankEndpoint("worker-8", 1, nil)
	for i, ep := range []fwkdl.Endpoint{ep0, ep1} {
		if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
			t.Fatalf("epH.Extract rank %d: %v", i, err)
		}
	}

	pod := makePod("worker-8", testNodeHostname, true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	for i, ep := range []fwkdl.Endpoint{ep0, ep1} {
		got, ok := readTopology(ep)
		if !ok {
			t.Fatalf("rank %d: expected Topology attribute", i)
		}
		if got.Hostname != testNodeHostname {
			t.Errorf("rank %d: hostname = %q, want %q", i, got.Hostname, testNodeHostname)
		}
	}
}

// --- no-label path: readiness filtering ---

func TestNoLabel_NotReadyPod_NotCached(t *testing.T) {
	ext, epH, podH := getHandlers(t, nil)

	// Not-ready pod notification — hostname must not be cached.
	pod := makePod("worker-9", testNodeHostname, false)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	ep := newRankEndpoint("worker-9", 0, nil)
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("epH.Extract: %v", err)
	}

	if _, ok := readTopology(ep); ok {
		t.Error("expected no Topology attribute for not-ready pod")
	}

	ext.mu.Lock()
	_, cached := ext.hostnames[types.NamespacedName{Name: "worker-9", Namespace: "default"}]
	ext.mu.Unlock()
	if cached {
		t.Error("expected hostname not cached for not-ready pod")
	}
}

func TestNoLabel_NotReadyEvictsCache(t *testing.T) {
	ext, _, podH := getHandlers(t, nil)

	podKey := types.NamespacedName{Name: "worker-10", Namespace: "default"}

	// Pod is ready — hostname cached.
	ready := makePod("worker-10", testNodeHostname, true)
	_ = podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, ready))

	ext.mu.Lock()
	_, cached := ext.hostnames[podKey]
	ext.mu.Unlock()
	if !cached {
		t.Fatal("expected hostname cached after ready notification")
	}

	// Pod becomes not-ready — cache evicted.
	notReady := makePod("worker-10", testNodeHostname, false)
	_ = podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, notReady))

	ext.mu.Lock()
	_, still := ext.hostnames[podKey]
	ext.mu.Unlock()
	if still {
		t.Error("expected hostname cache evicted when pod becomes not-ready")
	}
}

// --- no-label path: cache lifecycle ---

func TestNoLabel_EmptyHostname_NoAttribute(t *testing.T) {
	_, epH, podH := getHandlers(t, nil)

	ep := newRankEndpoint("worker-11", 0, nil)
	_ = epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep))

	// Ready pod but spec.hostname not set.
	pod := makePod("worker-11", "", true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	if _, ok := readTopology(ep); ok {
		t.Error("expected no Topology attribute when spec.hostname is empty")
	}
}

func TestNoLabel_PodDelete_ClearsHostnameCache(t *testing.T) {
	ext, _, podH := getHandlers(t, nil)

	podKey := types.NamespacedName{Name: "worker-12", Namespace: "default"}
	ready := makePod("worker-12", testNodeHostname, true)
	_ = podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, ready))

	ext.mu.Lock()
	_, cached := ext.hostnames[podKey]
	ext.mu.Unlock()
	if !cached {
		t.Fatal("precondition: expected hostname cached")
	}

	deleted := makePod("worker-12", "", false)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventDelete, deleted)); err != nil {
		t.Fatalf("podH.Extract delete: %v", err)
	}

	ext.mu.Lock()
	_, still := ext.hostnames[podKey]
	ext.mu.Unlock()
	if still {
		t.Error("expected hostname cache cleared on pod delete")
	}
}

func TestNoLabel_EndpointDelete_RemovesFromMap(t *testing.T) {
	ext, epH, _ := getHandlers(t, nil)

	podKey := types.NamespacedName{Name: "worker-13", Namespace: "default"}
	ep0 := newRankEndpoint("worker-13", 0, nil)
	ep1 := newRankEndpoint("worker-13", 1, nil)

	_ = epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep0))
	_ = epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep1))

	ext.mu.Lock()
	count := len(ext.endpoints[podKey])
	ext.mu.Unlock()
	if count != 2 {
		t.Fatalf("precondition: expected 2 endpoints, got %d", count)
	}

	// Delete rank-0.
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventDelete, ep0)); err != nil {
		t.Fatalf("epH.Extract delete: %v", err)
	}
	ext.mu.Lock()
	count = len(ext.endpoints[podKey])
	ext.mu.Unlock()
	if count != 1 {
		t.Errorf("expected 1 endpoint after deleting rank-0, got %d", count)
	}

	// Delete rank-1 — pod entry should be gone.
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventDelete, ep1)); err != nil {
		t.Fatalf("epH.Extract delete: %v", err)
	}
	ext.mu.Lock()
	_, still := ext.endpoints[podKey]
	ext.mu.Unlock()
	if still {
		t.Error("expected pod entry removed from map when all endpoints deleted")
	}
}

// --- zone and region label extraction ---

func TestLabel_ZoneAndRegion(t *testing.T) {
	_, epH, _ := getHandlers(t, &params{
		Hostname: "my/hostname",
		Rack:     "my/rack",
		Zone:     "my/zone",
		Region:   "my/region",
	})

	ep := newRankEndpoint("worker-z1", 0, map[string]string{
		"my/hostname": "host-1",
		"my/rack":     "rack-7",
		"my/zone":     "us-east-1a",
		"my/region":   "us-east-1",
	})
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok := readTopology(ep)
	if !ok {
		t.Fatal("expected Topology attribute")
	}
	if got.Hostname != "host-1" {
		t.Errorf("hostname = %q, want %q", got.Hostname, "host-1")
	}
	if got.Rack != "rack-7" {
		t.Errorf("rack = %q, want %q", got.Rack, "rack-7")
	}
	if got.Zone != "us-east-1a" {
		t.Errorf("zone = %q, want %q", got.Zone, "us-east-1a")
	}
	if got.Region != "us-east-1" {
		t.Errorf("region = %q, want %q", got.Region, "us-east-1")
	}
}

func TestLabel_ZoneAndRegionDefault(t *testing.T) {
	_, epH, _ := getHandlers(t, nil)

	ep := newRankEndpoint("worker-z2", 0, map[string]string{
		defaultHostnameLabel: "host-2",
		defaultRackLabel:     "rack-2",
		defaultZoneLabel:     "eu-west-1b",
		defaultRegionLabel:   "eu-west-1",
	})
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok := readTopology(ep)
	if !ok {
		t.Fatal("expected Topology attribute")
	}
	if got.Hostname != "host-2" {
		t.Errorf("hostname = %q, want %q", got.Hostname, "host-2")
	}
	if got.Rack != "rack-2" {
		t.Errorf("rack = %q, want %q", got.Rack, "rack-2")
	}
	if got.Zone != "eu-west-1b" {
		t.Errorf("zone = %q, want %q", got.Zone, "eu-west-1b")
	}
	if got.Region != "eu-west-1" {
		t.Errorf("region = %q, want %q", got.Region, "eu-west-1")
	}
}

func TestLabel_ZoneOnlyNoHostname_FallsBackToSpecHostname(t *testing.T) {
	_, epH, podH := getHandlers(t, &params{Zone: "my/zone"})

	// Endpoint has zone label but not the hostname label (defaults to kubernetes.io/hostname).
	ep := newRankEndpoint("worker-z3", 0, map[string]string{
		"my/zone": "ap-south-1a",
	})
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("epH.Extract: %v", err)
	}

	// Zone set but hostname still empty — pod notification provides the fallback.
	pod := makePod("worker-z3", testNodeHostname, true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	got, ok := readTopology(ep)
	if !ok {
		t.Fatal("expected Topology attribute")
	}
	if got.Hostname != testNodeHostname {
		t.Errorf("hostname = %q, want %q", got.Hostname, testNodeHostname)
	}
	if got.Zone != "ap-south-1a" {
		t.Errorf("zone = %q, want %q", got.Zone, "ap-south-1a")
	}
}

func TestLabel_HostnameLabelAbsent_FallsBackToSpecHostname(t *testing.T) {
	_, epH, podH := getHandlers(t, &params{Hostname: "custom/hostname"})

	// Endpoint does not have the configured hostname label.
	ep := newRankEndpoint("worker-z4", 0, map[string]string{"other": "val"})
	if err := epH.Extract(context.Background(), epEvent(fwkdl.EventAddOrUpdate, ep)); err != nil {
		t.Fatalf("epH.Extract: %v", err)
	}

	// No attribute yet.
	if _, ok := readTopology(ep); ok {
		t.Error("expected no Topology attribute before pod notification")
	}

	// Pod notification provides spec.hostname fallback.
	pod := makePod("worker-z4", testNodeHostname, true)
	if err := podH.Extract(context.Background(), podEvent(fwkdl.EventAddOrUpdate, pod)); err != nil {
		t.Fatalf("podH.Extract: %v", err)
	}

	got, ok := readTopology(ep)
	if !ok {
		t.Fatal("expected Topology attribute after pod notification")
	}
	if got.Hostname != testNodeHostname {
		t.Errorf("hostname = %q, want %q", got.Hostname, testNodeHostname)
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
	if gvk.Group != "" || gvk.Version != "v1" || gvk.Kind != "Pod" {
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
