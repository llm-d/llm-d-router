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

	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
)

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

func TestExtract_PodName(t *testing.T) {
	plugin, err := Factory("test", nil, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	ext := plugin.(*TopologyExtractor)

	ep := newEndpoint("worker-1", nil)
	if err := ext.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok := readTopology(ep, "test")
	if !ok {
		t.Fatal("expected Topology attribute to be set")
	}
	if got.Hostname != "worker-1" {
		t.Errorf("hostname = %q, want %q", got.Hostname, "worker-1")
	}
}

func TestExtract_LabelPresent(t *testing.T) {
	plugin, err := Factory("test", makeDecoder(params{HostnameLabel: "topology.hostname"}), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	ext := plugin.(*TopologyExtractor)

	ep := newEndpoint("worker-2", map[string]string{"topology.hostname": "rack-42"})
	if err := ext.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok := readTopology(ep, "test")
	if !ok {
		t.Fatal("expected Topology attribute to be set")
	}
	if got.Hostname != "rack-42" {
		t.Errorf("hostname = %q, want %q", got.Hostname, "rack-42")
	}
}

func TestExtract_LabelMissing(t *testing.T) {
	plugin, err := Factory("test", makeDecoder(params{HostnameLabel: "topology.hostname"}), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	ext := plugin.(*TopologyExtractor)

	ep := newEndpoint("worker-3", map[string]string{"other-label": "value"})
	if err := ext.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute when configured label is absent")
	}
}

func TestExtract_NoLabelMap(t *testing.T) {
	plugin, err := Factory("test", makeDecoder(params{HostnameLabel: "topology.hostname"}), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	ext := plugin.(*TopologyExtractor)

	ep := newEndpoint("worker-4", nil)
	if err := ext.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute when pod has no labels")
	}
}

func TestExtract_DeleteIsNoOp(t *testing.T) {
	plugin, err := Factory("test", nil, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	ext := plugin.(*TopologyExtractor)

	ep := newEndpoint("worker-5", nil)
	if err := ext.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute for delete event")
	}
}

func TestExtract_EmptyPodName(t *testing.T) {
	plugin, err := Factory("test", nil, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	ext := plugin.(*TopologyExtractor)

	ep := newEndpoint("", nil)
	if err := ext.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if _, ok := readTopology(ep, "test"); ok {
		t.Error("expected no Topology attribute when PodName is empty")
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
