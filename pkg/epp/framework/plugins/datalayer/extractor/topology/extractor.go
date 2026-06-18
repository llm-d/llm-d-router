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

// Package topology provides a datalayer plugin that stamps each endpoint with
// a Topology attribute containing the endpoint's hostname.
//
// The plugin registers for both endpoint lifecycle events and Pod k8s notification
// events. When hostnameLabel is configured, the label value is read from the
// endpoint event's pod metadata and written as the Topology attribute; Pod
// notifications are a no-op. When hostnameLabel is empty, the endpoint event
// tracks live endpoints keyed by pod identity; the Pod notification (ready pods
// only) reads spec.hostname, caches it, and stamps all current endpoints for
// that pod. Whichever event fires first, the attribute is set once both have
// been processed.
package topology

import (
	"context"
	"encoding/json"
	"slices"
	"sync"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

var (
	_ fwkplugin.Plugin            = (*TopologyExtractor)(nil)
	_ fwkdl.Registrant            = (*TopologyExtractor)(nil)
	_ fwkdl.EndpointExtractor     = (*endpointHandler)(nil)
	_ fwkdl.NotificationExtractor = (*podNotificationHandler)(nil)
)

// podGVK is the core/v1 Pod GVK watched by the pod notification handler.
var podGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}

// params holds the user-facing configuration for the topology extractor.
type params struct {
	// HostnameLabel is the pod label whose value is used as the topology hostname.
	// When empty (default), spec.hostname from the Pod object is used instead.
	// When set but absent on the pod, the attribute is not added.
	HostnameLabel string `json:"hostnameLabel,omitempty"`
}

// TopologyExtractor stamps each endpoint with a Topology attribute.
// It registers for both endpoint lifecycle events and Pod k8s notifications.
type TopologyExtractor struct {
	typedName     fwkplugin.TypedName
	hostnameLabel string
	dk            fwkplugin.DataKey

	// mu guards endpoints and hostnames.
	mu sync.Mutex
	// endpoints maps pod identity to all live Endpoints for that pod.
	// Keyed by {PodName, Namespace} — one pod may have N rank endpoints.
	endpoints map[types.NamespacedName][]fwkdl.Endpoint
	// hostnames caches spec.hostname per pod (ready pods only).
	// Populated by Pod notifications; cleared on pod delete.
	hostnames map[types.NamespacedName]string
}

// Factory is the plugin factory for topology-extractor.
func Factory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p := &params{}
	if parameters != nil {
		if err := parameters.Decode(p); err != nil {
			return nil, err
		}
	}
	if name == "" {
		name = attrtopology.TopologyExtractorType
	}
	return &TopologyExtractor{
		typedName:     fwkplugin.TypedName{Type: attrtopology.TopologyExtractorType, Name: name},
		hostnameLabel: p.HostnameLabel,
		dk:            attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(name),
		endpoints:     make(map[types.NamespacedName][]fwkdl.Endpoint),
		hostnames:     make(map[types.NamespacedName]string),
	}, nil
}

// TypedName returns the plugin type and name.
func (e *TopologyExtractor) TypedName() fwkplugin.TypedName {
	return e.typedName
}

// RegisterDependencies wires this extractor to both an endpoint source and a Pod
// notification source. Both sources are auto-created if absent from user config.
func (e *TopologyExtractor) RegisterDependencies(r fwkdl.Registrar) error {
	if err := r.Register(fwkdl.PendingRegistration{
		Owner:      e.typedName,
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  &endpointHandler{ext: e},
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	}); err != nil {
		return err
	}
	return r.Register(fwkdl.PendingRegistration{
		Owner:      e.typedName,
		SourceType: sourcenotifications.NotificationSourceType,
		Extractor:  &podNotificationHandler{ext: e},
		DefaultSource: sourcenotifications.NewK8sNotificationSource(
			sourcenotifications.NotificationSourceType,
			e.typedName.Name+"/pod",
			podGVK,
		),
	})
}

// podKey returns the pod identity key from endpoint metadata.
// One pod may produce multiple endpoints (one per rank), all sharing the same key.
func podKey(meta *fwkdl.EndpointMetadata) types.NamespacedName {
	return types.NamespacedName{Name: meta.PodName, Namespace: meta.NamespacedName.Namespace}
}

// endpointHandler handles endpoint lifecycle events.
//
// With hostnameLabel set: extracts the label value and writes the Topology attribute.
// Without hostnameLabel: maintains the endpoint map for Pod notification lookups,
// and stamps the attribute immediately if a hostname is already cached.
type endpointHandler struct {
	ext *TopologyExtractor
}

func (h *endpointHandler) TypedName() fwkplugin.TypedName {
	tn := h.ext.typedName
	tn.Name += "/endpoint"
	return tn
}

func (h *endpointHandler) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	meta := event.Endpoint.GetMetadata()
	if meta == nil {
		return nil
	}

	if event.Type == fwkdl.EventDelete {
		if h.ext.hostnameLabel == "" {
			h.removeEndpoint(meta, event.Endpoint)
		}
		return nil
	}

	if h.ext.hostnameLabel != "" {
		val, ok := meta.Labels[h.ext.hostnameLabel]
		if !ok || val == "" {
			return nil
		}
		event.Endpoint.GetAttributes().Put(h.ext.dk.String(), &attrtopology.Topology{Hostname: val})
		return nil
	}

	// No-label path: track endpoint; stamp immediately if hostname already cached.
	key := podKey(meta)
	h.ext.mu.Lock()
	h.ext.endpoints[key] = append(h.ext.endpoints[key], event.Endpoint)
	hn := h.ext.hostnames[key]
	h.ext.mu.Unlock()

	if hn != "" {
		event.Endpoint.GetAttributes().Put(h.ext.dk.String(), &attrtopology.Topology{Hostname: hn})
	}
	return nil
}

func (h *endpointHandler) removeEndpoint(meta *fwkdl.EndpointMetadata, _ fwkdl.Endpoint) {
	key := podKey(meta)
	epKey := meta.GetNamespacedName()

	h.ext.mu.Lock()
	defer h.ext.mu.Unlock()

	eps := h.ext.endpoints[key]
	eps = slices.DeleteFunc(eps, func(e fwkdl.Endpoint) bool {
		return e.GetMetadata().GetNamespacedName() == epKey
	})
	if len(eps) == 0 {
		delete(h.ext.endpoints, key)
	} else {
		h.ext.endpoints[key] = eps
	}
}

// podNotificationHandler handles Pod k8s notification events.
//
// With hostnameLabel set: no-op.
// Without hostnameLabel: caches spec.hostname for ready pods and stamps all
// current endpoints for that pod. Evicts the cache entry when the pod is
// deleted or becomes not-ready.
type podNotificationHandler struct {
	ext *TopologyExtractor
}

func (h *podNotificationHandler) TypedName() fwkplugin.TypedName {
	tn := h.ext.typedName
	tn.Name += "/pod"
	return tn
}

func (h *podNotificationHandler) GVK() schema.GroupVersionKind {
	return podGVK
}

func (h *podNotificationHandler) Extract(_ context.Context, event fwkdl.NotificationEvent) error {
	if h.ext.hostnameLabel != "" {
		return nil
	}

	obj := event.Object
	key := types.NamespacedName{Name: obj.GetName(), Namespace: obj.GetNamespace()}

	if event.Type == fwkdl.EventDelete || !isPodReady(obj) {
		h.ext.mu.Lock()
		delete(h.ext.hostnames, key)
		h.ext.mu.Unlock()
		return nil
	}

	hostname, _, _ := unstructured.NestedString(obj.Object, "spec", "hostname")
	if hostname == "" {
		return nil
	}

	h.ext.mu.Lock()
	h.ext.hostnames[key] = hostname
	eps := slices.Clone(h.ext.endpoints[key])
	h.ext.mu.Unlock()

	for _, ep := range eps {
		ep.GetAttributes().Put(h.ext.dk.String(), &attrtopology.Topology{Hostname: hostname})
	}
	return nil
}

// isPodReady returns true if the pod's Ready condition is True.
func isPodReady(obj *unstructured.Unstructured) bool {
	conditions, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	for _, c := range conditions {
		cond, ok := c.(map[string]any)
		if !ok {
			continue
		}
		t, _, _ := unstructured.NestedString(cond, "type")
		if t != "Ready" {
			continue
		}
		s, _, _ := unstructured.NestedString(cond, "status")
		return s == "True"
	}
	return false
}
