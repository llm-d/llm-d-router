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
// tracks the endpoint in an internal map; the Pod notification reads
// spec.hostname and stamps the matching endpoint.
package topology

import (
	"context"
	"encoding/json"
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

	// endpoints maps pod NamespacedName to the live Endpoint.
	// Used in the no-label path so Pod notifications can find the endpoint to stamp.
	mu        sync.RWMutex
	endpoints map[types.NamespacedName]fwkdl.Endpoint
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
		endpoints:     make(map[types.NamespacedName]fwkdl.Endpoint),
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

// endpointHandler handles endpoint lifecycle events.
//
// With hostnameLabel set: extracts the label value and writes the Topology attribute.
// Without hostnameLabel: maintains the endpoint map for Pod notification lookups.
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
	key := meta.GetNamespacedName()

	if event.Type == fwkdl.EventDelete {
		if h.ext.hostnameLabel == "" {
			h.ext.mu.Lock()
			delete(h.ext.endpoints, key)
			h.ext.mu.Unlock()
		}
		return nil
	}

	if h.ext.hostnameLabel != "" {
		val, ok := meta.Labels[h.ext.hostnameLabel]
		if !ok || val == "" {
			return nil
		}
		event.Endpoint.GetAttributes().Put(h.ext.dk.String(), &attrtopology.Topology{Hostname: val})
	} else {
		h.ext.mu.Lock()
		h.ext.endpoints[key] = event.Endpoint
		h.ext.mu.Unlock()
	}
	return nil
}

// podNotificationHandler handles Pod k8s notification events.
//
// With hostnameLabel set: no-op.
// Without hostnameLabel: reads spec.hostname and stamps the matching endpoint.
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
	if h.ext.hostnameLabel != "" || event.Type == fwkdl.EventDelete {
		return nil
	}

	obj := event.Object
	key := types.NamespacedName{Name: obj.GetName(), Namespace: obj.GetNamespace()}

	h.ext.mu.RLock()
	ep, ok := h.ext.endpoints[key]
	h.ext.mu.RUnlock()
	if !ok {
		return nil
	}

	hostname, _, _ := unstructured.NestedString(obj.Object, "spec", "hostname")
	if hostname == "" {
		return nil
	}

	ep.GetAttributes().Put(h.ext.dk.String(), &attrtopology.Topology{Hostname: hostname})
	return nil
}
