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

// Package topology provides an EndpointExtractor that stamps each endpoint with
// a Topology attribute containing the endpoint's hostname.
//
// The hostname is resolved once when the endpoint is created and stored as a
// static attribute. When a hostnameLabel is configured the label's value is
// used; if the label is absent the attribute is not set. When no label is
// configured the Pod's hostname field (EndpointMetadata.PodName) is used
// directly, which also works for file-based endpoints.
package topology

import (
	"context"
	"encoding/json"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

var (
	_ fwkdl.EndpointExtractor = (*TopologyExtractor)(nil)
	_ fwkdl.Registrant        = (*TopologyExtractor)(nil)
)

// params holds the user-facing configuration for the topology extractor.
type params struct {
	// HostnameLabel is the pod label whose value is used as the hostname key.
	// When empty (default), the Pod hostname field is used.
	// When set but the label is absent on the pod, the attribute is not added.
	HostnameLabel string `json:"hostnameLabel,omitempty"`
}

// TopologyExtractor stamps each endpoint with a Topology attribute on creation.
type TopologyExtractor struct {
	typedName     fwkplugin.TypedName
	hostnameLabel string
	dk            string
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
		dk:            attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(name).String(),
	}, nil
}

// TypedName returns the plugin type and name.
func (e *TopologyExtractor) TypedName() fwkplugin.TypedName {
	return e.typedName
}

// RegisterDependencies wires this extractor to the endpoint notification source.
// If no endpoint-notification-source is configured, one is created automatically.
func (e *TopologyExtractor) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      e.typedName,
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  e,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract sets the Topology attribute on an endpoint when it is created or updated.
// On delete events and when the hostname cannot be determined, no attribute is written.
func (e *TopologyExtractor) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type == fwkdl.EventDelete {
		return nil
	}

	meta := event.Endpoint.GetMetadata()
	if meta == nil {
		return nil
	}

	var hostname string
	if e.hostnameLabel != "" {
		val, ok := meta.Labels[e.hostnameLabel]
		if !ok {
			// Configured label is absent; do not set the attribute.
			return nil
		}
		hostname = val
	} else {
		hostname = meta.PodName
	}

	if hostname == "" {
		return nil
	}

	event.Endpoint.GetAttributes().Put(e.dk, &attrtopology.Topology{Hostname: hostname})
	return nil
}
