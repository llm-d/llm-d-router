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

// Package sessionaffinity provides a filter that enforces hard session
// affinity. When the request's session is bound to a candidate endpoint, only
// that endpoint is kept; otherwise all candidates pass through unchanged.
package sessionaffinity

import (
	"context"
	"encoding/json"
	"fmt"
	"net"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

// PluginType is the registered type of the session-affinity filter.
const PluginType = "session-affinity-filter"

var _ fwksched.Filter = &Plugin{}

// Config holds the filter's configurable parameters.
type Config struct {
	// SessionIDProducerName names the session-id-producer instance whose
	// BoundEndpoint attribute this filter consumes. Empty selects the default
	// producer.
	SessionIDProducerName string `json:"sessionIDProducerName,omitempty"`
}

// Plugin is the session-affinity filter.
type Plugin struct {
	typedName fwkplugin.TypedName
	bindingDK fwkplugin.DataKey
}

// Factory builds a Plugin from raw plugin parameters.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := Config{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}

	return &Plugin{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		bindingDK: attrsession.BoundEndpointDataKey.WithNonEmptyProducerName(config.SessionIDProducerName),
	}, nil
}

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Filter narrows the candidate set to the session's bound endpoint when one
// is published and present. When the binding is absent or its endpoint is no
// longer in the candidate list, all candidates pass through unchanged so
// other plugins can pick a replacement.
func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx).V(logutil.DEBUG)

	if len(endpoints) <= 1 {
		return endpoints
	}

	bound, ok := p.readBinding(request)
	if !ok {
		logger.Info("session-affinity-filter: no binding, keeping all endpoints", "total", len(endpoints))
		return endpoints
	}

	for _, ep := range endpoints {
		if endpointHostPort(ep) == string(bound) {
			logger.Info("session-affinity-filter: binding matches a candidate, returning single endpoint",
				"endpoint", string(bound))
			return []fwksched.Endpoint{ep}
		}
	}

	logger.Info("session-affinity-filter: bound endpoint not in candidates, keeping all",
		"endpoint", string(bound), "total", len(endpoints))
	return endpoints
}

// Consumes declares the BoundEndpoint attribute key read by this filter.
func (p *Plugin) Consumes() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.bindingDK: attrsession.BoundEndpoint(""),
	}
}

func (p *Plugin) readBinding(request *fwksched.InferenceRequest) (attrsession.BoundEndpoint, bool) {
	if request == nil {
		return "", false
	}
	bound, ok := fwksched.ReadRequestAttribute[attrsession.BoundEndpoint](request, p.bindingDK.String())
	if !ok || bound == "" {
		return "", false
	}
	return bound, true
}

// endpointHostPort returns the canonical host:port form of an endpoint, or
// the empty string when either coordinate is missing.
func endpointHostPort(ep fwksched.Endpoint) string {
	meta := ep.GetMetadata()
	if meta.Address == "" || meta.Port == "" {
		return ""
	}
	return net.JoinHostPort(meta.Address, meta.Port)
}
