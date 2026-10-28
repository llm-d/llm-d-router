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

// Package reserveendpoint handles "Prefer: reserve-endpoint" requests: it
// records the endpoint EPP picked, so that the director answers the caller
// with it and forwards nothing.
package reserveendpoint

import (
	"context"
	"encoding/json"
	"net"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PluginType is the type of this plugin.
const PluginType = "reserve-endpoint"

var (
	_ fwkrc.PreRequest      = (*Plugin)(nil)
	_ plugin.ProducerPlugin = (*Plugin)(nil)
)

// Plugin handles requests that carry "Prefer: reserve-endpoint".
type Plugin struct {
	typedName plugin.TypedName
}

// Factory creates the plugin. It takes no parameters.
func Factory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return New().WithName(name), nil
}

// New returns the plugin under its default name.
func New() *Plugin {
	return &Plugin{typedName: plugin.TypedName{Type: PluginType, Name: PluginType}}
}

// WithName sets the name of the plugin.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces declares the request attribute the plugin writes.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{fwkrc.ReservedEndpointAttributeKey: ""}
}

// PreRequest records the <ip:port> of the primary profile's first target
// endpoint under ReservedEndpointAttributeKey for a request that carries the
// preference. The director answers the caller with it and forwards nothing. A
// request without the preference, and a result with no primary endpoint, are
// left to the director.
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if request == nil || !routing.HasPreference(request.Headers, routing.PreferReserveEndpoint) {
		return nil
	}
	endpoint := result.PrimaryEndpoint()
	if endpoint == nil {
		return nil
	}
	md := endpoint.GetMetadata()
	request.PutAttribute(fwkrc.ReservedEndpointAttributeKey, net.JoinHostPort(md.GetIPAddress(), md.GetPort()))
	return nil
}
