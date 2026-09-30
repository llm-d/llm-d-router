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

// Package reserveendpoint claims the "Prefer: reserve-endpoint" preference so
// that EPP answers the caller with the endpoint it picked instead of forwarding
// the request.
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

var _ fwkrc.PreRequest = (*Plugin)(nil)

// Plugin records the primary profile's picked endpoint on requests that carry
// "Prefer: reserve-endpoint".
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

// PreRequest stores the <ip:port> of the primary profile's first target
// endpoint under ReservedEndpointAttributeKey. Requests without the preference
// are a no-op. A result with no primary endpoint is left unclaimed, and the
// director rejects the request.
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if request == nil || result == nil || !routing.HasPreference(request.Headers, routing.PreferReserveEndpoint) {
		return nil
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	md := primary.TargetEndpoints[0].GetMetadata()
	request.PutAttribute(fwkrc.ReservedEndpointAttributeKey, net.JoinHostPort(md.GetIPAddress(), md.GetPort()))
	return nil
}
