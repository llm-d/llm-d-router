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

// Package reserveendpoint answers "Prefer: reserve-endpoint" requests with the
// endpoint EPP picked, so that EPP forwards nothing.
package reserveendpoint

import (
	"context"
	"encoding/json"
	"net"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PluginType is the type of this plugin.
const PluginType = "reserve-endpoint"

var _ fwkrc.PreRequest = (*Plugin)(nil)

// Plugin answers requests that carry "Prefer: reserve-endpoint".
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

// PreRequest answers a request that carries the preference with a NoContent
// error: 204, the <ip:port> of the primary profile's first target endpoint on
// ReservedEndpointHeader, and Preference-Applied. The director sends that
// answer to the caller and forwards nothing. A request without the
// preference, and a result with no primary endpoint, are left to the director.
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if request == nil || !routing.HasPreference(request.Headers, routing.PreferReserveEndpoint) {
		return nil
	}
	endpoint := primaryEndpoint(result)
	if endpoint == nil {
		return nil
	}
	md := endpoint.GetMetadata()
	return errcommon.Error{
		Code: errcommon.NoContent,
		Msg:  "reserve-endpoint request answered with the picked endpoint",
		Headers: map[string]string{
			routing.ReservedEndpointHeader:  net.JoinHostPort(md.GetIPAddress(), md.GetPort()),
			routing.PreferenceAppliedHeader: routing.PreferReserveEndpoint,
		},
	}
}

// primaryEndpoint returns the first endpoint the primary profile picked, or
// nil when the result has none.
func primaryEndpoint(result *fwksched.SchedulingResult) fwksched.Endpoint {
	if result == nil {
		return nil
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	return primary.TargetEndpoints[0]
}
