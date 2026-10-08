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

// Package destinationendpointserved writes the endpoint identity picked
// by the scheduler into a response header so downstream consumers can
// attribute per-request metrics to the served arm. It is the
// ORIGINAL_DST counterpart to destination-endpoint-served-verifier,
// which reads Envoy LB metadata that ORIGINAL_DST never populates.
package destinationendpointserved

import (
	"context"
	"encoding/json"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// PluginType is the plugin registry key.
	PluginType = "destination-endpoint-served"

	// ResultHeader is the response header the plugin writes. Matches
	// the header the destination-endpoint-served-verifier writes so a
	// client can use a single header key regardless of whether the
	// data plane load-balances (verifier) or uses ORIGINAL_DST (this
	// plugin).
	ResultHeader = "x-conformance-test-served-endpoint"

	// FailureNoEndpoint is written when the director invokes
	// ResponseHeader without a picked endpoint. Kept human-legible so
	// journal rows carry a diagnosable string.
	FailureNoEndpoint = "fail: no target endpoint"
)

var _ requestcontrol.ResponseHeaderProcessor = &Plugin{}

// Plugin writes the scheduler-picked endpoint's Name to the response
// header. It reads the endpoint from the ResponseHeaderProcessor's
// targetEndpoint argument, which the director populates from
// RequestContext.TargetPod regardless of Envoy's LB configuration.
type Plugin struct {
	typedName fwkplugin.TypedName
}

// New returns a Plugin with the type's default name.
func New() *Plugin {
	return &Plugin{typedName: fwkplugin.TypedName{Type: PluginType, Name: PluginType}}
}

// WithName sets the plugin's instance name.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Factory constructs a Plugin from the plugin registry. The plugin
// takes no parameters; the parameters decoder is ignored.
func Factory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return New().WithName(name), nil
}

// ResponseHeader writes the picked endpoint's Name to the result
// header. When targetEndpoint is nil the plugin writes a legible
// failure string so a consumer can distinguish "no attribution" from
// "attribution to an unknown arm".
func (p *Plugin) ResponseHeader(
	_ context.Context,
	_ *fwksched.InferenceRequest,
	response *requestcontrol.Response,
	targetEndpoint *fwkdl.EndpointMetadata,
) {
	if response.Headers == nil {
		response.Headers = map[string]string{}
	}
	if targetEndpoint == nil {
		response.Headers[ResultHeader] = FailureNoEndpoint
		return
	}
	response.Headers[ResultHeader] = targetEndpoint.Name
}
