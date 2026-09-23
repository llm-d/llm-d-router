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

// Package prefillpin screens a request down to the prefill endpoint named in
// its x-prefill-pin header.
package prefillpin

import (
	"context"
	"encoding/json"
	"net"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PluginType is the type of this plugin.
const PluginType = "prefill-pin-screener"

var _ fwkrc.Screener = (*Screener)(nil)

// Screener keeps only the endpoint whose <ip:port> equals the request's
// x-prefill-pin header.
type Screener struct {
	typedName plugin.TypedName
}

// Factory creates the screener. It takes no parameters.
func Factory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return New().WithName(name), nil
}

// New returns the screener under its default name.
func New() *Screener {
	return &Screener{typedName: plugin.TypedName{Type: PluginType, Name: PluginType}}
}

// WithName sets the name of the screener.
func (s *Screener) WithName(name string) *Screener {
	s.typedName.Name = name
	return s
}

// TypedName returns the typed name of the screener.
func (s *Screener) TypedName() plugin.TypedName {
	return s.typedName
}

// Screen returns every endpoint when the request carries no pin. With a pin it
// returns only the matching endpoint, and none when that endpoint is not a
// candidate: falling back to another pod would send the request to a pod its
// peer request does not name, so the director answers 503 instead.
func (s *Screener) Screen(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	if request == nil {
		return endpoints
	}
	pin := request.Headers[routing.PrefillPinHeader]
	if pin == "" {
		return endpoints
	}
	// A pin without a port matches no endpoint.
	if host, port, err := net.SplitHostPort(pin); err == nil {
		for _, endpoint := range endpoints {
			md := endpoint.GetMetadata()
			if md != nil && md.GetIPAddress() == host && md.GetPort() == port {
				return []fwksched.Endpoint{endpoint}
			}
		}
	}
	log.FromContext(ctx).V(logutil.DEBUG).Info("pinned prefill endpoint is not a candidate", "pin", pin)
	return nil
}
