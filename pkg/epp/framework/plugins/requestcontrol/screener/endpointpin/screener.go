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

// Package endpointpin screens a request down to the endpoint named in its
// x-pin-host-port header.
package endpointpin

import (
	"context"
	"encoding/json"
	"net/netip"
	"strconv"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PluginType is the type of this plugin.
const PluginType = "endpoint-pin-screener"

var _ fwkrc.Screener = (*Screener)(nil)

// Screener keeps only the endpoint whose <ip:port> equals the request's
// x-pin-host-port header.
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
	pin := request.Headers[routing.EndpointPinHeader]
	if pin == "" {
		return endpoints
	}
	addrPort, err := netip.ParseAddrPort(pin)
	if err != nil {
		log.FromContext(ctx).V(logutil.DEBUG).Info("pin is not an ip:port", "pin", pin)
		return nil
	}
	// Parsed so that any spelling of the address matches the one the endpoint stores.
	host, port := addrPort.Addr().String(), strconv.Itoa(int(addrPort.Port()))
	for _, endpoint := range endpoints {
		if md := endpoint.GetMetadata(); md != nil && md.GetIPAddress() == host && md.GetPort() == port {
			return []fwksched.Endpoint{endpoint}
		}
	}
	log.FromContext(ctx).V(logutil.DEBUG).Info("pinned endpoint is not a candidate", "pin", pin)
	return nil
}
