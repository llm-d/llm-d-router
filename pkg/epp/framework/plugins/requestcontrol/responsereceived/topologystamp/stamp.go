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

// Package topologystamp provides a response-header processor that stamps the
// prefill endpoint's topology onto the response, for a coordinator to forward
// to the decode request.
package topologystamp

import (
	"context"
	"encoding/json"
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	topoutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/topology"
)

// PluginType is the plugin type for the topology stamp handler.
const PluginType = "topology-stamp-handler"

// defaultHeaderName is the response header the encoded topology is written
// to, and the header topology-affinity-filter/-scorer read the peer topology
// from in coordinator deployments.
const defaultHeaderName = "x-peer-topology"

type parameters struct {
	// HeaderName is the response header the encoded topology is written to.
	// Defaults to "x-peer-topology".
	HeaderName string `json:"headerName,omitempty"`
	// ProfileName selects the scheduling profile whose selected endpoint's
	// topology is stamped. Empty selects the primary profile's endpoint.
	ProfileName string `json:"profileName,omitempty"`
	// TopologyProducerName selects the topology-extractor instance to read
	// endpoint topology from. Defaults to the extractor's default producer.
	TopologyProducerName string `json:"topologyProducerName,omitempty"`
}

var _ fwkrc.ResponseHeaderProcessor = &Handler{}

// Factory creates a topology stamp handler.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' handler - %w", PluginType, err)
		}
	}
	if params.HeaderName == "" {
		params.HeaderName = defaultHeaderName
	}
	if name == "" {
		name = PluginType
	}
	return &Handler{
		typedName:   fwkplugin.TypedName{Type: PluginType, Name: name},
		headerName:  params.HeaderName,
		profileName: params.ProfileName,
		dataKey:     attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(params.TopologyProducerName),
	}, nil
}

// Handler stamps the encoded Topology of a scheduling profile's selected
// endpoint onto the response headers.
type Handler struct {
	typedName   fwkplugin.TypedName
	headerName  string
	profileName string
	dataKey     fwkplugin.DataKey
}

func (h *Handler) TypedName() fwkplugin.TypedName {
	return h.typedName
}

// ResponseHeader stamps h.headerName on response with the encoded topology of
// the endpoint selected by h.profileName (the primary profile's endpoint when
// unset). It sets nothing when the profile did not run, the endpoint has no
// Topology attribute, or response has no header map to write to.
func (h *Handler) ResponseHeader(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if response == nil || response.Headers == nil || request == nil || request.SchedulingResult == nil {
		return
	}
	profileName := h.profileName
	if profileName == "" {
		profileName = request.SchedulingResult.PrimaryProfileName
	}
	result := request.SchedulingResult.ProfileResults[profileName]
	if result == nil || len(result.TargetEndpoints) == 0 || result.TargetEndpoints[0] == nil {
		return
	}
	topo, ok := fwkdl.ReadAttribute[*attrtopology.Topology](result.TargetEndpoints[0], h.dataKey.String())
	if !ok {
		return
	}
	response.Headers[h.headerName] = topoutil.Encode(topo)
}
