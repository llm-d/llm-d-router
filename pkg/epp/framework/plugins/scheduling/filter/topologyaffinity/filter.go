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

package topologyaffinity

import (
	"context"
	"encoding/json"
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	topoutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/topology"
)

// FilterType is the type of the topology affinity filter.
const FilterType = "topology-affinity-filter"

type parameters struct {
	// MinAffinity is the tightest-to-loosest floor an endpoint must meet
	// against the peer topology to pass: "host", "rack", "zone", or "region".
	// Defaults to "host".
	MinAffinity string `json:"minAffinity,omitempty"`
	// TopologyProducerName selects the topology-extractor instance to read
	// endpoint topology from. Defaults to the extractor's default producer.
	TopologyProducerName string `json:"topologyProducerName,omitempty"`
}

var _ fwksched.Filter = &Filter{}
var _ fwkplugin.ConsumerPlugin = &Filter{}

// Factory creates a topology affinity filter.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", FilterType, err)
		}
	}
	if params.MinAffinity == "" {
		params.MinAffinity = "host"
	}
	minAffinity, err := topoutil.ParseLevel(params.MinAffinity)
	if err != nil {
		return nil, fmt.Errorf("invalid configuration for '%s' filter: %w", FilterType, err)
	}
	if name == "" {
		name = FilterType
	}
	return &Filter{
		typedName:   fwkplugin.TypedName{Type: FilterType, Name: name},
		minAffinity: minAffinity,
		dataKey:     attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(params.TopologyProducerName),
	}, nil
}

// Filter keeps candidate endpoints whose topology is co-located with the peer
// endpoint at minAffinity or tighter. An endpoint missing the topology
// attribute, or missing the field being compared, never matches.
//
// The filter fails open in two cases: when no peer topology is available (the
// peer is unknown, or has no non-empty field), and when no candidate meets
// minAffinity. In both cases the unfiltered candidates are returned, since
// topology affinity is a preference and must never make a request
// unroutable.
type Filter struct {
	typedName   fwkplugin.TypedName
	minAffinity topoutil.Level
	dataKey     fwkplugin.DataKey
}

func (f *Filter) TypedName() fwkplugin.TypedName {
	return f.typedName
}

// Consumes returns the Topology attribute as optional: a missing producer
// logs a startup warning rather than an error, since the filter fails open
// (no peer topology means the candidates pass through unfiltered) rather
// than depending on the attribute to function.
func (f *Filter) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{f.dataKey: attrtopology.Topology{}},
	}
}

func (f *Filter) Filter(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	peer, ok := topoutil.PeerTopology(request, f.dataKey.String())
	if !ok {
		return endpoints
	}

	filtered := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		candidate, ok := fwkdl.ReadAttribute[*attrtopology.Topology](endpoint, f.dataKey.String())
		if !ok {
			continue
		}
		if topoutil.Compare(peer, candidate) <= f.minAffinity {
			filtered = append(filtered, endpoint)
		}
	}

	if len(filtered) == 0 {
		return endpoints
	}
	return filtered
}
