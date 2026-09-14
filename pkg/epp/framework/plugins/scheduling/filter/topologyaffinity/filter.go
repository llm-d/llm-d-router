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
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
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
	// LoadAllowance limits the extra in-flight requests tolerated on the least-loaded
	// local endpoint relative to the least-loaded remote endpoint. Nil disables the gate.
	LoadAllowance *int64 `json:"loadAllowance,omitempty"`
	// InFlightLoadProducerName selects the producer supplying the request counts.
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`
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
	if params.LoadAllowance != nil && *params.LoadAllowance < 0 {
		return nil, fmt.Errorf("invalid configuration for '%s' filter: loadAllowance must be non-negative", FilterType)
	}
	minAffinity, err := topoutil.ParseLevel(params.MinAffinity)
	if err != nil {
		return nil, fmt.Errorf("invalid configuration for '%s' filter: %w", FilterType, err)
	}
	if name == "" {
		name = FilterType
	}
	return &Filter{
		typedName:     fwkplugin.TypedName{Type: FilterType, Name: name},
		minAffinity:   minAffinity,
		dataKey:       attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(params.TopologyProducerName),
		loadAllowance: params.LoadAllowance,
		loadDataKey:   attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(params.InFlightLoadProducerName),
	}, nil
}

// Filter applies peer-topology affinity with an optional in-flight load allowance.
// Missing topology attributes or fields do not establish a match.
//
// The filter fails open in two cases: when no peer topology is available (the
// peer is unknown, or has no non-empty field), and when no candidate meets
// minAffinity. In both cases the unfiltered candidates are returned, since
// topology affinity is a preference and must never make a request
// unroutable.
type Filter struct {
	typedName     fwkplugin.TypedName
	minAffinity   topoutil.Level
	dataKey       fwkplugin.DataKey
	loadAllowance *int64
	loadDataKey   fwkplugin.DataKey
}

func (f *Filter) TypedName() fwkplugin.TypedName {
	return f.typedName
}

// Consumes returns the Topology attribute as optional: a missing producer
// logs a startup warning rather than an error, since the filter fails open
// (no peer topology means the candidates pass through unfiltered) rather
// than depending on the attribute to function.
func (f *Filter) Consumes() fwkplugin.DataDependencies {
	dependencies := fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{f.dataKey: attrtopology.Topology{}},
	}
	if f.loadAllowance != nil {
		dependencies.Required = map[fwkplugin.DataKey]any{f.loadDataKey: attrconcurrency.InFlightLoad{}}
	}
	return dependencies
}

func (f *Filter) Filter(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	peer, ok := topoutil.PeerTopology(request, f.dataKey)
	if !ok {
		return endpoints
	}

	filtered := make([]fwksched.Endpoint, 0, len(endpoints))
	var localLoad, remoteLoad int64
	localLoadSet, remoteLoadSet := false, false
	loadsKnown := true
	for _, endpoint := range endpoints {
		candidate, ok := fwkdl.ReadAttribute[*attrtopology.Topology](endpoint, f.dataKey)
		if !ok || candidate == nil {
			loadsKnown = false
			continue
		}
		local := topoutil.Compare(peer, candidate) <= f.minAffinity
		if local {
			filtered = append(filtered, endpoint)
		}
		if f.loadAllowance == nil {
			continue
		}
		// A missing topology field cannot establish that an endpoint is remote.
		if !local && !knownAtLevel(peer, candidate, f.minAffinity) {
			loadsKnown = false
			continue
		}
		load, ok := fwkdl.ReadAttribute[*attrconcurrency.InFlightLoad](endpoint, f.loadDataKey)
		if !ok || load == nil || load.Requests < 0 {
			loadsKnown = false
			continue
		}
		if local {
			if !localLoadSet || load.Requests < localLoad {
				localLoad = load.Requests
			}
			localLoadSet = true
		} else {
			if !remoteLoadSet || load.Requests < remoteLoad {
				remoteLoad = load.Requests
			}
			remoteLoadSet = true
		}
	}

	if len(filtered) == 0 {
		return endpoints
	}
	if f.loadAllowance != nil && loadsKnown && localLoadSet && remoteLoadSet && localLoad-remoteLoad > *f.loadAllowance {
		return endpoints
	}
	return filtered
}

func knownAtLevel(peer, candidate *attrtopology.Topology, level topoutil.Level) bool {
	if peer == nil || candidate == nil {
		return false
	}
	switch level {
	case topoutil.LevelHost:
		return peer.Hostname != "" && candidate.Hostname != ""
	case topoutil.LevelRack:
		return peer.Rack != "" && candidate.Rack != ""
	case topoutil.LevelZone:
		return peer.Zone != "" && candidate.Zone != ""
	case topoutil.LevelRegion:
		return peer.Region != "" && candidate.Region != ""
	default:
		return false
	}
}
