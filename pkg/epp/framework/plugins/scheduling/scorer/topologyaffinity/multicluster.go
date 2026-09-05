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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	topoutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/topology"
)

// MultiClusterPluginType is the cross-cluster topology scorer. Unlike
// topology-affinity-scorer, which scores proximity to a peer endpoint, this
// scorer scores against a configured weight table and needs no peer.
const MultiClusterPluginType = "multicluster-topology-scorer"

type multiClusterParameters struct {
	// Field selects which Topology field the weights key on: region (default),
	// zone, rack, or host.
	Field string `json:"field,omitempty"`
	// Weights maps a field value to its score. A value absent from the map
	// scores 0, so an unlisted or unlabeled cluster is deprioritized, not dropped.
	Weights map[string]float64 `json:"weights"`
	// TopologyProducerName selects the topology-extractor instance to read from.
	TopologyProducerName string `json:"topologyProducerName,omitempty"`
}

var (
	_ fwksched.Scorer          = &MultiClusterScorer{}
	_ fwkplugin.ConsumerPlugin = &MultiClusterScorer{}
)

// MultiClusterFactory builds the cross-cluster topology scorer.
func MultiClusterFactory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if rawParameters == nil {
		return nil, errors.New(MultiClusterPluginType + " requires parameters")
	}
	params := multiClusterParameters{}
	if err := rawParameters.Decode(&params); err != nil {
		return nil, fmt.Errorf("decode %s parameters: %w", MultiClusterPluginType, err)
	}
	if len(params.Weights) == 0 {
		return nil, errors.New(MultiClusterPluginType + " requires a non-empty weights map")
	}
	// The scheduler clamps each scorer output to [0,1] before weighting, so
	// normalize by the max to keep the ratio between regions on any scale.
	weights, err := normalizeWeights(params.Weights)
	if err != nil {
		return nil, err
	}
	level, err := topoutil.ParseLevel(cmp.Or(params.Field, topoutil.LevelRegion.String()))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", MultiClusterPluginType, err)
	}
	if name == "" {
		name = MultiClusterPluginType
	}
	return &MultiClusterScorer{
		typedName: fwkplugin.TypedName{Type: MultiClusterPluginType, Name: name},
		selector:  fieldSelector(level),
		weights:   weights,
		dataKey:   attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(params.TopologyProducerName),
	}, nil
}

// MultiClusterScorer scores cluster endpoints by a static weight table over a
// Topology field, instead of by proximity to a peer.
type MultiClusterScorer struct {
	typedName fwkplugin.TypedName
	selector  func(*attrtopology.Topology) string
	weights   map[string]float64
	dataKey   fwkplugin.DataKey
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *MultiClusterScorer) TypedName() fwkplugin.TypedName { return s.typedName }

// Category reports that this scorer expresses an affinity preference.
func (s *MultiClusterScorer) Category() fwksched.ScorerCategory { return fwksched.Affinity }

// Consumes marks Topology optional: a missing producer scores every endpoint 0
// rather than failing, matching the topology-affinity-scorer.
func (s *MultiClusterScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{s.dataKey: attrtopology.Topology{}},
	}
}

// Score returns the configured weight for each endpoint's topology field value.
// Endpoints with no Topology attribute score 0.
func (s *MultiClusterScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		topology, ok := fwkdl.ReadAttribute[*attrtopology.Topology](endpoint, s.dataKey.String())
		if !ok {
			scores[endpoint] = 0
			continue
		}
		scores[endpoint] = s.weights[s.selector(topology)]
	}
	return scores
}

// normalizeWeights scales the table so its largest value is 1.0, keeping the
// ratios between values. Weights must be non-negative with at least one positive.
func normalizeWeights(weights map[string]float64) (map[string]float64, error) {
	maxW := 0.0
	for value, w := range weights {
		if w < 0 {
			return nil, fmt.Errorf("%s weight for %q must be non-negative, got %v", MultiClusterPluginType, value, w)
		}
		maxW = max(maxW, w)
	}
	if maxW == 0 {
		return nil, errors.New(MultiClusterPluginType + " requires at least one positive weight")
	}
	normalized := make(map[string]float64, len(weights))
	for value, w := range weights {
		normalized[value] = w / maxW
	}
	return normalized, nil
}

// fieldSelector returns the accessor for a validated topology level. ParseLevel
// has already rejected any other value, so the default reads region.
func fieldSelector(level topoutil.Level) func(*attrtopology.Topology) string {
	switch level {
	case topoutil.LevelHost:
		return func(t *attrtopology.Topology) string { return t.Hostname }
	case topoutil.LevelRack:
		return func(t *attrtopology.Topology) string { return t.Rack }
	case topoutil.LevelZone:
		return func(t *attrtopology.Topology) string { return t.Zone }
	default:
		return func(t *attrtopology.Topology) string { return t.Region }
	}
}
