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

// Package topologyaffinity provides a scorer that grades candidate endpoints
// by topology proximity to the peer endpoint.
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

// ScorerType is the type of the topology affinity scorer.
const ScorerType = "topology-affinity-scorer"

// levelScores maps the tightest common topology level to its proximity
// score, shaped by the KV-transfer bandwidth available at that level rather
// than by even spacing: same host uses NVLink, same rack a NIC and a single
// switch, and each looser level adds switch hops. Host dominates rack 5:1 and
// zone 20:1 so co-location wins outright, while the looser levels still carry
// enough signal to break ties among non-colocated candidates.
//
// These ratios are hardcoded rather than configurable per level; if a real
// deployment shows they are wrong, add per-level weight parameters then.
var levelScores = map[topoutil.Level]float64{
	topoutil.LevelHost:   1.00,
	topoutil.LevelRack:   0.20,
	topoutil.LevelZone:   0.05,
	topoutil.LevelRegion: 0.02,
	topoutil.LevelNone:   0.00,
}

type parameters struct {
	// TopologyProducerName selects the topology-extractor instance to read
	// endpoint topology from. Defaults to the extractor's default producer.
	TopologyProducerName string `json:"topologyProducerName,omitempty"`
}

var _ fwksched.Scorer = &Scorer{}
var _ fwkplugin.ConsumerPlugin = &Scorer{}

// Factory creates a topology affinity scorer.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", ScorerType, err)
		}
	}
	if name == "" {
		name = ScorerType
	}
	return &Scorer{
		typedName: fwkplugin.TypedName{Type: ScorerType, Name: name},
		dataKey:   attrtopology.TopologyAttributeKey.WithNonEmptyProducerName(params.TopologyProducerName),
	}, nil
}

// Scorer grades candidate endpoints by topology proximity to the peer
// endpoint, using the fixed levelScores curve. Endpoints score 0 when no peer
// topology is available, or when the endpoint has no matching field.
type Scorer struct {
	typedName fwkplugin.TypedName
	dataKey   fwkplugin.DataKey
}

func (s *Scorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Consumes returns the Topology attribute as optional: a missing producer
// logs a startup warning rather than an error, since the scorer scores every
// endpoint 0 (no signal, not an error) rather than depending on the
// attribute to function.
func (s *Scorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{s.dataKey: attrtopology.Topology{}},
	}
}

func (s *Scorer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

func (s *Scorer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))

	peer, ok := topoutil.PeerTopology(request, s.dataKey.String())
	if !ok {
		for _, endpoint := range endpoints {
			scores[endpoint] = 0
		}
		return scores
	}

	for _, endpoint := range endpoints {
		candidate, ok := fwkdl.ReadAttribute[*attrtopology.Topology](endpoint, s.dataKey.String())
		if !ok {
			scores[endpoint] = 0
			continue
		}
		scores[endpoint] = levelScores[topoutil.Compare(peer, candidate)]
	}
	return scores
}
