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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
)

func makeEndpoint(t *testing.T, name string, topo *attrtopology.Topology) fwksched.Endpoint {
	t.Helper()
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name, Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if topo != nil {
		ep.Put(attrtopology.TopologyAttributeKey.String(), topo)
	}
	return ep
}

func requestWithPeer(topo *attrtopology.Topology) *fwksched.InferenceRequest {
	req := &fwksched.InferenceRequest{}
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "peer", Namespace: "default"}}
	peer := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if topo != nil {
		peer.Put(attrtopology.TopologyAttributeKey.String(), topo)
	}
	req.PutAttribute(disagg.PeerEndpointAttributeKey, peer)
	return req
}

func newTestScorer() *Scorer {
	return &Scorer{
		typedName: fwkplugin.TypedName{Type: ScorerType, Name: "test"},
		dataKey:   attrtopology.TopologyAttributeKey,
	}
}

func TestScorer_LevelCurve(t *testing.T) {
	peerTopo := &attrtopology.Topology{Hostname: "h1", Rack: "r1", Zone: "z1", Region: "reg1"}
	sameHost := makeEndpoint(t, "same-host", &attrtopology.Topology{Hostname: "h1", Rack: "r1", Zone: "z1", Region: "reg1"})
	sameRack := makeEndpoint(t, "same-rack", &attrtopology.Topology{Hostname: "h2", Rack: "r1", Zone: "z1", Region: "reg1"})
	sameZone := makeEndpoint(t, "same-zone", &attrtopology.Topology{Hostname: "h2", Rack: "r2", Zone: "z1", Region: "reg1"})
	sameRegion := makeEndpoint(t, "same-region", &attrtopology.Topology{Hostname: "h2", Rack: "r2", Zone: "z2", Region: "reg1"})
	noMatch := makeEndpoint(t, "no-match", &attrtopology.Topology{Hostname: "h2", Rack: "r2", Zone: "z2", Region: "reg2"})

	s := newTestScorer()
	req := requestWithPeer(peerTopo)
	got := s.Score(context.Background(), req, []fwksched.Endpoint{sameHost, sameRack, sameZone, sameRegion, noMatch})

	assert.Equal(t, 1.00, got[sameHost])
	assert.Equal(t, 0.20, got[sameRack])
	assert.Equal(t, 0.05, got[sameZone])
	assert.Equal(t, 0.02, got[sameRegion])
	assert.Equal(t, 0.00, got[noMatch])
}

func TestScorer_SparseCandidateScoresAtItsOnlyPopulatedLevel(t *testing.T) {
	peerTopo := &attrtopology.Topology{Hostname: "h1", Rack: "r1", Zone: "z1", Region: "reg1"}
	sameRackOnly := makeEndpoint(t, "same-rack-only", &attrtopology.Topology{Rack: "r1"})

	s := newTestScorer()
	req := requestWithPeer(peerTopo)
	got := s.Score(context.Background(), req, []fwksched.Endpoint{sameRackOnly})
	assert.Equal(t, 0.20, got[sameRackOnly], "candidate with only Rack set scores at rack, not host")
}

func TestScorer_AllZeroWhenNoPeerTopology(t *testing.T) {
	endpoints := []fwksched.Endpoint{
		makeEndpoint(t, "a", &attrtopology.Topology{Hostname: "h1"}),
		makeEndpoint(t, "b", &attrtopology.Topology{Hostname: "h2"}),
	}

	s := newTestScorer()
	got := s.Score(context.Background(), &fwksched.InferenceRequest{}, endpoints)
	for _, ep := range endpoints {
		assert.Equal(t, 0.00, got[ep])
	}
}

func TestScorer_MissingAttributeScoresZero(t *testing.T) {
	peerTopo := &attrtopology.Topology{Hostname: "h1"}
	noAttribute := makeEndpoint(t, "no-attribute", nil)

	s := newTestScorer()
	req := requestWithPeer(peerTopo)
	got := s.Score(context.Background(), req, []fwksched.Endpoint{noAttribute})
	assert.Equal(t, 0.00, got[noAttribute])
}

func TestScorer_Category(t *testing.T) {
	s := newTestScorer()
	assert.Equal(t, fwksched.Affinity, s.Category())
}

func TestScorer_Consumes(t *testing.T) {
	s := newTestScorer()

	consumes := s.Consumes()

	assert.Empty(t, consumes.Required)
	require.Len(t, consumes.Optional, 1)
	assert.Equal(t, attrtopology.Topology{}, consumes.Optional[s.dataKey])
}

func TestFactory_Defaults(t *testing.T) {
	p, err := Factory("", nil, nil)
	require.NoError(t, err)
	s, ok := p.(*Scorer)
	require.True(t, ok)
	assert.Equal(t, ScorerType, s.TypedName().Name)
}
