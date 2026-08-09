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
	topoutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/topology"
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

func newTestFilter(minAffinity topoutil.Level) *Filter {
	return &Filter{
		typedName:   fwkplugin.TypedName{Type: FilterType, Name: "test"},
		minAffinity: minAffinity,
		dataKey:     attrtopology.TopologyAttributeKey,
	}
}

func TestFilter_KeepsOnlyMatchingMinAffinity(t *testing.T) {
	peerTopo := &attrtopology.Topology{Hostname: "h1", Rack: "r1", Zone: "z1", Region: "reg1"}
	sameHost := makeEndpoint(t, "same-host", &attrtopology.Topology{Hostname: "h1", Rack: "r1", Zone: "z1", Region: "reg1"})
	sameRack := makeEndpoint(t, "same-rack", &attrtopology.Topology{Hostname: "h2", Rack: "r1", Zone: "z1", Region: "reg1"})
	noMatch := makeEndpoint(t, "no-match", &attrtopology.Topology{Hostname: "h3", Rack: "r3", Zone: "z3", Region: "reg3"})

	f := newTestFilter(topoutil.LevelHost)
	req := requestWithPeer(peerTopo)
	got := f.Filter(context.Background(), req, []fwksched.Endpoint{sameHost, sameRack, noMatch})
	require.Len(t, got, 1)
	assert.Equal(t, "same-host", got[0].GetMetadata().ID.Name)

	f = newTestFilter(topoutil.LevelRack)
	got = f.Filter(context.Background(), req, []fwksched.Endpoint{sameHost, sameRack, noMatch})
	require.Len(t, got, 2)
}

func TestFilter_SparseCandidateMatchesAtItsOnlyPopulatedLevel(t *testing.T) {
	peerTopo := &attrtopology.Topology{Hostname: "h1", Rack: "r1", Zone: "z1", Region: "reg1"}
	sameRackOnly := makeEndpoint(t, "same-rack-only", &attrtopology.Topology{Rack: "r1"})

	f := newTestFilter(topoutil.LevelRack)
	req := requestWithPeer(peerTopo)
	got := f.Filter(context.Background(), req, []fwksched.Endpoint{sameRackOnly})
	require.Len(t, got, 1, "candidate with only Rack set still matches at rack")

	f = newTestFilter(topoutil.LevelHost)
	got = f.Filter(context.Background(), req, []fwksched.Endpoint{sameRackOnly})
	require.Len(t, got, 1, "fails open: rack-only match does not meet host floor")
}

func TestFilter_FailsOpenWhenNoCandidateMatches(t *testing.T) {
	peerTopo := &attrtopology.Topology{Hostname: "h1"}
	noMatch := makeEndpoint(t, "no-match", &attrtopology.Topology{Hostname: "h2"})

	f := newTestFilter(topoutil.LevelHost)
	req := requestWithPeer(peerTopo)
	got := f.Filter(context.Background(), req, []fwksched.Endpoint{noMatch})
	require.Len(t, got, 1, "fails open: no candidate meets minAffinity")
}

func TestFilter_FailsOpenWhenNoPeerTopology(t *testing.T) {
	endpoints := []fwksched.Endpoint{
		makeEndpoint(t, "a", &attrtopology.Topology{Hostname: "h1"}),
		makeEndpoint(t, "b", &attrtopology.Topology{Hostname: "h2"}),
	}

	f := newTestFilter(topoutil.LevelHost)
	got := f.Filter(context.Background(), &fwksched.InferenceRequest{}, endpoints)
	assert.Equal(t, endpoints, got, "fails open: no peer topology at all")
}

func TestFilter_EndpointMissingAttributeIsDropped(t *testing.T) {
	peerTopo := &attrtopology.Topology{Hostname: "h1"}
	sameHost := makeEndpoint(t, "same-host", &attrtopology.Topology{Hostname: "h1"})
	noAttribute := makeEndpoint(t, "no-attribute", nil)

	f := newTestFilter(topoutil.LevelHost)
	req := requestWithPeer(peerTopo)
	got := f.Filter(context.Background(), req, []fwksched.Endpoint{sameHost, noAttribute})
	require.Len(t, got, 1)
	assert.Equal(t, "same-host", got[0].GetMetadata().ID.Name)
}

func TestFilter_Consumes(t *testing.T) {
	f := newTestFilter(topoutil.LevelHost)

	consumes := f.Consumes()

	assert.Empty(t, consumes.Required)
	require.Len(t, consumes.Optional, 1)
	assert.Equal(t, attrtopology.Topology{}, consumes.Optional[f.dataKey])
}

func TestFactory_Defaults(t *testing.T) {
	p, err := Factory("", nil, nil)
	require.NoError(t, err)
	f, ok := p.(*Filter)
	require.True(t, ok)
	assert.Equal(t, topoutil.LevelHost, f.minAffinity)
	assert.Equal(t, FilterType, f.TypedName().Name)
}

func TestFactory_InvalidMinAffinity(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"minAffinity": "datacenter"}`)), nil)
	assert.Error(t, err)
}
