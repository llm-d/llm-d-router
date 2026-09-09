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

package topology

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
)

func TestPeerTopology_NilRequest(t *testing.T) {
	_, ok := PeerTopology(nil, attrtopology.TopologyAttributeKey)
	assert.False(t, ok)
}

func TestPeerTopology_NoPeerAttribute(t *testing.T) {
	req := &fwksched.InferenceRequest{}
	_, ok := PeerTopology(req, attrtopology.TopologyAttributeKey)
	assert.False(t, ok)
}

func TestPeerTopology_FromPeerEndpointAttribute(t *testing.T) {
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "peer", Namespace: "default"}}
	peerEndpoint := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	topo := &attrtopology.Topology{Hostname: "h1"}
	peerEndpoint.Put(attrtopology.TopologyAttributeKey, topo)

	req := &fwksched.InferenceRequest{}
	req.PutAttribute(disagg.PeerEndpointAttributeKey, peerEndpoint)

	got, ok := PeerTopology(req, attrtopology.TopologyAttributeKey)
	assert.True(t, ok)
	assert.Equal(t, topo, got)
}

func TestPeerTopology_PeerEndpointMissingTopologyAttribute(t *testing.T) {
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "peer", Namespace: "default"}}
	peerEndpoint := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())

	req := &fwksched.InferenceRequest{}
	req.PutAttribute(disagg.PeerEndpointAttributeKey, peerEndpoint)

	_, ok := PeerTopology(req, attrtopology.TopologyAttributeKey)
	assert.False(t, ok)
}
