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
	"strings"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
)

// PeerTopology returns the topology of the endpoint selected in the peer
// scheduling phase, or false when no peer topology is available.
//
// Single-EPP deployments: disagg-profile-handler publishes the peer Endpoint
// as the disagg.PeerEndpointAttributeKey request attribute before running
// the prefill profile; its Topology attribute (dataKey) is read directly.
//
// Coordinator deployments, where prefill and decode are picked by separate
// EPPs, carry the peer's topology on headerName instead, stamped by
// topology-stamp-handler on the prefill response and forwarded by the
// coordinator to the decode request. The attribute is preferred over the
// header when both are present.
func PeerTopology(request *fwksched.InferenceRequest, dataKey, headerName string) (*attrtopology.Topology, bool) {
	if request == nil {
		return nil, false
	}
	if peer, ok := fwksched.ReadRequestAttribute[fwksched.Endpoint](request, disagg.PeerEndpointAttributeKey); ok && peer != nil {
		if topo, ok := fwkdl.ReadAttribute[*attrtopology.Topology](peer, dataKey); ok {
			return topo, true
		}
	}
	if headerName == "" {
		return nil, false
	}
	header := request.Headers[strings.ToLower(headerName)]
	if header == "" {
		return nil, false
	}
	return Decode(header), true
}
