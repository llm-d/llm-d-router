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
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
)

// PeerTopology returns the topology of the endpoint selected in the peer
// scheduling phase, or false when no peer topology is available.
//
// disagg-profile-handler publishes the peer Endpoint as the
// disagg.PeerEndpointAttributeKey request attribute before running the
// prefill profile; its Topology attribute (dataKey) is read directly. Scoped
// to single-EPP deployments; coordinator deployments, where the peer's
// topology arrives on a request header instead, are not yet supported.
func PeerTopology(request *fwksched.InferenceRequest, dataKey fwkplugin.DataKey) (*attrtopology.Topology, bool) {
	if request == nil {
		return nil, false
	}
	peer, ok := fwksched.ReadRequestAttribute[fwksched.Endpoint](request, disagg.PeerEndpointAttributeKey)
	if !ok || peer == nil {
		return nil, false
	}
	return fwkdl.ReadAttribute[*attrtopology.Topology](peer, dataKey)
}
