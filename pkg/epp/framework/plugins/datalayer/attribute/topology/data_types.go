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

// Package topology declares the Topology attribute that carries endpoint
// locality information for topology-aware routing.
package topology

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	// TopologyExtractorType is the plugin type identifier for the topology extractor.
	TopologyExtractorType = "topology-extractor"
)

// TopologyAttributeKey identifies the Topology attribute stored on each endpoint.
var TopologyAttributeKey = plugin.NewDataKey("Topology", TopologyExtractorType)

// Topology carries the locality information for an endpoint.
// The Hostname field is populated once when the endpoint is created and
// does not change unless the endpoint is re-created.
type Topology struct {
	// Hostname identifies the node or host on which the endpoint runs.
	// Derived from the Pod's hostname field or from a user-configured label.
	Hostname string
}

// Clone returns an independent copy of the Topology.
func (t *Topology) Clone() fwkdl.Cloneable {
	if t == nil {
		return nil
	}
	cp := *t
	return &cp
}
