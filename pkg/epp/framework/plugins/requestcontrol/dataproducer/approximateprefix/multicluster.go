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

package approximateprefix

import (
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// MultiClusterPluginType is the explicit cluster-scoped approx-prefix producer.
const MultiClusterPluginType = "multicluster-approx-prefix-cache-producer"

// MultiClusterProducer tracks (cluster, block) instead of (pod, block): with
// cluster-endpoints as the candidate set the stock indexer already keys on
// endpoint identity, so this delegates without change. It exists as an explicit
// cluster-scoped type for a complete multicluster plugin family.
type MultiClusterProducer struct {
	*dataProducer
	name string
}

// MultiClusterFactory builds the cluster-scoped approx-prefix producer.
func MultiClusterFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	inner, err := ApproxPrefixCacheFactory(name, rawParameters, handle)
	if err != nil {
		return nil, err
	}
	return &MultiClusterProducer{dataProducer: inner.(*dataProducer), name: name}, nil
}

// TypedName reports the multi-cluster type with this instance's name. The
// published data key stays name-based, so the prefix scorer binds unchanged.
func (p *MultiClusterProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: MultiClusterPluginType, Name: p.name}
}
