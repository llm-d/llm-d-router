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

package sessionaffinity

import (
	"encoding/json"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// MultiClusterFilterType is the cross-cluster session-affinity filter.
const MultiClusterFilterType = "multicluster-session-affinity-filter"

var _ fwksched.Filter = &MultiClusterFilter{}

// MultiClusterFactory builds the cross-cluster session-affinity filter.
func MultiClusterFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	inner, err := Factory(name, params, handle)
	if err != nil {
		return nil, err
	}
	return &MultiClusterFilter{SessionAffinity: inner.(*SessionAffinity), name: name}, nil
}

// MultiClusterFilter is the explicit cluster-scoped session-affinity filter.
// Cluster identity comes from discovery, so the stock filter already pins to the
// right cluster-endpoint and this delegates without translation.
type MultiClusterFilter struct {
	*SessionAffinity
	name string
}

// TypedName reports the multi-cluster type with this instance's name.
func (f *MultiClusterFilter) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: MultiClusterFilterType, Name: f.name}
}
