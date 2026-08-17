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

package prefix

import (
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// MultiClusterScorerType is the explicit cluster-scoped prefix-cache scorer.
const MultiClusterScorerType = "multicluster-prefix-cache-scorer"

var _ fwksched.Scorer = &MultiClusterScorer{}

// MultiClusterScorer scores cluster endpoints by prefix-cache affinity. With
// cluster-endpoints the stock scorer already routes a repeated prefix back to
// the cluster that served it, so this delegates without change. It exists as an
// explicit cluster-scoped type for a complete multicluster plugin family.
type MultiClusterScorer struct {
	*Plugin
}

// MultiClusterScorerFactory builds the cluster-scoped prefix-cache scorer.
func MultiClusterScorerFactory(name string, decoder *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	inner, err := PrefixCachePluginFactory(name, decoder, handle)
	if err != nil {
		return nil, err
	}
	return &MultiClusterScorer{Plugin: inner.(*Plugin)}, nil
}

// TypedName reports the multi-cluster type with this instance's name.
func (s *MultiClusterScorer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: MultiClusterScorerType, Name: s.Plugin.TypedName().Name}
}
