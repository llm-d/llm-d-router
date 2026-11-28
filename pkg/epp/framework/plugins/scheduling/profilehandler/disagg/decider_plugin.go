/*
Copyright 2026 The llm-d Authors.

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

// Package disagg provides profile handler plugins for the epp.
package disagg

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// deciderPlugin decides whether the disaggregated stage should run for the request.
type deciderPlugin interface {
	plugin.Plugin
	disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool
}

// prefixMatchInfoConsumer is implemented by deciders that read PrefixCacheMatchInfo
// from endpoint attributes. The profile handler declares the decider's key in its own
// Consumes() so the data layer wires the producer the decider reads.
type prefixMatchInfoConsumer interface {
	prefixMatchInfoDataKey() plugin.DataKey
}

// handlerScopedConsumer is implemented by deciders that read data only inside the
// profile handler's extension points. The handler declares the decider's reads in its
// own Consumes(). Such a decider is not a plugin.ConsumerPlugin, because the data graph
// orders consumers by the extension points they implement and a decider implements none.
type handlerScopedConsumer interface {
	deciderConsumes() plugin.DataDependencies
}

// deciderConsumes returns the data decider reads, and whether it declares any.
func deciderConsumes(decider deciderPlugin) (plugin.DataDependencies, bool) {
	switch d := decider.(type) {
	case plugin.ConsumerPlugin:
		return d.Consumes(), true
	case handlerScopedConsumer:
		return d.deciderConsumes(), true
	}
	return plugin.DataDependencies{}, false
}
