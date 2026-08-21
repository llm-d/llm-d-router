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

// Package gwsubset implements a request-control Screener that narrows the
// candidate endpoint set to the subset indicated by the inference-gateway
// dynamic-metadata key envoy.lb.subset_hint, surfaced to the EPP via the
// ext_proc header x-gateway-destination-endpoint-subset.
//
// The plugin is a system built-in: the runner registers it directly on
// startup, gated only by the --disable-endpoint-subset-filter CLI flag. It is
// not exposed via the plugin registry and is not user-configurable.
package gwsubset

import (
	"context"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// PluginType is the plugin type name. Reserved for diagnostic surfaces even
// though the plugin is not registered in the global plugin registry.
const PluginType = "gw-subset-screener"

// Screener narrows the endpoint candidate set to the addresses supplied via
// the destination-endpoint-subset dynamic metadata.
type Screener struct {
	typedName string
}

var (
	_ fwkplugin.Plugin = (*Screener)(nil)
	_ fwkrc.Screener   = (*Screener)(nil)
)

// NewScreener returns a new gwsubset Screener.
func NewScreener() *Screener {
	return &Screener{typedName: PluginType}
}

// TypedName implements plugin.Plugin.
func (s *Screener) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: PluginType, Name: s.typedName}
}

// Screen filters endpoints to those whose IP address is listed in the
// Envoy-supplied subset. When the request carries no subset hint the input
// set is returned unchanged.
func (s *Screener) Screen(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	if request == nil || request.Metadata == nil {
		return endpoints
	}

	subsetMap, ok := request.Metadata[metadata.SubsetFilterNamespace].(map[string]any)
	if !ok {
		return endpoints
	}
	rawList, ok := subsetMap[metadata.SubsetFilterKey].([]any)
	if !ok {
		return endpoints
	}

	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)

	allowed := sets.New[string]()
	for _, entry := range rawList {
		epStr, ok := entry.(string)
		if !ok {
			loggerTrace.Info("ignoring non-string entry in subset list", "value", entry)
			continue
		}
		if idx := strings.LastIndexByte(epStr, ':'); idx >= 0 {
			epStr = epStr[:idx]
		}
		allowed.Insert(epStr)
	}
	if allowed.Len() == 0 {
		// Subset key was present but contained no usable addresses; treat as
		// an empty candidate set so upstream returns 503.
		loggerTrace.Info("subset filter contained no usable addresses, returning empty candidate set")
		return []fwksched.Endpoint{}
	}

	filtered := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		md := endpoint.GetMetadata()
		if md == nil {
			continue
		}
		if allowed.Has(md.GetIPAddress()) {
			filtered = append(filtered, endpoint)
		}
	}
	return filtered
}
