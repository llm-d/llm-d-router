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

package modelaffinity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// PluginType is the registered type name for the model-affinity filter.
	PluginType = "model-affinity-filter"

	// DefaultLabelKey is the default endpoint label key inspected by the filter.
	DefaultLabelKey = "model"

	// DefaultModelHeader is the default request header from which the target
	// model name is read. This is the header set by IPP's body-field-to-header
	// plugin (model extractor) in the ext_proc chain.
	DefaultModelHeader = "x-gateway-model-name"
)

// parameters is the user-facing configuration for the filter.
type parameters struct {
	// LabelKey is the endpoint label key whose value is compared against the
	// resolved model name. Defaults to "model".
	LabelKey string `json:"labelKey"`
	// ModelHeader is the request header from which the target model name is
	// read. When empty, the filter falls back to InferenceRequest.TargetModel
	// (populated from the body's "model" field or x-llm-d-model-name-rewrite
	// header). Set to "x-gateway-model-name" (default) when IPP's
	// body-field-to-header plugin runs before EPP in the ext_proc chain.
	ModelHeader string `json:"modelHeader"`
}

// compile-time interface assertion
var _ scheduling.Filter = &ModelAffinityFilter{}

// Factory is the plugin factory registered with the EPP framework.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	params := parameters{
		LabelKey:    DefaultLabelKey,
		ModelHeader: DefaultModelHeader,
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for '%s' filter: %w", PluginType, err)
		}
	}
	return New(name, params)
}

// New validates the parameters and returns a new ModelAffinityFilter.
func New(name string, params parameters) (*ModelAffinityFilter, error) {
	if name == "" {
		return nil, errors.New("model-affinity-filter: name cannot be empty")
	}
	if params.LabelKey == "" {
		params.LabelKey = DefaultLabelKey
	}
	return &ModelAffinityFilter{
		typedName:   plugin.TypedName{Type: PluginType, Name: name},
		labelKey:    params.LabelKey,
		modelHeader: params.ModelHeader,
	}, nil
}

// ModelAffinityFilter retains only candidate endpoints whose configured label
// value matches the model name extracted from the request. The model name is
// resolved from a configurable request header (set by IPP's body-field-to-header
// plugin), falling back to InferenceRequest.TargetModel. In a multi-cluster hub
// topology where endpoints are discovered via file-discovery, this enables
// model-aware routing by labelling each endpoint entry with the model it serves.
type ModelAffinityFilter struct {
	typedName   plugin.TypedName
	labelKey    string
	modelHeader string
}

// TypedName returns the typed name of the plugin.
func (f *ModelAffinityFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Filter retains endpoints whose label value for the configured key matches
// the resolved model name. The model is resolved by reading the configured
// request header (typically set by IPP), falling back to TargetModel from the
// request body. If no endpoint matches, an empty slice is returned so the
// scheduler reports a routing error rather than sending to the wrong model.
func (f *ModelAffinityFilter) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	targetModel := f.resolveModelName(request)
	if targetModel == "" {
		log.FromContext(ctx).V(4).Info("model-affinity-filter: no target model resolved, passing all endpoints through")
		return endpoints
	}

	filtered := make([]scheduling.Endpoint, 0, len(endpoints))

	for _, ep := range endpoints {
		labels := ep.GetMetadata().Labels
		if labels == nil {
			continue
		}
		if labels[f.labelKey] == targetModel {
			filtered = append(filtered, ep)
		}
	}

	if len(filtered) == 0 {
		log.FromContext(ctx).V(2).Info("model-affinity-filter: no endpoints matched target model",
			"targetModel", targetModel, "labelKey", f.labelKey, "candidateCount", len(endpoints))
	}

	return filtered
}

// resolveModelName extracts the target model name from the request. It first
// checks the configured header (set by IPP's body-field-to-header plugin in
// the ext_proc chain), then falls back to InferenceRequest.TargetModel (parsed
// from body or x-llm-d-model-name-rewrite header by the EPP director).
func (f *ModelAffinityFilter) resolveModelName(request *scheduling.InferenceRequest) string {
	if request == nil {
		return ""
	}
	if f.modelHeader != "" && request.Headers != nil {
		if headerValue := request.Headers[f.modelHeader]; headerValue != "" {
			return headerValue
		}
	}
	return request.TargetModel
}
