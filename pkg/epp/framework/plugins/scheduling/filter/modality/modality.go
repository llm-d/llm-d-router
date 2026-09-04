/*
Copyright 2025 The llm-d Authors.

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

package modality

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/common"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const ModalityFilterType = "modality-filter"

var _ scheduling.Filter = &ModalityFilter{}

// ModalityFilterFactory creates a ModalityFilter instance.
func ModalityFilterFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return NewModalityFilter().WithName(name), nil
}

// ModalityFilter selects endpoints whose llm-d.ai/model-arch label matches the
// architectures compatible with the request path. Paths not in PathToModelArch
// pass all endpoints through unchanged.
type ModalityFilter struct {
	typedName plugin.TypedName
}

// NewModalityFilter returns a ModalityFilter with the default type name.
func NewModalityFilter() *ModalityFilter {
	return &ModalityFilter{
		typedName: plugin.TypedName{Type: ModalityFilterType},
	}
}

// WithName sets the plugin instance name.
func (f *ModalityFilter) WithName(name string) *ModalityFilter {
	f.typedName.Name = name
	return f
}

// TypedName returns the typed name of the plugin.
func (f *ModalityFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Filter keeps only endpoints whose model-arch label is compatible with the
// request path. Unknown paths return all endpoints unfiltered.
func (f *ModalityFilter) Filter(_ context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	path := request.Headers[common.EnvoyPathHeader]
	if idx := strings.IndexByte(path, '?'); idx >= 0 {
		path = path[:idx]
	}

	validArchs, known := common.PathToModelArch[path]
	if !known {
		return endpoints
	}

	archSet := make(map[string]struct{}, len(validArchs))
	for _, arch := range validArchs {
		archSet[arch] = struct{}{}
	}

	filtered := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		archLabel := ep.GetMetadata().Labels[common.ModelArchLabel]
		if _, ok := archSet[archLabel]; ok {
			filtered = append(filtered, ep)
		}
	}

	return filtered
}
