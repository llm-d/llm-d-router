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

package bylabel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// LabelSelectorFilterType is the canonical type of the label selector filter.
const LabelSelectorFilterType = "label-selector-filter"

// compile-time type assertion
var _ scheduling.Filter = &Selector{}

// LabelSelectorFilterFactory is an alias for SelectorFactory using the canonical name.
var LabelSelectorFilterFactory = SelectorFactory

// SelectorFactory defines the factory function for the Selector filter.
func SelectorFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	parameters := metav1.LabelSelector{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", LabelSelectorFilterType, err)
		}
	}
	return NewSelector(name, &parameters)
}

// NewSelector returns a new filter instance, configured with the provided
// name and label selector.
func NewSelector(name string, selector *metav1.LabelSelector) (*Selector, error) {
	if name == "" {
		return nil, errors.New("Selector: missing filter name")
	}
	labelSelector, err := metav1.LabelSelectorAsSelector(selector)
	if err != nil {
		return nil, err
	}

	return &Selector{
		typedName: plugin.TypedName{Type: LabelSelectorFilterType, Name: name},
		selector:  labelSelector,
	}, nil
}

// Selector filters out endpoints that do not match its label selector criteria.
type Selector struct {
	typedName plugin.TypedName
	selector  labels.Selector
}

// TypedName returns the typed name of the plugin
func (blf *Selector) TypedName() plugin.TypedName {
	return blf.typedName
}

// Filter filters out all endpoints that do not satisfy the label selector
func (blf *Selector) Filter(_ context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filtered := []scheduling.Endpoint{}

	for _, endpoint := range endpoints {
		labels := labels.Set(endpoint.GetMetadata().Labels)
		if blf.selector.Matches(labels) {
			filtered = append(filtered, endpoint)
		}
	}
	return filtered
}
