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

package disagg

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// AlwaysDisaggMulimodalPluginType is the type-name of the AlwaysDisaggMultimodalDecider plugin.
	AlwaysDisaggMulimodalPluginType = "always-disagg-multimodal-decider"
)

// compile-time type assertion
var _ deciderPlugin = &AlwaysDisaggMultimodalDecider{}

// AlwaysDisaggMultimodalDecider is an EP decider plugin which always decides to encode.
type AlwaysDisaggMultimodalDecider struct {
	typedName plugin.TypedName
}

// AlwaysDisaggMulimodalDeciderPluginFactory defines the factory function for creating
// a new instance of the AlwaysDisaggEncodeDecider.
func AlwaysDisaggMulimodalDeciderPluginFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return NewAlwaysDisaggEncodeDecider().WithName(name), nil
}

// NewAlwaysDisaggEncodeDecider creates a new AlwaysDisaggMultimodalDecider.
func NewAlwaysDisaggEncodeDecider() *AlwaysDisaggMultimodalDecider {
	return &AlwaysDisaggMultimodalDecider{
		typedName: plugin.TypedName{Type: AlwaysDisaggMulimodalPluginType},
	}
}

// TypedName returns the typed name of the plugin.
func (d *AlwaysDisaggMultimodalDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *AlwaysDisaggMultimodalDecider) WithName(name string) *AlwaysDisaggMultimodalDecider {
	d.typedName.Name = name
	return d
}

func (d *AlwaysDisaggMultimodalDecider) disaggregate(_ context.Context, request *scheduling.InferenceRequest, _ scheduling.Endpoint) bool {
	return hasMultimodalContent(request)
}
