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

package iodisagg

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	AlwaysDisaggInferenceDeciderPluginType string = "always-disagg-inference-decider"
)

// compile-time type assertion
var _ ioDeciderPlugin = &AlwaysDisaggInferenceDecider{}

// AlwaysDisaggInferenceDecider is an inference decider plugin which always decide to disaggregate inference
type AlwaysDisaggInferenceDecider struct {
	typedName plugin.TypedName
}

// AlwaysDisaggInferenceDeciderPluginFactory defines the factory function for creating
// a new instance of the AlwaysDisaggInferenceDecider.
func AlwaysDisaggInferenceDeciderPluginFactory(name string, _ *json.Decoder,
	_ plugin.Handle) (plugin.Plugin, error) {
	return newAlwaysDisaggInferenceDecider().WithName(name), nil
}

func newAlwaysDisaggInferenceDecider() *AlwaysDisaggInferenceDecider {
	return &AlwaysDisaggInferenceDecider{
		typedName: plugin.TypedName{Type: AlwaysDisaggInferenceDeciderPluginType},
	}
}

// TypedName returns the typed name of the plugin.
func (d *AlwaysDisaggInferenceDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *AlwaysDisaggInferenceDecider) WithName(name string) *AlwaysDisaggInferenceDecider {
	d.typedName.Name = name
	return d
}

func (d *AlwaysDisaggInferenceDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, post_process_endpoint scheduling.Endpoint, pre_process_endpoint ...scheduling.Endpoint) bool {
	return true
}
