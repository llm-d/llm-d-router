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
	AlwaysDisaggPreprocessDeciderPluginType string = "always-disagg-preprocess-decider"
)

// compile-time type assertion
var _ ioDeciderPlugin = &AlwaysDisaggPreprocessDecider{}

// AlwaysDisaggPreprocessDecider is a preprocess decider plugin which always decide to disaggregate preprocess
type AlwaysDisaggPreprocessDecider struct {
	typedName plugin.TypedName
}

// AlwaysDisaggPreprocessDeciderPluginFactory defines the factory function for creating
// a new instance of the AlwaysDisaggPreprocessDecider.
func AlwaysDisaggPreprocessDeciderPluginFactory(name string, _ *json.Decoder,
	_ plugin.Handle) (plugin.Plugin, error) {
	return newAlwaysDisaggPreprocessDecider().WithName(name), nil
}

func newAlwaysDisaggPreprocessDecider() *AlwaysDisaggPreprocessDecider {
	return &AlwaysDisaggPreprocessDecider{
		typedName: plugin.TypedName{Type: AlwaysDisaggPreprocessDeciderPluginType},
	}
}

// TypedName returns the typed name of the plugin.
func (d *AlwaysDisaggPreprocessDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *AlwaysDisaggPreprocessDecider) WithName(name string) *AlwaysDisaggPreprocessDecider {
	d.typedName.Name = name
	return d
}

func (d *AlwaysDisaggPreprocessDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, postProcessEndpoint scheduling.Endpoint, preProcessEndpoint ...scheduling.Endpoint) bool {
	return true
}
