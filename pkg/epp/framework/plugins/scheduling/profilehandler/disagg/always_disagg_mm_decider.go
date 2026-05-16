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
func AlwaysDisaggMulimodalDeciderPluginFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return newAlwaysDisaggEncodeDecider().WithName(name), nil
}

func newAlwaysDisaggEncodeDecider() *AlwaysDisaggMultimodalDecider {
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
