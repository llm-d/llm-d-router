package disagg

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

const (
	// AlwaysDisaggPDDeciderPluginType is the type-name of the alwaysDisaggPDDecider plugin.
	AlwaysDisaggPDDeciderPluginType = "always-disagg-pd-decider"
)

// compile-time type assertion
var _ deciderPlugin = &AlwaysDisaggPDDecider{}

// AlwaysDisaggPDDecider is a PD decider plugin which always decide to disaggregate PD
type AlwaysDisaggPDDecider struct {
	typedName plugin.TypedName
}

// AlwaysDisaggPDDeciderPluginFactory defines the factory function for creating
// a new instance of the AlwaysDisaggPDDecider.
func AlwaysDisaggPDDeciderPluginFactory(name string, _ *json.Decoder,
	_ plugin.Handle) (plugin.Plugin, error) {
	return newAlwaysDisaggPDDecider().WithName(name), nil
}

func newAlwaysDisaggPDDecider() *AlwaysDisaggPDDecider {
	return &AlwaysDisaggPDDecider{
		typedName: plugin.TypedName{Type: AlwaysDisaggPDDeciderPluginType},
	}
}

// TypedName returns the typed name of the plugin.
func (d *AlwaysDisaggPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *AlwaysDisaggPDDecider) WithName(name string) *AlwaysDisaggPDDecider {
	d.typedName.Name = name
	return d
}

func (d *AlwaysDisaggPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	return true
}
