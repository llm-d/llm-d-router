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

package groupsched_test

// THE GROUP SCHEDULER STAYS IN THE FLOW-CONTROL LAYER.
//
// The scheduler is a FairnessPolicy that also implements the scheduling Filter (the seat pin)
// and, to hear RequestAbandoned, the DataProducer interface. The data layer classifies a
// plugin by the FIRST interface it matches, and FairnessPolicy is checked first — so none of
// the others may move it. If one ever did, a data key it reads would be validated against a
// later layer and rejected at startup (measured live in wave 1: "producer thread-identity needs
// to be executed before consumer group/group-scheduler"), and the policy would stop being
// constructed as the flow-control policy before admission runs.
//
// The layer function is unexported, so the placement is read the only way the data layer
// exposes it: through the order check. A probe wrapping the scheduler consumes a key produced
// by a DataProducer in the request-handling phase (layer 1). That edge is rejected exactly
// when the consumer sits in the flow-control layer (0) — a scheduler that had slid to layer 1
// or later would pass, and this test would fail.

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/groupsched"
)

// layerProbeKey is the key the stub producer publishes and the probes consume.
var layerProbeKey = fwkplugin.NewDataKey("layer-probe", "")

// stubProducer is a DataProducer assigned to the request-handling phase (layer 1).
type stubProducer struct{}

func (stubProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "layer-probe-producer", Name: "layer-probe-producer"}
}
func (stubProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{layerProbeKey: ""}
}
func (stubProducer) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

// schedulerProbe is the real scheduler plus one declared read: same interfaces, same layer.
type schedulerProbe struct{ fwkplugin.Plugin }

func (schedulerProbe) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{layerProbeKey: ""}}
}

// filterProbe is the control: a plain scheduling Filter with the same declared read, which
// lives in the scheduling layer (4) and must pass the same check.
type filterProbe struct{ name string }

func (f filterProbe) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "filter-probe", Name: f.name}
}
func (filterProbe) Filter(_ context.Context, _ *fwksched.InferenceRequest, e []fwksched.Endpoint) []fwksched.Endpoint {
	return e
}
func (filterProbe) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{layerProbeKey: ""}}
}

func TestGroupSchedulerLayerStaysFlowControl(t *testing.T) {
	phases := map[string]fwkrh.DataProducerPhase{
		"layer-probe-producer": fwkrh.DataProducerPhaseRequestHandling,
	}

	t.Run("the control passes: a scheduling filter may read a request-handling producer", func(t *testing.T) {
		_, err := datalayer.ValidateAndOrderDataDependencies(
			[]fwkplugin.Plugin{stubProducer{}, filterProbe{name: "control"}},
			datalayer.WithDataProducerPhases(phases))
		require.NoError(t, err, "the probe key and the phases are well-formed: a Filter consumer passes")
	})

	t.Run("the scheduler is rejected as a consumer: it sits ahead of every producer layer", func(t *testing.T) {
		p, err := groupsched.Factory("group", nil, nil)
		require.NoError(t, err)
		// the strictest case: the scheduler is itself listed as a request-handling producer,
		// and the FairnessPolicy check must still win over both producer phases
		phases := map[string]fwkrh.DataProducerPhase{
			"layer-probe-producer": fwkrh.DataProducerPhaseRequestHandling,
			p.TypedName().Name:     fwkrh.DataProducerPhaseRequestHandling,
		}
		_, err = datalayer.ValidateAndOrderDataDependencies(
			[]fwkplugin.Plugin{stubProducer{}, schedulerProbe{Plugin: p}},
			datalayer.WithDataProducerPhases(phases))
		require.Error(t, err,
			"the scheduler must stay in the flow-control layer: implementing the scheduling Filter or the DataProducer interface must NOT move it")
		require.Contains(t, err.Error(), "needs to be executed before consumer")
	})
}
