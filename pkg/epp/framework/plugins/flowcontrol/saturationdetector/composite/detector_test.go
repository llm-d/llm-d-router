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

package composite

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// notADetector is a plugin that does not implement flowcontrol.SaturationDetector.
type notADetector struct{}

func (n *notADetector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "not-a-detector", Name: "not-a-detector"}
}

func newHandle(t *testing.T) fwkplugin.Handle {
	t.Helper()
	return fwkplugin.NewEppHandle(context.Background(), nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
}

func mockDetector(name string, saturation float64) *fwkfcmocks.MockSaturationDetector {
	return &fwkfcmocks.MockSaturationDetector{
		TypedNameV:  fwkplugin.TypedName{Type: "mock-detector", Name: name},
		SaturationV: saturation,
	}
}

func TestFactory_ReturnsMaxOfChildren(t *testing.T) {
	handle := newHandle(t)
	handle.AddPlugin("low", mockDetector("low", 0.25))
	handle.AddPlugin("high", mockDetector("high", 0.75))

	p, err := MaxSaturationDetectorFactory("combined",
		fwkplugin.StrictDecoder([]byte(`{"detectors":["low","high"]}`)), handle)
	require.NoError(t, err)

	d, ok := p.(flowcontrol.SaturationDetector)
	require.True(t, ok)
	assert.Equal(t, MaxSaturationDetectorType, d.TypedName().Type)
	assert.Equal(t, "combined", d.TypedName().Name)
	assert.InDelta(t, 0.75, d.Saturation(context.Background(), nil), 1e-9)
}

func TestFactory_SingleChildPassesThrough(t *testing.T) {
	handle := newHandle(t)
	handle.AddPlugin("only", mockDetector("only", 1.5))

	p, err := MaxSaturationDetectorFactory("combined",
		fwkplugin.StrictDecoder([]byte(`{"detectors":["only"]}`)), handle)
	require.NoError(t, err)

	d := p.(flowcontrol.SaturationDetector)
	// Values above 1.0 (oversubscription magnitude) must pass through unclamped.
	assert.InDelta(t, 1.5, d.Saturation(context.Background(), nil), 1e-9)
}

func TestFactory_Errors(t *testing.T) {
	tests := []struct {
		name    string
		params  string
		setup   func(h fwkplugin.Handle)
		wantErr string
	}{
		{
			name:    "no detectors configured",
			params:  `{}`,
			wantErr: "at least one entry",
		},
		{
			name:    "empty detectors list",
			params:  `{"detectors":[]}`,
			wantErr: "at least one entry",
		},
		{
			name:    "missing child",
			params:  `{"detectors":["absent"]}`,
			wantErr: "plugin not found: absent",
		},
		{
			name:   "child of wrong type",
			params: `{"detectors":["wrong"]}`,
			setup: func(h fwkplugin.Handle) {
				h.AddPlugin("wrong", &notADetector{})
			},
			wantErr: "does not implement flowcontrol.SaturationDetector",
		},
		{
			name:   "duplicate child",
			params: `{"detectors":["a","a"]}`,
			setup: func(h fwkplugin.Handle) {
				h.AddPlugin("a", mockDetector("a", 0.5))
			},
			wantErr: "duplicate entry",
		},
		{
			name:    "malformed parameters",
			params:  `{"detectors":"not-a-list"}`,
			wantErr: "failed to unmarshal",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handle := newHandle(t)
			if tt.setup != nil {
				tt.setup(handle)
			}
			_, err := MaxSaturationDetectorFactory("combined",
				fwkplugin.StrictDecoder([]byte(tt.params)), handle)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

func TestFactory_NilHandle(t *testing.T) {
	_, err := MaxSaturationDetectorFactory("combined",
		fwkplugin.StrictDecoder([]byte(`{"detectors":["a"]}`)), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "plugin handle is required")
}

// TestNotAConsumerOrSchedulingPlugin pins two deliberate design choices:
//
//   - The composite must not implement ConsumerPlugin. Its children are plugins in the
//     configuration themselves, so their data dependencies are validated directly; declaring
//     the union here would put the keys through the data graph's layer-order check, which
//     places a plugin with no scheduling or requestcontrol interface before every producer
//     and rejects otherwise valid configurations.
//   - The composite must not implement the scheduling Filter extension point: the config
//     loader only auto-injects a gating detector into scheduling profiles when it implements
//     Filter, and per-endpoint filtering belongs to the children, listed in profiles
//     explicitly.
func TestNotAConsumerOrSchedulingPlugin(t *testing.T) {
	handle := newHandle(t)
	handle.AddPlugin("only", mockDetector("only", 0.5))

	p, err := MaxSaturationDetectorFactory("combined",
		fwkplugin.StrictDecoder([]byte(`{"detectors":["only"]}`)), handle)
	require.NoError(t, err)

	_, isConsumer := p.(fwkplugin.ConsumerPlugin)
	assert.False(t, isConsumer, "composite must not declare data dependencies of its own")
	_, isFilter := p.(fwksched.Filter)
	assert.False(t, isFilter, "composite must not be auto-injected into scheduling profiles as a filter")
}
