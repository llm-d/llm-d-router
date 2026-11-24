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

package loader

import (
	"fmt"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	configapiv1alpha1 "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

func TestPerFlowLimitsLoad(t *testing.T) {
	for _, version := range []string{"v1", "v1alpha1"} {
		for _, limits := range []struct {
			name, yaml      string
			requests, bytes uint64
		}{
			{"unset", "", 0, 0},
			{"zero", `maxRequestsPerFlow: "0", maxBytesPerFlow: "0"`, 0, 0},
			{"count", `maxRequestsPerFlow: "1k"`, 1000, 0},
			{"bytes", `maxBytesPerFlow: "2Mi"`, 0, 2 * 1024 * 1024},
			{"both", `maxRequestsPerFlow: "7", maxBytesPerFlow: "3Ki"`, 7, 3 * 1024},
		} {
			t.Run(version+"/"+limits.name, func(t *testing.T) {
				raw := fmt.Sprintf(`apiVersion: llm-d.ai/%s
kind: EndpointPickerConfig
flowControl:
  defaultPriorityBand: {%s}
  defaultNegativePriorityBand: {%s}
  priorityBands:
  - {priority: 10, %s}
`, version, limits.yaml, limits.yaml, limits.yaml)
				api, _, err := LoadRawConfig([]byte(raw), logr.Discard())
				require.NoError(t, err)
				handle := newFlowControlTestHandle(t)
				defaults, err := buildPriorityBandPolicyDefaults(handle)
				require.NoError(t, err)
				cfg, err := buildRegistryConfig(api.FlowControl, defaults, handle)
				require.NoError(t, err)
				for _, band := range []*registry.PriorityBandConfig{cfg.PriorityBands[10], cfg.PriorityBands[0], cfg.DefaultPriorityBand, cfg.DefaultNegativePriorityBand} {
					require.Equal(t, limits.requests, band.MaxRequestsPerFlow)
					require.Equal(t, limits.bytes, band.MaxBytesPerFlow)
				}
				reg := registry.NewFlowRegistry(cfg, logr.Discard())
				reg.ApplyDesiredPriorities(map[int]struct{}{7: {}, -7: {}})
				for _, priority := range []int{0, 10, 7, -7} {
					snapshot, err := reg.CapacitySnapshot(priority)
					require.NoError(t, err)
					require.Equal(t, limits.requests, snapshot.MaxRequestsPerFlow)
					require.Equal(t, limits.bytes, snapshot.MaxBytesPerFlow)
				}
			})
		}
	}
}

// TestFairAdmissionLoad checks that fairAdmission reaches the bands it is set on in both API versions, including bands
// the registry later creates from a template, and no others.
func TestFairAdmissionLoad(t *testing.T) {
	for _, version := range []string{"v1", "v1alpha1"} {
		t.Run(version, func(t *testing.T) {
			raw := fmt.Sprintf(`apiVersion: llm-d.ai/%s
kind: EndpointPickerConfig
flowControl:
  defaultPriorityBand: {fairAdmission: true}
  defaultNegativePriorityBand: {}
  priorityBands:
  - {priority: 10, fairAdmission: true}
  - {priority: 20}
`, version)
			api, _, err := LoadRawConfig([]byte(raw), logr.Discard())
			require.NoError(t, err)
			handle := newFlowControlTestHandle(t)
			defaults, err := buildPriorityBandPolicyDefaults(handle)
			require.NoError(t, err)
			cfg, err := buildRegistryConfig(api.FlowControl, defaults, handle)
			require.NoError(t, err)
			require.True(t, cfg.PriorityBands[10].FairAdmission)
			require.False(t, cfg.PriorityBands[20].FairAdmission)
			require.True(t, cfg.DefaultPriorityBand.FairAdmission)
			require.False(t, cfg.DefaultNegativePriorityBand.FairAdmission)

			reg := registry.NewFlowRegistry(cfg, logr.Discard())
			reg.ApplyDesiredPriorities(map[int]struct{}{7: {}, -7: {}})
			for priority, want := range map[int]bool{7: true, -7: false} {
				key := fwkfc.FlowKey{ID: "flow", Priority: priority}
				require.NoError(t, reg.WithConnection(key, func(contracts.ActiveFlowConnection) error {
					mq, err := reg.ManagedQueue(key)
					require.NoError(t, err)
					return mq.Add(fwkfcmocks.NewMockQueueItemAccessor(1, "req", key))
				}))
				require.Equal(t, want, reg.LongestQueue(priority) != nil, "template-created priority %d", priority)
			}
		})
	}
}

// TestFairAdmissionRejectsComposedFairnessIDs checks that the loader refuses fairAdmission on a priority band or band
// template while requestHandler.composeFairnessIDs is on, in both API versions, and accepts either setting alone.
func TestFairAdmissionRejectsComposedFairnessIDs(t *testing.T) {
	for _, version := range []string{"v1", "v1alpha1"} {
		for _, tc := range []struct {
			name        string
			compose     bool
			flowControl string
			// wantField is the band field the error must name; empty means the configuration is accepted.
			wantField string
		}{
			{
				name: "both on", compose: true,
				flowControl: "{priorityBands: [{priority: 10}, {priority: 20, fairAdmission: true}]}",
				wantField:   "flowControl.priorityBands[1].fairAdmission",
			},
			{
				name: "default template", compose: true,
				flowControl: "{defaultPriorityBand: {fairAdmission: true}}",
				wantField:   "flowControl.defaultPriorityBand.fairAdmission",
			},
			{
				name: "negative template", compose: true,
				flowControl: "{defaultNegativePriorityBand: {fairAdmission: true}}",
				wantField:   "flowControl.defaultNegativePriorityBand.fairAdmission",
			},
			{
				name:        "fair admission alone",
				flowControl: "{defaultPriorityBand: {fairAdmission: true}, priorityBands: [{priority: 10, fairAdmission: true}]}",
			},
			{
				name: "composed IDs alone", compose: true,
				flowControl: "{defaultPriorityBand: {}, priorityBands: [{priority: 10}]}",
			},
		} {
			t.Run(version+"/"+tc.name, func(t *testing.T) {
				raw := fmt.Sprintf(`apiVersion: llm-d.ai/%s
kind: EndpointPickerConfig
requestHandler: {composeFairnessIDs: %t}
flowControl: %s
plugins:
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins: []
`, version, tc.compose, tc.flowControl)
				api, _, err := LoadRawConfig([]byte(raw), logr.Discard())
				require.NoError(t, err)
				err = validateConfig(api)
				if tc.wantField == "" {
					require.NoError(t, err)
					return
				}
				require.ErrorContains(t, err, tc.wantField)
				require.ErrorContains(t, err, "requestHandler.composeFairnessIDs")
			})
		}
	}
}

func TestPerFlowLimitsInvalid(t *testing.T) {
	for _, version := range []string{"v1", "v1alpha1"} {
		for _, field := range []string{"maxRequestsPerFlow", "maxBytesPerFlow"} {
			for _, scope := range []string{"defaultPriorityBand", "defaultNegativePriorityBand", "priorityBands"} {
				for _, value := range []string{"-1", "invalid"} {
					t.Run(version+"/"+field+"/"+scope+"/"+value, func(t *testing.T) {
						band := fmt.Sprintf(`{%s: %q}`, field, value)
						if scope == "priorityBands" {
							band = "[" + band + "]"
						}
						raw := fmt.Sprintf("apiVersion: llm-d.ai/%s\nkind: EndpointPickerConfig\nflowControl:\n  %s: %s\n", version, scope, band)
						api, _, err := LoadRawConfig([]byte(raw), logr.Discard())
						if value == "invalid" {
							require.Error(t, err)
							return
						}
						require.NoError(t, err)
						handle := newFlowControlTestHandle(t)
						defaults, err := buildPriorityBandPolicyDefaults(handle)
						require.NoError(t, err)
						_, err = buildRegistryConfig(api.FlowControl, defaults, handle)
						require.ErrorContains(t, err, "must be non-negative")
					})
				}
			}
		}
	}
}

func TestPerFlowLimitsDeepCopy(t *testing.T) {
	quantity := resource.MustParse("1234567890123456789.123")
	original := quantity.DeepCopy()
	t.Run("v1", func(t *testing.T) {
		band := configapiv1.PriorityBandConfig{MaxRequestsPerFlow: &quantity, MaxBytesPerFlow: &quantity}
		copied := band.DeepCopy()
		copied.MaxRequestsPerFlow.Add(quantity)
		copied.MaxBytesPerFlow.Add(quantity)
		require.True(t, band.MaxRequestsPerFlow.Equal(original))
		require.True(t, band.MaxBytesPerFlow.Equal(original))
		require.NotSame(t, band.MaxRequestsPerFlow, copied.MaxRequestsPerFlow)
		require.NotSame(t, band.MaxBytesPerFlow, copied.MaxBytesPerFlow)
	})
	t.Run("v1alpha1 conversion", func(t *testing.T) {
		band := configapiv1alpha1.PriorityBandConfig{MaxRequestsPerFlow: &quantity, MaxBytesPerFlow: &quantity}
		copied := band.DeepCopy()
		converted := convertPriorityBand(band)
		copied.MaxRequestsPerFlow.Add(quantity)
		copied.MaxBytesPerFlow.Add(quantity)
		converted.MaxRequestsPerFlow.Add(quantity)
		converted.MaxBytesPerFlow.Add(quantity)
		require.True(t, band.MaxRequestsPerFlow.Equal(original))
		require.True(t, band.MaxBytesPerFlow.Equal(original))
		require.NotSame(t, band.MaxRequestsPerFlow, converted.MaxRequestsPerFlow)
		require.NotSame(t, band.MaxBytesPerFlow, converted.MaxBytesPerFlow)
	})
}
