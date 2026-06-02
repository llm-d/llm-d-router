/*
Copyright 2026 The Kubernetes Authors.

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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/roundrobin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/edf"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

func newFlowControlTestHandle(t *testing.T) fwkplugin.Handle {
	t.Helper()
	handle := igwtestutils.NewTestHandle(t.Context())
	handle.AddPlugin(globalstrict.GlobalStrictFairnessPolicyType, &fwkfcmocks.MockFairnessPolicy{
		TypedNameV: fwkplugin.TypedName{
			Type: globalstrict.GlobalStrictFairnessPolicyType,
			Name: globalstrict.GlobalStrictFairnessPolicyType,
		},
	})
	handle.AddPlugin(roundrobin.RoundRobinFairnessPolicyType, &fwkfcmocks.MockFairnessPolicy{
		TypedNameV: fwkplugin.TypedName{
			Type: roundrobin.RoundRobinFairnessPolicyType,
			Name: roundrobin.RoundRobinFairnessPolicyType,
		},
	})
	handle.AddPlugin(fcfs.FCFSOrderingPolicyType, &fwkfcmocks.MockOrderingPolicy{
		TypedNameV: fwkplugin.TypedName{
			Type: fcfs.FCFSOrderingPolicyType,
			Name: fcfs.FCFSOrderingPolicyType,
		},
	})
	handle.AddPlugin(edf.EDFOrderingPolicyType, &fwkfcmocks.MockOrderingPolicy{
		TypedNameV: fwkplugin.TypedName{
			Type: edf.EDFOrderingPolicyType,
			Name: edf.EDFOrderingPolicyType,
		},
		RequiredQueueCapabilitiesV: []fwkfc.QueueCapability{fwkfc.CapabilityPriorityConfigurable},
	})
	handle.AddPlugin(usagelimits.StaticUsageLimitPolicyType, usagelimits.DefaultPolicy())
	return handle
}

func TestBuildRegistryConfig(t *testing.T) {
	t.Parallel()
	handle := newFlowControlTestHandle(t)
	defaults, err := buildPriorityBandPolicyDefaults(handle)
	require.NoError(t, err)

	testCases := []struct {
		name        string
		apiConfig   *configapi.FlowControlConfig
		assertion   func(*testing.T, *registry.Config)
		expectedErr string
	}{
		// --- Happy Paths ---
		{
			name: "ShouldSucceed_WithFullConfiguration",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes: ptr.To(resource.MustParse("100")),
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: ptr.To(resource.MustParse("50")),
					},
				},
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(resource.MustParse("10")),
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				assert.Equal(t, uint64(100), cfg.MaxBytes, "Global MaxBytes should be correctly translated")

				// Verify Explicit Band
				require.Contains(t, cfg.PriorityBands, 1, "Configured priority band should be present")
				assert.Equal(t, uint64(50), cfg.PriorityBands[1].MaxBytes, "Band MaxBytes should be correctly translated")

				// Verify Default Template
				require.NotNil(t, cfg.DefaultPriorityBand, "DefaultPriorityBand should be configured")
				assert.Equal(t, uint64(10), cfg.DefaultPriorityBand.MaxBytes,
					"DefaultPriorityBand template MaxBytes should be translated")
			},
		},
		{
			name: "ShouldSucceed_WithKubernetesQuantityFormat",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes: ptr.To(resource.MustParse("1Gi")),
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: ptr.To(resource.MustParse("500Mi")),
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				assert.Equal(t, uint64(1073741824), cfg.MaxBytes,
					"1Gi should be correctly parsed as 1073741824 bytes")
				require.Contains(t, cfg.PriorityBands, 1)
				assert.Equal(t, uint64(524288000), cfg.PriorityBands[1].MaxBytes,
					"500Mi should be correctly parsed as 524288000 bytes")
			},
		},
		{
			name: "ShouldSucceed_WithPolicyReferences",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority:          1,
						OrderingPolicyRef: edf.EDFOrderingPolicyType,
						FairnessPolicyRef: roundrobin.RoundRobinFairnessPolicyType,
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.Contains(t, cfg.PriorityBands, 1, "Configured priority band should be present")
				band := cfg.PriorityBands[1]
				assert.Equal(t, edf.EDFOrderingPolicyType, band.OrderingPolicy.TypedName().Name,
					"OrderingPolicy should be correctly translated")
				assert.Equal(t, roundrobin.RoundRobinFairnessPolicyType, band.FairnessPolicy.TypedName().Name,
					"FairnessPolicy should be correctly translated")
			},
		},
		{
			name:      "ShouldSucceed_WithNilConfig_AndApplySystemDefaults",
			apiConfig: nil,
			assertion: func(t *testing.T, cfg *registry.Config) {
				assert.Equal(t, uint64(0), cfg.MaxBytes, "Default global limit should be 0 (unlimited)")
				require.NotNil(t, cfg.DefaultPriorityBand,
					"Default priority band template should be initialized automatically")
				assert.Equal(t, uint64(1_000_000_000) /* registry default: 1 GB */, cfg.DefaultPriorityBand.MaxBytes,
					"Default template should use system default capacity")
			},
		},

		// --- Defaulting Logic (Nil vs Zero) ---
		{
			name: "ShouldApplyDefault_WhenBandMaxBytesIsNil",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						// MaxBytes and MaxRequests omitted
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				assert.Equal(t, uint64(1_000_000_000) /* registry default: 1 GB */, cfg.PriorityBands[1].MaxBytes,
					"Omitted MaxBytes (nil) should result in system default capacity (1GB)")
			},
		},
		{
			name: "ShouldApplyDefault_WhenBandMaxBytesIsZero",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: ptr.To(resource.MustParse("0")), // Explicitly zero,
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				assert.Equal(t, uint64(1_000_000_000) /* registry default: 1 GB */, cfg.PriorityBands[1].MaxBytes,
					"Explicit MaxBytes (0) should be treated as 'Use Default' (1GB)")
			},
		},
		{
			name: "ShouldApplyDefault_WhenDefaultPriorityBandMaxBytesIsZero",
			apiConfig: &configapi.FlowControlConfig{
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(resource.MustParse("0")), // Explicitly zero,
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.NotNil(t, cfg.DefaultPriorityBand)
				assert.Equal(t, uint64(1_000_000_000) /* registry default: 1 GB */, cfg.DefaultPriorityBand.MaxBytes,
					"Explicit 0 in DefaultPriorityBand template should be treated as 'Use Default'")
			},
		},

		// --- Validation Errors ---
		{
			name: "ShouldError_WithNegativeGlobalMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes: ptr.To(resource.MustParse("-1")),
			},
			expectedErr: "global MaxBytes must be non-negative",
		},
		{
			name: "ShouldError_WithNegativePriorityBandMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: ptr.To(resource.MustParse("-100")),
					},
				},
			},
			expectedErr: "priority band 1 MaxBytes must be non-negative",
		},
		{
			name: "ShouldError_WithNegativeDefaultPriorityBandMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(resource.MustParse("-5")),
				},
			},
			expectedErr: "default priority band MaxBytes must be non-negative",
		},

		// --- MaxRequests: Happy Paths ---
		{
			name: "ShouldSucceed_WithMaxBytesAndMaxRequests",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes:    ptr.To(resource.MustParse("1Gi")),
				MaxRequests: ptr.To(resource.MustParse("5000")),
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority:    100,
						MaxBytes:    ptr.To(resource.MustParse("1Gi")),
						MaxRequests: ptr.To(resource.MustParse("5000")),
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				assert.Equal(t, uint64(1073741824), cfg.MaxBytes, "Global MaxBytes should be 1Gi")
				assert.Equal(t, uint64(5000), cfg.MaxRequests, "Global MaxRequests should be 5000")

				require.Contains(t, cfg.PriorityBands, 100, "Priority band 100 should be present")
				band := cfg.PriorityBands[100]
				assert.Equal(t, uint64(1073741824), band.MaxBytes, "Band MaxBytes should be 1Gi")
				assert.Equal(t, uint64(5000), band.MaxRequests, "Band MaxRequests should be 5000")
			},
		},
		{
			name: "ShouldSucceed_WithOnlyMaxRequests_NoMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				MaxRequests: ptr.To(resource.MustParse("1000")),
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority:    1,
						MaxRequests: ptr.To(resource.MustParse("500")),
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				assert.Equal(t, uint64(0), cfg.MaxBytes, "Global MaxBytes should be 0 (no global byte limit)")
				assert.Equal(t, uint64(1000), cfg.MaxRequests, "Global MaxRequests should be 1000")

				require.Contains(t, cfg.PriorityBands, 1)
				band := cfg.PriorityBands[1]
				assert.Equal(t, uint64(1_000_000_000) /* registry default: 1 GB */, band.MaxBytes,
					"Band MaxBytes should fall back to system default when not configured")
				assert.Equal(t, uint64(500), band.MaxRequests, "Band MaxRequests should be 500")
			},
		},
		{
			name: "ShouldSucceed_WithDefaultPriorityBandMaxRequests",
			apiConfig: &configapi.FlowControlConfig{
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxRequests: ptr.To(resource.MustParse("200")),
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.NotNil(t, cfg.DefaultPriorityBand)
				assert.Equal(t, uint64(200), cfg.DefaultPriorityBand.MaxRequests,
					"DefaultPriorityBand MaxRequests should be translated")
			},
		},

		// --- MaxRequests: Defaulting Logic ---
		{
			name: "ShouldDefaultToZero_WhenMaxRequestsIsNil",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				assert.Equal(t, uint64(0), cfg.PriorityBands[1].MaxRequests,
					"Omitted MaxRequests should default to 0 (no request limit)")
			},
		},
		{
			name: "ShouldDefaultToZero_WhenMaxRequestsIsZero",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority:    1,
						MaxRequests: ptr.To(resource.MustParse("0")),
					},
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				assert.Equal(t, uint64(0), cfg.PriorityBands[1].MaxRequests,
					"Explicit MaxRequests=0 should remain 0 (no request limit)")
			},
		},

		// --- MaxRequests: Validation Errors ---
		{
			name: "ShouldError_WithNegativeGlobalMaxRequests",
			apiConfig: &configapi.FlowControlConfig{
				MaxRequests: ptr.To(resource.MustParse("-1")),
			},
			expectedErr: "global MaxRequests must be non-negative",
		},
		{
			name: "ShouldError_WithNegativePriorityBandMaxRequests",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority:    1,
						MaxRequests: ptr.To(resource.MustParse("-100")),
					},
				},
			},
			expectedErr: "priority band 1 MaxRequests must be non-negative",
		},
		{
			name: "ShouldError_WithNegativeDefaultPriorityBandMaxRequests",
			apiConfig: &configapi.FlowControlConfig{
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxRequests: ptr.To(resource.MustParse("-5")),
				},
			},
			expectedErr: "default priority band MaxRequests must be non-negative",
		},

		// --- DefaultNegativePriorityBand ---
		{
			name: "ShouldSucceed_WithDefaultNegativePriorityBand",
			apiConfig: &configapi.FlowControlConfig{
				DefaultNegativePriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(resource.MustParse("100")),
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				require.NotNil(t, cfg.DefaultNegativePriorityBand)
				assert.Equal(t, uint64(100), cfg.DefaultNegativePriorityBand.MaxBytes,
					"DefaultNegativePriorityBand MaxBytes should be translated")
				assert.NotNil(t, cfg.DefaultNegativePriorityBand.OrderingPolicy,
					"DefaultNegativePriorityBand should have defaults applied")
			},
		},
		{
			name: "ShouldFallBackToDefaultBand_WhenNegativeBandIsNil",
			apiConfig: &configapi.FlowControlConfig{
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(resource.MustParse("500")),
				},
			},
			assertion: func(t *testing.T, cfg *registry.Config) {
				assert.Nil(t, cfg.DefaultNegativePriorityBand,
					"DefaultNegativePriorityBand should remain nil when not configured")
				require.NotNil(t, cfg.DefaultPriorityBand)
				assert.Equal(t, uint64(500), cfg.DefaultPriorityBand.MaxBytes)
			},
		},
		{
			name: "ShouldError_WithNegativeDefaultNegativePriorityBandMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				DefaultNegativePriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(resource.MustParse("-1")),
				},
			},
			expectedErr: "default negative priority band MaxBytes must be non-negative",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg, err := buildRegistryConfig(tc.apiConfig, defaults, handle)

			if tc.expectedErr != "" {
				require.Error(t, err, "buildRegistryConfig should return an error")
				assert.Contains(t, err.Error(), tc.expectedErr, "Error message should contain expected text")
				assert.Nil(t, cfg, "Config should be nil when error occurs")
			} else {
				require.NoError(t, err, "buildRegistryConfig should not return an error for valid input")
				require.NotNil(t, cfg, "Config should not be nil on success")
				if tc.assertion != nil {
					tc.assertion(t, cfg)
				}
			}
		})
	}
}

// --- Tests for buildFlowControlConfig ---
// Moved from flowcontrol/config_test.go TestNewConfigFromAPI.
// Table-driven to match the original style.

func TestBuildFlowControlConfig(t *testing.T) {
	t.Parallel()
	handle := newFlowControlTestHandle(t)

	const funcPolicyName = "func-policy"
	handle.AddPlugin(funcPolicyName, usagelimits.NewPolicyFunc(funcPolicyName, func(_ context.Context, _ float64, priorities []int) []float64 {
		result := make([]float64, len(priorities))
		for i := range result {
			result[i] = 0.8
		}
		return result
	}))

	const structPolicyName = "struct-policy"
	handle.AddPlugin(structPolicyName, &constantPointEightPolicy{})

	testCases := []struct {
		name      string
		apiConfig *configapi.FlowControlConfig
		assertion func(*testing.T, *flowcontrol.Config)
	}{
		{
			name:      "Success - Nil Config defaults",
			apiConfig: nil,
			assertion: func(t *testing.T, cfg *flowcontrol.Config) {
				assert.NotNil(t, cfg.Registry, "Registry config sub-struct should be initialized even when API config is nil")
				assert.NotNil(t, cfg.Controller,
					"Controller config sub-struct should be initialized even when API config is nil")
				assert.NotZero(t, cfg.Controller.EnqueueChannelBufferSize,
					"Controller should contain default values (EnqueueChannelBufferSize) when API config is nil")
			},
		},
		{
			name: "Success - Explicit Values",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes: ptr.To(resource.MustParse("2048")),
			},
			assertion: func(t *testing.T, cfg *flowcontrol.Config) {
				assert.Equal(t, uint64(2048), cfg.Registry.MaxBytes,
					"MaxBytes should be correctly translated from resource.Quantity in API to uint64 in internal config")
			},
		},
		{
			name:      "Success - Default UsageLimitPolicy when UsageLimit is nil",
			apiConfig: nil,
			assertion: func(t *testing.T, cfg *flowcontrol.Config) {
				require.NotNil(t, cfg.UsageLimitPolicy, "UsageLimitPolicy should be resolved even when not explicitly configured")
				ceilings := cfg.UsageLimitPolicy.ComputeLimit(context.Background(), 0.5, []int{0})
				assert.Equal(t, []float64{1.0}, ceilings, "Default noop policy should return 1.0 (no gating)")
			},
		},
		{
			name: "Success - UsageLimitPolicyPluginRef is resolved",
			apiConfig: &configapi.FlowControlConfig{
				UsageLimitPolicyPluginRef: usagelimits.StaticUsageLimitPolicyType,
			},
			assertion: func(t *testing.T, cfg *flowcontrol.Config) {
				require.NotNil(t, cfg.UsageLimitPolicy, "UsageLimitPolicy should be resolved from the handle")
				ceilings := cfg.UsageLimitPolicy.ComputeLimit(context.Background(), 0.5, []int{0})
				assert.Equal(t, []float64{1.0}, ceilings, "Noop policy should return 1.0 (no gating)")
			},
		},
		{
			name: "Success - Func-based UsageLimitPolicy resolved via PluginRef",
			apiConfig: &configapi.FlowControlConfig{
				UsageLimitPolicyPluginRef: funcPolicyName,
			},
			assertion: func(t *testing.T, cfg *flowcontrol.Config) {
				require.NotNil(t, cfg.UsageLimitPolicy)
				ctx := context.Background()
				for _, tc := range []struct {
					name       string
					priority   int
					saturation float64
				}{
					{"zero saturation", 0, 0.0},
					{"half saturation", 1, 0.5},
					{"full saturation", 5, 1.0},
				} {
					assert.Equal(t, []float64{0.8}, cfg.UsageLimitPolicy.ComputeLimit(ctx, tc.saturation, []int{tc.priority}),
						"func-based policy should return 0.8 at %s", tc.name)
				}
			},
		},
		{
			name: "Success - Struct-based UsageLimitPolicy resolved via PluginRef",
			apiConfig: &configapi.FlowControlConfig{
				UsageLimitPolicyPluginRef: structPolicyName,
			},
			assertion: func(t *testing.T, cfg *flowcontrol.Config) {
				require.NotNil(t, cfg.UsageLimitPolicy)
				ctx := context.Background()
				for _, tc := range []struct {
					name       string
					priority   int
					saturation float64
				}{
					{"zero saturation", 0, 0.0},
					{"half saturation", 1, 0.5},
					{"full saturation", 5, 1.0},
				} {
					assert.Equal(t, []float64{0.8}, cfg.UsageLimitPolicy.ComputeLimit(ctx, tc.saturation, []int{tc.priority}),
						"struct-based policy should return 0.8 at %s", tc.name)
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := buildFlowControlConfig(tc.apiConfig, handle)

			require.NoError(t, err, "buildFlowControlConfig should not return an error for valid configuration")
			require.NotNil(t, cfg, "buildFlowControlConfig should return a non-nil Config object on success")

			if tc.assertion != nil {
				tc.assertion(t, cfg)
			}
		})
	}
}

func TestBuildFlowControlConfig_Errors(t *testing.T) {
	t.Parallel()
	handle := newFlowControlTestHandle(t)

	t.Run("Error - UsageLimitPolicy plugin not found", func(t *testing.T) {
		t.Parallel()
		_, err := buildFlowControlConfig(&configapi.FlowControlConfig{
			UsageLimitPolicyPluginRef: "non-existent-policy",
		}, handle)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "non-existent-policy")
	})
}

func TestBuildPriorityBandPolicyDefaults(t *testing.T) {
	t.Parallel()

	t.Run("ShouldResolveDefaultOrderingAndFairnessPolicies", func(t *testing.T) {
		t.Parallel()
		handle := newFlowControlTestHandle(t)
		defaults, err := buildPriorityBandPolicyDefaults(handle)
		require.NoError(t, err)
		require.NotNil(t, defaults.OrderingPolicy)
		assert.Equal(t, fcfs.FCFSOrderingPolicyType, defaults.OrderingPolicy.TypedName().Name)
		require.NotNil(t, defaults.FairnessPolicy)
		assert.Equal(t, globalstrict.GlobalStrictFairnessPolicyType, defaults.FairnessPolicy.TypedName().Name)
	})
}

// constantPointEightPolicy is a hand-rolled UsageLimitPolicy implementation that always returns 0.8.
// It exists to show that any struct satisfying the interface can be registered and resolved,
// without relying on the usagelimits.NewPolicyFunc helper.
type constantPointEightPolicy struct{}

func (p *constantPointEightPolicy) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: "constant-point-eight-policy-type",
		Name: "constant-point-eight-policy",
	}
}

func (p *constantPointEightPolicy) ComputeLimit(_ context.Context, _ float64, priorities []int) []float64 {
	result := make([]float64, len(priorities))
	for i := range result {
		result[i] = 0.8
	}
	return result
}

var _ fwkfc.UsageLimitPolicy = (*constantPointEightPolicy)(nil)
