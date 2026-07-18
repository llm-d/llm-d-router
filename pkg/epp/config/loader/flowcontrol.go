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

	"k8s.io/apimachinery/pkg/api/resource"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits/priorityholdback"
)

// ExplicitPolicyInfo represents an interface to introspect priority holdback policies.
type ExplicitPolicyInfo interface {
	Domain() string
	Ceilings() map[int]float64
}

// buildFlowControlConfig resolves all flow-control policy plugins
// and returns the Config.
func buildFlowControlConfig(
	apiConfig *configapi.FlowControlConfig,
	handle fwkplugin.Handle,
) (*flowcontrol.Config, error) {
	defaults, err := buildPriorityBandPolicyDefaults(handle)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve default flow-control policies: %w", err)
	}

	registryConfig, err := buildRegistryConfig(apiConfig, defaults, handle)
	if err != nil {
		return nil, fmt.Errorf("failed to create registry config: %w", err)
	}

	ctrlCfg, err := controller.NewConfigFromAPI(apiConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create controller config: %w", err)
	}

	usageLimitRef := registry.DefaultUsageLimitPolicyRef
	if apiConfig != nil && apiConfig.UsageLimitPolicyPluginRef != "" {
		usageLimitRef = apiConfig.UsageLimitPolicyPluginRef
	}
	usageLimitPolicy, err := resolvePlugin[fwkfc.UsageLimitPolicy](handle, usageLimitRef)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve usage limit policy: %w", err)
	}

	if phPolicy, ok := usageLimitPolicy.(ExplicitPolicyInfo); ok && phPolicy.Domain() == priorityholdback.DomainExplicit {
		ceilings := phPolicy.Ceilings()
		if apiConfig != nil {
			if apiConfig.DefaultPriorityBand != nil || apiConfig.DefaultNegativePriorityBand != nil {
				return nil, fmt.Errorf("priority holdback policy with explicit domain cannot be used with dynamic priority band templates")
			}
			if _, ok := ceilings[0]; !ok {
				return nil, fmt.Errorf("priority band 0 must have a configured ceiling in explicit domain")
			}
			for _, band := range apiConfig.PriorityBands {
				if _, ok := ceilings[band.Priority]; !ok {
					return nil, fmt.Errorf("priority band %d must have a configured ceiling in explicit domain", band.Priority)
				}
			}
		} else {
			if _, ok := ceilings[0]; !ok {
				return nil, fmt.Errorf("priority band 0 must have a configured ceiling in explicit domain")
			}
		}
	}

	return flowcontrol.NewConfig(ctrlCfg, registryConfig, usageLimitPolicy), nil
}

func buildPriorityBandPolicyDefaults(handle fwkplugin.Handle) (registry.PriorityBandPolicyDefaults, error) {
	orderingPolicy, err := resolvePlugin[fwkfc.OrderingPolicy](handle, registry.DefaultOrderingPolicyRef)
	if err != nil {
		return registry.PriorityBandPolicyDefaults{}, err
	}
	fairnessPolicy, err := resolvePlugin[fwkfc.FairnessPolicy](handle, registry.DefaultFairnessPolicyRef)
	if err != nil {
		return registry.PriorityBandPolicyDefaults{}, err
	}

	return registry.PriorityBandPolicyDefaults{
		OrderingPolicy: orderingPolicy,
		FairnessPolicy: fairnessPolicy,
	}, nil
}

// buildRegistryConfig translates the API flow-control configuration into a
// registry.Config, resolving per-band policy overrides via the handle.
func buildRegistryConfig(
	apiConfig *configapi.FlowControlConfig,
	defaults registry.PriorityBandPolicyDefaults,
	handle fwkplugin.Handle,
) (*registry.Config, error) {
	if apiConfig == nil {
		return registry.NewConfig(defaults)
	}

	opts := make([]registry.ConfigOption, 0, len(apiConfig.PriorityBands)+3)

	maxBytes, err := resolveQuantity(apiConfig.MaxBytes, "global MaxBytes")
	if err != nil {
		return nil, err
	}
	if maxBytes > 0 {
		opts = append(opts, registry.WithMaxBytes(maxBytes))
	}

	maxRequests, err := resolveQuantity(apiConfig.MaxRequests, "global MaxRequests")
	if err != nil {
		return nil, err
	}
	if maxRequests > 0 {
		opts = append(opts, registry.WithMaxRequests(maxRequests))
	}

	if apiConfig.DefaultPriorityBand != nil {
		pb, err := buildPriorityBand(defaults, handle, apiConfig.DefaultPriorityBand, "default priority band")
		if err != nil {
			return nil, err
		}
		opts = append(opts, registry.WithDefaultPriorityBand(pb))
	}

	if apiConfig.DefaultNegativePriorityBand != nil {
		pb, err := buildPriorityBand(defaults, handle, apiConfig.DefaultNegativePriorityBand, "default negative priority band")
		if err != nil {
			return nil, err
		}
		opts = append(opts, registry.WithDefaultNegativePriorityBand(pb))
	}

	for i := range apiConfig.PriorityBands {
		band := &apiConfig.PriorityBands[i]
		label := fmt.Sprintf("priority band %d", band.Priority)
		pb, err := buildPriorityBand(defaults, handle, band, label)
		if err != nil {
			return nil, err
		}
		opts = append(opts, registry.WithPriorityBand(pb))
	}

	return registry.NewConfig(defaults, opts...)
}

// buildPriorityBand translates a single API PriorityBandConfig into a registry.PriorityBandConfig,
// resolving any per-band policy overrides via the handle. The label is used in error messages
// (e.g., "default priority band", "priority band 5").
func buildPriorityBand(
	defaults registry.PriorityBandPolicyDefaults,
	handle fwkplugin.Handle,
	band *configapi.PriorityBandConfig,
	label string,
) (*registry.PriorityBandConfig, error) {
	bandOpts := make([]registry.PriorityBandConfigOption, 0, 4)

	maxBytes, err := resolveQuantity(band.MaxBytes, label+" MaxBytes")
	if err != nil {
		return nil, err
	}
	if maxBytes > 0 {
		bandOpts = append(bandOpts, registry.WithBandMaxBytes(maxBytes))
	}

	maxRequests, err := resolveQuantity(band.MaxRequests, label+" MaxRequests")
	if err != nil {
		return nil, err
	}
	if maxRequests > 0 {
		bandOpts = append(bandOpts, registry.WithBandMaxRequests(maxRequests))
	}

	if band.OrderingPolicyRef != "" {
		policy, err := resolvePlugin[fwkfc.OrderingPolicy](handle, band.OrderingPolicyRef)
		if err != nil {
			return nil, err
		}
		bandOpts = append(bandOpts, registry.WithOrderingPolicy(policy))
	}
	if band.FairnessPolicyRef != "" {
		policy, err := resolvePlugin[fwkfc.FairnessPolicy](handle, band.FairnessPolicyRef)
		if err != nil {
			return nil, err
		}
		bandOpts = append(bandOpts, registry.WithFairnessPolicy(policy))
	}

	pb, err := registry.NewPriorityBandConfig(band.Priority, defaults, bandOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s config: %w", label, err)
	}
	return pb, nil
}

func resolveQuantity(q *resource.Quantity, fieldName string) (uint64, error) {
	if q == nil {
		return 0, nil
	}
	v := q.Value()
	if v < 0 {
		return 0, fmt.Errorf("%s must be non-negative, got %d", fieldName, v)
	}
	return uint64(v), nil
}

// resolvePlugin looks up a plugin by name in the handle and asserts its type.
func resolvePlugin[T fwkplugin.Plugin](handle fwkplugin.Handle, ref string) (T, error) {
	var zero T
	v := handle.Plugin(ref)
	if v == nil {
		return zero, fmt.Errorf("plugin %q not found", ref)
	}
	typed, ok := v.(T)
	if !ok {
		return zero, fmt.Errorf("plugin %q does not implement %T (actual type: %T)", ref, zero, v)
	}
	return typed, nil
}
