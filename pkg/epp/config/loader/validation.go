/*
Copyright 2025 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// validatePlugins validates the plugins section of the configuration file.
func validatePlugins(configuredPlugins []configapi.PluginSpec) error {
	pluginNames := sets.New[string]()
	for _, spec := range configuredPlugins {
		if spec.Type == "" {
			return fmt.Errorf("plugin '%s' is missing a type", spec.Name)
		}
		if pluginNames.Has(spec.Name) {
			return fmt.Errorf("duplicate plugin name '%s'", spec.Name)
		}
		pluginNames.Insert(spec.Name)

		_, ok := fwkplugin.Registry[spec.Type]
		if !ok {
			return fmt.Errorf("plugin type '%s' is not registered", spec.Type)
		}
	}
	return nil
}

// validateConfig performs a deep validation of the configuration integrity.
// It checks relationships between profiles, plugins, and feature gates.
func validateConfig(cfg *configapi.EndpointPickerConfig) error {
	if err := validateFeatureGates(cfg.FeatureGates); err != nil {
		return fmt.Errorf("feature gate validation failed: %w", err)
	}
	if err := validateSchedulingProfiles(cfg); err != nil {
		return fmt.Errorf("scheduling profile validation failed: %w", err)
	}
	if err := validateSaturationDetector(cfg); err != nil {
		return fmt.Errorf("saturation detector validation failed: %w", err)
	}
	if err := validateParsers(cfg); err != nil {
		return fmt.Errorf("parser validation failed: %w", err)
	}
	return nil
}

func validateParsers(cfg *configapi.EndpointPickerConfig) error {
	if cfg.RequestHandler == nil || len(cfg.RequestHandler.Parsers) == 0 {
		return nil
	}

	definedPlugins := sets.New[string]()
	for _, p := range cfg.Plugins {
		definedPlugins.Insert(p.Name)
	}

	for _, pc := range cfg.RequestHandler.Parsers {
		if !definedPlugins.Has(pc.PluginRef) {
			return fmt.Errorf("parser references undefined plugin '%s'", pc.PluginRef)
		}
	}

	return nil
}

func validateSaturationDetector(cfg *configapi.EndpointPickerConfig) error {
	if cfg.FlowControl == nil || cfg.FlowControl.SaturationDetector == nil {
		return nil
	}
	if cfg.FlowControl.SaturationDetector.PluginRef == "" {
		return errors.New("saturation detector plugin reference is empty")
	}

	definedPlugins := sets.New[string]()
	for _, p := range cfg.Plugins {
		definedPlugins.Insert(p.Name)
	}

	if !definedPlugins.Has(cfg.FlowControl.SaturationDetector.PluginRef) {
		return fmt.Errorf("saturation detector references undefined plugin '%s'", cfg.FlowControl.SaturationDetector.PluginRef)
	}

	return nil
}

func validateSchedulingProfiles(cfg *configapi.EndpointPickerConfig) error {
	definedPlugins := sets.New[string]()
	for _, p := range cfg.Plugins {
		definedPlugins.Insert(p.Name)
	}
	seenProfileNames := sets.New[string]()

	for i, profile := range cfg.SchedulingProfiles {
		if profile.Name == "" {
			return fmt.Errorf("schedulingProfiles[%d] is missing a name", i)
		}
		if seenProfileNames.Has(profile.Name) {
			return fmt.Errorf("schedulingProfiles[%d] has duplicate name '%s'", i, profile.Name)
		}
		seenProfileNames.Insert(profile.Name)

		for j, pluginRef := range profile.Plugins {
			if pluginRef.PluginRef == "" {
				return fmt.Errorf("schedulingProfiles[%s].plugins[%d] is missing a 'pluginRef'", profile.Name, j)
			}

			if !definedPlugins.Has(pluginRef.PluginRef) {
				return fmt.Errorf("schedulingProfiles[%s] references undefined plugin '%s'",
					profile.Name, pluginRef.PluginRef)
			}
		}
	}
	return nil
}

func validateFeatureGates(gates configapi.FeatureGates) error {
	if gates == nil {
		return nil
	}

	registeredFeatureGatesMu.RLock()
	defer registeredFeatureGatesMu.RUnlock()
	for _, gate := range gates {
		parts := strings.Split(gate, "=")
		if _, ok := registeredFeatureGates[parts[0]]; !ok {
			return fmt.Errorf("feature gate '%s' is unknown or unregistered", parts[0])
		}
		if len(parts) > 1 {
			_, err := strconv.ParseBool(strings.ToLower(strings.TrimSpace(parts[1])))
			if err != nil {
				return fmt.Errorf("%s is not a valid value for the feature gate %s (error: %w)", parts[1], parts[0], err)
			}
		}
	}

	return nil
}
