/*
Copyright 2025 The Kubernetes Authors.
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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"

	"github.com/go-logr/logr"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

// ErrUnsupportedReload indicates that a candidate changes startup-only configuration.
var ErrUnsupportedReload = errors.New("unsupported configuration reload")

// BuildSchedulerForReload validates a candidate against startup configuration and reuses the existing plugin handle.
func BuildSchedulerForReload(
	configBytes []byte,
	startupSource *configapiv1.EndpointPickerConfig,
	startupProfiles []configapiv1.SchedulingProfile,
	handle fwkplugin.Handle,
	extraGates ...string,
) (*scheduling.Scheduler, error) {
	if len(configBytes) == 0 {
		return nil, errors.New("configuration file is empty")
	}
	candidate, _, err := LoadRawConfig(configBytes, logr.Discard(), extraGates...)
	if err != nil {
		return nil, err
	}

	if err := validateStaticReloadConfig(candidate, startupSource); err != nil {
		return nil, err
	}
	if err := applySchedulingDefaultsForReload(candidate, startupProfiles, handle); err != nil {
		return nil, fmt.Errorf("scheduling defaults: %w", err)
	}
	if err := ensureSaturationDetectorWithRegistration(candidate, handle, handle.GetAllPluginsWithNames(), false); err != nil {
		return nil, fmt.Errorf("saturation detector defaults: %w", err)
	}
	if err := validateReloadProfileChanges(candidate.SchedulingProfiles, startupProfiles, handle); err != nil {
		return nil, err
	}

	schedulerConfig, err := buildSchedulerConfig(candidate.SchedulingProfiles, handle)
	if err != nil {
		return nil, err
	}
	return scheduling.NewSchedulerWithConfig(schedulerConfig), nil
}

func validateStaticReloadConfig(candidate, startupSource *configapiv1.EndpointPickerConfig) error {
	staticCandidate := candidate.DeepCopy()
	staticCandidate.SchedulingProfiles = nil
	staticStartup := startupSource.DeepCopy()
	staticStartup.SchedulingProfiles = nil
	equal, err := semanticallyEqual(staticCandidate, staticStartup)
	if err != nil {
		return fmt.Errorf("compare reload configuration: %w", err)
	}
	if !equal {
		return fmt.Errorf("%w: fields outside schedulingProfiles changed", ErrUnsupportedReload)
	}
	if len(candidate.SchedulingProfiles) == 0 && len(startupSource.SchedulingProfiles) != 0 {
		return fmt.Errorf("%w: scheduling profiles removed", ErrUnsupportedReload)
	}
	return nil
}

func validateReloadProfileChanges(candidateProfiles, startupProfiles []configapiv1.SchedulingProfile, handle fwkplugin.Handle) error {
	wantNames := make(map[string]struct{}, len(startupProfiles))
	for _, profile := range startupProfiles {
		wantNames[profile.Name] = struct{}{}
	}
	candidateNames := make(map[string]struct{}, len(candidateProfiles))
	for i, profile := range candidateProfiles {
		if _, exists := candidateNames[profile.Name]; exists {
			return fmt.Errorf("schedulingProfiles[%d] has duplicate name '%s'", i, profile.Name)
		}
		candidateNames[profile.Name] = struct{}{}
		for _, pluginRef := range profile.Plugins {
			if pluginRef.Weight == nil {
				continue
			}
			if _, ok := handle.Plugin(pluginRef.PluginRef).(fwksched.Scorer); !ok {
				continue
			}
			if math.IsNaN(*pluginRef.Weight) || math.IsInf(*pluginRef.Weight, 0) {
				return fmt.Errorf("weight for plugin %q in profile %q must be finite", pluginRef.PluginRef, profile.Name)
			}
		}
	}
	if len(candidateNames) != len(wantNames) {
		return fmt.Errorf("%w: scheduling profile names changed", ErrUnsupportedReload)
	}
	for name := range candidateNames {
		if _, ok := wantNames[name]; !ok {
			return fmt.Errorf("%w: scheduling profile names changed", ErrUnsupportedReload)
		}
	}
	return nil
}

func applySchedulingDefaultsForReload(cfg *configapiv1.EndpointPickerConfig, startupProfiles []configapiv1.SchedulingProfile, handle fwkplugin.Handle) error {
	if len(cfg.SchedulingProfiles) == 0 {
		cfg.SchedulingProfiles = make([]configapiv1.SchedulingProfile, len(startupProfiles))
		for i := range startupProfiles {
			cfg.SchedulingProfiles[i] = *startupProfiles[i].DeepCopy()
		}
	}
	// Picker defaults must remain stable across reloads when multiple pickers exist.
	startupPickers := make(map[string]string, len(startupProfiles))
	for _, profile := range startupProfiles {
		for _, pluginRef := range profile.Plugins {
			if _, ok := handle.Plugin(pluginRef.PluginRef).(fwksched.Picker); ok {
				startupPickers[profile.Name] = pluginRef.PluginRef
				break
			}
		}
	}
	for i := range cfg.SchedulingProfiles {
		profile := &cfg.SchedulingProfiles[i]
		hasPicker := false
		for _, pluginRef := range profile.Plugins {
			if _, ok := handle.Plugin(pluginRef.PluginRef).(fwksched.Picker); ok {
				hasPicker = true
				break
			}
		}
		if !hasPicker {
			if picker, ok := startupPickers[profile.Name]; ok {
				profile.Plugins = append(profile.Plugins, configapiv1.SchedulingPlugin{PluginRef: picker})
			}
		}
	}
	return ensureSchedulingLayerWithRegistration(cfg, handle, handle.GetAllPluginsWithNames(), false)
}

func semanticallyEqual(a, b *configapiv1.EndpointPickerConfig) (bool, error) {
	canonical := func(cfg *configapiv1.EndpointPickerConfig) (any, error) {
		data, err := json.Marshal(cfg)
		if err != nil {
			return nil, err
		}
		var value any
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.UseNumber()
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		return value, nil
	}
	left, err := canonical(a)
	if err != nil {
		return false, err
	}
	right, err := canonical(b)
	if err != nil {
		return false, err
	}
	return reflect.DeepEqual(left, right), nil
}
