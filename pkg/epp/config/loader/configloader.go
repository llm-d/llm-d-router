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
	"sync"

	"github.com/go-logr/logr"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/epp/config"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

var (
	scheme                       = runtime.NewScheme()
	registeredFeatureGatesMu     sync.RWMutex
	registeredFeatureGates       = sets.New[string]()
	deprecatedSchemeGroupVersion = schema.GroupVersion{Group: "inference.networking.x-k8s.io", Version: "v1alpha1"}
)

func init() {
	// Support deprecated pseudo config CRD
	var builder runtime.SchemeBuilder
	(&builder).Register(func(scheme *runtime.Scheme) error {
		scheme.AddKnownTypes(deprecatedSchemeGroupVersion,
			&configapi.EndpointPickerConfig{},
		)
		// AddToGroupVersion allows the serialization of client types like ListOptions.
		v1.AddToGroupVersion(scheme, deprecatedSchemeGroupVersion)
		return nil
	})

	utilruntime.Must(configapi.Install(scheme))
	utilruntime.Must((&builder).AddToScheme(scheme))

}

// RegisterFeatureGate registers a feature gate name for validation purposes.
func RegisterFeatureGate(gate string) {
	registeredFeatureGatesMu.Lock()
	defer registeredFeatureGatesMu.Unlock()
	registeredFeatureGates.Insert(gate)
}

// LoadRawConfig parses the raw configuration bytes, applies initial defaults, and extracts feature gates.
// It does not instantiate plugins.
func LoadRawConfig(configBytes []byte, logger logr.Logger) (*configapi.EndpointPickerConfig, map[string]bool, error) {
	var rawConfig *configapi.EndpointPickerConfig
	var err error
	if len(configBytes) != 0 {
		rawConfig, err = decodeRawConfig(configBytes)
		if err != nil {
			return nil, nil, err
		}

		if rawConfig.GroupVersionKind().GroupVersion() == deprecatedSchemeGroupVersion {
			logger.Info("DEPRECATION: apiVersion inference.networking.x-k8s.io/v1alpha1/EndpointPickerConfig is deprecated",
				"replacement", "llm-d.ai/v1alpha1/EndpointPickerConfig")
		}

		logger.Info("Loaded raw configuration", "config", rawConfig.String())
	} else {
		logger.Info("A configuration wasn't specified. A default one is being used.")
		rawConfig = loadDefaultConfig()
		logger.Info("Default raw configuration used", "config", rawConfig.String())
	}

	applyStaticDefaults(rawConfig)

	// We validate gates early because they might dictate downstream loading logic.
	if err := validateFeatureGates(rawConfig.FeatureGates); err != nil {
		return nil, nil, fmt.Errorf("feature gate validation failed: %w", err)
	}

	featureConfig := loadFeatureConfig(rawConfig.FeatureGates)
	return rawConfig, featureConfig, nil
}

// InstantiateAndConfigure performs the heavy lifting of plugin instantiation, system architecture injection, and
// scheduler construction.
func InstantiateAndConfigure(
	rawConfig *configapi.EndpointPickerConfig,
	handle fwkplugin.Handle,
	logger logr.Logger,
) (*config.Config, error) {

	if err := instantiatePlugins(rawConfig.Plugins, handle); err != nil {
		return nil, fmt.Errorf("plugin instantiation failed: %w", err)
	}

	if err := applySystemDefaults(rawConfig, handle); err != nil {
		return nil, fmt.Errorf("system default application failed: %w", err)
	}
	logger.Info("Instantiated all plugins and applied system defaults. Effective raw configuration", "config", rawConfig.String())

	if err := validateConfig(rawConfig); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	schedulerConfig, err := buildSchedulerConfig(rawConfig.SchedulingProfiles, handle)
	if err != nil {
		return nil, fmt.Errorf("scheduler config build failed: %w", err)
	}

	featureGates := loadFeatureConfig(rawConfig.FeatureGates)
	var dataConfig *datalayer.Config
	if !featureGates[datalayer.EnableLegacyMetricsFeatureGate] {
		var err error
		dataConfig, err = buildDataLayerConfig(rawConfig.DataLayer, handle)
		if err != nil {
			return nil, fmt.Errorf("data layer config build failed: %w", err)
		}
		if len(dataConfig.Sources) == 0 {
			logger.Info("No data sources configured; metrics collection is disabled")
		}
	}

	var flowControlConfig *flowcontrol.Config
	if featureGates[flowcontrol.FeatureGate] {
		var err error
		flowControlConfig, err = flowcontrol.NewConfigFromAPI(rawConfig.FlowControl, handle)
		if err != nil {
			return nil, fmt.Errorf("failed to load flow control config: %w", err)
		}
	}

	parserConfig, err := buildParserConfig(rawConfig.Parser, handle)
	if err != nil {
		return nil, fmt.Errorf("parse config build failed: %w", err)
	}

	plugin, ok := handle.GetAllPluginsWithNames()[rawConfig.SaturationDetector.PluginRef]
	if !ok {
		return nil, fmt.Errorf("saturation detector plugin '%s' not found", rawConfig.SaturationDetector.PluginRef)
	}
	saturationDetector, ok := plugin.(fwkfc.SaturationDetector)
	if !ok {
		return nil, fmt.Errorf("plugin '%s' is not a fwkfc.SaturationDetector", rawConfig.SaturationDetector.PluginRef)
	}

	return &config.Config{
		SchedulerConfig:    schedulerConfig,
		SaturationDetector: saturationDetector,
		DataConfig:         dataConfig,
		FlowControlConfig:  flowControlConfig,
		ParserConfig:       parserConfig,
	}, nil
}

func decodeRawConfig(configBytes []byte) (*configapi.EndpointPickerConfig, error) {
	cfg := &configapi.EndpointPickerConfig{}
	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)
	if err := runtime.DecodeInto(codecs.UniversalDecoder(), configBytes, cfg); err != nil {
		return nil, fmt.Errorf("failed to decode configuration JSON/YAML: %w", err)
	}
	return cfg, nil
}

func instantiatePlugins(configuredPlugins []configapi.PluginSpec, handle fwkplugin.Handle) error {
	pluginNames := sets.New[string]()
	for _, spec := range configuredPlugins {
		if spec.Type == "" {
			return fmt.Errorf("plugin '%s' is missing a type", spec.Name)
		}
		if pluginNames.Has(spec.Name) {
			return fmt.Errorf("duplicate plugin name '%s'", spec.Name)
		}
		pluginNames.Insert(spec.Name)

		factory, ok := fwkplugin.Registry[spec.Type]
		if !ok {
			return fmt.Errorf("plugin type '%s' is not registered", spec.Type)
		}
		plugin, err := factory(spec.Name, spec.Parameters, handle)
		if err != nil {
			return fmt.Errorf("failed to create plugin '%s' (type: %s): %w", spec.Name, spec.Type, err)
		}

		handle.AddPlugin(spec.Name, plugin)
	}

	return nil
}

func buildSchedulerConfig(
	configProfiles []configapi.SchedulingProfile,
	handle fwkplugin.Handle,
) (*scheduling.SchedulerConfig, error) {

	profiles := make(map[string]fwksched.SchedulerProfile)

	for _, cfgProfile := range configProfiles {
		fwProfile := scheduling.NewSchedulerProfile()

		for _, pluginRef := range cfgProfile.Plugins {
			plugin := handle.Plugin(pluginRef.PluginRef)
			if plugin == nil { // Should be caught by validation, but defensive check.
				return nil, fmt.Errorf(
					"plugin '%s' referenced in profile '%s' not found in handle",
					pluginRef.PluginRef, cfgProfile.Name)
			}

			// Wrap Scorers with weights.
			if scorer, ok := plugin.(fwksched.Scorer); ok {
				weight := DefaultScorerWeight
				if pluginRef.Weight != nil {
					weight = *pluginRef.Weight
				}
				plugin = scheduling.NewWeightedScorer(scorer, weight)
			}

			if err := fwProfile.AddPlugins(plugin); err != nil {
				return nil, fmt.Errorf("failed to add plugin '%s' to profile '%s': %w", pluginRef.PluginRef, cfgProfile.Name, err)
			}
		}
		profiles[cfgProfile.Name] = fwProfile
	}

	var profileHandler fwksched.ProfileHandler
	for name, plugin := range handle.GetAllPluginsWithNames() {
		if ph, ok := plugin.(fwksched.ProfileHandler); ok {
			if profileHandler != nil {
				return nil, fmt.Errorf("multiple profile handlers found ('%s', '%s'); only one is allowed",
					profileHandler.TypedName().Name, name)
			}
			profileHandler = ph
		}
	}

	if profileHandler == nil {
		return nil, errors.New("no profile handler configured")
	}

	if profileHandler.TypedName().Type == single.SingleProfileHandlerType && len(profiles) > 1 {
		return nil, errors.New("SingleProfileHandler cannot support multiple scheduling profiles")
	}

	return scheduling.NewSchedulerConfig(profileHandler, profiles), nil
}

func loadFeatureConfig(gates configapi.FeatureGates) map[string]bool {
	registeredFeatureGatesMu.RLock()
	defer registeredFeatureGatesMu.RUnlock()
	config := make(map[string]bool, len(registeredFeatureGates))
	for gate := range registeredFeatureGates {
		config[gate] = false
	}
	for _, gate := range gates {
		config[gate] = true
	}
	return config
}

func buildParserConfig(rawParserConfig *configapi.ParserConfig, handle fwkplugin.Handle) (*handlers.Config, error) {
	if rawParserConfig == nil {
		return nil, errors.New("parserConfig is not configured")
	}
	plugin, ok := handle.GetAllPluginsWithNames()[rawParserConfig.PluginRef]
	if !ok {
		return nil, errors.New("the configured parser is not loaded")
	}
	v, ok := plugin.(fwkrh.Parser)
	if !ok {
		return nil, errors.New("the specified plugin is not a parser plugin in the config")
	}
	return &handlers.Config{
		Parser: v,
	}, nil
}

func buildDataLayerConfig(rawDataConfig *configapi.DataLayerConfig, handle fwkplugin.Handle) (*datalayer.Config, error) {
	cfg := datalayer.Config{
		Sources: []datalayer.DataSourceConfig{},
	}

	if rawDataConfig == nil { // metrics data collection not enabled and no additional configuration
		return &cfg, nil
	}

	for _, source := range rawDataConfig.Sources {
		if sourcePlugin, ok := handle.Plugin(source.PluginRef).(fwkdl.DataSource); ok {
			sourceConfig := datalayer.DataSourceConfig{
				Plugin:     sourcePlugin,
				Extractors: []fwkdl.ExtractorBase{},
			}
			for _, extractor := range source.Extractors {
				if extractorPlugin, ok := handle.Plugin(extractor.PluginRef).(fwkdl.ExtractorBase); ok {
					sourceConfig.Extractors = append(sourceConfig.Extractors, extractorPlugin)
				} else {
					return nil, fmt.Errorf("the plugin %s is not a fwkdl.ExtractorBase", source.PluginRef)
				}
			}
			cfg.Sources = append(cfg.Sources, sourceConfig)
		} else {
			return nil, fmt.Errorf("the plugin %s is not a fwkdl.DataSource", source.PluginRef)
		}
	}
	return &cfg, nil
}
