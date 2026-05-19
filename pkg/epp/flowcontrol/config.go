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

package flowcontrol

import (
	"fmt"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const FeatureGate = "flowControl"

// Config is the top-level configuration for the entire flow control module.
// It embeds the configurations for the controller and the registry, providing a single point of entry for validation
// and initialization.
type Config struct {
	Controller       *controller.Config
	Registry         *registry.Config
	UsageLimitPolicy flowcontrol.UsageLimitPolicy
}

// NewConfigFromAPI creates a new Config by translating the top-level API configuration.
func NewConfigFromAPI(apiConfig *configapi.FlowControlConfig, handle plugin.Handle) (*Config, error) {
	registryConfig, err := registry.NewConfigFromAPI(apiConfig, handle)
	if err != nil {
		return nil, fmt.Errorf("failed to create registry config: %w", err)
	}
	ctrlCfg, err := controller.NewConfigFromAPI(apiConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create controller config: %w", err)
	}
	usageLimitPolicy, err := ensureUsageLimitPolicy(apiConfig, handle)
	if err != nil {
		return nil, err
	}
	cfg := &Config{
		Controller:       ctrlCfg,
		Registry:         registryConfig,
		UsageLimitPolicy: usageLimitPolicy,
	}
	return cfg, nil
}

func ensureUsageLimitPolicy(apiConfig *configapi.FlowControlConfig, handle plugin.Handle) (flowcontrol.UsageLimitPolicy, error) {
	ref := registry.DefaultUsageLimitPolicyRef
	if apiConfig != nil && apiConfig.UsageLimitPolicyPluginRef != "" {
		ref = apiConfig.UsageLimitPolicyPluginRef
	}
	p := handle.Plugin(ref)
	if p == nil {
		return nil, fmt.Errorf("usage limit policy plugin '%s' not found", ref)
	}
	usageLimitPolicy, ok := p.(flowcontrol.UsageLimitPolicy)
	if !ok {
		return nil, fmt.Errorf("plugin '%s' does not implement UsageLimitPolicy", ref)
	}
	return usageLimitPolicy, nil
}
