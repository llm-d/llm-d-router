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

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
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

func (c *Config) String() string {
	if c == nil {
		return "<nil>"
	}
	// Define a local type definition to prevent infinite recursion when calling Sprintf("%+v").
	// A new type definition inherits the struct fields but does not copy its methods,
	// bypassing the Stringer check and allowing a safe reflection-based field dump.
	type temp Config
	return fmt.Sprintf("%+v", temp(*c))
}

// NewConfig constructs a Config from pre-resolved components.
// All plugin resolution is performed by the config loader before calling this constructor.
func NewConfig(ctrl *controller.Config, reg *registry.Config, ulp flowcontrol.UsageLimitPolicy) *Config {
	return &Config{
		Controller:       ctrl,
		Registry:         reg,
		UsageLimitPolicy: ulp,
	}
}
