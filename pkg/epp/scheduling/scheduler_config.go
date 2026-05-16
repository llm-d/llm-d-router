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

package scheduling

import (
	"fmt"

	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

// NewSchedulerConfig creates a new SchedulerConfig object and returns its pointer.
func NewSchedulerConfig(profileHandler fwksched.ProfileHandler, profiles map[string]fwksched.SchedulerProfile) *SchedulerConfig {
	return &SchedulerConfig{
		profileHandler: profileHandler,
		profiles:       profiles,
	}
}

// WithFallbacks records the fallback profile for each primary profile name. The
// scheduler consults this map when a profile's Run sets NoSignal=true; if a
// matching fallback exists, it runs and its result replaces the primary's.
// Mutates and returns the receiver for fluent construction. See #1139.
func (c *SchedulerConfig) WithFallbacks(fallbacks map[string]fwksched.SchedulerProfile) *SchedulerConfig {
	c.fallbacks = fallbacks
	return c
}

// SchedulerConfig provides a configuration for the scheduler which influence routing decisions.
type SchedulerConfig struct {
	profileHandler fwksched.ProfileHandler
	profiles       map[string]fwksched.SchedulerProfile
	fallbacks      map[string]fwksched.SchedulerProfile
}

func (c *SchedulerConfig) String() string {
	return fmt.Sprintf(
		"{ProfileHandler: %s, Profiles: %v}",
		c.profileHandler.TypedName(),
		c.profiles,
	)
}
