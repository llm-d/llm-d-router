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

package scheduling

import (
	"fmt"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// NewSchedulerConfig creates a new SchedulerConfig object and returns its pointer.
func NewSchedulerConfig(profileHandler fwksched.ProfileHandler, profiles map[string]fwksched.SchedulerProfile) *SchedulerConfig {
	return &SchedulerConfig{
		profileHandler: profileHandler,
		profiles:       profiles,
	}
}

// SchedulerConfig provides a configuration for the scheduler which influence routing decisions.
type SchedulerConfig struct {
	profileHandler fwksched.ProfileHandler
	profiles       map[string]fwksched.SchedulerProfile
	// reservers are consulted after each profile run to reserve the picked
	// endpoint for the request, so concurrently-scheduled requests observe
	// each pick immediately. Wired at construction; empty means no reserving.
	reservers []fwkrh.EndpointReserver
}

// WithReservers sets the EndpointReserver implementations the scheduler
// consults after each profile run.
func (c *SchedulerConfig) WithReservers(reservers []fwkrh.EndpointReserver) *SchedulerConfig {
	c.reservers = reservers
	return c
}

func (c *SchedulerConfig) String() string {
	return fmt.Sprintf(
		"{ProfileHandler: %s, Profiles: %v}",
		c.profileHandler.TypedName(),
		c.profiles,
	)
}
