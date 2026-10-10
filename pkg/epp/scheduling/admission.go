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

package scheduling

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
)

// FilterForAdmission checks projected capacity without running scheduling plugins.
// Unknown handlers retain their scheduling behavior without this additional gate.
func (s *Scheduler) FilterForAdmission(ctx context.Context, request *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint, filter func(string, []fwksched.Endpoint) []fwksched.Endpoint,
) (bool, map[string][]fwksched.Endpoint) {
	if len(endpoints) == 0 {
		return true, nil
	}
	switch handler := s.profileHandler.(type) {
	case *disagg.Handler:
		return handler.FilterForAdmission(ctx, request, s.profiles, endpoints, filter)
	case *single.SingleProfileHandler:
		if len(s.profiles) != 1 {
			return true, nil
		}
		// Single-profile admission is limited to monolithic decode serving.
		decode := bylabel.NewDecodeRole().Filter(ctx, request, endpoints)
		if len(decode) != len(endpoints) {
			return true, nil
		}
		admitted := filter(flowcontrol.SaturationStageDecode, endpoints)
		if len(admitted) == 0 {
			return false, nil
		}
		for name := range s.profiles {
			return true, map[string][]fwksched.Endpoint{name: admitted}
		}
	default:
		return true, nil
	}
	return true, nil
}

// HasRoleFilter identifies canonical role selection without invoking filters.
func (p *SchedulerProfile) HasRoleFilter(roleType string) bool {
	for _, filter := range p.filters {
		if roleFilter, ok := filter.(*bylabel.RoleFilter); ok && roleFilter.TypedName().Type == roleType {
			return true
		}
	}
	return false
}
