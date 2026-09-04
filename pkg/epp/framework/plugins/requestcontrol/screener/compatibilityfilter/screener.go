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

package compatibilityfilter

import (
	"context"
	"math/rand/v2"
	"sort"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Screen filters endpoints to a single compatibility group.
//
// If the request carries the configured header (e.g. forwarded from a prefill
// response), only endpoints matching that value survive (strict pinning).
// Otherwise, endpoints are grouped by compatibility value, groups missing any
// required role are dropped, and one surviving group is chosen at random
// weighted by endpoint count.
func (s *Screener) Screen(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	if request == nil || len(endpoints) == 0 {
		return endpoints
	}

	// Strict pinning: if the request already carries a compatibility value
	// (e.g. decode request forwarded from a prefill response), filter to
	// endpoints with that exact value.
	if pinned := request.Headers[s.config.HeaderName]; pinned != "" {
		return s.filterByValue(endpoints, pinned)
	}

	// Group endpoints by compatibility value.
	groups := s.groupEndpoints(endpoints)
	if len(groups) == 0 {
		return nil
	}

	// Drop groups missing required roles.
	eligible := s.filterCoveredGroups(groups)
	if len(eligible) == 0 {
		logger := log.FromContext(ctx)
		logger.V(logutil.DEBUG).Info("No compatibility group covers all required roles",
			"plugin", s.typedName.Name,
			"groups", len(groups),
			"requireRoles", s.config.RequireRoles,
		)
		return nil
	}

	// Pick one group, weighted by endpoint count.
	chosen := pickGroup(eligible, rand.Float64())

	// Cache the chosen value for ResponseHeader().
	s.mu.Lock()
	s.chosenValues[request.RequestID] = chosen
	s.mu.Unlock()

	return s.filterByValue(endpoints, chosen)
}

type compatGroup struct {
	value    string
	count    int
	roleSeen map[string]bool
}

func (s *Screener) groupEndpoints(endpoints []fwksched.Endpoint) map[string]*compatGroup {
	groups := make(map[string]*compatGroup)
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			continue
		}
		value := s.getCompatValue(ep)
		if value == "" {
			continue
		}
		g, ok := groups[value]
		if !ok {
			g = &compatGroup{value: value, roleSeen: make(map[string]bool)}
			groups[value] = g
		}
		g.count++
		if s.config.RoleLabelKey != "" {
			if role := ep.GetMetadata().Labels[s.config.RoleLabelKey]; role != "" {
				g.roleSeen[role] = true
			}
		}
	}
	return groups
}

func (s *Screener) filterCoveredGroups(groups map[string]*compatGroup) map[string]*compatGroup {
	if len(s.config.RequireRoles) == 0 {
		return groups
	}
	covered := make(map[string]*compatGroup, len(groups))
	for value, g := range groups {
		if hasCoverage(g, s.config.RequireRoles) {
			covered[value] = g
		}
	}
	return covered
}

func hasCoverage(g *compatGroup, requiredRoles []string) bool {
	for _, role := range requiredRoles {
		if !g.roleSeen[role] {
			return false
		}
	}
	return true
}

func (s *Screener) filterByValue(endpoints []fwksched.Endpoint, value string) []fwksched.Endpoint {
	result := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			continue
		}
		if s.getCompatValue(ep) == value {
			result = append(result, ep)
		}
	}
	return result
}

// pickGroup selects one compatibility value from the eligible groups, weighted
// by endpoint count. Deterministic for a given draw value.
func pickGroup(groups map[string]*compatGroup, draw float64) string {
	values := make([]string, 0, len(groups))
	total := 0
	for value, g := range groups {
		values = append(values, value)
		total += g.count
	}
	if total == 0 {
		return ""
	}
	sort.Strings(values)
	if len(values) == 1 {
		return values[0]
	}
	x := draw * float64(total)
	cumulative := 0.0
	for _, value := range values {
		cumulative += float64(groups[value].count)
		if x < cumulative {
			return value
		}
	}
	return values[len(values)-1]
}
