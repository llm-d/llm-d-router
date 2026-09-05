/*
Copyright 2026 The Kubernetes Authors.

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

package disaggregatedsetrollout

import (
	"context"
	"math/rand/v2"
	"sort"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Screen applies revision gating and strict selectors before scheduling
// profiles observe the endpoint set.
func (c *Screener) Screen(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	current := endpoints
	if c.config.RevisionGating.Active() {
		if request == nil {
			return nil
		}
		seenRevisions := uniqueRevisions(current, c.revisionLabelKey)
		distribution := c.distributionSnapshot()
		// TODO: Make the initial Pod snapshot a condition of EPP readiness.
		// Until a generic readiness hook exists, an empty snapshot fails closed.
		shares := make(map[string]float64, len(seenRevisions))
		allowedRevisions := make(map[string]struct{}, len(seenRevisions))
		for revision := range seenRevisions {
			share := distribution.shares[revision]
			// If one required role is missing, this revision cannot serve a
			// disaggregated request yet. Missing roles and revisions absent
			// from the warmed distribution both resolve to zero and fail closed.
			if share == 0 {
				continue
			}
			shares[revision] = share
			allowedRevisions[revision] = struct{}{}
		}
		chosenRevision := ""
		if !c.hasStrictHeader(request) {
			chosenRevision = pickWeightedRevision(shares, rand.Float64())
		}
		current = c.applyRevisionDecision(current, allowedRevisions, chosenRevision)
		if len(current) == 0 {
			return nil
		}
	}
	return c.screenStrictSelectors(ctx, request, current)
}

func (c *Screener) applyRevisionDecision(
	endpoints []fwksched.Endpoint,
	allowedRevisions map[string]struct{},
	chosenRevision string,
) []fwksched.Endpoint {
	result := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		revision := endpoint.GetMetadata().Labels[c.revisionLabelKey]
		if _, covered := allowedRevisions[revision]; !covered {
			continue
		}
		if chosenRevision != "" && revision != chosenRevision {
			continue
		}
		result = append(result, endpoint)
	}
	return result
}

func (c *Screener) screenStrictSelectors(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	current := endpoints
	if request == nil || len(current) == 0 {
		return current
	}
	for _, selector := range c.config.HeaderSelectors {
		if selector.Mode != ModeStrict {
			continue
		}
		requested := request.Headers[selector.HeaderName]
		if requested == "" {
			continue
		}
		matched := make([]fwksched.Endpoint, 0, len(current))
		for _, endpoint := range current {
			if endpoint == nil || endpoint.GetMetadata() == nil {
				continue
			}
			if endpoint.GetMetadata().Labels[selector.LabelKey] == requested {
				matched = append(matched, endpoint)
			}
		}
		current = matched
		if len(matched) == 0 {
			recordStrictHeaderNoMatch(c.typedName.Name, selector.Name)
		}
		if len(current) == 0 {
			return current
		}
	}
	return current
}

func (c *Screener) hasStrictHeader(request *fwksched.InferenceRequest) bool {
	if request == nil {
		return false
	}
	for _, selector := range c.config.HeaderSelectors {
		if selector.Mode == ModeStrict && request.Headers[selector.HeaderName] != "" {
			return true
		}
	}
	return false
}

func uniqueRevisions(endpoints []fwksched.Endpoint, revisionLabelKey string) map[string]struct{} {
	seen := make(map[string]struct{})
	for _, endpoint := range endpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		if revision := endpoint.GetMetadata().Labels[revisionLabelKey]; revision != "" {
			seen[revision] = struct{}{}
		}
	}
	return seen
}

func revisionSumWeight(perRole map[string]int, required []string) (int, bool) {
	weight := 0
	for _, role := range required {
		count := perRole[role]
		if count == 0 {
			return 0, false
		}
		weight += count
	}
	return weight, true
}

func pickWeightedRevision(shares map[string]float64, draw float64) string {
	revisions := make([]string, 0, len(shares))
	total := 0.0
	for revision, share := range shares {
		revisions = append(revisions, revision)
		total += share
	}
	if total == 0 {
		return ""
	}
	sort.Strings(revisions)
	if len(revisions) == 1 {
		return revisions[0]
	}
	x := draw * total
	cumulative := 0.0
	for _, revision := range revisions {
		cumulative += shares[revision]
		if x < cumulative {
			return revision
		}
	}
	return revisions[len(revisions)-1]
}
