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
	"errors"
	"math/rand/v2"
	"sort"

	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const localRevisionDecisionStateKey fwkplugin.StateKey = "revision-decision"

var errInvalidRevisionDecision = errors.New("revision coordination returned an invalid rollout revision")

type revisionDecision string

func (r revisionDecision) Clone() fwkplugin.StateData {
	return r
}

// Screen applies revision gating and strict revision selection before
// scheduling profiles observe the endpoint set.
func (c *Screener) Screen(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	requestedRevision := c.requestedRevision(request)
	if !c.config.RevisionGating.Active() {
		if requestedRevision == "" {
			return endpoints
		}
		result := c.filterRevisions(endpoints, nil, requestedRevision)
		if len(result) == 0 {
			recordStrictRevisionNoMatch(c.typedName.Name)
		}
		return result
	}

	if request == nil {
		return nil
	}
	seenRevisions := uniqueRevisions(endpoints, c.revisionLabelKey)
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
	chosenRevision := requestedRevision
	// A forwarded revision header is authoritative. In particular, a decode
	// request in P/D must use the revision stamped by prefill; it must not depend
	// on the two EPPs sharing a CrossReplicaSyncer.
	if chosenRevision == "" {
		chosenRevision = pickWeightedRevision(shares, rand.Float64())
		if decisionID := revisionDecisionID(request); decisionID != "" && chosenRevision != "" {
			var err error
			chosenRevision, err = c.getOrSetRevision(ctx, decisionID, chosenRevision)
			if err != nil {
				log.FromContext(ctx).Error(err, "failed to coordinate rollout revision")
				return nil
			}
		}
	}
	result := c.filterRevisions(endpoints, allowedRevisions, chosenRevision)
	if requestedRevision != "" && len(result) == 0 {
		recordStrictRevisionNoMatch(c.typedName.Name)
	}
	return result
}

func (c *Screener) requestedRevision(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	return request.Headers[c.revisionHeaderName]
}

func revisionDecisionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	if decisionID := request.Headers[reqcommon.RevisionDecisionIDHeaderKey]; decisionID != "" {
		return decisionID
	}
	// A single downstream request needs no identifier beyond its request ID.
	return request.Headers[reqcommon.RequestIDHeaderKey]
}

func (c *Screener) getOrSetRevision(ctx context.Context, id, candidate string) (string, error) {
	syncer := c.crossReplicaSyncer()
	if syncer == nil {
		// Parallel encode requests can share a decision ID and reach this EPP
		// without an earlier x-llm-d-disagg-revision response to forward. Local state
		// makes them reuse the first revision selected for that decision ID.
		actual, _ := c.localRevisionDecisions.ReadOrWrite(id, localRevisionDecisionStateKey, revisionDecision(candidate))
		revision, ok := actual.(revisionDecision)
		if !ok || revision == "" {
			return "", errInvalidRevisionDecision
		}
		return string(revision), nil
	}

	actual, _, err := syncer.GetOrSet(ctx, c.revisionDecisionStateKey, id, candidate)
	if err != nil {
		return "", err
	}
	revision, ok := actual.(string)
	if !ok || revision == "" {
		return "", errInvalidRevisionDecision
	}
	return revision, nil
}

func (c *Screener) crossReplicaSyncer() fwkdl.CrossReplicaSyncer {
	if c.handle == nil {
		return nil
	}
	syncer, _ := c.handle.CrossReplicaSyncer().(fwkdl.CrossReplicaSyncer)
	return syncer
}

func (c *Screener) filterRevisions(
	endpoints []fwksched.Endpoint,
	allowedRevisions map[string]struct{},
	revisionDecision string,
) []fwksched.Endpoint {
	result := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		revision := endpoint.GetMetadata().Labels[c.revisionLabelKey]
		if allowedRevisions != nil {
			if _, covered := allowedRevisions[revision]; !covered {
				continue
			}
		}
		if revisionDecision != "" && revision != revisionDecision {
			continue
		}
		result = append(result, endpoint)
	}
	return result
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
