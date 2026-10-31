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

package disaggregatedsetrollout

import (
	"context"
	"errors"
	"math/rand/v2"
	"sort"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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
		} else {
			c.recordSelection(ctx, request, requestedRevision, sourceHeader)
		}
		return result
	}

	if request == nil {
		return nil
	}
	seenRevisions := uniqueRevisions(endpoints, c.revisionLabelKey)
	distribution := c.distributionSnapshot()
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
	source := sourceHeader
	// A forwarded revision header is authoritative. In particular, a decode
	// request in P/D must use the revision stamped by prefill; it must not depend
	// on the two EPPs sharing a CrossReplicaSyncer.
	if chosenRevision == "" {
		var drawn, overflowed string
		drawn, overflowed, source = c.chooseRevision(ctx, request, endpoints, shares)
		chosenRevision = overflowed
		decided := false
		if decisionID := revisionDecisionID(request); distribution.needsCoordination && decisionID != "" && chosenRevision != "" {
			var err error
			chosenRevision, decided, err = c.getOrSetRevision(ctx, decisionID, chosenRevision)
			if err != nil {
				log.FromContext(ctx).Error(err, "failed to coordinate rollout revision")
				return nil
			}
		}
		if decided {
			source = sourceCoordinated
		}
		// Count an overflow once per decision: a later call under the same
		// decision ID gets the stored revision and is not a new overflow.
		if overflowed != drawn && !decided {
			recordRevisionOverflow(c.typedName.Name, drawn, overflowed)
		}
	}
	result := c.filterRevisions(endpoints, allowedRevisions, chosenRevision)
	if len(result) > 0 {
		c.recordSelection(ctx, request, chosenRevision, source)
	} else if requestedRevision != "" {
		recordStrictRevisionNoMatch(c.typedName.Name)
	}
	return result
}

// recordSelection is called only for a selection that kept endpoints, so the
// revision label takes only values of revisions present in the pool.
func (c *Screener) recordSelection(ctx context.Context, request *fwksched.InferenceRequest, revision string, source selectionSource) {
	recordRevisionSelection(c.typedName.Name, revision, source)
	if logger := log.FromContext(ctx).V(logging.DEBUG); logger.Enabled() {
		logger.Info("Selected rollout revision", "requestID", request.RequestID, "revision", revision, "source", source)
	}
}

// chooseRevision returns the drawn revision, the revision after overflow, and
// the source of the draw.
func (c *Screener) chooseRevision(
	ctx context.Context,
	request *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint,
	shares map[string]float64,
) (string, string, selectionSource) {
	if len(shares) <= 1 {
		for revision := range shares {
			return revision, revision, sourceSingle
		}
		return "", "", sourceSingle
	}
	if key, source := c.affinityKey(ctx, request); source != sourceRandom {
		drawn := pickRendezvousRevision(shares, key)
		return drawn, c.overflowRevision(ctx, request, endpoints, shares, drawn, func(withRoom map[string]float64) string {
			return pickRendezvousRevision(withRoom, key)
		}), source
	}
	draw := rand.Float64() //#nosec G404 -- traffic split across revisions, not security-sensitive
	drawn := pickWeightedRevision(shares, draw)
	return drawn, c.overflowRevision(ctx, request, endpoints, shares, drawn, func(withRoom map[string]float64) string {
		return pickWeightedRevision(withRoom, drawWithin(shares, drawn, draw))
	}), sourceRandom
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

// getOrSetRevision returns the revision stored for id, storing candidate when
// none is, and whether a revision was already stored.
func (c *Screener) getOrSetRevision(ctx context.Context, id, candidate string) (string, bool, error) {
	syncer := c.crossReplicaSyncer()
	if syncer == nil {
		// Parallel encode requests can share a decision ID and reach this EPP
		// without an earlier x-llm-d-disagg-revision response to forward. Local state
		// makes them reuse the first revision selected for that decision ID.
		actual, existed := c.localRevisionDecisions.ReadOrWrite(id, localRevisionDecisionStateKey, revisionDecision(candidate))
		revision, ok := actual.(revisionDecision)
		if !ok || revision == "" {
			return "", false, errInvalidRevisionDecision
		}
		return string(revision), existed, nil
	}

	actual, existed, err := syncer.GetOrSet(ctx, c.revisionDecisionStateKey, id, candidate)
	if err != nil {
		return "", false, err
	}
	revision, ok := actual.(string)
	if !ok || revision == "" {
		return "", false, errInvalidRevisionDecision
	}
	return revision, existed, nil
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
