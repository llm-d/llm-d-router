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
	"sync"

	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// cacheOutcome labels the result of the prefix cache step.
type cacheOutcome string

const (
	// cacheChosen: the revision with the longest cached prefix was selected.
	cacheChosen cacheOutcome = "chosen"
	// cacheNoLead: no revision led every other by MinLeadBlocks.
	cacheNoLead cacheOutcome = "no_lead"
	// cacheOverBudget: the leading revision ran above its share plus
	// ShareSlack.
	cacheOverBudget cacheOutcome = "over_budget"
	// cacheMissingMatchInfo: no covered endpoint carried PrefixCacheMatchInfo.
	cacheMissingMatchInfo cacheOutcome = "missing_match_info"
)

// cacheRevision returns the covered revision whose in-scope endpoints hold the
// most matched prefix blocks, or "" when prefix cache affinity is off, the
// revision does not lead every other covered revision by MinLeadBlocks, or it
// is over its share budget.
func (c *Screener) cacheRevision(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint,
	shares map[string]float64) string {
	if c.prefixCache == nil {
		return ""
	}
	matched, found := c.matchedBlocks(endpoints, shares)
	revision, outcome := c.cacheDecision(matched, found, shares)
	recordCacheOutcome(c.typedName.Name, outcome)
	if logger := log.FromContext(ctx).V(logging.DEBUG); logger.Enabled() {
		logger.Info("Rollout revision prefix cache lookup", "requestID", request.RequestID,
			"matchedBlocks", matched, "outcome", outcome)
	}
	if outcome != cacheChosen {
		return ""
	}
	return revision
}

// matchedBlocks returns the most matched blocks on any in-scope endpoint of
// each covered revision, prefill and decode alike, and whether any such
// endpoint carried PrefixCacheMatchInfo.
func (c *Screener) matchedBlocks(endpoints []fwksched.Endpoint, shares map[string]float64) (map[string]int, bool) {
	matched := make(map[string]int, len(shares))
	found := false
	for _, endpoint := range endpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		endpointLabels := endpoint.GetMetadata().Labels
		revision := endpointLabels[c.revisionLabelKey]
		if _, covered := shares[revision]; !covered {
			continue
		}
		if c.scope != nil && !c.scope.Matches(labels.Set(endpointLabels)) {
			continue
		}
		raw, ok := endpoint.Get(c.prefixMatchDataKey)
		if !ok {
			continue
		}
		info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
		if !ok || info == nil {
			continue
		}
		found = true
		matched[revision] = max(matched[revision], info.MatchBlocks())
	}
	return matched, found
}

func (c *Screener) cacheDecision(matched map[string]int, found bool, shares map[string]float64) (string, cacheOutcome) {
	if !found {
		return "", cacheMissingMatchInfo
	}
	// A covered revision with no matching endpoint counts as zero blocks.
	best, bestBlocks, runnerUpBlocks := "", -1, -1
	for revision := range shares {
		switch blocks := matched[revision]; {
		case blocks > bestBlocks:
			best, bestBlocks, runnerUpBlocks = revision, blocks, bestBlocks
		case blocks > runnerUpBlocks:
			runnerUpBlocks = blocks
		}
	}
	if bestBlocks-runnerUpBlocks < c.prefixCache.MinLeadBlocks {
		return "", cacheNoLead
	}
	// Without the budget, a prefix shared by many sessions (such as a system
	// prompt) cached only on the old revision would keep the new revision from
	// ever receiving that traffic.
	if c.selections.fraction(best) > shares[best]+*c.prefixCache.ShareSlack {
		return "", cacheOverBudget
	}
	return best, cacheChosen
}

// selectionFractionAlpha weighs each selection in the moving average, which
// covers roughly the last 1/alpha selections.
const selectionFractionAlpha = 0.01

// selectionFractionFloor drops a revision once its fraction decays below it,
// so revisions that left the pool do not accumulate.
const selectionFractionFloor = 1e-6

// selectionFractions tracks an exponential moving average of the fraction of
// the Screener's own revision selections that went to each revision.
type selectionFractions struct {
	mu        sync.Mutex
	fractions map[string]float64
}

func newSelectionFractions() *selectionFractions {
	return &selectionFractions{fractions: map[string]float64{}}
}

func (s *selectionFractions) record(revision string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for r, fraction := range s.fractions {
		fraction *= 1 - selectionFractionAlpha
		if fraction < selectionFractionFloor {
			delete(s.fractions, r)
			continue
		}
		s.fractions[r] = fraction
	}
	s.fractions[revision] += selectionFractionAlpha
}

func (s *selectionFractions) fraction(revision string) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.fractions[revision]
}
