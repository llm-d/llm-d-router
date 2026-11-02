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
	"encoding/json"
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// noMatchInfo marks an endpoint that carries no PrefixCacheMatchInfo.
const noMatchInfo = -1

func prefixCacheConfig(prefixCache RevisionPrefixCache) Config {
	config := prefixAffinityConfig()
	config.RevisionGating.Affinity.PrefixCache = &prefixCache
	return config
}

// cachedEndpoint is an in-scope endpoint of revision and role whose
// PrefixCacheMatchInfo reports matchBlocks, or none for noMatchInfo.
func cachedEndpoint(name, revision, role string, matchBlocks int) fwksched.Endpoint {
	e := endpoint(name, map[string]string{testRevLabel: revision, testRoleLabel: role, "disaggregatedset.x-k8s.io/name": "my-set"})
	if matchBlocks != noMatchInfo {
		e.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(matchBlocks, 32, 16))
	}
	return e
}

// outOfScopeEndpoint is a decode endpoint of revision outside the Screener's
// scope whose PrefixCacheMatchInfo reports matchBlocks.
func outOfScopeEndpoint(name, revision string, matchBlocks int) fwksched.Endpoint {
	e := endpoint(name, map[string]string{testRevLabel: revision, testRoleLabel: "decode"})
	e.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(matchBlocks, 32, 16))
	return e
}

// cachedPool has one prefill and one decode endpoint per revision; the decode
// endpoint of each revision reports the given matched blocks and the prefill
// endpoint zero, or neither carries match info for noMatchInfo.
func cachedPool(v1Blocks, v2Blocks int) []fwksched.Endpoint {
	prefillBlocks := func(decodeBlocks int) int { return min(decodeBlocks, 0) }
	return []fwksched.Endpoint{
		cachedEndpoint("v1-prefill", "v1", "prefill", prefillBlocks(v1Blocks)),
		cachedEndpoint("v1-decode", "v1", "decode", v1Blocks),
		cachedEndpoint("v2-prefill", "v2", "prefill", prefillBlocks(v2Blocks)),
		cachedEndpoint("v2-decode", "v2", "decode", v2Blocks),
	}
}

// tokenizedRequest is a pre-tokenized generate request with no session
// identifier, which has no affinity key.
func tokenizedRequest(id string) *fwksched.InferenceRequest {
	tokens := []uint32{1, 2, 3}
	return &fwksched.InferenceRequest{
		RequestID: id,
		Headers:   map[string]string{reqcommon.RequestIDHeaderKey: id},
		Body: &fwkrequest.InferenceRequestBody{
			Generate: &fwkrequest.GenerateRequest{TokenIDs: tokens},
			Payload:  fwkrequest.PayloadMap{"token_ids": tokens},
		},
	}
}

func evenRevisions() map[string]map[string]int {
	return map[string]map[string]int{
		"v1": {"prefill": 1, "decode": 1},
		"v2": {"prefill": 1, "decode": 1},
	}
}

func newPrefixCacheScreener(t *testing.T, prefixCache RevisionPrefixCache) *Screener {
	t.Helper()
	screener := newTestScreener(prefixCacheConfig(prefixCache))
	seedCounts(t, screener, evenRevisions())
	return screener
}

func cacheOutcomeCount(outcome cacheOutcome) float64 {
	return testutil.ToFloat64(revisionCacheAffinityTotal.WithLabelValues(PluginType, "test-screener", string(outcome)))
}

// screenRevisions screens n tokenized requests and counts the serving revisions.
func screenRevisions(t *testing.T, screener *Screener, pool []fwksched.Endpoint, n int) map[string]int {
	t.Helper()
	counts := map[string]int{}
	for i := range n {
		counts[screenedRevision(t, screener, tokenizedRequest(fmt.Sprintf("%s-%d", t.Name(), i)), pool)]++
	}
	return counts
}

func TestScreenerPrefixCacheSelectsCachedRevisionForTokenizedRequests(t *testing.T) {
	resetMetrics(t)
	screener := newPrefixCacheScreener(t, RevisionPrefixCache{})
	// Below the share budget: 20 selections lift v2's fraction to about 0.18.
	const requests = 20
	counts := screenRevisions(t, screener, cachedPool(3, 10), requests)
	if counts["v2"] != requests {
		t.Fatalf("want every request on v2, which holds the longest cached prefix, got %v", counts)
	}
	if got := selectionCount("v2", sourceCache); got != requests {
		t.Fatalf("cache selections = %v, want %d", got, requests)
	}
	if got := cacheOutcomeCount(cacheChosen); got != requests {
		t.Fatalf("chosen outcomes = %v, want %d", got, requests)
	}
}

func TestScreenerPrefixCacheOffIgnoresMatchInfo(t *testing.T) {
	resetMetrics(t)
	screener := newTestScreener(prefixAffinityConfig())
	seedCounts(t, screener, evenRevisions())
	counts := screenRevisions(t, screener, cachedPool(0, 10), testDraws)
	if counts["v1"] == 0 || counts["v2"] == 0 {
		t.Fatalf("want the weighted random draw across both revisions, got %v", counts)
	}
	if got := selectionCount("v2", sourceCache) + selectionCount("v1", sourceCache); got != 0 {
		t.Fatalf("cache selections = %v without prefixCache", got)
	}
}

// TestScreenerPrefixCacheKeepsKeyedSelection checks that a request with a text
// conversation prefix keeps its rendezvous revision even when the other
// revision holds a longer cached prefix.
func TestScreenerPrefixCacheKeepsKeyedSelection(t *testing.T) {
	resetMetrics(t)
	screener := newPrefixCacheScreener(t, RevisionPrefixCache{})
	for session := range 50 {
		request := sessionTurns(session)[0]
		want := pickRendezvousRevision(map[string]float64{"v1": 0.5, "v2": 0.5}, mustKey(t, screener, request))
		other := map[string]string{"v1": "v2", "v2": "v1"}[want]
		pool := cachedPool(0, 10)
		if other == "v1" {
			pool = cachedPool(10, 0)
		}
		if got := screenedRevision(t, screener, request, pool); got != want {
			t.Fatalf("session %d: got %s, want rendezvous revision %s", session, got, want)
		}
	}
	if got := cacheOutcomeCount(cacheChosen) + cacheOutcomeCount(cacheNoLead); got != 0 {
		t.Fatalf("cache step ran %v times for keyed requests", got)
	}
}

func TestScreenerPrefixCacheFallsBackToRandom(t *testing.T) {
	tests := []struct {
		name        string
		prefixCache RevisionPrefixCache
		pool        []fwksched.Endpoint
		want        cacheOutcome
	}{
		{name: "tie", pool: cachedPool(6, 6), want: cacheNoLead},
		{name: "lead below the default of two blocks", pool: cachedPool(5, 6), want: cacheNoLead},
		{name: "lead below minLeadBlocks", prefixCache: RevisionPrefixCache{MinLeadBlocks: 8}, pool: cachedPool(3, 10), want: cacheNoLead},
		{name: "no match on any revision", pool: cachedPool(0, 0), want: cacheNoLead},
		{name: "no match info", pool: cachedPool(noMatchInfo, noMatchInfo), want: cacheMissingMatchInfo},
		{
			name: "out-of-scope endpoint does not count",
			pool: append(cachedPool(4, 4), outOfScopeEndpoint("v2-out-of-scope", "v2", 30)),
			want: cacheNoLead,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetMetrics(t)
			screener := newPrefixCacheScreener(t, tt.prefixCache)
			counts := screenRevisions(t, screener, tt.pool, testDraws)
			if counts["v1"] == 0 || counts["v2"] == 0 {
				t.Fatalf("want the weighted random draw across both revisions, got %v", counts)
			}
			if got := cacheOutcomeCount(tt.want); got != testDraws {
				t.Fatalf("%s outcomes = %v, want %d", tt.want, got, testDraws)
			}
			if got := selectionCount("v1", sourceRandom) + selectionCount("v2", sourceRandom); got != testDraws {
				t.Fatalf("random selections = %v, want %d", got, testDraws)
			}
		})
	}
}

// TestScreenerPrefixCacheCountsRevisionsWithoutMatchInfoAsZero checks that a
// covered revision whose endpoints carry no match info trails by the leader's
// full match, and that an uncovered revision is ignored.
func TestScreenerPrefixCacheCountsRevisionsWithoutMatchInfoAsZero(t *testing.T) {
	resetMetrics(t)
	screener := newPrefixCacheScreener(t, RevisionPrefixCache{})
	pool := append(cachedPool(4, noMatchInfo), cachedEndpoint("v3-decode", "v3", "decode", 30))
	if got := screenedRevision(t, screener, tokenizedRequest("r"), pool); got != "v1" {
		t.Fatalf("got %s, want v1", got)
	}
	if got := cacheOutcomeCount(cacheChosen); got != 1 {
		t.Fatalf("chosen outcomes = %v, want 1", got)
	}
}

// TestScreenerPrefixCacheShareBudget models a shared prompt cached only on v1:
// every request finds v1 ahead, and the share budget still sends v2 about its
// share minus the slack.
func TestScreenerPrefixCacheShareBudget(t *testing.T) {
	resetMetrics(t)
	screener := newPrefixCacheScreener(t, RevisionPrefixCache{})
	const requests = 2000
	counts := screenRevisions(t, screener, cachedPool(10, 0), requests)
	if share := float64(counts["v2"]) / requests; share < 0.3 || share > 0.5 {
		t.Fatalf("v2 share = %.3f, want about 0.4 (share 0.5 minus slack 0.1)", share)
	}
	if cacheOutcomeCount(cacheOverBudget) == 0 {
		t.Fatal("want over-budget outcomes once v1 runs above its share")
	}
}

func TestCacheDecisionShareBudget(t *testing.T) {
	shares := map[string]float64{"v1": 0.5, "v2": 0.5}
	matched := map[string]int{"v1": 10}
	tests := []struct {
		name     string
		slack    float64
		fraction float64
		want     cacheOutcome
	}{
		{name: "under budget", slack: 0.1, fraction: 0.55, want: cacheChosen},
		{name: "at budget", slack: 0.1, fraction: 0.6, want: cacheChosen},
		{name: "over budget", slack: 0.1, fraction: 0.61, want: cacheOverBudget},
		{name: "zero slack", slack: 0, fraction: 0.51, want: cacheOverBudget},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			screener := newTestScreener(prefixCacheConfig(RevisionPrefixCache{ShareSlack: &tt.slack}))
			screener.selections.fractions["v1"] = tt.fraction
			if _, got := screener.cacheDecision(matched, true, shares); got != tt.want {
				t.Fatalf("got %s, want %s", got, tt.want)
			}
		})
	}
}

func TestScreenerPrefixCacheBudgetSkipsStrictHeaders(t *testing.T) {
	screener := newPrefixCacheScreener(t, RevisionPrefixCache{})
	for i := range 100 {
		request := tokenizedRequest(fmt.Sprintf("strict-%d", i))
		request.Headers[DefaultRevisionHeader] = "v1"
		screenedRevision(t, screener, request, cachedPool(10, 0))
	}
	if got := screener.selections.fraction("v1"); got != 0 {
		t.Fatalf("v1 fraction = %v after strict header requests, want 0", got)
	}
	screenedRevision(t, screener, tokenizedRequest("chosen"), cachedPool(10, 0))
	if got := screener.selections.fraction("v1"); got != selectionFractionAlpha {
		t.Fatalf("v1 fraction = %v after one cache selection, want %v", got, selectionFractionAlpha)
	}
}

func TestSelectionFractions(t *testing.T) {
	s := newSelectionFractions()
	for range 1000 {
		s.record("v1")
	}
	if got := s.fraction("v1"); got < 0.99 {
		t.Fatalf("v1 fraction = %v after only v1 selections, want about 1", got)
	}
	for range 2000 {
		s.record("v2")
	}
	if got := s.fraction("v2"); got < 0.99 {
		t.Fatalf("v2 fraction = %v after only v2 selections, want about 1", got)
	}
	if _, tracked := s.fractions["v1"]; tracked {
		t.Fatal("v1 is still tracked after decaying below the floor")
	}
}

func TestScreenerPrefixCacheWithOverflow(t *testing.T) {
	resetMetrics(t)
	config := prefixCacheConfig(RevisionPrefixCache{})
	config.RevisionGating.Overflow = &RevisionOverflow{GuardPluginName: testGuardName}
	screener := seededOverflowScreener(t, config)
	pool := []fwksched.Endpoint{
		loadedEndpoint("v1-prefill", "v1", "prefill", 0),
		loadedEndpoint("v2-prefill", "v2", "prefill", testOverflowLimit),
	}
	pool[0].Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(0, 32, 16))
	pool[1].Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(10, 32, 16))
	if got := screenedRevision(t, screener, tokenizedRequest("r"), pool); got != "v1" {
		t.Fatalf("got %s, want v1: the cached revision v2 is full", got)
	}
	if got := overflowCount("v2", "v1"); got != 1 {
		t.Fatalf("overflow v2->v1 = %v, want 1", got)
	}
	if got := selectionCount("v1", sourceCache); got != 1 {
		t.Fatalf("cache selections on v1 = %v, want 1: overflow keeps the source of the draw", got)
	}
}

// TestScreenerPrefixCacheCoordinatesSiblings checks that the cache pick is the
// candidate stored under the decision ID, so a sibling with a different cache
// lead follows it.
func TestScreenerPrefixCacheCoordinatesSiblings(t *testing.T) {
	resetMetrics(t)
	screener := newPrefixCacheScreener(t, RevisionPrefixCache{})
	first := tokenizedRequest("first")
	first.Headers[reqcommon.RevisionDecisionIDHeaderKey] = "decision"
	if got := screenedRevision(t, screener, first, cachedPool(0, 10)); got != "v2" {
		t.Fatalf("first sibling got %s, want v2", got)
	}
	sibling := tokenizedRequest("sibling")
	sibling.Headers[reqcommon.RevisionDecisionIDHeaderKey] = "decision"
	if got := screenedRevision(t, screener, sibling, cachedPool(10, 0)); got != "v2" {
		t.Fatalf("sibling got %s, want the coordinated v2", got)
	}
	if got := selectionCount("v2", sourceCoordinated); got != 1 {
		t.Fatalf("coordinated selections = %v, want 1", got)
	}
}

func TestScreenerPrefixCacheConsumes(t *testing.T) {
	sources := []sessionaffinity.SessionIDSource{{Attribute: "agent-identity"}}
	tests := []struct {
		name         string
		prefixCache  *RevisionPrefixCache
		wantRequired fwkplugin.DataKey
	}{
		{name: "off"},
		{name: "default producer", prefixCache: &RevisionPrefixCache{}, wantRequired: attrprefix.PrefixCacheMatchInfoDataKey},
		{
			name:         "named producer",
			prefixCache:  &RevisionPrefixCache{ProducerName: "my-prefix"},
			wantRequired: attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("my-prefix"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := prefixAffinityConfig(sources...)
			config.RevisionGating.Affinity.PrefixCache = tt.prefixCache
			deps := newTestScreener(config).Consumes()
			if len(deps.Optional) != 1 {
				t.Fatalf("optional = %v, want the agent-identity attribute", deps.Optional)
			}
			if tt.prefixCache == nil {
				if len(deps.Required) != 0 {
					t.Fatalf("required = %v, want none", deps.Required)
				}
				return
			}
			if _, ok := deps.Required[tt.wantRequired]; !ok || len(deps.Required) != 1 {
				t.Fatalf("required = %v, want only %s", deps.Required, tt.wantRequired)
			}
		})
	}
}

type fakeProducer struct {
	name     string
	produces fwkplugin.DataKey
	value    any
	consumes map[fwkplugin.DataKey]any
}

func (p *fakeProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-producer", Name: p.name}
}
func (p *fakeProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.produces: p.value}
}
func (p *fakeProducer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: p.consumes}
}
func (p *fakeProducer) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

// TestScreenerPrefixCacheRunsProducersDuringRequestHandling checks the startup
// wiring: the prefix cache producer is auto-created for the Screener, and it
// and the tokenizer it consumes run during request handling with no
// requestHandler.dataProducers entry, while unrelated producers stay after
// screening.
func TestScreenerPrefixCacheRunsProducersDuringRequestHandling(t *testing.T) {
	type tokens struct{}
	tokensKey := fwkplugin.NewDataKey("test-tokens", "test-tokenizer")
	loadKey := fwkplugin.NewDataKey("test-load", "test-load")
	factories := map[string]fwkplugin.FactoryFunc{
		"test-prefix": func(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
			return &fakeProducer{name: name, produces: attrprefix.PrefixCacheMatchInfoDataKey, value: attrprefix.PrefixCacheMatchInfo{},
				consumes: map[fwkplugin.DataKey]any{tokensKey: tokens{}}}, nil
		},
		"test-tokenizer": func(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
			return &fakeProducer{name: name, produces: tokensKey, value: tokens{}}, nil
		},
	}
	defaults := map[string]string{
		attrprefix.PrefixCacheMatchInfoDataKey.String(): "test-prefix",
		tokensKey.String(): "test-tokenizer",
	}

	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	screener := newTestScreener(prefixCacheConfig(RevisionPrefixCache{}))
	handle.AddPlugin(screener.TypedName().Name, screener)
	handle.AddPlugin("load", &fakeProducer{name: "load", produces: loadKey, value: 0})
	if err := datalayer.CreateMissingDataProducers(context.Background(), defaults, factories, handle); err != nil {
		t.Fatalf("CreateMissingDataProducers: %v", err)
	}
	phases := datalayer.ResolveDataProducerPhases(handle.GetAllPlugins(), nil)
	want := map[string]fwkrh.DataProducerPhase{
		"test-prefix":    fwkrh.DataProducerPhaseRequestHandling,
		"test-tokenizer": fwkrh.DataProducerPhaseRequestHandling,
		"load":           fwkrh.DataProducerPhaseScheduling,
	}
	for name, phase := range want {
		if phases[name] != phase {
			t.Fatalf("phase of %s = %q, want %q (all phases: %v)", name, phases[name], phase, phases)
		}
	}
	if _, err := datalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins(), datalayer.WithDataProducerPhases(phases)); err != nil {
		t.Fatalf("ValidateAndOrderDataDependencies: %v", err)
	}
}
