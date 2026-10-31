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
	"math"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
)

// testOverflowLimit is the request limit of testGuardParams: 8 * (1 + 0.5).
const (
	testOverflowLimit = 12
	testGuardName     = "prefill-guard"
	testGuardParams   = `{"maxConcurrency": 8, "headroom": 0.5, "failOpen": false}`
	// testDraws is the number of randomly drawn requests a test screens; with
	// even shares the chance that none draws a given revision is 2^-200.
	testDraws = 200
)

func overflowConfig() Config {
	config := validConfig()
	config.RevisionGating.Overflow = &RevisionOverflow{GuardPluginName: testGuardName}
	return config
}

// newGuardedScreener builds a Screener whose handle holds a concurrency
// detector named testGuardName, configured by guardParams.
func newGuardedScreener(t *testing.T, config Config, guardParams string) (*Screener, error) {
	t.Helper()
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	guard, err := concurrency.ConcurrencyDetectorFactory(testGuardName, json.NewDecoder(strings.NewReader(guardParams)), handle)
	if err != nil {
		t.Fatal(err)
	}
	handle.AddPlugin(testGuardName, guard)
	scope, err := labels.Parse(config.Scope.LabelSelector)
	if err != nil {
		t.Fatal(err)
	}
	screener := newScreener("test-screener", config, scope, handle)
	return screener, screener.resolveOverflowGuard()
}

// loadedEndpoint is an endpoint of revision and role carrying requests in flight.
func loadedEndpoint(name, revision, role string, requests int64) fwksched.Endpoint {
	e := endpoint(name, map[string]string{testRevLabel: revision, testRoleLabel: role, "disaggregatedset.x-k8s.io/name": "my-set"})
	e.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: requests})
	return e
}

// overflowPool has two prefill and two decode endpoints per revision, with the
// given in-flight requests on every prefill endpoint of each revision.
func overflowPool(v1Prefill, v2Prefill int64) []fwksched.Endpoint {
	pool := make([]fwksched.Endpoint, 0, 8)
	for revision, requests := range map[string]int64{"v1": v1Prefill, "v2": v2Prefill} {
		for i := range 2 {
			pool = append(pool,
				loadedEndpoint(fmt.Sprintf("%s-prefill-%d", revision, i), revision, "prefill", requests),
				loadedEndpoint(fmt.Sprintf("%s-decode-%d", revision, i), revision, "decode", 0))
		}
	}
	return pool
}

func seededOverflowScreener(t *testing.T, config Config) *Screener {
	t.Helper()
	screener, err := newGuardedScreener(t, config, testGuardParams)
	if err != nil {
		t.Fatal(err)
	}
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 2, "decode": 2},
		"v2": {"prefill": 2, "decode": 2},
	})
	return screener
}

func screenedRevision(t *testing.T, screener *Screener, request *fwksched.InferenceRequest, pool []fwksched.Endpoint) string {
	t.Helper()
	got := screenCandidates(t, screener, request, pool)
	if len(got) == 0 {
		t.Fatal("empty survivor set")
	}
	return got[0].GetMetadata().Labels[testRevLabel]
}

func overflowCount(from, to string) float64 {
	return testutil.ToFloat64(revisionOverflowTotal.WithLabelValues(PluginType, "test-screener", from, to))
}

// screenDraws screens testDraws fresh requests and counts the serving revisions.
func screenDraws(t *testing.T, screener *Screener, pool []fwksched.Endpoint) map[string]int {
	t.Helper()
	counts := map[string]int{}
	for range testDraws {
		counts[screenedRevision(t, screener, &fwksched.InferenceRequest{Headers: map[string]string{}}, pool)]++
	}
	return counts
}

// assertNoOverflow fails when screening pool moves any request off its drawn
// revision, and requires both revisions to still be drawn.
func assertNoOverflow(t *testing.T, screener *Screener, pool []fwksched.Endpoint) {
	t.Helper()
	before := overflowCount("v1", "v2") + overflowCount("v2", "v1")
	counts := screenDraws(t, screener, pool)
	if after := overflowCount("v1", "v2") + overflowCount("v2", "v1"); after != before {
		t.Fatalf("%v requests overflowed, want none", after-before)
	}
	if counts["v1"] == 0 || counts["v2"] == 0 {
		t.Fatalf("served %v, want both revisions", counts)
	}
}

func TestOverflowMovesRequestsOffAFullRevision(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	before := overflowCount("v1", "v2")
	if counts := screenDraws(t, screener, overflowPool(testOverflowLimit, 0)); counts["v2"] != testDraws {
		t.Fatalf("v1 prefill full: served %v, want every request on v2", counts)
	}
	// Only the requests that drew v1 overflowed.
	if moved := overflowCount("v1", "v2") - before; moved == 0 || moved == testDraws {
		t.Fatalf("overflow metric moved by %v, want between 0 and %d", moved, testDraws)
	}
	// Nothing is stored: once v1 has room, requests draw v1 again.
	assertNoOverflow(t, screener, overflowPool(testOverflowLimit-1, 0))
}

func TestOverflowKeepsTheRevisionWhenAnyEndpointHasRoom(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	pool := overflowPool(testOverflowLimit, 0)
	// One v1 prefill endpoint frees a slot.
	for _, e := range pool {
		if e.GetMetadata().Name == "v1-prefill-0" {
			e.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: testOverflowLimit - 1})
		}
	}
	assertNoOverflow(t, screener, pool)
}

func TestOverflowKeepsTheRevisionWhenEveryRevisionIsFull(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	assertNoOverflow(t, screener, overflowPool(testOverflowLimit, testOverflowLimit))
}

func TestOverflowChecksOnlyTheConfiguredRole(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	pool := overflowPool(0, 0)
	for _, e := range pool {
		if e.GetMetadata().Labels[testRevLabel] == "v1" && e.GetMetadata().Labels[testRoleLabel] == "decode" {
			e.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: testOverflowLimit})
		}
	}
	assertNoOverflow(t, screener, pool)
}

func TestOverflowNeverOverridesAStrictRevisionHeader(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	request := &fwksched.InferenceRequest{Headers: map[string]string{"x-llm-d-disagg-revision": "v1"}}
	if got := screenedRevision(t, screener, request, overflowPool(testOverflowLimit, 0)); got != "v1" {
		t.Fatalf("pinned to v1: served by %s", got)
	}
}

func TestOverflowOffKeepsTheDrawnRevision(t *testing.T) {
	screener := seededOverflowScreener(t, validConfig())
	if counts := screenDraws(t, screener, overflowPool(testOverflowLimit, 0)); counts["v1"] == 0 {
		t.Fatalf("overflow off: served %v, want some requests on the full v1", counts)
	}
}

func TestOverflowUsesTheGuardTokenLimit(t *testing.T) {
	screener, err := newGuardedScreener(t, overflowConfig(),
		`{"concurrencyMode": "hybrid", "maxConcurrency": 8, "maxTokenConcurrency": 1000, "headroom": 0, "failOpen": false}`)
	if err != nil {
		t.Fatal(err)
	}
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 2, "decode": 2},
		"v2": {"prefill": 2, "decode": 2},
	})
	pool := overflowPool(1, 1)
	for _, e := range pool {
		if e.GetMetadata().Labels[testRevLabel] == "v1" && e.GetMetadata().Labels[testRoleLabel] == "prefill" {
			e.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: 1, Tokens: 1000})
		}
	}
	if counts := screenDraws(t, screener, pool); counts["v2"] != testDraws {
		t.Fatalf("v1 prefill at its token limit: served %v, want every request on v2", counts)
	}
}

func TestOverflowGuardMustBeAFailClosedFilter(t *testing.T) {
	if _, err := newGuardedScreener(t, overflowConfig(), `{"maxConcurrency": 8, "failOpen": true}`); err == nil ||
		!strings.Contains(err.Error(), "fails open") {
		t.Fatalf("fail-open guard: err = %v", err)
	}
	config := overflowConfig()
	config.RevisionGating.Overflow.GuardPluginName = "missing"
	if _, err := newGuardedScreener(t, config, testGuardParams); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("missing guard: err = %v", err)
	}
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	handle.AddPlugin(testGuardName, newTestScreener(validConfig()))
	config = overflowConfig()
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	screener := newScreener("test-screener", config, labels.Everything(), handle)
	if err := screener.resolveOverflowGuard(); err == nil || !strings.Contains(err.Error(), "not a scheduling filter") {
		t.Fatalf("non-filter guard: err = %v", err)
	}
}

// A shrinking old revision holding long sessions is the GLM failure mode: flow
// control admits on the fleet average, then a fail-closed prefill filter finds
// the old revision full. With overflow, no request is sent to a full revision
// while the other one has room.
func TestOverflowSendsNoRequestToAFullRevisionWhileAnotherHasRoom(t *testing.T) {
	for _, tc := range []struct {
		name     string
		config   Config
		wantFull bool
	}{
		{"overflow off", validConfig(), true},
		{"overflow on", overflowConfig(), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			screener := seededOverflowScreener(t, tc.config)
			pool := overflowPool(testOverflowLimit, 2)
			if toFull := screenDraws(t, screener, pool)["v1"]; (toFull > 0) != tc.wantFull {
				t.Fatalf("%d of %d requests sent to the full revision", toFull, testDraws)
			}
		})
	}
}

func TestOverflowSpreadsOverTheRemainingRevisionsByShare(t *testing.T) {
	config := overflowConfig()
	config.RevisionGating.Mode = GatingModeMaxRole
	screener, err := newGuardedScreener(t, config, testGuardParams)
	if err != nil {
		t.Fatal(err)
	}
	// a (full) holds 1 of 9 decode Pods; b and c hold 4 each.
	seedCounts(t, screener, map[string]map[string]int{
		"a": {"prefill": 1, "decode": 1},
		"b": {"prefill": 1, "decode": 4},
		"c": {"prefill": 1, "decode": 4},
	})
	pool := []fwksched.Endpoint{
		loadedEndpoint("a-prefill", "a", "prefill", testOverflowLimit),
		loadedEndpoint("b-prefill", "b", "prefill", 0),
		loadedEndpoint("c-prefill", "c", "prefill", 0),
	}
	counts := map[string]int{}
	for range 4000 {
		counts[screenedRevision(t, screener, &fwksched.InferenceRequest{Headers: map[string]string{}}, pool)]++
	}
	if counts["a"] != 0 {
		t.Fatalf("%d requests sent to the full revision", counts["a"])
	}
	if b, c := float64(counts["b"]), float64(counts["c"]); b/(b+c) < 0.45 || b/(b+c) > 0.55 {
		t.Fatalf("overflow split b=%v c=%v, want about even", b, c)
	}
}

func TestDrawWithinRescalesTheChosenSlice(t *testing.T) {
	shares := map[string]float64{"a": 0.1, "b": 0.45, "c": 0.45}
	for _, tc := range []struct{ draw, want float64 }{{0, 0}, {0.05, 0.5}, {0.0999, 0.999}} {
		if got := drawWithin(shares, "a", tc.draw); math.Abs(got-tc.want) > 1e-6 {
			t.Fatalf("drawWithin(a, %v) = %v, want %v", tc.draw, got, tc.want)
		}
	}
	if got := drawWithin(shares, "b", 0.1); got != 0 {
		t.Fatalf("the start of b's slice must map to 0, got %v", got)
	}
}

func TestOverflowCountsOnlyWhenItServes(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	request := &fwksched.InferenceRequest{Headers: map[string]string{"x-llm-d-revision-decision-id": "parallel-1"}}
	// The first request of the decision ID stores its revision while both have room.
	stored := screenedRevision(t, screener, request, overflowPool(0, 0))
	other := map[string]string{"v1": "v2", "v2": "v1"}[stored]
	pool := overflowPool(testOverflowLimit, 0)
	if stored == "v2" {
		pool = overflowPool(0, testOverflowLimit)
	}
	before := overflowCount(stored, other)
	// A sibling sees the stored revision full, but coordination keeps it.
	if got := screenedRevision(t, screener, request, pool); got != stored {
		t.Fatalf("coordinated sibling served by %s, want %s", got, stored)
	}
	if after := overflowCount(stored, other); after != before {
		t.Fatalf("an overflow overridden by coordination was counted: %v -> %v", before, after)
	}
}

func TestOverflowCountsOncePerDecision(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	overflowed := 0
	for i := range testDraws {
		request := &fwksched.InferenceRequest{Headers: map[string]string{"x-llm-d-revision-decision-id": fmt.Sprintf("phases-%d", i)}}
		before := overflowCount("v1", "v2")
		// Both phases of one request see v1 full; only the first decides.
		for phase := range 2 {
			if got := screenedRevision(t, screener, request, overflowPool(testOverflowLimit, 0)); got != "v2" {
				t.Fatalf("phase %d served by %s, want v2", phase, got)
			}
		}
		switch moved := overflowCount("v1", "v2") - before; moved {
		case 0:
		case 1:
			overflowed++
		default:
			t.Fatalf("decision %d counted %v overflows, want at most 1", i, moved)
		}
	}
	if overflowed == 0 {
		t.Fatal("no decision overflowed")
	}
}

func TestOverflowIgnoresOutOfScopeEndpoints(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	// Both revisions' in-scope prefill endpoints are full; only an
	// out-of-scope one, for example not yet released, has room.
	pool := overflowPool(testOverflowLimit, testOverflowLimit)
	unreleased := endpoint("v2-prefill-unreleased", map[string]string{testRevLabel: "v2", testRoleLabel: "prefill"})
	unreleased.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{})
	pool = append(pool, unreleased)
	assertNoOverflow(t, screener, pool)
}

func TestOverflowLeavesARevisionWithNoEndpointsOfTheRole(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	// v1 still holds a share, but its prefill endpoints left the pool.
	var pool []fwksched.Endpoint
	for _, e := range overflowPool(0, 0) {
		if e.GetMetadata().Labels[testRevLabel] == "v1" && e.GetMetadata().Labels[testRoleLabel] == "prefill" {
			continue
		}
		pool = append(pool, e)
	}
	if counts := screenDraws(t, screener, pool); counts["v2"] != testDraws {
		t.Fatalf("v1 has no prefill endpoint in the pool: served %v, want every request on v2", counts)
	}
}

func TestOverflowSkipsNilEndpoints(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	pool := append([]fwksched.Endpoint{nil}, overflowPool(testOverflowLimit, 0)...)
	if counts := screenDraws(t, screener, pool); counts["v2"] != testDraws {
		t.Fatalf("served %v, want every request on v2", counts)
	}
}

func TestOverflowMetricIsDeletedWithItsRevision(t *testing.T) {
	screener := seededOverflowScreener(t, overflowConfig())
	revisionOverflowTotal.DeletePartialMatch(prometheus.Labels{"plugin_name": "test-screener"})
	screenDraws(t, screener, overflowPool(testOverflowLimit, 0))
	if n := testutil.CollectAndCount(revisionOverflowTotal); n != 1 {
		t.Fatalf("want 1 overflow series, got %d", n)
	}
	screener.mu.RLock()
	var v1 []types.NamespacedName
	for key, pod := range screener.pods {
		if pod.revision == "v1" {
			v1 = append(v1, key)
		}
	}
	screener.mu.RUnlock()
	for _, key := range v1 {
		screener.removePod(key)
	}
	if n := testutil.CollectAndCount(revisionOverflowTotal); n != 0 {
		t.Fatalf("overflow series of a removed revision must be deleted, %d left", n)
	}
}

func TestConfigParserExposesTheGuardAsADependency(t *testing.T) {
	parsed, err := ConfigParser(json.NewDecoder(strings.NewReader(
		`{"revisionGating": {"mode": "sum", "overflow": {"guardPluginName": "prefill-guard"}}}`)), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := parsed.(*Config).RevisionGating.Overflow.GuardPluginName; got != "prefill-guard" {
		t.Fatalf("guardPluginName = %q", got)
	}
}
