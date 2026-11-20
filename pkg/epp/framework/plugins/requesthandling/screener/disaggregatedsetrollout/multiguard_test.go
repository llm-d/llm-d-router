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
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
)

func TestMultiGuardCapacity(t *testing.T) {
	for _, tc := range []struct {
		name           string
		ap, ad, bp, bd int64
		missing        string
		want           string
	}{
		{"decode saturated", 0, 12, 0, 0, "", "v2"},
		{"both available", 0, 0, 0, 0, "", "v1"},
		{"prefill saturated", 12, 0, 0, 0, "", "v2"},
		{"alternative decode full", 12, 0, 0, 12, "", "v1"},
		{"all full", 12, 12, 12, 12, "", "v1"},
		{"missing decode", 0, 0, 0, 0, "decode", "v2"},
		{"missing prefill", 0, 0, 0, 0, "prefill", "v2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := overflowConfig()
			cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: testGuardName}}}
			s := seededOverflowScreener(t, cfg)
			var pool []fwksched.Endpoint
			for _, r := range []string{"v1", "v2"} {
				for _, role := range []string{"prefill", "decode"} {
					if r == "v1" && role == tc.missing {
						continue
					}
					load := tc.ap
					if role == "decode" {
						load = tc.ad
					}
					if r == "v2" {
						load = tc.bp
						if role == "decode" {
							load = tc.bd
						}
					}
					pool = append(pool, loadedEndpoint(r+role, r, role, load))
				}
			}
			got := s.overflowRevision(context.Background(), &fwksched.InferenceRequest{}, pool, map[string]float64{"v1": .5, "v2": .5}, "v1", func(room map[string]float64) string { return pickWeightedRevision(room, .25) })
			if got != tc.want {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
		})
	}
}

func TestMultiGuardConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		valid     bool
	}{
		{"list", `{"guards":[{"role":"prefill","guardPluginName":"g"}]}`, true},
		{"empty", `{"guards":[]}`, false},
		{"null mixed legacy", `{"guards":null,"guardPluginName":"g"}`, false},
		{"case-insensitive mixed legacy", `{"Guards":[{"role":"decode","guardPluginName":"g"}],"Role":""}`, false},
		{"duplicate", `{"guards":[{"role":"prefill","guardPluginName":"g"},{"role":"prefill","guardPluginName":"h"}]}`, false},
		{"mixed", `{"guardPluginName":"g","guards":[{"role":"decode","guardPluginName":"h"}]}`, false},
		{"mixed empty legacy", `{"role":"","guards":[{"role":"decode","guardPluginName":"g"}]}`, false},
		{"missing role", `{"guards":[{"guardPluginName":"g"}]}`, false},
		{"missing name", `{"guards":[{"role":"decode"}]}`, false},
		{"unknown role", `{"guards":[{"role":"encode","guardPluginName":"g"}]}`, false},
		{"unknown field", `{"guards":[{"role":"decode","guardPluginName":"g","typo":1}]}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validConfig()
			var overflow RevisionOverflow
			d := json.NewDecoder(strings.NewReader(tc.raw))
			d.DisallowUnknownFields()
			err := d.Decode(&overflow)
			if err == nil {
				cfg.RevisionGating.Overflow = &overflow
				err = cfg.Validate()
			}
			if (err == nil) != tc.valid {
				t.Fatalf("valid=%v, error=%v", tc.valid, err)
			}
		})
	}
}

func TestMultiGuardSelectionPolicies(t *testing.T) {
	cfg := overflowConfig()
	cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: testGuardName}}}
	s := seededOverflowScreener(t, cfg)
	shares := map[string]float64{"a": .1, "b": .3, "c": .6, "d": .5}
	pool := []fwksched.Endpoint{
		loadedEndpoint("ap", "a", "prefill", 0), loadedEndpoint("ad", "a", "decode", 12),
		loadedEndpoint("bp", "b", "prefill", 0), loadedEndpoint("bd", "b", "decode", 0),
		loadedEndpoint("cp", "c", "prefill", 0), loadedEndpoint("cd", "c", "decode", 0),
		loadedEndpoint("dp", "d", "prefill", 0), loadedEndpoint("dd", "d", "decode", 12),
	}
	eligible := map[string]float64{"b": .3, "c": .6}
	for _, tc := range []struct {
		draw float64
		want string
	}{{0, "b"}, {.32, "b"}, {.34, "c"}, {.99, "c"}} {
		got := s.overflowRevision(t.Context(), &fwksched.InferenceRequest{}, pool, shares, "a", func(room map[string]float64) string {
			if len(room) != 2 || room["b"] != .3 || room["c"] != .6 {
				t.Fatalf("eligible shares=%v", room)
			}
			return pickWeightedRevision(room, tc.draw)
		})
		if got != tc.want {
			t.Fatalf("draw %v got %s want %s", tc.draw, got, tc.want)
		}
	}
	for _, key := range []uint64{0, 1, 42, 123456} {
		want := pickRendezvousRevision(eligible, key)
		for range 3 {
			got := s.overflowRevision(t.Context(), &fwksched.InferenceRequest{}, pool, shares, "a", func(room map[string]float64) string { return pickRendezvousRevision(room, key) })
			if got != want {
				t.Fatalf("key %d got %s want %s", key, got, want)
			}
		}
	}
}

func TestMultiGuardLegacyEquivalent(t *testing.T) {
	legacy := seededOverflowScreener(t, overflowConfig())
	cfg := overflowConfig()
	cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: DefaultOverflowRole, GuardPluginName: testGuardName}}}
	list := seededOverflowScreener(t, cfg)
	for _, load := range []int64{0, 11, 12} {
		for _, s := range []*Screener{legacy, list} {
			want := "v1"
			if load == 12 {
				want = "v2"
			}
			got := s.overflowRevision(t.Context(), &fwksched.InferenceRequest{}, overflowPool(load, 0), map[string]float64{"v1": .5, "v2": .5}, "v1", func(room map[string]float64) string { return pickWeightedRevision(room, .2) })
			if got != want {
				t.Fatalf("load %d got %s want %s", load, got, want)
			}
		}
	}
}

func TestMultiGuardScopeAndRevision(t *testing.T) {
	cfg := overflowConfig()
	cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: testGuardName}}}
	s := seededOverflowScreener(t, cfg)
	outside := endpoint("outside", map[string]string{testRevLabel: "v1", testRoleLabel: "decode"})
	pool := []fwksched.Endpoint{nil, outside, loadedEndpoint("p", "v1", "prefill", 0), loadedEndpoint("other", "wrong", "decode", 0), loadedEndpoint("bp", "v2", "prefill", 0), loadedEndpoint("bd", "v2", "decode", 0)}
	got := s.overflowRevision(t.Context(), &fwksched.InferenceRequest{}, pool, map[string]float64{"v1": .5, "v2": .5}, "v1", func(room map[string]float64) string { return pickWeightedRevision(room, .2) })
	if got != "v2" {
		t.Fatalf("out-of-scope or wrong revision satisfied guard: %s", got)
	}
}

func TestMultiGuardStrictAndStoredDecisions(t *testing.T) {
	cfg := overflowConfig()
	cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: testGuardName}}}
	s := seededOverflowScreener(t, cfg)
	strict := &fwksched.InferenceRequest{Headers: map[string]string{DefaultRevisionHeader: "v1"}}
	if got := screenedRevision(t, s, strict, overflowPool(12, 0)); got != "v1" {
		t.Fatalf("strict revision=%s", got)
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-llm-d-revision-decision-id": "multi-stored"}}
	stored := screenedRevision(t, s, req, overflowPool(0, 0))
	pool := overflowPool(12, 0)
	other := "v2"
	if stored == "v2" {
		pool = overflowPool(0, 12)
		other = "v1"
	}
	before := overflowCount(stored, other)
	if got := screenedRevision(t, s, req, pool); got != stored {
		t.Fatalf("stored %s changed to %s", stored, got)
	}
	if after := overflowCount(stored, other); after != before {
		t.Fatalf("stored decision counted overflow: %v to %v", before, after)
	}
}

func TestMultiGuardSecondGuardValidation(t *testing.T) {
	for _, name := range []string{"missing", "invalid", "fail-open"} {
		t.Run(name, func(t *testing.T) {
			cfg := overflowConfig()
			cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: name}}}
			s, err := newGuardedScreener(t, cfg, testGuardParams)
			if name == "missing" {
				if err == nil {
					t.Fatal("missing second guard accepted")
				}
				return
			}
			if name == "invalid" {
				s.handle.AddPlugin(name, newTestScreener(validConfig()))
			} else {
				guard, err := concurrency.ConcurrencyDetectorFactory(name, json.NewDecoder(strings.NewReader(`{"maxConcurrency":8,"failOpen":true}`)), s.handle)
				if err != nil {
					t.Fatal(err)
				}
				s.handle.AddPlugin(name, guard)
			}
			if err := s.resolveOverflowGuard(); err == nil {
				t.Fatalf("%s second guard accepted", name)
			}
		})
	}
}

func TestMultiGuardConfigRoundTrip(t *testing.T) {
	overflow := RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "decode", GuardPluginName: testGuardName}}}
	data, err := json.Marshal(overflow)
	if err != nil {
		t.Fatal(err)
	}
	var decoded RevisionOverflow
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	cfg := validConfig()
	cfg.RevisionGating.Overflow = &decoded
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestMultiGuardCacheSelection(t *testing.T) {
	for _, saturated := range []bool{false, true} {
		t.Run(fmt.Sprintf("decode-saturated=%v", saturated), func(t *testing.T) {
			resetMetrics(t)
			cfg := prefixCacheConfig(RevisionPrefixCache{})
			cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: testGuardName}}}
			s := seededOverflowScreener(t, cfg)
			load := int64(0)
			if saturated {
				load = testOverflowLimit
			}
			pool := []fwksched.Endpoint{loadedEndpoint("ap", "v1", "prefill", 0), loadedEndpoint("ad", "v1", "decode", load), loadedEndpoint("bp", "v2", "prefill", 0), loadedEndpoint("bd", "v2", "decode", 0)}
			pool[1].Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(10, 32, 16))
			pool[3].Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(3, 32, 16))
			want := "v1"
			if saturated {
				want = "v2"
			}
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{s})
			request, scoped, _ := datalayer.ScopeInvocation(logr.Discard(), fwkrh.ScreenerExtensionPoint, s, tokenizedRequest(t.Name()), pool)
			if got := screenedRevision(t, s, request, scoped); got != want {
				t.Fatalf("selected %s, want %s", got, want)
			}
			if got := selectionCount(want, sourceCache); got != 1 {
				t.Fatalf("cache selection metric=%v", got)
			}
			wantOverflow := float64(0)
			if saturated {
				wantOverflow = 1
			}
			if got := overflowCount("v1", "v2"); got != wantOverflow {
				t.Fatalf("overflow metric=%v, want %v", got, wantOverflow)
			}
			if got := cacheOutcomeCount(cacheChosen); got != 1 {
				t.Fatalf("cache outcome=%v", got)
			}
		})
	}
}

func TestMultiGuardScopedCapacity(t *testing.T) {
	for _, multi := range []bool{false, true} {
		t.Run(fmt.Sprintf("multi=%v", multi), func(t *testing.T) {
			cfg := overflowConfig()
			if multi {
				cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: testGuardName}}}
			}
			s := seededOverflowScreener(t, cfg)
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{s})
			req, scoped, _ := datalayer.ScopeInvocation(logr.Discard(), fwkrh.ScreenerExtensionPoint, s, &fwksched.InferenceRequest{}, overflowPool(testOverflowLimit, 0))
			got := s.overflowRevision(t.Context(), req, scoped, map[string]float64{"v1": .5, "v2": .5}, "v1", func(room map[string]float64) string { return pickWeightedRevision(room, .2) })
			if got != "v2" {
				t.Fatalf("scoped guard selected %s, want v2", got)
			}
		})
	}
}

// TestMultiGuardConsumesPrefersRequired folds guards that disagree on a key, in
// both orders, alongside the prefix cache requirement.
func TestMultiGuardConsumesPrefersRequired(t *testing.T) {
	sharedKey := fwkplugin.NewDataKey("shared", "")
	optionalKey := fwkplugin.NewDataKey("optional", "")
	requires := &fakeGuard{deps: fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{sharedKey: "required"}}}
	offers := &fakeGuard{deps: fwkplugin.DataDependencies{Optional: map[fwkplugin.DataKey]any{sharedKey: "optional", optionalKey: "optional"}}}
	for name, filters := range map[string][]fwksched.Filter{"required-first": {requires, offers}, "optional-first": {offers, requires}} {
		t.Run(name, func(t *testing.T) {
			s := newTestScreener(prefixCacheConfig(RevisionPrefixCache{}))
			s.overflowGuards = []resolvedOverflowGuard{{role: "prefill", filter: filters[0]}, {role: "decode", filter: filters[1]}}
			deps := s.Consumes()
			if _, ok := deps.Required[s.prefixMatchDataKey]; !ok {
				t.Fatalf("Required %v: want the prefix cache key", deps.Required)
			}
			if got := deps.Required[sharedKey]; got != "required" {
				t.Fatalf("Required[shared] = %v, want the requiring guard's witness", got)
			}
			if _, optional := deps.Optional[sharedKey]; optional {
				t.Fatal("a key one guard requires must not also be optional")
			}
			if got := deps.Optional[optionalKey]; got != "optional" {
				t.Fatalf("Optional[optional] = %v, want the guard's witness", got)
			}
		})
	}
}

func TestMultiGuardScopedSelectionPolicies(t *testing.T) {
	for _, mode := range []string{"random", "session", "conversation"} {
		t.Run(mode, func(t *testing.T) {
			cfg := overflowConfig()
			if mode != "random" {
				cfg = prefixAffinityConfig(testSources...)
			}
			cfg.RevisionGating.Overflow = &RevisionOverflow{Guards: []RevisionOverflowGuard{{Role: "prefill", GuardPluginName: testGuardName}, {Role: "decode", GuardPluginName: testGuardName}}}
			s := seededOverflowScreener(t, cfg)
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{s})
			shares := map[string]float64{"v1": .5, "v2": .5}
			for i := range 50 {
				req := &fwksched.InferenceRequest{Headers: map[string]string{}}
				if mode == "session" {
					req = chatRequest(map[string]string{testSessionHeader: fmt.Sprintf("session-%d", i)})
				}
				if mode == "conversation" {
					req = chatRequest(nil, chatMessage("user", fmt.Sprintf("prompt-%d", i)))
				}
				pool := []fwksched.Endpoint{loadedEndpoint("ap", "v1", "prefill", 0), loadedEndpoint("ad", "v1", "decode", testOverflowLimit), loadedEndpoint("bp", "v2", "prefill", 0), loadedEndpoint("bd", "v2", "decode", 0)}
				request, scoped, _ := datalayer.ScopeInvocation(logr.Discard(), fwkrh.ScreenerExtensionPoint, s, req, pool)
				drawn, selected, source := s.chooseRevision(t.Context(), request, scoped, shares)
				if selected != "v2" {
					t.Fatalf("drawn=%s selected=%s source=%s", drawn, selected, source)
				}
				if mode == "random" {
					if source != sourceRandom {
						t.Fatalf("source=%s", source)
					}
					continue
				}
				key := mustKey(t, s, req)
				if want := pickRendezvousRevision(shares, key); drawn != want {
					t.Fatalf("drawn=%s, affinity wants %s", drawn, want)
				}
				pool[1].Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{})
				_, selected, _ = s.chooseRevision(t.Context(), request, scoped, shares)
				if selected != drawn {
					t.Fatalf("available affinity moved %s to %s", drawn, selected)
				}
			}
		})
	}
}
