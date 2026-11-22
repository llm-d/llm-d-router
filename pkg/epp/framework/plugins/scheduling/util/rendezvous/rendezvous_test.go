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

package rendezvous

import (
	"context"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestOverflow(t *testing.T) {
	shares := map[string]float64{"a": 1, "b": 1, "c": 1}
	pickFirst := func(withRoom map[string]float64) string {
		best := ""
		for target := range withRoom {
			if best == "" || target < best {
				best = target
			}
		}
		return best
	}
	room := func(names ...string) func(string) bool {
		set := map[string]bool{}
		for _, n := range names {
			set[n] = true
		}
		return func(t string) bool { return set[t] }
	}
	for name, tc := range map[string]struct {
		chosen  string
		hasRoom func(string) bool
		want    string
	}{
		"chosen has room":         {"a", room("a", "b"), "a"},
		"chosen full, others":     {"a", room("b", "c"), "b"},
		"chosen full, one other":  {"a", room("c"), "c"},
		"all full keeps chosen":   {"a", room(), "a"},
		"empty chosen unchanged":  {"", room("a"), ""},
		"never returns chosen as": {"b", room("b"), "b"},
	} {
		if got := Overflow(tc.chosen, shares, tc.hasRoom, pickFirst); got != tc.want {
			t.Errorf("%s: got %q, want %q", name, got, tc.want)
		}
	}
	// The alternative set never contains the chosen target.
	Overflow("a", shares, room("b"), func(withRoom map[string]float64) string {
		if _, ok := withRoom["a"]; ok {
			t.Error("chosen target offered as its own alternative")
		}
		return ""
	})
}

func TestPickIsStableWhenOthersLeave(t *testing.T) {
	full := map[string]float64{"a": 2, "b": 1, "c": 1}
	for key := range uint64(2000) {
		before := Pick(full, key)
		without := map[string]float64{}
		for target, share := range full {
			if target != "c" {
				without[target] = share
			}
		}
		if before != "c" && Pick(without, key) != before {
			t.Fatalf("key %d on %s moved when c left", key, before)
		}
	}
}

type namedPlugin struct{ name string }

func (p namedPlugin) TypedName() plugin.TypedName { return plugin.TypedName{Type: "t", Name: p.name} }

type filterPlugin struct{ namedPlugin }

func (filterPlugin) Filter(_ context.Context, _ *fwksched.InferenceRequest, e []fwksched.Endpoint) []fwksched.Endpoint {
	return e
}

type failOpenFilter struct{ filterPlugin }

func (failOpenFilter) FailsOpen() bool { return true }

func TestResolveGuard(t *testing.T) {
	handle := utils.NewTestHandle(t.Context())
	handle.AddPlugin("guard", filterPlugin{namedPlugin{"guard"}})
	handle.AddPlugin("not-a-filter", namedPlugin{"not-a-filter"})
	handle.AddPlugin("open", failOpenFilter{filterPlugin{namedPlugin{"open"}}})

	if g, err := ResolveGuard(handle, "guard", "f"); err != nil || g == nil {
		t.Fatalf("guard: %v", err)
	}
	for name, want := range map[string]string{
		"missing":      "not found",
		"not-a-filter": "is not a scheduling filter",
		"open":         "fails open",
	} {
		_, err := ResolveGuard(handle, name, "f")
		if err == nil || !strings.Contains(err.Error(), want) || !strings.HasPrefix(err.Error(), "f: ") {
			t.Errorf("%s: got %v, want %q", name, err, want)
		}
	}
	if _, err := ResolveGuard(nil, "guard", "f"); err == nil {
		t.Error("nil handle accepted")
	}
}
