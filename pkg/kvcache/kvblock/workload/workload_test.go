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

package workload

import (
	"slices"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// collect drains a generator's stream.
func collect(g *Generator) []Op {
	var ops []Op
	for {
		op, ok := g.Next()
		if !ok {
			return ops
		}
		ops = append(ops, op)
	}
}

func TestGeneratorDeterministic(t *testing.T) {
	a := collect(New(Agentic(8, 2, 16)))
	b := collect(New(Agentic(8, 2, 16)))
	if len(a) == 0 || len(a) != len(b) {
		t.Fatalf("stream length %d vs %d", len(a), len(b))
	}
	for i := range a {
		x, y := a[i], b[i]
		if x.Kind != y.Kind ||
			!slices.Equal(x.EngineKeys, y.EngineKeys) ||
			!slices.Equal(x.RequestKeys, y.RequestKeys) ||
			!slices.Equal(x.Query, y.Query) ||
			x.Pod != y.Pod || x.KeyType != y.KeyType {
			t.Fatalf("op %d differs: %+v vs %+v", i, x, y)
		}
		for j := range x.Entries {
			if x.Entries[j] != y.Entries[j] {
				t.Fatalf("op %d entry %d differs: %+v vs %+v", i, j, x.Entries[j], y.Entries[j])
			}
		}
	}
}

func TestGeneratorStreamShape(t *testing.T) {
	for _, p := range []Profile{Agentic(8, 2, 16), Interactive(16, 1, 4)} {
		ops := collect(New(p))
		var adds, matches int
		for _, op := range ops {
			switch op.Kind {
			case AddOp:
				adds++
				if len(op.RequestKeys) == 0 || len(op.Entries) == 0 {
					t.Fatalf("%s: empty add: %+v", p.Name, op)
				}
				for _, rk := range op.RequestKeys {
					if rk == kvblock.EmptyBlockHash {
						t.Fatalf("%s: empty block hash in add", p.Name)
					}
				}
			case MatchOp:
				matches++
				if len(op.Query) < p.Depth {
					t.Fatalf("%s: match query shorter than depth", p.Name)
				}
			case ClearOp:
				if op.Pod == "" {
					t.Fatalf("%s: clear without pod", p.Name)
				}
			case EvictOp:
				if len(op.RequestKeys) == 0 && len(op.EngineKeys) == 0 {
					t.Fatalf("%s: evict without keys", p.Name)
				}
			}
		}
		if adds == 0 || matches == 0 {
			t.Fatalf("%s: no adds (%d) or matches (%d)", p.Name, adds, matches)
		}
	}
}

func TestGeneratorSubCanonicalGeometry(t *testing.T) {
	p := Agentic(4, 1, 8)
	p.EngineRatio = 4
	found := false
	for _, op := range collect(New(p)) {
		if op.Kind == AddOp && len(op.EngineKeys) == 4*len(op.RequestKeys) {
			found = true
		}
	}
	if !found {
		t.Fatal("no sub-canonical add emitted")
	}
}

func TestGeneratorRolloverReAddsPrefix(t *testing.T) {
	// ClearChance 1 forces every conversation after the first to open with
	// a pod rollover, whose following Add must re-cover the whole prefix.
	p := Agentic(4, 1, 8)
	p.ClearChance = 1
	var cleared bool
	for _, op := range collect(New(p)) {
		switch op.Kind {
		case ClearOp:
			cleared = true
		case AddOp:
			if cleared {
				if want := p.Depth + p.Grow; len(op.RequestKeys) != want {
					t.Fatalf("post-rollover add covered %d keys, want the %d-block prefix plus growth",
						len(op.RequestKeys), want)
				}
				return
			}
		}
	}
	t.Fatal("no rollover emitted")
}
