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

package bandselection

import (
	"context"
	"slices"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

func TestStrictPolicyFactory(t *testing.T) {
	p, err := StrictPolicyFactory("strict", nil, nil)
	if err != nil {
		t.Fatalf("StrictPolicyFactory returned error: %v", err)
	}
	if _, ok := p.(flowcontrol.BandSelectionPolicy); !ok {
		t.Fatalf("StrictPolicyFactory result %T does not implement BandSelectionPolicy", p)
	}
	tn := p.TypedName()
	if tn.Name != "strict" {
		t.Errorf("TypedName.Name = %q, want %q", tn.Name, "strict")
	}
	if tn.Type != StrictBandSelectionPolicyType {
		t.Errorf("TypedName.Type = %q, want %q", tn.Type, StrictBandSelectionPolicyType)
	}
}

func TestNewStrictPolicyDefaultsName(t *testing.T) {
	if got := NewStrictPolicy("").TypedName().Name; got != StrictBandSelectionPolicyType {
		t.Errorf("TypedName.Name = %q, want %q", got, StrictBandSelectionPolicyType)
	}
}

// Rank must leave the framework's pre-filled identity permutation intact: strict order is exactly the
// order the priorities slice already carries.
func TestStrictPolicyRankPreservesIdentityOrder(t *testing.T) {
	for _, n := range []int{0, 1, 5} {
		priorities := make([]int, n)
		ceilings := make([]float64, n)
		order := make([]int, n)
		want := make([]int, n)
		for i := range priorities {
			priorities[i] = 100 - i*10
			ceilings[i] = 1.0
			order[i] = i
			want[i] = i
		}

		DefaultPolicy().Rank(context.Background(), 0.5, priorities, ceilings, order)

		if !slices.Equal(order, want) {
			t.Errorf("Rank(n=%d) order = %v, want %v", n, order, want)
		}
	}
}

// A policy that writes nothing must not be able to corrupt the buffer it was handed, whatever it
// contained on entry.
func TestStrictPolicyRankIgnoresBufferContents(t *testing.T) {
	order := []int{2, 0, 1}
	DefaultPolicy().Rank(context.Background(), 0.9, []int{100, 50, 10}, []float64{1.0, 0.8, 0.5}, order)

	if !slices.Equal(order, []int{2, 0, 1}) {
		t.Errorf("Rank mutated the order buffer: got %v, want %v", order, []int{2, 0, 1})
	}
}

func TestStrictPolicyRecordDispatch(t *testing.T) {
	// Strict order keeps no accounting, so RecordDispatch is a no-op that must stay callable for any
	// priority, including ones absent from the last Rank call.
	p := DefaultPolicy()
	p.RecordDispatch(context.Background(), 100)
	p.RecordDispatch(context.Background(), -1)
}
