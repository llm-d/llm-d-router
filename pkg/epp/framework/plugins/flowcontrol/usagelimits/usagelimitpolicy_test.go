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

package usagelimits

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestStaticPolicyFactory(t *testing.T) {
	t.Parallel()
	p, err := StaticPolicyFactory("custom", nil, nil)
	if err != nil {
		t.Fatalf("StaticPolicyFactory returned error: %v", err)
	}
	policy, ok := p.(flowcontrol.UsageLimitPolicy)
	if !ok {
		t.Fatalf("StaticPolicyFactory result %T does not implement UsageLimitPolicy", p)
	}
	tn := policy.TypedName()
	if tn.Name != "custom" {
		t.Errorf("TypedName.Name = %q, want %q", tn.Name, "custom")
	}
	if tn.Type != StaticUsageLimitPolicyType {
		t.Errorf("TypedName.Type = %q, want %q", tn.Type, StaticUsageLimitPolicyType)
	}

	ceilings := make([]float64, 3)
	policy.ComputeLimit(context.Background(), 0.5, []int{100, 50, 10}, ceilings)
	if want := []float64{1.0, 1.0, 1.0}; !slices.Equal(ceilings, want) {
		t.Errorf("ComputeLimit ceilings = %v, want %v", ceilings, want)
	}
}

func TestStaticPolicyFactoryEmptyNameDefaultsToType(t *testing.T) {
	t.Parallel()
	p, err := StaticPolicyFactory("", nil, nil)
	if err != nil {
		t.Fatalf("StaticPolicyFactory returned error: %v", err)
	}
	tn := p.TypedName()
	if tn.Name != StaticUsageLimitPolicyType {
		t.Errorf("TypedName.Name = %q, want %q", tn.Name, StaticUsageLimitPolicyType)
	}
	if tn.Type != StaticUsageLimitPolicyType {
		t.Errorf("TypedName.Type = %q, want %q", tn.Type, StaticUsageLimitPolicyType)
	}
}

func TestStaticPolicyFactoryWithThreshold(t *testing.T) {
	t.Parallel()
	dec := json.NewDecoder(bytes.NewReader([]byte(`{"threshold": 0.75}`)))
	p, err := StaticPolicyFactory("custom-threshold", dec, nil)
	if err != nil {
		t.Fatalf("StaticPolicyFactory returned error: %v", err)
	}
	policy := p.(flowcontrol.UsageLimitPolicy)
	ceilings := make([]float64, 2)
	policy.ComputeLimit(context.Background(), 0.9, []int{10, 0}, ceilings)
	if want := []float64{0.75, 0.75}; !slices.Equal(ceilings, want) {
		t.Errorf("ComputeLimit ceilings = %v, want %v", ceilings, want)
	}
}

func TestStaticPolicyFactoryInvalidJSON(t *testing.T) {
	t.Parallel()
	dec := json.NewDecoder(bytes.NewReader([]byte(`{not-json`)))
	if _, err := StaticPolicyFactory("bad", dec, nil); err == nil {
		t.Fatal("StaticPolicyFactory with invalid JSON should return an error")
	}
}

func TestStaticPolicyFactoryAcceptsThreshold(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		raw  string
		want float64
	}{
		{name: "nil_decoder", raw: "", want: 1.0},
		{name: "threshold_omitted", raw: `{}`, want: 1.0},
		{name: "upper_bound", raw: `{"threshold": 1.0}`, want: 1.0},
		{name: "fraction", raw: `{"threshold": 0.8}`, want: 0.8},
		{name: "small_positive", raw: `{"threshold": 0.01}`, want: 0.01},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			p, err := StaticPolicyFactory("static", plugin.StrictDecoder([]byte(tc.raw)), nil)
			if err != nil {
				t.Fatalf("StaticPolicyFactory returned error: %v", err)
			}
			policy, ok := p.(flowcontrol.UsageLimitPolicy)
			if !ok {
				t.Fatalf("StaticPolicyFactory result %T does not implement UsageLimitPolicy", p)
			}
			ceilings := make([]float64, 2)
			policy.ComputeLimit(t.Context(), 0.5, []int{10, 0}, ceilings)
			if want := []float64{tc.want, tc.want}; !slices.Equal(ceilings, want) {
				t.Errorf("ComputeLimit ceilings = %v, want %v", ceilings, want)
			}
		})
	}
}

func TestStaticPolicyFactoryRejectsInvalidConfig(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		raw  string
	}{
		{name: "zero", raw: `{"threshold": 0.0}`},
		{name: "negative", raw: `{"threshold": -0.1}`},
		{name: "above_one", raw: `{"threshold": 1.1}`},
		{name: "percentage", raw: `{"threshold": 80}`},
		{name: "unknown_field", raw: `{"threshold": 0.8, "unknown": true}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			p, err := StaticPolicyFactory("static", plugin.StrictDecoder([]byte(tc.raw)), nil)
			if err == nil {
				t.Fatal("StaticPolicyFactory returned nil error, want error")
			}
			if p != nil {
				t.Errorf("StaticPolicyFactory returned plugin %v, want nil", p)
			}
		})
	}
}

func TestDefaultPolicy(t *testing.T) {
	t.Parallel()
	p := DefaultPolicy()
	tn := p.TypedName()
	if tn.Name != StaticUsageLimitPolicyType {
		t.Errorf("TypedName.Name = %q, want %q", tn.Name, StaticUsageLimitPolicyType)
	}
	if tn.Type != StaticUsageLimitPolicyType {
		t.Errorf("TypedName.Type = %q, want %q", tn.Type, StaticUsageLimitPolicyType)
	}
}

func TestNewPolicyFuncTypeIsNotStatic(t *testing.T) {
	t.Parallel()
	p := NewPolicyFunc("custom-func", func(context.Context, float64, []int, []float64) {})
	if tn := p.TypedName(); tn.Type == StaticUsageLimitPolicyType {
		t.Errorf("TypedName.Type = %q, a func policy must not report the static policy type", tn.Type)
	}
}
