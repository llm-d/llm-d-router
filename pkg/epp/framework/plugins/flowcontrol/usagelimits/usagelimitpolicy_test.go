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
