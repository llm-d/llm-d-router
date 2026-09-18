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

// Package bandselection implements the default BandSelectionPolicy, which offers bands a dispatch
// opportunity from highest to lowest priority.
package bandselection

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// StrictBandSelectionPolicyType is the registration type for the strict band selection policy.
const StrictBandSelectionPolicyType = "strict-band-selection-policy"

// StrictPolicyFactory creates a StrictBandSelectionPolicy. It takes no configuration.
func StrictPolicyFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return NewStrictPolicy(name), nil
}

// DefaultPolicy returns the default BandSelectionPolicy, which dispatches in strict priority order.
func DefaultPolicy() flowcontrol.BandSelectionPolicy {
	return NewStrictPolicy(StrictBandSelectionPolicyType)
}

// NewStrictPolicy implements a BandSelectionPolicy that offers bands a dispatch opportunity from
// highest to lowest priority.
func NewStrictPolicy(name string) flowcontrol.BandSelectionPolicy {
	if name == "" {
		name = StrictBandSelectionPolicyType
	}
	return &strictPolicy{name: name}
}

type strictPolicy struct {
	name string
}

var _ flowcontrol.BandSelectionPolicy = &strictPolicy{}

func (s *strictPolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: StrictBandSelectionPolicyType,
		Name: s.name,
	}
}

// Rank leaves the order buffer untouched. The framework pre-fills it with the identity permutation
// over priorities ordered highest first, which is already strict order.
func (s *strictPolicy) Rank(_ context.Context, _ float64, _ []int, _ []float64, _ []int) {}

// RecordDispatch does nothing. Strict order carries no share accounting to settle.
func (s *strictPolicy) RecordDispatch(_ context.Context, _ int) {}
