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

// Package disaggregatedsetrollout provides a request-control Screener for safe rollouts
// across roles of a disaggregated inference deployment.
//
// Two mechanisms shape the survivor set the scheduler picker sees:
//
//   - Header selectors: strict mode screens the candidate pool to endpoints
//     whose label matches the request header. Prefer-mode selectors are
//     consumed by the separate DisaggregatedSet preference scorer.
//
//   - Revision gating: a request-independent safety+load-shaping layer
//     that (a) drops revisions missing Ready pods on any required role
//     (rollout-drift protection) and (b) when no strict header pins the
//     revision axis, weighted-random-picks one surviving revision to
//     collapse the candidate pool to a single revision.
//
// The Screener observes Pods through the data layer's Kubernetes notification
// source, applies gating and strict selectors before scheduling, and stamps
// response headers.
package disaggregatedsetrollout

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/labels"
)

// DisaggregatedSet operator label conventions. Used as defaults when the
// operator omits RevisionGating.RevisionLabelKey / RoleLabelKey.
const (
	DefaultRevisionLabel = "disaggregatedset.x-k8s.io/revision"
	DefaultRoleLabel     = "disaggregatedset.x-k8s.io/role"
)

// Config is the parameters block of a disaggregatedset-rollout-screener plugin.
type Config struct {
	Scope           Scope            `json:"scope"`
	HeaderSelectors []HeaderSelector `json:"headerSelectors"`
	RevisionGating  *RevisionGating  `json:"revisionGating"`
}

// Scope constrains which Pod notifications contribute to revision gating.
type Scope struct {
	LabelSelector string `json:"labelSelector"`
}

// HeaderSelector defines one header/label pair to select and stamp. Strict
// selectors are consumed by the Screener; prefer selectors are stamped but
// require a separately configured affinity scorer. ResponseHeader stamps both.
type HeaderSelector struct {
	Name       string       `json:"name"`
	HeaderName string       `json:"headerName"`
	LabelKey   string       `json:"labelKey"`
	Mode       SelectorMode `json:"mode"`
}

// UnmarshalJSON normalises HeaderName to lowercase at parse time. The
// framework hands request headers to plugins already lowercased; storing
// them lowercase here means the hot-path lookup is a direct map read with
// no per-request ToLower.
func (s *HeaderSelector) UnmarshalJSON(data []byte) error {
	type raw HeaderSelector
	var parsed raw
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&parsed); err != nil {
		return err
	}
	parsed.HeaderName = strings.ToLower(parsed.HeaderName)
	*s = HeaderSelector(parsed)
	return nil
}

// SelectorMode selects whether the Screener enforces a header selector.
type SelectorMode string

const (
	// ModeStrict: no match produces an empty candidate set, so the framework's downstream
	// pipeline returns an error (503). Intentional under strict mode:
	// the client asked for something specific and we do not silently
	// substitute.
	ModeStrict SelectorMode = "strict"
	// ModePrefer leaves matching to a separately configured soft-affinity
	// scorer. The Screener only stamps this selector's response header.
	ModePrefer SelectorMode = "prefer"
)

// RevisionGating governs revision-axis candidate screening. Two things happen
// per request when Active:
//
//  1. Coverage check: drop candidates whose revision has zero Ready pods
//     for any role listed in RequireRoles.Values.
//  2. Load shaping: when no strict header is present, weighted-random-pick
//     ONE surviving revision and keep only its pods.
//
// RevisionLabelKey and RoleLabelKey are independent of HeaderSelectors.
// gating can operate even with no header selectors, and header selectors
// can operate on different label axes. If either is empty, the defaults
// (DefaultRevisionLabel / DefaultRoleLabel) are used.
type RevisionGating struct {
	RevisionLabelKey string        `json:"revisionLabelKey,omitempty"`
	RoleLabelKey     string        `json:"roleLabelKey,omitempty"`
	Mode             GatingMode    `json:"mode"`
	RequireRoles     *RequireRoles `json:"requireRoles,omitempty"`
}

// GatingMode selects the gating algorithm. Open enum so future modes can
// land with their own sub-block without breaking the config surface.
type GatingMode string

const (
	// GatingModeSum does two things:
	//   1. Drop any revision missing Ready pods on any listed role
	//      (rollout drift safety).
	//   2. Weighted-random-pick ONE surviving revision, weighted by the
	//      sum of Ready pod counts across every listed role, and keep
	//      only that revision's candidates.
	// Traffic converges on (sum(crossRolePods(rev)) / sum over all
	// revisions), independent of the picker downstream. The "sum" name
	// refers to the per-revision sum used as weight.
	// For A={prefill:2,decode:9} and B={prefill:1,decode:1}, the weights
	// are 11 and 2, producing shares of 11/13 and 2/13.
	GatingModeSum GatingMode = "sum"
	// GatingModeMaxRole applies the same coverage check as GatingModeSum, sums
	// each required role across every covered revision, and selects the role
	// with the largest total. Every revision is then weighted by its Ready pod
	// count for that same role. RequireRoles order resolves equal totals.
	GatingModeMaxRole GatingMode = "max-role"
	// GatingModeDisabled turns off revision gating while preserving any
	// configured header selectors.
	GatingModeDisabled GatingMode = "disabled"
)

// RequireRoles lists the roles that must each have at least one Ready pod on a
// revision for that revision's candidates to survive screening. The label key
// that identifies a pod's role is RevisionGating.RoleLabelKey (parent scope).
type RequireRoles struct {
	Values []string `json:"values"`
}

// Active reports whether revision gating is enabled.
func (g *RevisionGating) Active() bool {
	if g == nil {
		return false
	}
	switch g.Mode {
	case GatingModeSum, GatingModeMaxRole:
		return g.RequireRoles != nil
	case GatingModeDisabled:
		return false
	}
	return false
}

// Validate performs static config checks and fills in label-key defaults.
func (c *Config) Validate() error {
	if c.Scope.LabelSelector == "" {
		return errors.New("scope.labelSelector is required")
	}
	if _, err := labels.Parse(c.Scope.LabelSelector); err != nil {
		return fmt.Errorf("scope.labelSelector is not a valid label selector: %w", err)
	}

	seenNames := make(map[string]struct{}, len(c.HeaderSelectors))
	seenHeaders := make(map[string]struct{}, len(c.HeaderSelectors))
	for index := range c.HeaderSelectors {
		selector := &c.HeaderSelectors[index]
		if selector.Name == "" {
			return fmt.Errorf("headerSelectors[%d].name is required", index)
		}
		if selector.HeaderName == "" {
			return fmt.Errorf("headerSelectors[%d].headerName is required", index)
		}
		if selector.LabelKey == "" {
			return fmt.Errorf("headerSelectors[%d].labelKey is required", index)
		}
		switch selector.Mode {
		case ModeStrict, ModePrefer:
		default:
			return fmt.Errorf("headerSelectors[%d].mode %q must be one of strict|prefer", index, selector.Mode)
		}
		if _, duplicate := seenNames[selector.Name]; duplicate {
			return fmt.Errorf("headerSelectors: duplicate name %q", selector.Name)
		}
		seenNames[selector.Name] = struct{}{}
		if _, duplicate := seenHeaders[selector.HeaderName]; duplicate {
			return fmt.Errorf("headerSelectors: duplicate headerName %q", selector.HeaderName)
		}
		seenHeaders[selector.HeaderName] = struct{}{}
	}

	if c.RevisionGating == nil {
		return errors.New("revisionGating is required")
	}
	if c.RevisionGating.RevisionLabelKey == "" {
		c.RevisionGating.RevisionLabelKey = DefaultRevisionLabel
	}
	if c.RevisionGating.RoleLabelKey == "" {
		c.RevisionGating.RoleLabelKey = DefaultRoleLabel
	}
	switch c.RevisionGating.Mode {
	case GatingModeDisabled:
		// Nothing else to validate: disabled skips wiring.
	case GatingModeSum, GatingModeMaxRole:
		if c.RevisionGating.RequireRoles == nil {
			return fmt.Errorf("revisionGating.requireRoles is required when revisionGating.mode=%s", c.RevisionGating.Mode)
		}
		if len(c.RevisionGating.RequireRoles.Values) == 0 {
			return errors.New("revisionGating.requireRoles.values must contain at least one role")
		}
		seenRoles := make(map[string]struct{}, len(c.RevisionGating.RequireRoles.Values))
		for i, role := range c.RevisionGating.RequireRoles.Values {
			if role == "" {
				return fmt.Errorf("revisionGating.requireRoles.values[%d] must not be empty", i)
			}
			if _, exists := seenRoles[role]; exists {
				return fmt.Errorf("revisionGating.requireRoles.values contains duplicate role %q", role)
			}
			seenRoles[role] = struct{}{}
		}
	default:
		return fmt.Errorf("revisionGating.mode %q must be one of sum|max-role|disabled", c.RevisionGating.Mode)
	}

	return nil
}
