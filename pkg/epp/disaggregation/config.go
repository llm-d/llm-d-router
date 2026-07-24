// Package disaggregation implements native, label-aware routing across roles
// of a disaggregated inference deployment (e.g. prefill/decode).
//
// A single Controller implements requestcontrol.ResponseHeaderProcessor;
// Filter behaviour is composed of three wrappers built by WireInto:
// strict-mode selectors run at the head of the scheduler profile's filter
// chain, prefer-mode selectors at the tail, and (when configured) a gating
// filter drops candidates whose revision fails the gating check.
//
// Boot flow: Register (validate config, start pod cache, verify config
// against observed cluster state) then WireInto (attach filters and the
// response-header processor). Both are called from
// cmd/epp/runner/disaggregation.go.
package disaggregation

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/labels"
)

// Config is the top-level disaggregation block loaded from the EPP YAML.
type Config struct {
	Enabled   bool       `json:"enabled"`
	Scope     Scope      `json:"scope"`
	Selectors []Selector `json:"selectors"`
	Gating    *Gating    `json:"gating,omitempty"`
}

// Scope constrains which pods the informer observes.
type Scope struct {
	LabelSelector string `json:"labelSelector"`
	// Namespace defaults to the InferencePool's namespace when empty.
	Namespace string `json:"namespace,omitempty"`
}

// Selector defines one header/label pair to filter and tag on.
type Selector struct {
	Name       string       `json:"name"`
	HeaderName string       `json:"headerName"`
	LabelKey   string       `json:"labelKey"`
	Mode       SelectorMode `json:"mode"`
}

// UnmarshalJSON normalises HeaderName to lowercase at parse time. Downstream
// code compares against `request.Headers[headerName]` which are lowercased
// by the framework; keeping normalisation at the boundary means callers of
// Validate see a pure check with no side effects.
func (s *Selector) UnmarshalJSON(data []byte) error {
	type raw Selector
	var parsed raw
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}
	parsed.HeaderName = strings.ToLower(parsed.HeaderName)
	*s = Selector(parsed)
	return nil
}

// SelectorMode governs filter behaviour when the header carries a value with
// no matching candidates.
type SelectorMode string

const (
	// ModeStrict: no match → empty candidate set → framework returns 503.
	ModeStrict SelectorMode = "strict"
	// ModePrefer: no match → fall back to the unfiltered candidate set.
	// (Escape-hatch semantics, not a spectrum; when at least one candidate
	// matches, ModePrefer behaves identically to ModeStrict.)
	ModePrefer SelectorMode = "prefer"
)

// Gating governs the filter that drops candidates whose revision fails the
// mode's liveness check. Appended after strict and prefer selector filters
// so the picker choice is decoupled from gating behaviour.
type Gating struct {
	Mode         GatingMode    `json:"mode"`
	RequireRoles *RequireRoles `json:"requireRoles,omitempty"`
}

// GatingMode selects the gating algorithm. Extensible: new modes land as
// constants with their own sub-blocks (e.g. GatingModeMinThreshold with a
// dedicated MinThreshold sub-block) without breaking the config surface.
type GatingMode string

const (
	// GatingModeSum does two things per Filter call:
	//   1. Drop any revision missing Ready pods on any listed role
	//      (rollout drift safety).
	//   2. Weighted-random-pick ONE surviving revision, weighted by the
	//      SUM of Ready pod counts across every listed role, and keep
	//      only that revision's candidates.
	// Traffic converges on (Σ crossRolePods(rev) / Σ Σ crossRolePods),
	// independent of the picker downstream. The "sum" name refers to the
	// per-revision sum used as weight.
	GatingModeSum GatingMode = "sum"
	// GatingModeDisabled skips wiring the filter even when a Gating block
	// is present. Lets operators keep the block for documentation while
	// turning the gate off.
	GatingModeDisabled GatingMode = "disabled"
)

// RequireRoles is the sub-config for GatingModeSum: the list of roles that
// must each have at least one Ready pod on a revision for that revision's
// candidates to survive the filter.
type RequireRoles struct {
	LabelKey string   `json:"labelKey"`
	Values   []string `json:"values"`
}

// Active reports whether the gating filter should be wired at boot.
func (g *Gating) Active() bool {
	if g == nil {
		return false
	}
	switch g.Mode {
	case GatingModeSum:
		return g.RequireRoles != nil
	case GatingModeDisabled:
		return false
	}
	return false
}

// Validate performs static config checks. A disabled config validates
// trivially. All returned errors carry the offending field path so operators
// can locate the problem quickly. Pure — no side effects; header
// normalisation happens at JSON unmarshal time via Selector.UnmarshalJSON.
func (c *Config) Validate() error {
	if !c.Enabled {
		return nil
	}

	if c.Scope.LabelSelector == "" {
		return errors.New("disaggregation.scope.labelSelector is required when enabled")
	}
	if _, err := labels.Parse(c.Scope.LabelSelector); err != nil {
		return fmt.Errorf("disaggregation.scope.labelSelector is not a valid label selector: %w", err)
	}

	if len(c.Selectors) == 0 {
		return errors.New("disaggregation.selectors must contain at least one entry")
	}
	seenNames := make(map[string]struct{}, len(c.Selectors))
	seenHeaders := make(map[string]struct{}, len(c.Selectors))
	for index := range c.Selectors {
		selector := &c.Selectors[index]
		if selector.Name == "" {
			return fmt.Errorf("disaggregation.selectors[%d].name is required", index)
		}
		if selector.HeaderName == "" {
			return fmt.Errorf("disaggregation.selectors[%d].headerName is required", index)
		}
		if selector.LabelKey == "" {
			return fmt.Errorf("disaggregation.selectors[%d].labelKey is required", index)
		}
		switch selector.Mode {
		case ModeStrict, ModePrefer:
		default:
			return fmt.Errorf("disaggregation.selectors[%d].mode %q must be one of strict|prefer", index, selector.Mode)
		}

		if _, duplicate := seenNames[selector.Name]; duplicate {
			return fmt.Errorf("disaggregation.selectors: duplicate selector name %q", selector.Name)
		}
		seenNames[selector.Name] = struct{}{}

		if _, duplicate := seenHeaders[selector.HeaderName]; duplicate {
			return fmt.Errorf("disaggregation.selectors: duplicate header name %q", selector.HeaderName)
		}
		seenHeaders[selector.HeaderName] = struct{}{}
	}

	if c.Gating != nil {
		switch c.Gating.Mode {
		case GatingModeDisabled:
			// Nothing else to validate: disabled skips wiring.
		case GatingModeSum:
			if c.Gating.RequireRoles == nil {
				return errors.New("disaggregation.gating.requireRoles is required when gating.mode=sum")
			}
			if c.Gating.RequireRoles.LabelKey == "" {
				return errors.New("disaggregation.gating.requireRoles.labelKey is required")
			}
			if len(c.Gating.RequireRoles.Values) == 0 {
				return errors.New("disaggregation.gating.requireRoles.values must contain at least one role")
			}
		default:
			return fmt.Errorf("disaggregation.gating.mode %q must be one of sum|disabled", c.Gating.Mode)
		}
	}

	return nil
}

// HasSelectorsInMode reports whether any selector in this config runs in the
// given mode. Used by wiring to skip registration of a strict/prefer wrapper
// that would have nothing to do.
func (c *Config) HasSelectorsInMode(mode SelectorMode) bool {
	for _, selector := range c.Selectors {
		if selector.Mode == mode {
			return true
		}
	}
	return false
}
