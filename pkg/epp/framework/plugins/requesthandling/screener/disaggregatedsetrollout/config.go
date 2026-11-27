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

// Package disaggregatedsetrollout provides a request-control Screener for safe
// rollouts across roles of a disaggregated inference deployment. It filters a
// requested revision strictly, or selects one complete revision using Ready
// Pod counts when the request does not specify one.
package disaggregatedsetrollout

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/labels"

	"github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// DisaggregatedSet rollout protocol and operator label defaults.
const (
	DefaultRevisionHeader = request.DisaggregatedRevisionHeaderKey
	DefaultRevisionLabel  = "disaggregatedset.x-k8s.io/revision"
	DefaultRoleLabel      = "disaggregatedset.x-k8s.io/role"
)

// Config is the parameters block of a disaggregatedset-rollout-screener plugin.
type Config struct {
	Scope          Scope           `json:"scope"`
	RevisionGating *RevisionGating `json:"revisionGating"`
}

// Scope constrains which Pod notifications contribute to revision gating.
type Scope struct {
	LabelSelector string `json:"labelSelector"`
}

// RevisionGating governs revision selection. Two things happen per request
// when Active:
//
//  1. Coverage check: drop candidates whose revision has zero Ready pods
//     for any role listed in RequiredRoles.
//  2. Load shaping: when no revision header is present, weighted-random-pick
//     one surviving revision and keep only its pods. Affinity can key that
//     draw so related requests pick the same revision.
//
// A request carrying RevisionHeaderName always selects that revision strictly.
// The selected endpoint's revision is stamped into the same response header.
type RevisionGating struct {
	RevisionHeaderName  string            `json:"revisionHeaderName,omitempty"`
	RevisionLabelKey    string            `json:"revisionLabelKey,omitempty"`
	RoleLabelKey        string            `json:"roleLabelKey,omitempty"`
	DisableCoordination bool              `json:"disableCoordination,omitempty"`
	Mode                GatingMode        `json:"mode"`
	RequiredRoles       []string          `json:"requiredRoles,omitempty"`
	Overflow            *RevisionOverflow `json:"overflow,omitempty"`
	Affinity            *RevisionAffinity `json:"affinity,omitempty"`
}

// RevisionOverflow selects an alternative when a configured role has no room.
type RevisionOverflow struct {
	Guards       []RevisionOverflowGuard `json:"guards,omitempty"`
	legacyFields bool
	// Role is the role whose endpoints must have room. Defaults to prefill.
	Role string `json:"role,omitempty"`
	// GuardPluginName names the fail-closed scheduling filter that decides
	// room, typically the concurrency-detector of Role's profile.
	GuardPluginName string `json:"guardPluginName,omitempty" pluginRef:""`
}

// UnmarshalJSON retains legacy field presence to reject mixed configuration forms.
func (o *RevisionOverflow) UnmarshalJSON(data []byte) error {
	type raw RevisionOverflow
	var parsed raw
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&parsed); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for key, value := range fields {
		switch {
		case strings.EqualFold(key, "role"), strings.EqualFold(key, "guardPluginName"):
			parsed.legacyFields = true
		case strings.EqualFold(key, "guards"):
			if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return errors.New("revisionGating.overflow.guards must not be null")
			}
		}
	}
	*o = RevisionOverflow(parsed)
	return nil
}

// RevisionOverflowGuard checks capacity for one required role.
type RevisionOverflowGuard struct {
	Role            string `json:"role"`
	GuardPluginName string `json:"guardPluginName"`
}

func (o *RevisionOverflow) guards() []RevisionOverflowGuard {
	if o.Guards != nil {
		return o.Guards
	}
	return []RevisionOverflowGuard{{Role: o.Role, GuardPluginName: o.GuardPluginName}}
}

// DefaultOverflowRole is the role checked for room by default.
const DefaultOverflowRole = "prefill"

// RevisionAffinity controls how the weighted revision choice is keyed; see the
// README's Revision Affinity.
type RevisionAffinity struct {
	Mode AffinityMode `json:"mode,omitempty"`
	// Sources optionally resolve a session identifier, in priority order,
	// that keys the revision choice ahead of the conversation prefix.
	Sources []sessionaffinity.SessionIDSource `json:"sources,omitempty"`
	// PrefixCache optionally selects the revision whose endpoints hold the
	// longest cached prefix for requests that resolve no session identifier
	// and no text conversation prefix, such as pre-tokenized requests.
	PrefixCache *RevisionPrefixCache `json:"prefixCache,omitempty"`
}

// RevisionPrefixCache configures cache-aware revision selection from
// PrefixCacheMatchInfo; see the README's Prefix Cache Affinity.
type RevisionPrefixCache struct {
	// ProducerName names the PrefixCacheMatchInfo producer, as in
	// prefix-cache-scorer. Empty selects the default producer.
	ProducerName string `json:"producerName,omitempty"`
	// MinLeadBlocks is how many more matched blocks the best revision must
	// hold than every other covered revision. Defaults to 2.
	MinLeadBlocks int `json:"minLeadBlocks,omitempty"`
	// ShareSlack is how far a revision's recent fraction of selections may
	// run above its share before the cache step stops choosing it. Defaults
	// to 0.1.
	ShareSlack *float64 `json:"shareSlack,omitempty"`
}

// Prefix cache affinity defaults.
const (
	DefaultPrefixCacheMinLeadBlocks = 2
	DefaultPrefixCacheShareSlack    = 0.1
)

// AffinityMode selects how the revision choice is keyed.
type AffinityMode string

const (
	// AffinityModeNone draws every request independently at random.
	AffinityModeNone AffinityMode = "none"
	// AffinityModePrefix keys weighted rendezvous selection on the first
	// session identifier resolved from Sources, else the text conversation
	// prefix. Otherwise PrefixCache, when set, may pick the revision holding
	// the request's cached prefix, falling back to a random draw.
	AffinityModePrefix AffinityMode = "prefix"
)

func (g *RevisionGating) affinityMode() AffinityMode {
	if g == nil || g.Affinity == nil || g.Affinity.Mode == "" {
		return AffinityModeNone
	}
	return g.Affinity.Mode
}

// UnmarshalJSON normalizes RevisionHeaderName to lowercase because request
// headers are lowercase in the scheduling request.
func (g *RevisionGating) UnmarshalJSON(data []byte) error {
	type raw RevisionGating
	var parsed raw
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&parsed); err != nil {
		return err
	}
	parsed.RevisionHeaderName = strings.ToLower(parsed.RevisionHeaderName)
	*g = RevisionGating(parsed)
	return nil
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
	// count for that same role. RequiredRoles order resolves equal totals.
	GatingModeMaxRole GatingMode = "max-role"
	// GatingModeDisabled turns off revision coverage and weighted selection.
	// Revision header filtering and response stamping remain enabled.
	GatingModeDisabled GatingMode = "disabled"
)

// Active reports whether revision gating is enabled.
func (g *RevisionGating) Active() bool {
	if g == nil {
		return false
	}
	switch g.Mode {
	case GatingModeSum, GatingModeMaxRole:
		return len(g.RequiredRoles) > 0
	case GatingModeDisabled:
		return false
	}
	return false
}

func (g *RevisionGating) coordinationEnabled() bool {
	return g != nil && !g.DisableCoordination
}

// Validate performs static config checks and fills in label-key defaults.
func (c *Config) Validate() error {
	if c.Scope.LabelSelector == "" {
		return errors.New("scope.labelSelector is required")
	}
	if _, err := labels.Parse(c.Scope.LabelSelector); err != nil {
		return fmt.Errorf("scope.labelSelector is not a valid label selector: %w", err)
	}

	if c.RevisionGating == nil {
		return errors.New("revisionGating is required")
	}
	if c.RevisionGating.RevisionHeaderName == "" {
		c.RevisionGating.RevisionHeaderName = DefaultRevisionHeader
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
		if len(c.RevisionGating.RequiredRoles) == 0 {
			return fmt.Errorf("revisionGating.requiredRoles must contain at least one role when revisionGating.mode=%s", c.RevisionGating.Mode)
		}
		seenRoles := make(map[string]struct{}, len(c.RevisionGating.RequiredRoles))
		for i, role := range c.RevisionGating.RequiredRoles {
			if role == "" {
				return fmt.Errorf("revisionGating.requiredRoles[%d] must not be empty", i)
			}
			if _, exists := seenRoles[role]; exists {
				return fmt.Errorf("revisionGating.requiredRoles contains duplicate role %q", role)
			}
			seenRoles[role] = struct{}{}
		}
	default:
		return fmt.Errorf("revisionGating.mode %q must be one of sum|max-role|disabled", c.RevisionGating.Mode)
	}

	if overflow := c.RevisionGating.Overflow; overflow != nil {
		if !c.RevisionGating.Active() {
			return fmt.Errorf("revisionGating.overflow requires revisionGating.mode %s or %s", GatingModeSum, GatingModeMaxRole)
		}
		if overflow.Guards != nil {
			if len(overflow.Guards) == 0 {
				return errors.New("revisionGating.overflow.guards must not be empty")
			}
			if overflow.legacyFields || overflow.Role != "" || overflow.GuardPluginName != "" {
				return errors.New("revisionGating.overflow cannot mix guards with role or guardPluginName")
			}
		} else {
			if overflow.GuardPluginName == "" {
				return errors.New("revisionGating.overflow.guardPluginName is required")
			}
			if overflow.Role == "" {
				overflow.Role = DefaultOverflowRole
			}
		}
		seen := map[string]bool{}
		for _, guard := range overflow.guards() {
			if guard.Role == "" || guard.GuardPluginName == "" {
				return errors.New("revisionGating.overflow guards require role and guardPluginName")
			}
			if seen[guard.Role] {
				return fmt.Errorf("revisionGating.overflow duplicate role %q", guard.Role)
			}
			seen[guard.Role] = true
			if !slices.Contains(c.RevisionGating.RequiredRoles, guard.Role) {
				return fmt.Errorf("revisionGating.overflow.role %q must be one of revisionGating.requiredRoles", guard.Role)
			}
		}
	}

	if affinity := c.RevisionGating.Affinity; affinity != nil {
		switch affinity.Mode {
		case "", AffinityModeNone:
			if len(affinity.Sources) > 0 {
				return fmt.Errorf("revisionGating.affinity.sources requires revisionGating.affinity.mode=%s", AffinityModePrefix)
			}
			if affinity.PrefixCache != nil {
				return fmt.Errorf("revisionGating.affinity.prefixCache requires revisionGating.affinity.mode=%s", AffinityModePrefix)
			}
		case AffinityModePrefix:
			if err := sessionaffinity.ValidateSources(affinity.Sources); err != nil {
				return fmt.Errorf("revisionGating.affinity.%w", err)
			}
			if err := c.RevisionGating.validatePrefixCache(); err != nil {
				return err
			}
		default:
			return fmt.Errorf("revisionGating.affinity.mode %q must be one of none|prefix", affinity.Mode)
		}
	}

	return nil
}

// validatePrefixCache checks affinity.prefixCache and fills in its defaults.
func (g *RevisionGating) validatePrefixCache() error {
	prefixCache := g.Affinity.PrefixCache
	if prefixCache == nil {
		return nil
	}
	if !g.Active() {
		return fmt.Errorf("revisionGating.affinity.prefixCache requires revisionGating.mode %s or %s", GatingModeSum, GatingModeMaxRole)
	}
	switch {
	case prefixCache.MinLeadBlocks < 0:
		return fmt.Errorf("revisionGating.affinity.prefixCache.minLeadBlocks must be positive, got %d", prefixCache.MinLeadBlocks)
	case prefixCache.MinLeadBlocks == 0:
		prefixCache.MinLeadBlocks = DefaultPrefixCacheMinLeadBlocks
	}
	if prefixCache.ShareSlack == nil {
		slack := DefaultPrefixCacheShareSlack
		prefixCache.ShareSlack = &slack
	}
	if slack := *prefixCache.ShareSlack; slack < 0 || slack > 1 {
		return fmt.Errorf("revisionGating.affinity.prefixCache.shareSlack must be between 0 and 1, got %v", slack)
	}
	return nil
}
