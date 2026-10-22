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

package qosadmission

import (
	"errors"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// ModeEnforce applies admission decisions.
	ModeEnforce = "enforce"
	// ModeShadow computes and records decisions without acting on them.
	ModeShadow = "shadow"

	defaultOutputTokenWeight = 2
	defaultBurstSeconds      = 30
)

// ClassConfig is one service class, mapped onto a flow-control priority band.
type ClassConfig struct {
	Name     string `json:"name"`
	Priority int    `json:"priority"`
}

// TeamConfig is one team's QoS contract. Zero-valued limits are unenforced.
type TeamConfig struct {
	ID string `json:"id"`
	// Class is the name of the class the team's committed traffic is served at.
	Class string `json:"class"`
	// CommitTokensPerMinute is the committed weighted-token rate. 0 disables the token bucket.
	CommitTokensPerMinute float64 `json:"commitTokensPerMinute,omitempty"`
	// BurstSeconds sizes the bucket as BurstSeconds of committed rate. Default 30.
	BurstSeconds float64 `json:"burstSeconds,omitempty"`
	// MaxConcurrency caps the team's requests from admission to completion. 0 is unlimited.
	MaxConcurrency int `json:"maxConcurrency,omitempty"`
	// MaxPromptTokens rejects larger prompts with 400. 0 is unlimited.
	MaxPromptTokens int64 `json:"maxPromptTokens,omitempty"`
	// MaxOutputTokens rejects larger output caps with 400, and is the reservation for a request
	// that sets no output cap. 0 is unlimited.
	MaxOutputTokens int64 `json:"maxOutputTokens,omitempty"`
	// DailyTokens is a weighted-token budget per UTC day. 0 is unlimited.
	DailyTokens float64 `json:"dailyTokens,omitempty"`
}

// Parameters is the plugin configuration.
type Parameters struct {
	// Mode is "enforce" (default) or "shadow".
	Mode string `json:"mode,omitempty"`
	// TeamHeader names the request header carrying the team ID. Defaults to the fairness ID
	// header, so the team is also the fairness flow.
	TeamHeader string `json:"teamHeader,omitempty"`
	// OutputTokenWeight is the cost of one output token relative to one input token. Default 2.
	OutputTokenWeight float64 `json:"outputTokenWeight,omitempty"`
	// DefaultOutputReservation is the output tokens reserved for a request that sets no output
	// cap and whose team sets no MaxOutputTokens.
	DefaultOutputReservation int64 `json:"defaultOutputReservation,omitempty"`
	// Classes are ordered from highest to lowest. Over-commit traffic moves to the next class.
	Classes []ClassConfig `json:"classes"`
	// DefaultTeam applies to requests whose team is absent or not listed.
	DefaultTeam TeamConfig `json:"defaultTeam"`
	// Teams lists the teams with a contract.
	Teams []TeamConfig `json:"teams,omitempty"`
}

func defaultParameters() Parameters {
	return Parameters{
		Mode:                     ModeEnforce,
		TeamHeader:               metadata.FlowFairnessIDKey,
		OutputTokenWeight:        defaultOutputTokenWeight,
		DefaultOutputReservation: 1000,
	}
}

func (p *Parameters) validate() error {
	if p.Mode != ModeEnforce && p.Mode != ModeShadow {
		return fmt.Errorf("mode must be %q or %q, got %q", ModeEnforce, ModeShadow, p.Mode)
	}
	if p.TeamHeader == "" {
		return errors.New("teamHeader must not be empty")
	}
	if p.OutputTokenWeight <= 0 {
		return fmt.Errorf("outputTokenWeight must be > 0, got %v", p.OutputTokenWeight)
	}
	if p.DefaultOutputReservation < 0 {
		return fmt.Errorf("defaultOutputReservation must be >= 0, got %d", p.DefaultOutputReservation)
	}
	if len(p.Classes) == 0 {
		return errors.New("at least one class is required")
	}
	classes := make(map[string]struct{}, len(p.Classes))
	for i, c := range p.Classes {
		if c.Name == "" {
			return fmt.Errorf("classes[%d]: name must not be empty", i)
		}
		if _, dup := classes[c.Name]; dup {
			return fmt.Errorf("classes[%d]: duplicate class %q", i, c.Name)
		}
		if i > 0 && c.Priority >= p.Classes[i-1].Priority {
			return fmt.Errorf("classes[%d] %q: priorities must strictly decrease, %d follows %d", i, c.Name, c.Priority, p.Classes[i-1].Priority)
		}
		classes[c.Name] = struct{}{}
	}
	if err := validateTeam(&p.DefaultTeam, classes); err != nil {
		return fmt.Errorf("defaultTeam: %w", err)
	}
	teams := make(map[string]struct{}, len(p.Teams))
	for i := range p.Teams {
		t := &p.Teams[i]
		if t.ID == "" {
			return fmt.Errorf("teams[%d]: id must not be empty", i)
		}
		if _, dup := teams[t.ID]; dup {
			return fmt.Errorf("teams[%d]: duplicate team %q", i, t.ID)
		}
		teams[t.ID] = struct{}{}
		if err := validateTeam(t, classes); err != nil {
			return fmt.Errorf("teams[%d] %q: %w", i, t.ID, err)
		}
	}
	return nil
}

func validateTeam(t *TeamConfig, classes map[string]struct{}) error {
	if _, ok := classes[t.Class]; !ok {
		return fmt.Errorf("unknown class %q", t.Class)
	}
	if t.CommitTokensPerMinute < 0 || t.BurstSeconds < 0 || t.MaxConcurrency < 0 ||
		t.MaxPromptTokens < 0 || t.MaxOutputTokens < 0 || t.DailyTokens < 0 {
		return errors.New("limits must be >= 0")
	}
	if t.BurstSeconds == 0 {
		t.BurstSeconds = defaultBurstSeconds
	}
	return nil
}
