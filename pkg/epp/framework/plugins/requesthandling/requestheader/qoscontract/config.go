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

package qoscontract

import (
	"errors"
	"fmt"
	"math"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// ModeEnforceLimits applies contract decisions: it serves each team at its class, moves traffic
	// outside a contract one class down, and rejects it in the lowest class.
	ModeEnforceLimits = "enforce-limits"
	// ModeReportOnly records the decisions enforce-limits would apply without applying any of them:
	// a request keeps its InferenceObjective priority and its fairness ID. It is the default, so a
	// contract is observed against real traffic before it is enforced.
	ModeReportOnly = "report-only"

	defaultOutputTokenWeight = 2
	defaultRateWindow        = time.Second
	// maxRateWindowRequests bounds the requests a rate window holds, and so its memory.
	maxRateWindowRequests = 1 << 20
)

// ClassConfig is one service class, mapped onto a flow-control priority band.
type ClassConfig struct {
	Name     string `json:"name"`
	Priority int    `json:"priority"`
}

// TeamConfig is one team's QoS contract. Zero-valued limits are unenforced. MaxConcurrency,
// DailyTokens, MaxReqsPerSec, MaxInputTokensPerSec and MaxUncachedTokensPerSec are the team's
// commits, and MaxPromptTokens and MaxOutputTokens its size limits:
// traffic outside any of them moves one class down, and is rejected in the lowest class.
type TeamConfig struct {
	ID string `json:"id"`
	// Class is the name of the class the team's committed traffic is served at.
	Class string `json:"class"`
	// MaxConcurrency is the committed number of requests from the request header to completion. 0 is
	// unlimited.
	MaxConcurrency int `json:"maxConcurrency,omitempty"`
	// MaxPromptTokens is the largest prompt served at the team's class. Larger prompts move one
	// class down, or get 400 in the lowest class. 0 is unlimited.
	MaxPromptTokens int64 `json:"maxPromptTokens,omitempty"`
	// MaxOutputTokens is the largest output cap served at the team's class. Larger caps move one
	// class down, or get 400 in the lowest class. It is also the reservation for a request that
	// sets no output cap. 0 is unlimited.
	MaxOutputTokens int64 `json:"maxOutputTokens,omitempty"`
	// DailyTokens is the committed weighted-token budget per UTC day. 0 is unlimited.
	DailyTokens float64 `json:"dailyTokens,omitempty"`
	// MaxReqsPerSec is the committed request rate, measured over a sliding RateWindow: at most
	// floor(MaxReqsPerSec * RateWindow) requests are served at the team's class in any window.
	// 0 is unlimited.
	MaxReqsPerSec float64 `json:"maxReqsPerSec,omitempty"`
	// MaxInputTokensPerSec is the committed prompt-token rate, a token bucket that refills at this
	// rate and holds at most MaxInputTokensPerSec * RateWindow tokens, the largest burst. A request
	// takes its estimated prompt tokens and is settled to the reported prompt tokens when it
	// completes. 0 is unlimited.
	MaxInputTokensPerSec float64 `json:"maxInputTokensPerSec,omitempty"`
	// MaxUncachedTokensPerSec is the committed rate of prompt tokens not served from the prefix
	// cache, a token bucket like MaxInputTokensPerSec. A request takes its estimated prompt tokens
	// less the team's observed cache-hit ratio and is settled to the reported prompt tokens less
	// the reported cached tokens. 0 is unlimited.
	MaxUncachedTokensPerSec float64 `json:"maxUncachedTokensPerSec,omitempty"`
	// RateWindow is the sliding window MaxReqsPerSec is measured over, and the burst of
	// MaxInputTokensPerSec and MaxUncachedTokensPerSec. Unset uses the plugin's RateWindow.
	RateWindow *metav1.Duration `json:"rateWindow,omitempty"`
}

// Parameters is the plugin configuration.
type Parameters struct {
	// Mode is "report-only" (default) or "enforce-limits".
	Mode string `json:"mode,omitempty"`
	// TeamHeader names the request header carrying the team ID. Defaults to the fairness ID
	// header, so the team is also the fairness flow.
	TeamHeader string `json:"teamHeader,omitempty"`
	// OutputTokenWeight is the cost of one output token relative to one input token. Default 2.
	OutputTokenWeight float64 `json:"outputTokenWeight,omitempty"`
	// DefaultOutputReservation is the output tokens reserved for a request that sets no output
	// cap and whose team sets no MaxOutputTokens.
	DefaultOutputReservation int64 `json:"defaultOutputReservation,omitempty"`
	// RateWindow is the sliding window a team's MaxReqsPerSec is measured over, and the burst of its
	// token rates, unless the team sets its own. Default 1s.
	RateWindow metav1.Duration `json:"rateWindow,omitempty"`
	// Classes are ordered from highest to lowest. Traffic outside the contract moves to the next
	// class.
	Classes []ClassConfig `json:"classes"`
	// DefaultTeam applies to requests whose team is absent or not listed.
	DefaultTeam TeamConfig `json:"defaultTeam"`
	// Teams lists the teams with a contract.
	Teams []TeamConfig `json:"teams,omitempty"`
}

func defaultParameters() Parameters {
	return Parameters{
		Mode:                     ModeReportOnly,
		TeamHeader:               metadata.FlowFairnessIDKey,
		OutputTokenWeight:        defaultOutputTokenWeight,
		DefaultOutputReservation: 1000,
		RateWindow:               metav1.Duration{Duration: defaultRateWindow},
	}
}

func (p *Parameters) validate() error {
	if p.Mode != ModeEnforceLimits && p.Mode != ModeReportOnly {
		return fmt.Errorf("mode must be %q or %q, got %q", ModeEnforceLimits, ModeReportOnly, p.Mode)
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
	if p.RateWindow.Duration <= 0 {
		return fmt.Errorf("rateWindow must be > 0, got %s", p.RateWindow.Duration)
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
	if err := validateTeam(&p.DefaultTeam, classes, p.RateWindow.Duration); err != nil {
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
		// The team label of an unlisted team is the defaultTeam ID, so a listed team with that ID
		// would share its metric series.
		if t.ID == p.DefaultTeam.ID {
			return fmt.Errorf("teams[%d]: id %q is the defaultTeam id", i, t.ID)
		}
		teams[t.ID] = struct{}{}
		if err := validateTeam(t, classes, p.RateWindow.Duration); err != nil {
			return fmt.Errorf("teams[%d] %q: %w", i, t.ID, err)
		}
	}
	return nil
}

func validateTeam(t *TeamConfig, classes map[string]struct{}, defaultWindow time.Duration) error {
	if _, ok := classes[t.Class]; !ok {
		return fmt.Errorf("unknown class %q", t.Class)
	}
	if t.MaxConcurrency < 0 || t.MaxPromptTokens < 0 || t.MaxOutputTokens < 0 || t.DailyTokens < 0 || t.MaxReqsPerSec < 0 ||
		t.MaxInputTokensPerSec < 0 || t.MaxUncachedTokensPerSec < 0 {
		return errors.New("limits must be >= 0")
	}
	if t.RateWindow != nil && t.RateWindow.Duration <= 0 {
		return fmt.Errorf("rateWindow must be > 0, got %s", t.RateWindow.Duration)
	}
	window := rateWindowOf(*t, defaultWindow)
	for _, r := range []struct {
		name string
		rate float64
	}{{"maxInputTokensPerSec", t.MaxInputTokensPerSec}, {"maxUncachedTokensPerSec", t.MaxUncachedTokensPerSec}} {
		if r.rate > 0 && r.rate*window.Seconds() < 1 {
			return fmt.Errorf("%s %v over rateWindow %s allows no token: use a larger rateWindow", r.name, r.rate, window)
		}
	}
	if t.MaxReqsPerSec == 0 {
		return nil
	}
	window, limit := rateLimit(*t, defaultWindow)
	if limit < 1 {
		return fmt.Errorf("maxReqsPerSec %v over rateWindow %s allows no request: use a larger rateWindow", t.MaxReqsPerSec, window)
	}
	if limit > maxRateWindowRequests {
		return fmt.Errorf("maxReqsPerSec %v over rateWindow %s allows more than %d requests per window: use a smaller rateWindow", t.MaxReqsPerSec, window, maxRateWindowRequests)
	}
	return nil
}

// rateLimit returns the sliding window a team's request rate is measured over and the requests
// it allows in that window.
func rateLimit(t TeamConfig, defaultWindow time.Duration) (window time.Duration, limit int) {
	window = rateWindowOf(t, defaultWindow)
	// The epsilon keeps a product such as 0.29 * 100 = 28.999999999999996 from flooring to 28. The
	// clamp keeps a huge product from overflowing int; validation rejects it.
	n := math.Floor(t.MaxReqsPerSec*window.Seconds() + 1e-9)
	if n > maxRateWindowRequests {
		return window, maxRateWindowRequests + 1
	}
	return window, int(n)
}

// rateWindowOf returns the team's rate window: its own, else the plugin's.
func rateWindowOf(t TeamConfig, defaultWindow time.Duration) time.Duration {
	if t.RateWindow != nil {
		return t.RateWindow.Duration
	}
	return defaultWindow
}
