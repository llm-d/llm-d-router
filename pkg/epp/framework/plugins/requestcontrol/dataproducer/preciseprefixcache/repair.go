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

package preciseprefixcache

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/utils/clock"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

const (
	defaultFullReportThreshold = 0.80
	defaultMinMissingBlocks    = 32
	defaultReportCooldown      = 10 * time.Second
)

// FullReportRepairConfig enables bounded per-request full KV-cache reports
// for endpoints whose event-derived index may be incomplete.
type FullReportRepairConfig struct {
	FullReportThreshold float64 `json:"fullReportThreshold,omitempty"`
	MinMissingBlocks    int     `json:"minMissingBlocks,omitempty"`
	// Cooldown is the minimum interval between full-report requests per
	// endpoint. Go duration string; defaults to defaultReportCooldown when
	// empty.
	Cooldown string `json:"cooldown,omitempty"`
}

func normalizeFullReportRepairConfig(config FullReportRepairConfig) (FullReportRepairConfig, time.Duration, error) {
	if config.FullReportThreshold == 0 {
		config.FullReportThreshold = defaultFullReportThreshold
	}
	if config.MinMissingBlocks == 0 {
		config.MinMissingBlocks = defaultMinMissingBlocks
	}
	if config.FullReportThreshold <= 0 || config.FullReportThreshold > 1 {
		return FullReportRepairConfig{}, 0, fmt.Errorf("fullReportThreshold must be in (0, 1], got %g", config.FullReportThreshold)
	}
	if config.MinMissingBlocks < 1 {
		return FullReportRepairConfig{}, 0, fmt.Errorf("minMissingBlocks must be positive, got %d", config.MinMissingBlocks)
	}
	cooldown := defaultReportCooldown
	if config.Cooldown != "" {
		parsed, err := time.ParseDuration(config.Cooldown)
		if err != nil {
			return FullReportRepairConfig{}, 0, fmt.Errorf("invalid cooldown: %w", err)
		}
		if parsed <= 0 {
			return FullReportRepairConfig{}, 0, fmt.Errorf("cooldown must be positive, got %s", parsed)
		}
		cooldown = parsed
	}
	return config, cooldown, nil
}

func validateFullReportRepairPrerequisites(config *kvevents.Config) error {
	if config == nil || !config.DiscoverPods || config.PodDiscoveryConfig == nil {
		return errors.New("fullReportRepair requires kvEventsConfig.discoverPods with podDiscoveryConfig")
	}
	if config.ZMQEndpoint != "" {
		return errors.New("fullReportRepair does not support kvEventsConfig.zmqEndpoint global-socket mode")
	}
	if config.EngineType != "" && config.EngineType != engineadapter.EngineTypeVLLM {
		return fmt.Errorf("fullReportRepair requires kvEventsConfig.engineType %q", engineadapter.EngineTypeVLLM)
	}
	if config.PodDiscoveryConfig.EffectiveReplayPort() > 0 {
		return errors.New("fullReportRepair does not support kvEventsConfig.podDiscoveryConfig.replaySocketPort")
	}
	return nil
}

type endpointRepairState struct {
	force       bool
	lastRequest time.Time
}

// fullReportRepair tracks, per endpoint, whether the event-derived index may
// be missing state and a report request is warranted. The cooldown bounds
// report frequency: a report lands only after the flagged request completes
// and its events propagate, so every request in that window would otherwise
// re-request the same report, and each re-announced store inflates the event
// dedup filter's reference counts. A cooldown of zero disables the bound.
type fullReportRepair struct {
	mu         sync.Mutex
	endpoints  map[string]endpointRepairState
	threshold  float64
	minMissing int
	cooldown   time.Duration
	clock      clock.PassiveClock
}

func newFullReportRepair(config FullReportRepairConfig, cooldown time.Duration) *fullReportRepair {
	return &fullReportRepair{
		endpoints:  make(map[string]endpointRepairState),
		threshold:  config.FullReportThreshold,
		minMissing: config.MinMissingBlocks,
		cooldown:   cooldown,
		clock:      clock.RealClock{},
	}
}

func (r *fullReportRepair) observe(endpoint string, event kvevents.StreamEvent) {
	if r == nil || endpoint == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	switch event {
	case kvevents.StreamEventAttached:
		if _, exists := r.endpoints[endpoint]; !exists {
			r.endpoints[endpoint] = endpointRepairState{}
		}
	case kvevents.StreamEventMissingParent:
		// Guarded on existence so a late event from a detached stream cannot
		// resurrect a removed endpoint.
		if state, exists := r.endpoints[endpoint]; exists {
			state.force = true
			r.endpoints[endpoint] = state
		}
	case kvevents.StreamEventDetached:
		delete(r.endpoints, endpoint)
	}
}

func (r *fullReportRepair) shouldRequest(endpoint string, match repairMatch) (bool, string) {
	if r == nil {
		return false, ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	state, eligible := r.endpoints[endpoint]
	if !eligible {
		return false, ""
	}
	// Ignore small gaps before considering either repair path.
	missing := match.total - match.confirmed
	if missing < r.minMissing || match.total <= 0 {
		return false, ""
	}
	// The cooldown check precedes the force check so a fault observed during
	// the cooldown window is preserved for the next eligible request.
	if r.cooldown > 0 && !state.lastRequest.IsZero() && r.clock.Since(state.lastRequest) < r.cooldown {
		return false, ""
	}
	if state.force {
		// Consume the fault only when it produces a report.
		state.force = false
		state.lastRequest = r.clock.Now()
		r.endpoints[endpoint] = state
		return true, "integrity"
	}
	if float64(match.confirmed)/float64(match.total) < r.threshold {
		state.lastRequest = r.clock.Now()
		r.endpoints[endpoint] = state
		return true, "threshold"
	}
	return false, ""
}
