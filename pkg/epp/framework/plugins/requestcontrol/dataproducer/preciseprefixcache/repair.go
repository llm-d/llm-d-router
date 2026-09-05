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
	PrefillProfile      string  `json:"prefillProfile,omitempty"`
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
	missing         map[kvevents.StreamBlock]struct{}
	reportSupported bool
	lastRequest     time.Time
}

// fullReportRepair retains missing block identities until a store, removal,
// or cache reset resolves them. Report requests share an endpoint cooldown.
type fullReportRepair struct {
	mu             sync.Mutex
	endpoints      map[string]endpointRepairState
	threshold      float64
	minMissing     int
	cooldown       time.Duration
	prefillProfile string
	clock          clock.PassiveClock
}

func newFullReportRepair(config FullReportRepairConfig, cooldown time.Duration) *fullReportRepair {
	if config.PrefillProfile == "" {
		config.PrefillProfile = experimentalPrefillProfile
	}
	return &fullReportRepair{
		endpoints:      make(map[string]endpointRepairState),
		threshold:      config.FullReportThreshold,
		minMissing:     config.MinMissingBlocks,
		cooldown:       cooldown,
		prefillProfile: config.PrefillProfile,
		clock:          clock.RealClock{},
	}
}

func (r *fullReportRepair) observe(endpoint string, event kvevents.StreamEvent, blocks ...kvevents.StreamBlock) {
	if r == nil || endpoint == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if event == kvevents.StreamEventDetached {
		delete(r.endpoints, endpoint)
		return
	}
	state, exists := r.endpoints[endpoint]
	if !exists && event != kvevents.StreamEventAttached {
		return
	}
	switch event {
	case kvevents.StreamEventReportSupported:
		state.reportSupported = true
	case kvevents.StreamEventMissingParent:
		if state.missing == nil {
			state.missing = make(map[kvevents.StreamBlock]struct{})
		}
		for _, block := range blocks {
			state.missing[block] = struct{}{}
		}
	case kvevents.StreamEventStored, kvevents.StreamEventRemoved:
		for _, block := range blocks {
			delete(state.missing, block)
		}
	case kvevents.StreamEventCleared:
		state.missing = nil
		state.reportSupported = false
	}
	r.endpoints[endpoint] = state
}

func (r *fullReportRepair) shouldRequest(endpoint string, match repairMatch) (bool, string) {
	if r == nil {
		return false, ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	state, eligible := r.endpoints[endpoint]
	if !eligible || !state.reportSupported || match.total <= 0 {
		return false, ""
	}
	if r.cooldown > 0 && !state.lastRequest.IsZero() && r.clock.Since(state.lastRequest) < r.cooldown {
		return false, ""
	}
	reason := "integrity"
	if len(state.missing) == 0 {
		missing := match.total - match.confirmed
		if missing < r.minMissing || float64(match.confirmed)/float64(match.total) >= r.threshold {
			return false, ""
		}
		reason = "threshold"
	}
	state.lastRequest = r.clock.Now()
	r.endpoints[endpoint] = state
	return true, reason
}
