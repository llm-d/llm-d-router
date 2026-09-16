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

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
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

// fullReportRequest returns the payload mutation that asks the serving engine
// for a full KV-cache report, or false when the request body cannot carry it.
type fullReportRequest func(body *fwkrh.InferenceRequestBody) (func(fwkrh.PayloadMap), bool)

// fullReportRequestFor validates the KV-events prerequisites of repair and
// returns the report request of the configured engine.
func fullReportRequestFor(config *kvevents.Config) (fullReportRequest, error) {
	if config == nil || !config.DiscoverPods || config.PodDiscoveryConfig == nil {
		return nil, errors.New("fullReportRepair requires kvEventsConfig.discoverPods with podDiscoveryConfig")
	}
	if config.ZMQEndpoint != "" {
		return nil, errors.New("fullReportRepair does not support kvEventsConfig.zmqEndpoint global-socket mode")
	}
	if config.PodDiscoveryConfig.EffectiveReplayPort() > 0 {
		return nil, errors.New("fullReportRepair does not support kvEventsConfig.podDiscoveryConfig.replaySocketPort")
	}
	switch config.EngineType {
	case "", engineadapter.EngineTypeVLLM:
		return vllmFullReport, nil
	default:
		return nil, fmt.Errorf("fullReportRepair requires kvEventsConfig.engineType %q, got %q",
			engineadapter.EngineTypeVLLM, config.EngineType)
	}
}

// endpointRepairState is one endpoint's report eligibility and open faults.
type endpointRepairState struct {
	// missing holds blocks dropped for a missing parent that no later store,
	// removal, or cache reset has resolved.
	missing map[kvevents.StreamBlock]struct{}
	// reportSupported records that the endpoint's stream tags store origins.
	reportSupported bool
	lastRequest     time.Time
}

// fullReportRepair decides when a request asks its endpoint for a full
// KV-cache report. State is keyed by endpoint address and follows the pool's
// stream transitions: an origin-tagged store makes an endpoint eligible, a
// missing-parent drop arms a fault, and a cache reset, including the reset
// queued when a subscriber is removed, deletes the endpoint's state.
type fullReportRepair struct {
	mu             sync.Mutex
	endpoints      map[string]*endpointRepairState
	threshold      float64
	minMissing     int
	cooldown       time.Duration
	prefillProfile string
	request        fullReportRequest
	clock          clock.PassiveClock
}

func newFullReportRepair(config FullReportRepairConfig, cooldown time.Duration, request fullReportRequest) *fullReportRepair {
	if config.PrefillProfile == "" {
		config.PrefillProfile = experimentalPrefillProfile
	}
	return &fullReportRepair{
		endpoints:      make(map[string]*endpointRepairState),
		threshold:      config.FullReportThreshold,
		minMissing:     config.MinMissingBlocks,
		cooldown:       cooldown,
		prefillProfile: config.PrefillProfile,
		request:        request,
		clock:          clock.RealClock{},
	}
}

func (r *fullReportRepair) observe(endpoint string, event kvevents.StreamEvent, blocks ...kvevents.StreamBlock) {
	r.mu.Lock()
	defer r.mu.Unlock()
	state := r.endpoints[endpoint]
	if state == nil {
		if event != kvevents.StreamEventReportSupported && event != kvevents.StreamEventMissingParent {
			return
		}
		state = &endpointRepairState{}
		r.endpoints[endpoint] = state
	}
	switch event {
	case kvevents.StreamEventCleared:
		delete(r.endpoints, endpoint)
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
	}
}

// shouldRequest reports whether a request routed to endpoint should ask for a
// full report, and why. It does not start the cooldown; reserve does.
//
// The endpoint's stream must tag store origins, because only tagged reports
// restore residency without adding dedup references, and its cooldown must
// have elapsed. The reason is then one of:
//
//   - "integrity" while a missing-parent fault is open. The request needs at
//     least one complete block but no minimum gap, since the dropped blocks'
//     prefix is unknown. A fault stays open until its blocks are stored,
//     removed, or cleared, so reports repeat at the cooldown rate until one
//     comes from a request that reuses the affected prefix.
//   - "threshold" when the request's confirmed prefix on the endpoint covers
//     less than threshold of its blocks and misses at least minMissing. The
//     gap is either an uncached prompt or stores the index missed, such as
//     those published before the subscriber attached. The engine re-announces
//     only blocks the request reuses, so an uncached prompt yields a small
//     report and an under-indexed one restores its prefix. The floor and ratio
//     keep small gaps and mostly indexed prompts from spending the cooldown.
func (r *fullReportRepair) shouldRequest(endpoint string, match repairMatch) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	state := r.endpoints[endpoint]
	if state == nil || !state.reportSupported || match.total <= 0 || r.coolingDown(state) {
		return "", false
	}
	if len(state.missing) > 0 {
		return "integrity", true
	}
	missing := match.total - match.confirmed
	if missing < r.minMissing || float64(match.confirmed)/float64(match.total) >= r.threshold {
		return "", false
	}
	return "threshold", true
}

// reserve starts endpoint's cooldown and reports whether no concurrent request
// started it first.
func (r *fullReportRepair) reserve(endpoint string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	state := r.endpoints[endpoint]
	if state == nil || r.coolingDown(state) {
		return false
	}
	state.lastRequest = r.clock.Now()
	return true
}

func (r *fullReportRepair) coolingDown(state *endpointRepairState) bool {
	return !state.lastRequest.IsZero() && r.clock.Since(state.lastRequest) < r.cooldown
}
