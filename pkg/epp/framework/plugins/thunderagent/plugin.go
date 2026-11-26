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

// Package thunderagent tracks the KV cache footprint of agentic sessions on
// each pod, based on ThunderAgent (arXiv 2602.13692).
//
// A session (an agent trajectory identified by the session id the
// agent-identity plugin publishes) has its pod binding and KV token footprint
// tracked by session-state-producer. The engine's own KV utilization cannot
// serve as an admission ceiling: a session waiting on a tool call still owns
// its context in the prefix cache, but those blocks sit on the free list and
// are reported as unused, so on an agentic workload the reported utilization
// stays low while the cache is in fact full. ThunderAgent counts idle
// sessions too and admits, pauses, and pins turns using those session facts.
//
// This file wires the plugin; policy overlays and pool projection live in
// policy.go.
package thunderagent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"k8s.io/utils/clock"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

// ThunderAgentPluginType is the plugin type registered with the framework.
const ThunderAgentPluginType = "thunder-agent"

var (
	_ fwkplugin.ConsumerPlugin = &ThunderAgent{}
	_ fwkplugin.StateDumper    = &ThunderAgent{}
	_ fwksched.Scorer          = &ThunderAgent{}
	_ fwkfc.SaturationDetector = &ThunderAgent{}
	_ fwkfc.FairnessPolicy     = &ThunderAgent{}
)

// ThunderAgent is a single named instance shared by every hookup, so all of
// them read and write the same admission policy state.
type ThunderAgent struct {
	typedName fwkplugin.TypedName
	clock     clock.PassiveClock

	fallbackCapacityTokens float64
	usableCapacityRatio    float64
	idleGracePeriod        time.Duration
	forceAdmitTimeout      time.Duration

	mu          sync.Mutex
	endpoints   map[string]*trackedEndpoint
	local       map[string]*localSession
	prefillSeen map[string]bool

	metrics *thunderMetrics
}

// Factory builds a ThunderAgent from raw plugin parameters, registers its
// metrics and starts the policy state sweep for the plugin's lifetime.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", ThunderAgentPluginType, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", ThunderAgentPluginType, err)
	}

	a := newThunderAgent(name, cfg)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := a.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", ThunderAgentPluginType, err)
			}
		}
		go a.runSweep(handle.Context(), sweepInterval)
	}
	return a, nil
}

// runSweep runs the local state sweep every interval until ctx is cancelled.
func (a *ThunderAgent) runSweep(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			a.sweep()
		}
	}
}

func newThunderAgent(name string, cfg Config) *ThunderAgent {
	return newThunderAgentWithClock(name, cfg, clock.RealClock{})
}

func newThunderAgentWithClock(name string, cfg Config, clk clock.PassiveClock) *ThunderAgent {
	a := &ThunderAgent{
		typedName:              fwkplugin.TypedName{Type: ThunderAgentPluginType, Name: name},
		clock:                  clk,
		fallbackCapacityTokens: float64(cfg.FallbackCapacityTokens),
		usableCapacityRatio:    cfg.UsableCapacityRatio,
		idleGracePeriod:        cfg.IdleGracePeriod.Duration,
		forceAdmitTimeout:      cfg.ForceAdmitTimeout.Duration,
		endpoints:              make(map[string]*trackedEndpoint),
		local:                  make(map[string]*localSession),
		prefillSeen:            make(map[string]bool),
	}
	a.metrics = newThunderMetrics(a)
	return a
}

func (a *ThunderAgent) TypedName() fwkplugin.TypedName {
	return a.typedName
}

// Consumes declares agent identity and session-state-producer outputs as
// required inputs so configuration loading auto-provisions or validates them.
func (a *ThunderAgent) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			agentidentity.AgentIdentityKey:       "",
			sessionstate.SessionStateDataKey:     sessionstate.SessionState{},
			sessionstate.EndpointSessionsDataKey: sessionstate.EndpointSessions{},
		},
	}
}

// sessionID returns the session identifier published by the agent-identity
// plugin, or "" for requests carrying none. The fairness ID is not used: an
// explicit fairness header often carries a tenant rather than a session.
// Anonymous traffic is not tracked.
func sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	id, _ := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return id
}

// stateDump is the snapshot returned by DumpState. Session ids come from
// client headers, so they are omitted; sessions are only counted.
type stateDump struct {
	RunningSessions int                     `json:"runningSessions"`
	IdleSessions    int                     `json:"idleSessions"`
	PausedSessions  int                     `json:"pausedSessions"`
	Endpoints       map[string]endpointDump `json:"endpoints"`
}

type endpointDump struct {
	WorkingSetTokens float64 `json:"workingSetTokens"`
	CapacityTokens   float64 `json:"capacityTokens"`
}

// DumpState reports session counts and each endpoint's working set and
// capacity, the same values as the metrics.
func (a *ThunderAgent) DumpState() (json.RawMessage, error) {
	snap := a.snapshot()
	dump := stateDump{
		RunningSessions: snap.running,
		IdleSessions:    snap.idle,
		PausedSessions:  snap.paused,
		Endpoints:       make(map[string]endpointDump, len(snap.endpoints)),
	}
	for id, e := range snap.endpoints {
		dump.Endpoints[id] = endpointDump{WorkingSetTokens: e.workingSet, CapacityTokens: e.capacity}
	}
	return json.Marshal(dump)
}
