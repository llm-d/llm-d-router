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

// Package sessionstate provides a DataProducer that tracks agentic session
// history and publishes it for session-aware scheduling plugins.
package sessionstate

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	sessionstateconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionstate/constants"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

const (
	defaultEvictionTTL           = time.Hour
	defaultEvictionSweepInterval = 5 * time.Minute
)

// SessionStateProducerType is the plugin type registered with the framework.
const SessionStateProducerType = sessionstateconstants.SessionStateProducerType

// Parameters configures idle session eviction.
type Parameters struct {
	// EvictionTTLSeconds is the maximum idle time before session state is removed.
	// Zero disables eviction.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds,omitempty"`
	// EvictionSweepSeconds is how often idle session state is scanned.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds,omitempty"`
}

func (p Parameters) validate() error {
	if p.EvictionTTLSeconds < 0 {
		return fmt.Errorf("evictionTtlSeconds must be >= 0, got %v", p.EvictionTTLSeconds)
	}
	if p.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", p.EvictionSweepSeconds)
	}
	return nil
}

var (
	_ requestcontrol.DataProducer = &Producer{}
	_ requestcontrol.PreRequest   = &Producer{}
)

type sessionRecord struct {
	turnsTaken  int64
	firstSeenAt time.Time
	lastSeenAt  time.Time
}

// Producer tracks session history within one EPP instance.
type Producer struct {
	typedName fwkplugin.TypedName
	dk        fwkplugin.DataKey

	mu       sync.Mutex
	sessions map[string]*sessionRecord

	evictionTTL           time.Duration
	evictionSweepInterval time.Duration
}

// Factory builds a session-state producer.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{
		EvictionTTLSeconds:   defaultEvictionTTL.Seconds(),
		EvictionSweepSeconds: defaultEvictionSweepInterval.Seconds(),
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", SessionStateProducerType, name, err)
		}
	}
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", SessionStateProducerType, name, err)
	}

	p := &Producer{
		typedName:             fwkplugin.TypedName{Type: SessionStateProducerType, Name: name},
		dk:                    attrsession.SessionStateDataKey.WithNonEmptyProducerName(name),
		sessions:              make(map[string]*sessionRecord),
		evictionTTL:           time.Duration(params.EvictionTTLSeconds * float64(time.Second)),
		evictionSweepInterval: time.Duration(params.EvictionSweepSeconds * float64(time.Second)),
	}
	if handle != nil && p.evictionTTL > 0 {
		go p.runEviction(handle.Context())
	}
	return p, nil
}

// TypedName returns the type and name of the plugin.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the SessionState request attribute written by this producer.
func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.dk: attrsession.SessionState{}}
}

// Produce publishes the history observed before the current request is
// dispatched, then marks the session as seen at the current time.
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	identity, ok := readAgentIdentity(request)
	if !ok {
		return nil
	}

	now := time.Now()
	p.mu.Lock()
	record, exists := p.sessions[identity]
	if !exists {
		record = &sessionRecord{
			firstSeenAt: now,
			lastSeenAt:  now,
		}
		p.sessions[identity] = record
	}
	state := attrsession.SessionState{
		TurnsTaken: record.turnsTaken,
		Duration:   now.Sub(record.firstSeenAt),
		LastSeenAt: record.lastSeenAt,
	}
	record.lastSeenAt = now
	p.mu.Unlock()

	request.PutAttribute(p.dk, state)
	return nil
}

// PreRequest records one dispatched turn. A request counts once even when
// several profiles run.
func (p *Producer) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	identity, ok := readAgentIdentity(request)
	if !ok {
		return nil
	}

	now := time.Now()
	p.mu.Lock()
	record, exists := p.sessions[identity]
	if !exists {
		record = &sessionRecord{
			firstSeenAt: now,
			lastSeenAt:  now,
		}
		p.sessions[identity] = record
	}
	record.turnsTaken++
	p.mu.Unlock()
	return nil
}

func readAgentIdentity(request *fwksched.InferenceRequest) (string, bool) {
	if request == nil {
		return "", false
	}
	identity, ok := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return identity, ok && identity != ""
}

func (p *Producer) runEviction(ctx context.Context) {
	ticker := time.NewTicker(p.evictionSweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.evictIdle(time.Now())
		}
	}
}

func (p *Producer) evictIdle(now time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for identity, record := range p.sessions {
		if p.evictionTTL > 0 && now.Sub(record.lastSeenAt) > p.evictionTTL {
			delete(p.sessions, identity)
		}
	}
}
