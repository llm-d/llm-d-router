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

package thunderagent

import (
	"sync"
	"time"
)

// bytesPerToken converts a request body size to an estimated prompt token
// count.
const bytesPerToken = 4.0

// maintenanceInterval bounds how often the full-table maintenance (TTL
// eviction, stale endpoint cleanup) runs.
const maintenanceInterval = time.Second

// endpointStaleAfter drops a pod entry that holds no sessions and has not been
// seen for this long.
const endpointStaleAfter = 5 * time.Second

// session is one agent trajectory, identified by the request FairnessID.
// All fields are guarded by sessionManager.mu.
type session struct {
	// endpoint the session is bound to; nil before the first dispatch and
	// after the endpoint leaves the pool.
	endpoint *endpointState
	// committedTokens is usage.total_tokens of the last completed turn.
	committedTokens int64
	// inflightTokens is the sum of the estimates of the session's turns
	// currently being processed.
	inflightTokens int64
	lastResponseAt time.Time
	lastActivity   time.Time
	turnCount      int64
}

// undecayed is the session's KV footprint in tokens.
func (s *session) undecayed() float64 {
	if f := float64(s.inflightTokens); f > float64(s.committedTokens) {
		return f
	}
	return float64(s.committedTokens)
}

// endpointState is the plugin's own record of one endpoint.
type endpointState struct {
	id       string
	capacity float64
	// sessions are the sessions bound to this pod.
	sessions map[string]*session
	// updatedAt is the last time this pod was seen.
	updatedAt time.Time
}

// undecayedTokens is the endpoint's working set: the footprints of all its
// sessions, running and idle.
func (p *endpointState) undecayedTokens() float64 {
	var total float64
	for _, s := range p.sessions {
		total += s.undecayed()
	}
	return total
}

// sessionManager is the ledger shared by all of the thunder agent plugin's hooks/
type sessionManager struct {
	// mu guards everything below, including all session and endpointState
	// fields. The Locked suffix and the session / endpointState helpers all
	// assume the caller holds it.
	mu        sync.Mutex
	sessions  map[string]*session
	endpoints map[string]*endpointState

	ttl             time.Duration
	lastMaintenance time.Time
}

func newSessionManager(cfg Config) *sessionManager {
	return &sessionManager{
		sessions:  make(map[string]*session),
		endpoints: make(map[string]*endpointState),
		ttl:       time.Duration(cfg.EvictionTTLSeconds * float64(time.Second)),
	}
}

// ensureEndpointLocked returns the ledger entry for an endpoint, creating
// it on first sight and refreshing its capacity.
func (m *sessionManager) ensureEndpointLocked(id string, capacity float64, now time.Time) *endpointState {
	p, ok := m.endpoints[id]
	if !ok {
		p = &endpointState{id: id, sessions: make(map[string]*session)}
		m.endpoints[id] = p
	}
	p.capacity = capacity
	p.updatedAt = now
	return p
}

// bindLocked returns the session for id bound to the given endpoint,
// creating the session on first sight and moving it if it was bound
// elsewhere.
func (m *sessionManager) bindLocked(id string, ep *endpointState) *session {
	s, ok := m.sessions[id]
	if !ok {
		s = &session{}
		m.sessions[id] = s
	}
	if s.endpoint != ep {
		if s.endpoint != nil {
			delete(s.endpoint.sessions, id)
		}
		s.endpoint = ep
		ep.sessions[id] = s
	}
	return s
}

// removeLocked drops a session from the ledger and from its endpoint.
func (m *sessionManager) removeLocked(id string) {
	s, ok := m.sessions[id]
	if !ok {
		return
	}
	if s.endpoint != nil {
		delete(s.endpoint.sessions, id)
	}
	delete(m.sessions, id)
}

// estimateTokens converts a request body size to a token estimate.
func estimateTokens(sizeBytes int) int64 {
	if sizeBytes <= 0 {
		return 0
	}
	return int64(float64(sizeBytes) / bytesPerToken)
}

// gaugeSnapshot is the ledger state the metrics collector reports.
type gaugeSnapshot struct {
	endpoints     map[string]endpointGauge
	running, idle int
}

type endpointGauge struct {
	undecayed float64
	capacity  float64
}

// maintainLocked is the housekeeping pass, run at most once per
// maintenanceInterval by whichever hook holds the lock: drop sessions idle
// past the TTL and drop empty stale endpoints.
func (m *sessionManager) maintainLocked(now time.Time) {
	if now.Sub(m.lastMaintenance) < maintenanceInterval {
		return
	}
	m.lastMaintenance = now

	for id, s := range m.sessions {
		if s.inflightTokens > 0 {
			continue
		}
		if now.Sub(s.lastActivity) <= m.ttl {
			continue
		}
		m.removeLocked(id)
	}
	for id, p := range m.endpoints {
		if len(p.sessions) == 0 && now.Sub(p.updatedAt) > endpointStaleAfter {
			delete(m.endpoints, id)
		}
	}
}

// snapshot runs due maintenance, so a scrape keeps the ledger current without
// traffic, and returns the values to report.
func (m *sessionManager) snapshot(now time.Time) gaugeSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.maintainLocked(now)

	snap := gaugeSnapshot{endpoints: make(map[string]endpointGauge, len(m.endpoints))}
	for _, s := range m.sessions {
		if s.inflightTokens > 0 {
			snap.running++
		} else {
			snap.idle++
		}
	}
	for id, p := range m.endpoints {
		snap.endpoints[id] = endpointGauge{undecayed: p.undecayedTokens(), capacity: p.capacity}
	}
	return snap
}
