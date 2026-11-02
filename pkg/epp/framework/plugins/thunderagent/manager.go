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
	"sort"
	"sync"
	"time"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// endpointStaleAfter drops a pod entry that holds no sessions and has not been
// seen for this long.
const endpointStaleAfter = 5 * time.Second

// reservationTTL bounds an admission reservation whose session never reached
// PreRequest, such as a request cancelled between dispatch and binding.
// Counting a cancelled request as occupancy would be phantom load.
const reservationTTL = 5 * time.Second

// sessionClass ranks a waiting session for admission. Lower dispatches
// first.
type sessionClass int

const (
	// classAdmitted is a session already admitted to a pod: bound to it,
	// at least one turn dispatched, not paused. Its footprint is already
	// counted and finishing its trajectory is what frees capacity, so its
	// turns always dispatch.
	classAdmitted sessionClass = iota
	// classPaused gave its room to another session's turn: its next turn
	// must fit its own pod again, but it outranks never-admitted sessions.
	classPaused
	// classNew is not admitted to any pod: it never dispatched, or its pod
	// left the pool. Admitted only when a pod has room.
	classNew
)

func (c sessionClass) String() string {
	switch c {
	case classAdmitted:
		return "admitted"
	case classPaused:
		return "paused"
	default:
		return "new"
	}
}

// lane holds the accounting of one agent inside a session. Turns that overlap
// in time belong to different agents, because a turn cannot start before the
// previous turn of its own conversation answers, so a lane serves at most one
// turn at a time and its totals are never shared between agents.
type lane struct {
	// committed is usage.total_tokens of the last turn settled in this lane.
	committed int64
	// inflight is the estimate of the turn in flight in this lane, zero when
	// the lane is free.
	inflight int64
}

// size is the lane's KV footprint in tokens. The turn in flight resends the
// history the committed total already covers, so the larger of the two is
// taken.
func (l lane) size() float64 {
	if f := float64(l.inflight); f > float64(l.committed) {
		return f
	}
	return float64(l.committed)
}

func (l lane) free() bool {
	return l.inflight == 0
}

// session is one agent trajectory, identified by its agent-identity session id.
// All fields are guarded by sessionManager.mu.
type session struct {
	endpoint *endpointState
	// lanes holds one entry per agent observed in this session: the main agent
	// and any sub-agent that ran in parallel with it. Their count is bounded by
	// the session's peak number of simultaneous turns.
	lanes          []lane
	lastActivity   time.Time
	lastResponseAt time.Time
	turnCount      int64
	// paused is set when Pick reclaims the session's room for another turn:
	// the session stops counting against its endpoint, and its next turn
	// must pass the fit check again before it dispatches.
	paused bool
	// Until reservedUntil, a session whose turn Pick has dispatched but
	// PreRequest has not yet charged counts reservedTokens against its
	// endpoint.
	reservedTokens float64
	reservedUntil  time.Time
}

// size is the session's KV footprint in tokens: its agents hold separate
// histories on the pod, so their lanes add up.
func (s *session) size() float64 {
	var total float64
	for _, l := range s.lanes {
		total += l.size()
	}
	return total
}

// sizeWithNewTurn is the footprint the session would have once a turn of the
// given estimate is charged, which is what the admission gate weighs before
// PreRequest picks the lane. Both read the same rule through this method.
func (s *session) sizeWithNewTurn(estimate int64) float64 {
	i := s.laneFor()
	var total float64
	for j, l := range s.lanes {
		if j == i {
			l.inflight += estimate
		}
		total += l.size()
	}
	if i == len(s.lanes) {
		total += lane{inflight: estimate}.size()
	}
	return total
}

// laneFor returns the lane a turn takes: the first free one, or len(s.lanes)
// when every lane is busy, which opens a lane for a turn that overlaps the
// others and so belongs to another agent. Which free lane a turn reuses does
// not matter, because collapse folds the free lanes together as soon as the
// session falls quiet.
func (s *session) laneFor() int {
	for i, l := range s.lanes {
		if l.free() {
			return i
		}
	}
	return len(s.lanes)
}

// collapse folds a quiet session's lanes into one holding the largest
// committed total.
//
// Overlap is the only evidence the ledger has that a session carries several
// agents, and it is gone once nothing is in flight. Keeping a lane past that
// point would hold a history no later turn can be matched to, so a session
// that opened lanes once would count them for as long as it stayed active:
// its turns refresh lastActivity, so the idle sweep never reaches it. Folding
// to the largest total keeps the biggest agent's history, which is the part
// the engine is most likely still holding.
func (s *session) collapse() {
	if len(s.lanes) <= 1 {
		return
	}
	var largest int64
	for _, l := range s.lanes {
		if !l.free() {
			return
		}
		largest = max(largest, l.committed)
	}
	s.lanes = s.lanes[:1]
	s.lanes[0] = lane{committed: largest}
}

// inflightTokens is the estimate of every turn the session has in flight.
func (s *session) inflightTokens() int64 {
	var total int64
	for _, l := range s.lanes {
		total += l.inflight
	}
	return total
}

// committedTokens is the settled total across the session's agents.
func (s *session) committedTokens() int64 {
	var total int64
	for _, l := range s.lanes {
		total += l.committed
	}
	return total
}

// charge adds a turn's estimate to its lane and returns the lane's index, so
// the matching settle lands on the lane that was charged.
func (s *session) charge(estimate int64) int {
	i := s.laneFor()
	if i == len(s.lanes) {
		s.lanes = append(s.lanes, lane{})
	}
	s.lanes[i].inflight += estimate
	return i
}

// settle removes a turn's estimate from its lane and replaces that lane's
// committed total, keeping the replace semantics one agent needs: a compacted
// history reports a smaller total and the footprint follows it down.
//
// Lanes are never removed, because turns in flight hold their lane index: a
// free lane is a slot the session's next turn reuses, and the whole session is
// dropped by the idle sweep.
func (s *session) settle(index int, estimate, total int64) {
	if index < 0 || index >= len(s.lanes) {
		return
	}
	l := &s.lanes[index]
	l.inflight = max(l.inflight-estimate, 0)
	switch {
	case total > 0:
		l.committed = total
	case estimate > l.committed:
		l.committed = estimate
	}
	s.collapse()
}

// class ranks the session for admission. The order of the checks matters: a
// session whose pod left the pool re-enters as new even if it was paused.
func (s *session) class() sessionClass {
	if s.endpoint == nil || s.turnCount == 0 {
		return classNew
	}
	if s.paused {
		return classPaused
	}
	return classAdmitted
}

// footprint is the accounting rule: an unexpired reservation counts its
// reserved size, a paused session counts nothing, everything else counts
// its full footprint.
func (s *session) footprint(now time.Time) float64 {
	switch {
	case now.Before(s.reservedUntil):
		return s.reservedTokens
	case s.paused:
		return 0
	}
	return s.size()
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

// occupancy is the endpoint's working set under the accounting rule.
func (p *endpointState) occupancy(now time.Time) float64 {
	var total float64
	for _, s := range p.sessions {
		total += s.footprint(now)
	}
	return total
}

// reclaimableLocked reports whether the session's room can be taken for
// another turn: it is admitted and counted in full, has no turn in flight,
// and has been idle for at least the idle lease since its last response.
func (m *sessionManager) reclaimableLocked(s *session, now time.Time) bool {
	return s.turnCount > 0 && !s.paused && !now.Before(s.reservedUntil) && s.inflightTokens() == 0 &&
		now.Sub(s.lastResponseAt) >= m.idleLease
}

// idleSessionsLocked returns the endpoint's sessions whose room can be
// reclaimed. Sessions with a turn queued are skipped: they are about to be
// active.
func (m *sessionManager) idleSessionsLocked(p *endpointState, now time.Time, queued map[string]bool) []*session {
	var idle []*session
	for id, s := range p.sessions {
		if m.reclaimableLocked(s, now) && !queued[id] {
			idle = append(idle, s)
		}
	}
	return idle
}

func (m *sessionManager) reclaimableTokensLocked(p *endpointState, now time.Time, queued map[string]bool) float64 {
	var total float64
	for id, s := range p.sessions {
		if m.reclaimableLocked(s, now) && !queued[id] {
			total += s.size()
		}
	}
	return total
}

// reclaimLocked pauses the endpoint's reclaimable sessions, longest idle
// first, until room covers need, and returns how many it paused.
func (m *sessionManager) reclaimLocked(p *endpointState, now time.Time, queued map[string]bool, room, need float64) int {
	if room >= need {
		return 0
	}
	idle := m.idleSessionsLocked(p, now, queued)
	sort.Slice(idle, func(i, j int) bool { return idle[i].lastResponseAt.Before(idle[j].lastResponseAt) })
	paused := 0
	for _, s := range idle {
		if room >= need {
			break
		}
		s.paused = true
		room += s.size()
		paused++
	}
	return paused
}

// sessionManager is the ledger shared by all of the thunder agent plugin's hooks.
type sessionManager struct {
	// mu guards everything below, including all session and endpointState
	// fields. The Locked suffix and the session / endpointState helpers all
	// assume the caller holds it.
	mu        sync.Mutex
	sessions  map[string]*session
	endpoints map[string]*endpointState

	ttl time.Duration
	// idleLease is how long a session keeps its room after its last response.
	idleLease time.Duration
}

func newSessionManager(cfg Config) *sessionManager {
	return &sessionManager{
		sessions:  make(map[string]*session),
		endpoints: make(map[string]*endpointState),
		ttl:       time.Duration(cfg.EvictionTTLSeconds * float64(time.Second)),
		idleLease: time.Duration(cfg.IdleLeaseSeconds * float64(time.Second)),
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

// estimateTokens converts a request body size to a token estimate. The body
// also holds the JSON envelope, tool schemas and images, so the estimate is
// above the prompt token count.
// TODO(#3221): use the request's token count once it is available before flow
// control.
func estimateTokens(sizeBytes int) int64 {
	if sizeBytes <= 0 {
		return 0
	}
	return int64(sizeBytes / fwkrh.BytesPerToken)
}

// gaugeSnapshot is the ledger state the metrics collector reports.
type gaugeSnapshot struct {
	endpoints             map[string]endpointGauge
	running, idle, paused int
	// lanes is the total across sessions, so the state dump shows how many
	// agents the ledger is tracking without naming any of them.
	lanes int
}

type endpointGauge struct {
	workingSet float64
	capacity   float64
}

// sweep drops sessions with no turn in flight that have been idle past the
// TTL, and empty endpoints not seen for endpointStaleAfter.
func (m *sessionManager) sweep(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for id, s := range m.sessions {
		// Keep live reservations: a session Pick admitted has no activity yet,
		// so once its reservation expires unconfirmed the TTL below drops it.
		if now.Before(s.reservedUntil) {
			continue
		}
		if s.inflightTokens() > 0 {
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

// removeEndpoint drops an endpoint that left the pool. Its sessions are
// unbound and keep their footprint, so their next turn binds them to the pod
// that serves it.
func (m *sessionManager) removeEndpoint(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.endpoints[id]
	if !ok {
		return
	}
	for _, s := range p.sessions {
		s.endpoint = nil
	}
	delete(m.endpoints, id)
}

// snapshot returns the values to report.
func (m *sessionManager) snapshot(now time.Time) gaugeSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()

	snap := gaugeSnapshot{endpoints: make(map[string]endpointGauge, len(m.endpoints))}
	for _, s := range m.sessions {
		snap.lanes += len(s.lanes)
		switch {
		case s.paused:
			snap.paused++
		case s.inflightTokens() > 0:
			snap.running++
		default:
			snap.idle++
		}
	}
	for id, p := range m.endpoints {
		snap.endpoints[id] = endpointGauge{workingSet: p.occupancy(now), capacity: p.capacity}
	}
	return snap
}
