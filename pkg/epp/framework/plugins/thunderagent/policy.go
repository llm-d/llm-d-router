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
	"slices"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
)

// reservationTTL caps how long Pick's room reservation holds between Pick
// and PreRequest so a cancelled request cannot hold pod capacity indefinitely.
const reservationTTL = 5 * time.Second

type sessionClass int

const (
	classAdmitted sessionClass = iota
	classPaused
	classNew
)

func (c sessionClass) String() string {
	switch c {
	case classAdmitted:
		return "admitted"
	case classPaused:
		return "paused"
	case classNew:
		return "new"
	default:
		return "unknown"
	}
}

// localSession holds ThunderAgent's policy overlays (pause state and
// short-lived dispatch reservations) on top of session-state-producer's facts.
type localSession struct {
	paused           bool
	pausedAtTurns    int64
	reservedEndpoint string
	reservedTokens   float64
	reservedUntil    time.Time
	reservedAtTurns  int64
	reservedUnbound  bool
}

func (l *localSession) pausedLive(state sessionstate.SessionState) bool {
	return l != nil && l.paused && state.Endpoint != "" && state.TurnsTaken == l.pausedAtTurns
}

func (l *localSession) reservationLive(state sessionstate.SessionState, now time.Time) bool {
	return l != nil && !l.reservedUntil.IsZero() && now.Before(l.reservedUntil) && state.TurnsTaken == l.reservedAtTurns
}

func (l *localSession) clearReservation() {
	l.reservedEndpoint = ""
	l.reservedTokens = 0
	l.reservedUntil = time.Time{}
	l.reservedAtTurns = 0
	l.reservedUnbound = false
}

type endpointView struct {
	id       string
	capacity float64
	load     float64
	sessions []sessionstate.SessionState
}

type poolView struct {
	pods  map[string]*endpointView
	bound map[string]sessionstate.SessionState
}

func (a *ThunderAgent) ensureLocalLocked(id string) *localSession {
	l, ok := a.local[id]
	if !ok {
		l = &localSession{}
		a.local[id] = l
	}
	return l
}

func knownStateLocked(id string, bound, candStates map[string]sessionstate.SessionState, l *localSession) (sessionstate.SessionState, bool) {
	if s, ok := bound[id]; ok {
		return s, true
	}
	if candStates != nil {
		if s, ok := candStates[id]; ok {
			return s, true
		}
	}
	if l != nil && l.reservedUnbound {
		return sessionstate.SessionState{TurnsTaken: l.reservedAtTurns}, true
	}
	return sessionstate.SessionState{}, false
}

func (a *ThunderAgent) refreshLocked(id string, state sessionstate.SessionState, hasState bool, pods map[string]*endpointView, now time.Time) {
	l := a.local[id]
	if l == nil {
		return
	}
	if l.paused && (!hasState || !l.pausedLive(state) || (pods != nil && state.Endpoint != "" && pods[state.Endpoint] == nil)) {
		l.paused = false
		l.pausedAtTurns = 0
	}
	if !l.reservedUntil.IsZero() {
		stillReserved := now.Before(l.reservedUntil) &&
			hasState &&
			state.TurnsTaken == l.reservedAtTurns &&
			(l.reservedUnbound || (state.Endpoint == l.reservedEndpoint && (pods == nil || pods[state.Endpoint] != nil))) &&
			(pods == nil || pods[l.reservedEndpoint] != nil)
		if !stillReserved {
			l.clearReservation()
		}
	}
	if !l.paused && l.reservedUntil.IsZero() {
		delete(a.local, id)
	}
}

func (a *ThunderAgent) refreshPoolLocked(pool *poolView, candStates map[string]sessionstate.SessionState, now time.Time) {
	for id, l := range a.local {
		state, hasState := knownStateLocked(id, pool.bound, candStates, l)
		a.refreshLocked(id, state, hasState, pool.pods, now)
	}
}

func (a *ThunderAgent) readPoolLocked(now time.Time, candStates map[string]sessionstate.SessionState) *poolView {
	pool := &poolView{
		pods:  make(map[string]*endpointView, len(a.endpoints)),
		bound: make(map[string]sessionstate.SessionState),
	}
	for id, entry := range a.endpoints {
		sessions, _ := sessionstate.ReadEndpointSessions(entry.ep)
		pv := &endpointView{
			id:       id,
			capacity: entry.capacity,
			sessions: sessions,
		}
		pool.pods[id] = pv
		for _, s := range sessions {
			pool.bound[s.SessionID] = s
		}
	}

	a.refreshPoolLocked(pool, candStates, now)

	for _, pv := range pool.pods {
		for _, s := range pv.sessions {
			pv.load += a.footprintLocked(s, a.local[s.SessionID], now)
		}
	}
	for id, l := range a.local {
		pv := pool.pods[l.reservedEndpoint]
		if pv == nil {
			continue
		}
		state, hasState := knownStateLocked(id, pool.bound, candStates, l)
		if !hasState || !l.reservationLive(state, now) {
			continue
		}
		if boundState, onSamePod := pool.bound[id]; onSamePod && boundState.Endpoint == pv.id {
			continue
		}
		pv.load += l.reservedTokens
	}
	return pool
}

func (a *ThunderAgent) footprintLocked(state sessionstate.SessionState, l *localSession, now time.Time) float64 {
	if l.pausedLive(state) {
		return 0
	}
	fp := float64(state.ContextSize())
	if l.reservationLive(state, now) && l.reservedEndpoint == state.Endpoint {
		return max(fp, l.reservedTokens)
	}
	return fp
}

func (a *ThunderAgent) reclaimableLocked(state sessionstate.SessionState, l *localSession, now time.Time, queued map[string]bool) bool {
	if queued[state.SessionID] {
		return false
	}
	if state.TurnsTaken <= 0 || state.InFlightRequests > 0 {
		return false
	}
	if l.pausedLive(state) || l.reservationLive(state, now) {
		return false
	}
	return now.Sub(state.LastResponseAt) >= a.idleGracePeriod
}

func (a *ThunderAgent) reclaimableTokensLocked(pv *endpointView, now time.Time, queued map[string]bool) float64 {
	var total float64
	for _, s := range pv.sessions {
		l := a.local[s.SessionID]
		if a.reclaimableLocked(s, l, now, queued) {
			total += a.footprintLocked(s, l, now)
		}
	}
	return total
}

func (a *ThunderAgent) reclaimLocked(pv *endpointView, now time.Time, queued map[string]bool, need float64) int {
	room := a.roomLocked(pv)
	if need <= room {
		return 0
	}
	deficit := need - room

	var victims []sessionstate.SessionState
	for _, s := range pv.sessions {
		if a.reclaimableLocked(s, a.local[s.SessionID], now, queued) && s.ContextSize() > 0 {
			victims = append(victims, s)
		}
	}
	slices.SortFunc(victims, func(x, y sessionstate.SessionState) int {
		if cmp := x.LastResponseAt.Compare(y.LastResponseAt); cmp != 0 {
			return cmp
		}
		return strings.Compare(x.SessionID, y.SessionID)
	})

	paused := 0
	for _, v := range victims {
		if deficit <= 0 {
			break
		}
		l := a.ensureLocalLocked(v.SessionID)
		fp := a.footprintLocked(v, l, now)
		l.paused = true
		l.pausedAtTurns = v.TurnsTaken
		pv.load -= fp
		deficit -= fp
		paused++
	}
	return paused
}

func (a *ThunderAgent) findBoundSessionLocked(id string) (sessionstate.SessionState, bool) {
	for _, entry := range a.endpoints {
		sessions, ok := sessionstate.ReadEndpointSessions(entry.ep)
		if !ok {
			continue
		}
		for _, s := range sessions {
			if s.SessionID == id {
				return s, true
			}
		}
	}
	return sessionstate.SessionState{}, false
}

func (a *ThunderAgent) deleteEndpointLocked(id string) {
	delete(a.endpoints, id)
	delete(a.prefillSeen, id)
}

func (a *ThunderAgent) pruneStaleEndpointsLocked(now time.Time) {
	for id, entry := range a.endpoints {
		if now.Sub(entry.updatedAt) > endpointStaleAfter {
			a.deleteEndpointLocked(id)
		}
	}
}

func (a *ThunderAgent) sweep() {
	a.mu.Lock()
	defer a.mu.Unlock()
	now := a.clock.Now()
	a.pruneStaleEndpointsLocked(now)
	_ = a.readPoolLocked(now, nil)
}

type ledgerSnapshot struct {
	running   int
	idle      int
	paused    int
	endpoints map[string]endpointSnapshot
}

type endpointSnapshot struct {
	workingSet float64
	capacity   float64
}

func (a *ThunderAgent) snapshot() ledgerSnapshot {
	a.mu.Lock()
	defer a.mu.Unlock()

	now := a.clock.Now()
	a.pruneStaleEndpointsLocked(now)
	pool := a.readPoolLocked(now, nil)

	snap := ledgerSnapshot{
		endpoints: make(map[string]endpointSnapshot, len(a.endpoints)),
	}
	for id, entry := range a.endpoints {
		workingSet := 0.0
		if pv := pool.pods[id]; pv != nil {
			workingSet = pv.load
		}
		snap.endpoints[id] = endpointSnapshot{
			workingSet: workingSet,
			capacity:   entry.capacity,
		}
	}
	for id, s := range pool.bound {
		switch {
		case s.InFlightRequests > 0:
			snap.running++
		case a.local[id].pausedLive(s):
			snap.paused++
		default:
			snap.idle++
		}
	}
	return snap
}
