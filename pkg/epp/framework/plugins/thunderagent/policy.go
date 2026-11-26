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
	endpointHint     string
	paused           bool
	pausedAtTurns    int64
	reservedEndpoint string
	reservedTokens   float64
	reservedInflight float64
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

func (l *localSession) clearPause() {
	l.paused = false
	l.pausedAtTurns = 0
}

func (l *localSession) clearReservation() {
	l.reservedEndpoint = ""
	l.reservedTokens = 0
	l.reservedInflight = 0
	l.reservedUntil = time.Time{}
	l.reservedAtTurns = 0
	l.reservedUnbound = false
}

func (l *localSession) empty() bool {
	return !l.paused && l.reservedUntil.IsZero()
}

type resolvedSession struct {
	state    sessionstate.SessionState
	hasState bool
	isBound  bool
}

type endpointView struct {
	id       string
	capacity float64
	load     float64
	room     float64
	spare    float64
	hasAttr  bool
	view     sessionstate.EndpointSessions
}

func (a *ThunderAgent) ensureLocalLocked(id string) *localSession {
	l, ok := a.local[id]
	if !ok {
		l = &localSession{}
		a.local[id] = l
	}
	return l
}

// boundSessionLocked looks up id on hintEndpoint first in O(1), falling back to
// per-endpoint lookups when hintEndpoint is empty or stale.
func (a *ThunderAgent) boundSessionLocked(id, hintEndpoint string) (sessionstate.SessionState, bool) {
	if id == "" {
		return sessionstate.SessionState{}, false
	}
	if hintEndpoint != "" {
		if entry := a.endpoints[hintEndpoint]; entry != nil {
			if view, ok := sessionstate.PeekEndpointSessions(entry.ep); ok {
				if s, ok := view.Session(id); ok {
					return s, true
				}
			}
		}
	}
	for epID, entry := range a.endpoints {
		if epID == hintEndpoint {
			continue
		}
		if view, ok := sessionstate.PeekEndpointSessions(entry.ep); ok {
			if s, ok := view.Session(id); ok {
				return s, true
			}
		}
	}
	return sessionstate.SessionState{}, false
}

func (a *ThunderAgent) localStateLocked(id string, l *localSession, candByID map[string]*candidate) resolvedSession {
	hint := ""
	if l != nil {
		hint = l.endpointHint
		if hint == "" {
			hint = l.reservedEndpoint
		}
	}
	var (
		candState sessionstate.SessionState
		hasCand   bool
	)
	if c := candByID[id]; c != nil && c.hasState {
		candState, hasCand = c.state, true
	}
	if hint == "" && hasCand {
		hint = candState.Endpoint
	}
	if s, ok := a.boundSessionLocked(id, hint); ok {
		return resolvedSession{state: s, hasState: true, isBound: true}
	}
	if hasCand {
		return resolvedSession{state: candState, hasState: true}
	}
	if l != nil && l.reservedUnbound {
		return resolvedSession{
			state:    sessionstate.SessionState{TurnsTaken: l.reservedAtTurns},
			hasState: true,
		}
	}
	return resolvedSession{}
}

func (a *ThunderAgent) refreshOneLocked(id string, l *localSession, res resolvedSession, checkPool bool, now time.Time) bool {
	if l == nil {
		return false
	}
	boundLive := res.hasState && res.state.Endpoint != "" && (!checkPool || (res.isBound && a.endpoints[res.state.Endpoint] != nil))
	resumed := false
	if l.paused {
		switch {
		case !boundLive:
			l.clearPause()
		case res.state.TurnsTaken > l.pausedAtTurns:
			l.clearPause()
			l.endpointHint = res.state.Endpoint
			resumed = true
		}
	}
	if !l.reservedUntil.IsZero() {
		stillReserved := now.Before(l.reservedUntil) &&
			res.hasState &&
			res.state.TurnsTaken == l.reservedAtTurns &&
			(l.reservedUnbound || (boundLive && res.state.Endpoint == l.reservedEndpoint)) &&
			(!checkPool || a.endpoints[l.reservedEndpoint] != nil)
		if !stillReserved {
			l.clearReservation()
		}
	}
	if l.empty() {
		delete(a.local, id)
	} else if boundLive {
		l.endpointHint = res.state.Endpoint
	}
	return resumed
}

func (a *ThunderAgent) refreshLocalLocked(candByID map[string]*candidate, now time.Time) int {
	if len(a.local) == 0 {
		return 0
	}
	resumed := 0
	for id, l := range a.local {
		res := a.localStateLocked(id, l, candByID)
		if a.refreshOneLocked(id, l, res, true, now) {
			resumed++
		}
	}
	return resumed
}

func (a *ThunderAgent) sessionLoadDeltaLocked(res resolvedSession, l *localSession, now time.Time) (boundDelta, reservedLoad float64) {
	if res.isBound {
		boundDelta = a.footprintLocked(res.state, l, now) - float64(res.state.ContextSize())
	}
	if res.hasState && l.reservationLive(res.state, now) && (!res.isBound || res.state.Endpoint != l.reservedEndpoint) {
		reservedLoad = l.reservedTokens
	}
	return boundDelta, reservedLoad
}

// singleEndpointViewLocked builds an endpointView for one pod in O(1 + |local|).
func (a *ThunderAgent) singleEndpointViewLocked(id string, candByID map[string]*candidate, now time.Time) *endpointView {
	entry := a.endpoints[id]
	if entry == nil {
		return nil
	}
	view, hasAttr := sessionstate.PeekEndpointSessions(entry.ep)
	load := 0.0
	if hasAttr {
		load = float64(view.TotalContextTokens())
	}
	for sid, l := range a.local {
		var res resolvedSession
		if hasAttr {
			if bs, ok := view.Session(sid); ok {
				res = resolvedSession{state: bs, hasState: true, isBound: true}
			}
		}
		if !res.isBound && l.reservedEndpoint == id {
			res = a.localStateLocked(sid, l, candByID)
		}
		boundDelta, reservedLoad := a.sessionLoadDeltaLocked(res, l, now)
		if res.isBound && res.state.Endpoint == id {
			load += boundDelta
		}
		if l.reservedEndpoint == id {
			load += reservedLoad
		}
	}
	pv := &endpointView{
		id:       id,
		capacity: entry.capacity,
		load:     max(load, 0),
		hasAttr:  hasAttr,
		view:     view,
	}
	pv.room = a.roomLocked(pv)
	return pv
}

// buildPoolViewsLocked projects all tracked endpoints in O(pods + |local|).
func (a *ThunderAgent) buildPoolViewsLocked(now time.Time, candByID map[string]*candidate) map[string]*endpointView {
	if len(a.endpoints) == 0 {
		return nil
	}
	backing := make([]endpointView, len(a.endpoints))
	pods := make(map[string]*endpointView, len(a.endpoints))
	i := 0
	for id, entry := range a.endpoints {
		view, hasAttr := sessionstate.PeekEndpointSessions(entry.ep)
		pv := &backing[i]
		i++
		pv.id = id
		pv.capacity = entry.capacity
		pv.hasAttr = hasAttr
		pv.view = view
		if hasAttr {
			pv.load = float64(view.TotalContextTokens())
		}
		pods[id] = pv
	}

	for sid, l := range a.local {
		res := a.localStateLocked(sid, l, candByID)
		boundDelta, reservedLoad := a.sessionLoadDeltaLocked(res, l, now)
		if boundDelta != 0 {
			if pv := pods[res.state.Endpoint]; pv != nil {
				pv.load += boundDelta
			}
		}
		if reservedLoad != 0 {
			if pv := pods[l.reservedEndpoint]; pv != nil {
				pv.load += reservedLoad
			}
		}
	}

	for _, pv := range pods {
		pv.load = max(pv.load, 0)
		pv.room = a.roomLocked(pv)
		pv.spare = a.reclaimableTokensLocked(pv, now, candByID)
	}
	return pods
}

func (a *ThunderAgent) footprintLocked(state sessionstate.SessionState, l *localSession, now time.Time) float64 {
	fp := float64(state.ContextSize())
	if l.reservationLive(state, now) && l.reservedEndpoint == state.Endpoint {
		return max(fp, l.reservedTokens)
	}
	if l.pausedLive(state) {
		return 0
	}
	return fp
}

func (a *ThunderAgent) reclaimableLocked(state sessionstate.SessionState, l *localSession, now time.Time, queued map[string]*candidate) bool {
	if queued[state.SessionID] != nil {
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

func (a *ThunderAgent) reclaimableTokensLocked(pv *endpointView, now time.Time, queued map[string]*candidate) float64 {
	if !pv.hasAttr {
		return 0
	}
	var total float64
	for _, s := range pv.view.Sessions() {
		l := a.local[s.SessionID]
		if a.reclaimableLocked(s, l, now, queued) {
			total += a.footprintLocked(s, l, now)
		}
	}
	return total
}

func (a *ThunderAgent) reclaimLocked(pv *endpointView, now time.Time, queued map[string]*candidate, need float64) int {
	room := a.roomLocked(pv)
	if need <= room || !pv.hasAttr {
		return 0
	}
	deficit := need - room

	var victims []sessionstate.SessionState
	for _, s := range pv.view.Sessions() {
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
		l.endpointHint = pv.id
		l.paused = true
		l.pausedAtTurns = v.TurnsTaken
		pv.load -= fp
		pv.room += fp
		deficit -= fp
		paused++
	}
	return paused
}

func (a *ThunderAgent) deleteEndpointLocked(id string) {
	delete(a.endpoints, id)
	delete(a.prefillSeen, id)
	for sid, l := range a.local {
		if l.endpointHint == id {
			l.clearPause()
			l.endpointHint = ""
		}
		if l.reservedEndpoint == id {
			l.clearReservation()
		}
		if l.empty() {
			delete(a.local, sid)
		}
	}
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
	now := a.clock.Now()
	a.pruneStaleEndpointsLocked(now)
	resumed := a.refreshLocalLocked(nil, now)
	a.mu.Unlock()

	if resumed > 0 {
		a.metrics.resumes.Add(float64(resumed))
	}
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

// snapshot computes metrics and debug state without mutating policy overlays.
func (a *ThunderAgent) snapshot() ledgerSnapshot {
	a.mu.Lock()
	defer a.mu.Unlock()

	now := a.clock.Now()
	snap := ledgerSnapshot{
		endpoints: make(map[string]endpointSnapshot, len(a.endpoints)),
	}
	loads := make(map[string]float64, len(a.endpoints))
	liveCaps := make(map[string]float64, len(a.endpoints))

	for id, entry := range a.endpoints {
		if now.Sub(entry.updatedAt) > endpointStaleAfter {
			continue
		}
		liveCaps[id] = entry.capacity
		view, hasAttr := sessionstate.PeekEndpointSessions(entry.ep)
		if !hasAttr {
			continue
		}
		loads[id] = float64(view.TotalContextTokens())
		for _, s := range view.Sessions() {
			switch {
			case s.InFlightRequests > 0:
				snap.running++
			case a.local[s.SessionID].pausedLive(s):
				snap.paused++
			default:
				snap.idle++
			}
		}
	}

	for sid, l := range a.local {
		res := a.localStateLocked(sid, l, nil)
		boundDelta, reservedLoad := a.sessionLoadDeltaLocked(res, l, now)
		loads[res.state.Endpoint] += boundDelta
		loads[l.reservedEndpoint] += reservedLoad
	}

	for id, capTokens := range liveCaps {
		snap.endpoints[id] = endpointSnapshot{
			workingSet: max(loads[id], 0),
			capacity:   capTokens,
		}
	}
	return snap
}
