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
	"context"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// holdWaitFloor separates admission holds from normal dispatch latency in
// the holds metric.
const holdWaitFloor = time.Second

// NewState satisfies the FairnessPolicy contract; Pick reads the shared
// sessionManager instead of per-band state.
func (a *ThunderAgent) NewState(_ context.Context) any { return nil }

// candidate is one dispatchable queue head under consideration.
type candidate struct {
	queue     fwkfc.FlowQueueAccessor
	id        string
	sizeBytes int
	class     sessionClass
	// tokens is the session's size once PreRequest charges this turn.
	tokens   float64
	waitMs   float64
	starving bool
	fitPod   *endpointState
}

// Pick is the admission gate for thunder agent.
//
// Turns of admitted sessions always dispatch. A paused session's
// next turn dispatches only when its own pod has room again. A new session dispatches
// only when some pod has room, and is then reserved onto that pod.
//
// Order among the sessions allowed to dispatch:
// waited past headWaitStarvationMs (oldest first) ->
// class (admitted -> paused -> new) -> smallest footprint -> oldest.
// A session that fits no pod does not block the others, so smaller sessions
// can take room a larger paused one waits for; the starvation backstop bounds
// that wait.
//
// If no session is allowed (every waiting session is paused or new and fits no
// pod), Pick returns nil. The requests stay queued and the
// processor tries again on its next dispatch cycle.
//
// Room is capacity * utilThreshold minus the working set. When the picked
// session does not fit, idle sessions give up their room, longest idle first,
// and are paused: for a paused or new session only those idle past
// idleLeaseSeconds, for an admitted session's turn that pushes its pod over
// the ceiling any idle session. Sessions with a turn queued are never paused,
// and nothing is paused until a session is picked.
func (a *ThunderAgent) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := time.Now()

	var candidates []*candidate
	queued := map[string]bool{}
	band.IterateQueues(func(queue fwkfc.FlowQueueAccessor) bool {
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}
		id := headSessionID(head)
		waitMs := float64(now.Sub(head.EnqueueTime()).Milliseconds())
		if id != "" {
			queued[id] = true
		}
		candidates = append(candidates, &candidate{
			queue:     queue,
			id:        id,
			sizeBytes: headSizeBytes(head),
			waitMs:    waitMs,
			starving:  a.headWaitStarvationMs > 0 && waitMs >= a.headWaitStarvationMs,
		})
		return true
	})
	// The processor calls Pick every dispatch cycle, queued work or not.
	if len(candidates) == 0 {
		return nil, nil //nolint:nilnil
	}

	m := a.mgr
	m.mu.Lock()
	rooms := make(map[*endpointState]float64, len(m.endpoints))
	spare := make(map[*endpointState]float64, len(m.endpoints))
	for _, p := range m.endpoints {
		rooms[p] = p.capacity*a.utilThreshold - p.occupancy(now)
		spare[p] = p.reclaimableTokens(now, a.idleLease, queued)
	}

	var best *candidate
	held := 0
	for _, c := range candidates {
		if !a.sizeAndFitLocked(c, rooms, spare) {
			held++
			continue
		}
		if best == nil || betterThan(c, best) {
			best = c
		}
	}
	paused := 0
	if best != nil {
		paused = a.admitLocked(best, now, rooms, queued)
	}
	m.mu.Unlock()

	if paused > 0 {
		a.metrics.pauses.Add(float64(paused))
		log.FromContext(ctx).V(logutil.DEBUG).Info("thunderagent.reclaim", "class", best.class.String(), "paused", paused)
	}
	if best == nil {
		if held > 0 {
			log.FromContext(ctx).V(logutil.DEBUG).Info("thunderagent.hold", "held", held)
		}
		return nil, nil //nolint:nilnil
	}
	a.metrics.releases.WithLabelValues(best.class.String()).Inc()
	if best.waitMs >= float64(holdWaitFloor.Milliseconds()) {
		a.metrics.holds.WithLabelValues(best.class.String()).Inc()
	}
	if best.starving {
		a.metrics.starvationPromotions.Inc()
	}
	return best.queue, nil
}

// sizeAndFitLocked classifies and sizes a candidate and, for a paused or new
// session, finds the pod it fits once reclaimable idle sessions give up
// their room. Returns false when the candidate must hold.
func (a *ThunderAgent) sizeAndFitLocked(c *candidate, rooms, spare map[*endpointState]float64) bool {
	s := a.mgr.sessions[c.id]
	var committed float64
	switch {
	case c.id == "":
		// Anonymous traffic is not tracked and passes through.
		c.class = classAdmitted
		return true
	case s == nil:
		c.class = classNew
	default:
		c.class = s.class()
		committed = float64(s.committedTokens)
	}

	// The new turn resends the whole history, so its estimate is the
	// session's size from now on; committed tokens are the floor.
	c.tokens = max(committed, float64(estimateTokens(c.sizeBytes)))
	if c.class == classAdmitted || len(rooms) == 0 {
		return true
	}
	switch c.class {
	case classPaused:
		if rooms[s.endpoint]+spare[s.endpoint] >= a.fitTokens(s.endpoint, c.tokens) {
			c.fitPod = s.endpoint
		}
	case classNew:
		c.fitPod = a.newSessionPod(c.tokens, rooms, spare)
	}
	return c.fitPod != nil || c.starving
}

// fitTokens is the room a session needs on a pod: its size, capped at the
// pod's ceiling. A session larger than the ceiling (it grew there through
// turns that skip the fit check) then fits once every other session on the
// pod is paused or reclaimable, instead of never.
func (a *ThunderAgent) fitTokens(p *endpointState, tokens float64) float64 {
	return min(tokens, p.capacity*a.utilThreshold)
}

// newSessionPod picks the pod for a new session: among pods it fits once
// reclaimable idle sessions give up their room, the one with the most free
// room. If any pod fits it without pausing anyone, that pod has the most
// free room, so no one is paused.
func (a *ThunderAgent) newSessionPod(tokens float64, rooms, spare map[*endpointState]float64) *endpointState {
	var best *endpointState
	for p, room := range rooms {
		if room+spare[p] >= a.fitTokens(p, tokens) && (best == nil || room > rooms[best]) {
			best = p
		}
	}
	return best
}

// admitLocked commits the picked session. It pauses idle sessions on the
// session's pod until the turn fits, then reserves the turn's room until
// PreRequest charges it. The reservation keeps the next cycles from admitting
// into the same room twice, and keeps a dispatched admitted session, no
// longer queued and not yet in flight, from looking idle and being reclaimed.
// Returns how many sessions it paused.
func (a *ThunderAgent) admitLocked(best *candidate, now time.Time, rooms map[*endpointState]float64, queued map[string]bool) int {
	m := a.mgr
	s := m.sessions[best.id]
	paused := 0
	switch {
	case best.class == classAdmitted:
		if s == nil {
			return 0 // anonymous traffic
		}
		// Its current size is already counted; only the turn's growth is new.
		paused = s.endpoint.reclaim(now, 0, queued, rooms[s.endpoint], max(best.tokens-s.size(), 0))
	case best.fitPod != nil:
		paused = best.fitPod.reclaim(now, a.idleLease, queued, rooms[best.fitPod], a.fitTokens(best.fitPod, best.tokens))
		s = m.bindLocked(best.id, best.fitPod)
	case s == nil:
		// A force-admitted new session that fits no pod: the scheduler
		// places it and PreRequest binds it.
		return 0
	}
	s.reservedTokens = best.tokens
	s.reservedUntil = now.Add(reservationTTL)
	return paused
}

// betterThan reports whether a outranks b:
// waited past headWaitStarvationMs (oldest first) ->
// class (admitted -> paused -> new) -> smallest footprint -> oldest.
func betterThan(a, b *candidate) bool {
	if a.starving != b.starving {
		return a.starving
	}
	if a.starving {
		return a.waitMs > b.waitMs
	}
	if a.class != b.class {
		return a.class < b.class
	}
	if a.tokens != b.tokens {
		return a.tokens < b.tokens
	}
	return a.waitMs > b.waitMs
}

// headSizeBytes returns the request body size of a queue head, from the
// scheduling request when the item carries one (the production adapter
// does), else from the flow-control item's byte size.
// headSessionID returns the session of a queue head, read the same way as in
// PreRequest, or "" for anonymous traffic. A queue is named by the fairness
// ID, which can group several sessions.
func headSessionID(head fwkfc.QueueItemAccessor) string {
	req := head.OriginalRequest()
	if req == nil {
		return ""
	}
	return sessionID(req.InferenceRequest())
}

func headSizeBytes(head fwkfc.QueueItemAccessor) int {
	req := head.OriginalRequest()
	if req == nil {
		return 0
	}
	if ir := req.InferenceRequest(); ir != nil && ir.RequestSizeBytes > 0 {
		return ir.RequestSizeBytes
	}
	return int(req.ByteSize())
}
