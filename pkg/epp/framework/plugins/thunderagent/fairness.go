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
	"math"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
)

// holdWaitFloor separates admission holds from normal dispatch latency in
// the delayed dispatch metric.
const holdWaitFloor = time.Second

// NewState satisfies the FairnessPolicy contract; Pick reads the shared plugin
// policy state instead of per-band state.
func (a *ThunderAgent) NewState(_ context.Context) any { return nil }

// candidate is one dispatchable queue head under consideration.
type candidate struct {
	queue     fwkfc.FlowQueueAccessor
	id        string
	sizeBytes int
	class     sessionClass
	// tokens is the session's size once PreRequest charges this turn.
	tokens        float64
	wait          time.Duration
	starving      bool
	hasState      bool
	isBound       bool
	state         sessionstate.SessionState
	boundEndpoint string
	fitPod        *endpointView
}

// Pick is the admission gate for thunder agent.
//
// Turns of admitted sessions always dispatch. A paused session's
// next turn dispatches only when its own pod has room again. A new session dispatches
// only when some pod has room, and is then reserved onto that pod.
//
// Unlike the original ThunderAgent, which restores a paused session onto any
// pod with room, a paused session waits for its own pod, so its warm prefix
// is kept. The wait is bounded by forceAdmitTimeout.
// TODO(#3221): add an option to restore a paused session onto another pod
// with room when its own pod stays full.
//
// The sessions allowed to dispatch are ranked by betterThan.
// A session that fits no pod does not block the others, so smaller sessions
// can take room a larger paused one waits for; the starvation backstop bounds
// that wait.
//
// If no session is allowed (every waiting session is paused or new and fits no
// pod), Pick returns nil. The requests stay queued and the
// processor tries again on its next dispatch cycle.
//
// Room is capacity * usableCapacityRatio minus the working set. When the picked
// session does not fit (including a session force-admitted onto its own pod),
// or an admitted session's turn pushes its pod over the ceiling, sessions idle
// past idleGracePeriod give up their room, longest idle first, and are
// paused. Sessions with a turn queued are never paused,
// and nothing is paused until a session is picked.
func (a *ThunderAgent) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := a.clock.Now()

	var candidates []candidate
	band.IterateQueues(func(queue fwkfc.FlowQueueAccessor) bool {
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}
		ir := headRequest(head)
		id := sessionID(ir)
		wait := now.Sub(head.EnqueueTime())
		var (
			state    sessionstate.SessionState
			hasState bool
		)
		if id != "" {
			state, hasState = sessionstate.ReadSessionState(ir)
		}
		candidates = append(candidates, candidate{
			queue:     queue,
			id:        id,
			sizeBytes: headSizeBytes(head),
			wait:      wait,
			starving:  a.forceAdmitTimeout > 0 && wait >= a.forceAdmitTimeout,
			hasState:  hasState,
			state:     state,
		})
		return true
	})
	// The processor calls Pick every dispatch cycle, queued work or not.
	if len(candidates) == 0 {
		return nil, nil //nolint:nilnil
	}

	candByID := make(map[string]*candidate, len(candidates))
	for i := range candidates {
		c := &candidates[i]
		if c.id == "" {
			continue
		}
		if prev := candByID[c.id]; prev == nil || (!prev.hasState && c.hasState) {
			candByID[c.id] = c
		}
	}

	a.mu.Lock()
	resumed := a.refreshLocalLocked(candByID, now)
	needFit := false
	for i := range candidates {
		a.classifyLocked(&candidates[i], now)
		if candidates[i].class != classAdmitted {
			needFit = true
		}
	}

	var pods map[string]*endpointView
	if needFit {
		pods = a.buildPoolViewsLocked(now, candByID)
	}

	var best *candidate
	for i := range candidates {
		c := &candidates[i]
		if !a.fitsLocked(c, pods) {
			continue
		}
		if best == nil || betterThan(c, best) {
			best = c
		}
	}
	paused := 0
	if best != nil {
		paused = a.admitLocked(best, pods, candByID, now)
	}
	a.mu.Unlock()

	if resumed > 0 {
		a.metrics.resumes.Add(float64(resumed))
	}
	if paused > 0 {
		a.metrics.pauses.Add(float64(paused))
		log.FromContext(ctx).V(logutil.DEBUG).Info("Paused idle sessions to make room", "sessionClass", best.class.String(), "pausedSessions", paused)
	}
	if best == nil {
		return nil, nil //nolint:nilnil
	}
	a.metrics.releases.WithLabelValues(best.class.String()).Inc()
	if best.wait >= holdWaitFloor {
		a.metrics.delayedDispatches.WithLabelValues(best.class.String()).Inc()
	}
	if best.starving {
		a.metrics.starvationPromotions.Inc()
	}
	return best.queue, nil
}

// roomLocked is the pod's ceiling minus its working set.
func (a *ThunderAgent) roomLocked(pv *endpointView) float64 {
	return pv.capacity*a.usableCapacityRatio - pv.load
}

// classifyLocked sets c's admission class, bound endpoint, and projected token
// footprint using live endpoint bindings first and request-header state second.
func (a *ThunderAgent) classifyLocked(c *candidate, now time.Time) {
	if c.id == "" {
		// Anonymous traffic is not tracked and passes through.
		c.class = classAdmitted
		return
	}

	candState, hasCand := c.state, c.hasState
	var (
		state   sessionstate.SessionState
		ok      bool
		isBound bool
	)
	if hasCand && candState.Endpoint != "" {
		if s, found := a.boundSessionLocked(c.id, candState.Endpoint); found {
			state, ok, isBound = s, true, true
		}
	} else if !hasCand {
		if s, found := a.boundSessionLocked(c.id, ""); found {
			state, ok, isBound = s, true, true
		}
	}
	if !ok && hasCand {
		state, ok = candState, true
	}
	l := a.local[c.id]
	if !ok && l != nil && l.reservedUnbound {
		state = sessionstate.SessionState{TurnsTaken: l.reservedAtTurns}
	}
	c.state = state
	c.hasState = ok
	c.isBound = isBound && a.endpoints[state.Endpoint] != nil
	if c.isBound {
		c.boundEndpoint = state.Endpoint
	}

	switch {
	case c.boundEndpoint != "" && l.pausedLive(state):
		c.class = classPaused
	case c.boundEndpoint != "":
		c.class = classAdmitted
	case l.reservationLive(state, now) && l.reservedUnbound && a.endpoints[l.reservedEndpoint] != nil:
		c.boundEndpoint = l.reservedEndpoint
		c.class = classAdmitted
	default:
		c.class = classNew
	}

	var committed, inflight int64
	if ok {
		committed = state.ContextTokens
		inflight = state.InFlightTokens
	}
	prevInflight := 0.0
	if l.reservationLive(state, now) && c.boundEndpoint != "" && l.reservedEndpoint == c.boundEndpoint {
		prevInflight = l.reservedInflight
	}
	c.tokens = max(float64(committed), float64(inflight)+prevInflight+float64(sessionstate.EstimatePromptTokens(c.sizeBytes)))
}

// fitsLocked reports whether c may dispatch under the current pod room, or
// skips the check for admitted sessions or when no pool data is available.
func (a *ThunderAgent) fitsLocked(c *candidate, pods map[string]*endpointView) bool {
	if c.class == classAdmitted || len(pods) == 0 {
		return true
	}
	switch c.class {
	case classPaused:
		if pv := pods[c.boundEndpoint]; pv != nil && pv.room+pv.spare >= a.fitTokens(pv, c.tokens) {
			c.fitPod = pv
		}
	case classNew:
		c.fitPod = a.newSessionPod(c.tokens, pods)
	}
	return c.fitPod != nil || c.starving
}

// fitTokens is the room a session needs on a pod: its size, capped at the
// pod's ceiling. A session larger than the ceiling (it grew there through
// turns that skip the fit check) then fits once every other session on the
// pod is paused or reclaimable, instead of never.
func (a *ThunderAgent) fitTokens(pv *endpointView, tokens float64) float64 {
	return min(tokens, pv.capacity*a.usableCapacityRatio)
}

// newSessionPod picks the pod for a new session: among pods it fits once
// reclaimable idle sessions give up their room, the one with the most free
// room. If any pod fits it without pausing anyone, that pod has the most
// free room, so no one is paused.
func (a *ThunderAgent) newSessionPod(tokens float64, pods map[string]*endpointView) *endpointView {
	var best *endpointView
	for _, pv := range pods {
		if pv.room+pv.spare >= a.fitTokens(pv, tokens) && (best == nil || pv.room > best.room) {
			best = pv
		}
	}
	return best
}

// admitLocked commits the picked session. It pauses idle sessions on the
// session's pod until the turn fits, then reserves the turn's room until
// PreRequest charges it. The reservation keeps the next cycles from admitting
// into the same room twice, and keeps a dispatched admitted session, no
// longer queued and not yet in flight, from looking idle and being reclaimed.
// A paused session keeps its pause overlay until PreRequest advances TurnsTaken
// so a cancelled resume does not permanently drop paused status.
func (a *ThunderAgent) admitLocked(best *candidate, pods map[string]*endpointView, candByID map[string]*candidate, now time.Time) int {
	if best.id == "" {
		return 0
	}

	podOf := func(id string) *endpointView {
		if id == "" {
			return nil
		}
		if pods != nil {
			return pods[id]
		}
		return a.singleEndpointViewLocked(id, candByID, now)
	}

	var targetPod *endpointView
	var need float64

	switch best.class {
	case classAdmitted:
		targetPod = podOf(best.boundEndpoint)
		if targetPod == nil {
			return 0
		}
		cur := 0.0
		l := a.local[best.id]
		if best.isBound {
			cur = a.footprintLocked(best.state, l, now)
		} else if l.reservationLive(best.state, now) && l.reservedEndpoint == targetPod.id {
			cur = l.reservedTokens
		}
		// Its current size is already counted; only the turn's growth is new.
		need = max(best.tokens-cur, 0)
	case classPaused:
		targetPod = best.fitPod
		if targetPod == nil {
			// A force-admitted paused session that fits no pod returns to its own pod.
			targetPod = podOf(best.boundEndpoint)
		}
		if targetPod == nil {
			return 0
		}
		cur := a.footprintLocked(best.state, a.local[best.id], now)
		need = max(a.fitTokens(targetPod, best.tokens)-cur, 0)
	case classNew:
		if best.fitPod == nil {
			// A force-admitted new session that fits no pod: the scheduler
			// places it and session-state-producer binds it.
			return 0
		}
		targetPod = best.fitPod
		need = a.fitTokens(targetPod, best.tokens)
	}

	paused := a.reclaimLocked(targetPod, now, candByID, need)

	l := a.ensureLocalLocked(best.id)
	if best.isBound {
		l.endpointHint = best.boundEndpoint
	}
	prevReserved := 0.0
	prevInflight := 0.0
	if l.reservationLive(best.state, now) && l.reservedEndpoint == targetPod.id {
		prevReserved = l.reservedTokens
		prevInflight = l.reservedInflight
	}
	l.reservedEndpoint = targetPod.id
	l.reservedTokens = max(best.tokens, prevReserved)
	l.reservedInflight = prevInflight + float64(sessionstate.EstimatePromptTokens(best.sizeBytes))
	l.reservedUntil = now.Add(reservationTTL)
	l.reservedAtTurns = best.state.TurnsTaken
	l.reservedUnbound = !best.isBound

	return paused
}

// betterThan reports whether a outranks b:
// waited past forceAdmitTimeout (oldest first) ->
// class (admitted -> paused -> new) -> smallest footprint -> oldest.
func betterThan(a, b *candidate) bool {
	if a.starving != b.starving {
		return a.starving
	}
	if a.starving {
		return a.wait > b.wait
	}
	if a.class != b.class {
		return a.class < b.class
	}
	if a.tokens != b.tokens {
		return a.tokens < b.tokens
	}
	return a.wait > b.wait
}

func headRequest(head fwkfc.QueueItemAccessor) *fwksched.InferenceRequest {
	req := head.OriginalRequest()
	if req == nil {
		return nil
	}
	return req.InferenceRequest()
}

// headSizeBytes returns the request body size of a queue head, from the
// scheduling request when the item carries one (the production adapter
// does), else from the flow-control item's byte size.
func headSizeBytes(head fwkfc.QueueItemAccessor) int {
	req := head.OriginalRequest()
	if req == nil {
		return 0
	}
	if ir := req.InferenceRequest(); ir != nil && ir.RequestSizeBytes > 0 {
		return ir.RequestSizeBytes
	}
	if n := req.ByteSize(); n <= math.MaxInt {
		return int(n)
	}
	return math.MaxInt
}
