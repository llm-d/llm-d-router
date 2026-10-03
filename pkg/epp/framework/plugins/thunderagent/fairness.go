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
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// holdWaitFloor separates admission holds from normal dispatch latency in
// the holds metric.
const holdWaitFloor = time.Second

// NewState satisfies the FairnessPolicy contract; Pick reads the shared
// sessionManager instead of per-band state.
func (a *ThunderAgent) NewState(_ context.Context) any { return nil }

// candidate is one dispatchable queue head under consideration.
type candidate struct {
	queue    fwkfc.FlowQueueAccessor
	class    sessionClass
	tokens   float64
	waitMs   float64
	starving bool
	fitPod   *endpointState
}

// Pick is the admission gate for thunder agent.
//
// Turns of admitted (reasoning) sessions always dispatch. A paused session's
// next turn dispatches only when its own pod has room again. A new session dispatches
// only when some pod has room, and is then reserved onto the pod with the most room.
//
// Order among the sessions allowed to dispatch:
// waited past headWaitStarvationMs (oldest first) ->
// class (reasoning -> paused -> new) -> smallest footprint -> oldest.
//
// If no session is allowed (every waiting session is paused or new and fits no
// pod), Pick returns nil. The requests stay queued and the
// processor tries again on its next dispatch cycle.
//
// Room is capacity * utilThreshold minus the decayed occupancy.
func (a *ThunderAgent) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := time.Now()
	m := a.mgr

	m.mu.Lock()
	rooms := make(map[*endpointState]float64, len(m.endpoints))
	for _, p := range m.endpoints {
		_, decayed := p.occupancy(now, m.halfLife)
		rooms[p] = p.capacity*a.utilThreshold - decayed
	}
	haveView := len(rooms) > 0

	var best *candidate
	held := 0
	band.IterateQueues(func(queue fwkfc.FlowQueueAccessor) bool {
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}
		id := queue.FlowKey().ID
		waitMs := float64(now.Sub(head.EnqueueTime()).Milliseconds())
		starving := a.headWaitStarvationMs > 0 && waitMs >= a.headWaitStarvationMs

		c := &candidate{queue: queue, waitMs: waitMs, starving: starving}
		s := m.sessions[id]
		switch {
		case id == metadata.DefaultFairnessID:
			// Anonymous traffic is not tracked and passes through.
			c.class = classReasoning
		case s == nil:
			c.class = classNew
		default:
			c.class = s.class()
			if c.class == classReasoning {
				c.tokens = s.undecayed()
			} else {
				c.tokens = float64(s.committedTokens)
			}
		}

		if c.class != classReasoning {
			// The new turn resends the whole history, so its estimate is the
			// session's size from now on; committed tokens are the floor.
			if est := float64(estimateTokens(headSizeBytes(head))); est > c.tokens {
				c.tokens = est
			}
			if haveView && !starving {
				switch c.class {
				case classPaused:
					if rooms[s.endpoint] < c.tokens {
						held++
						return true
					}
					c.fitPod = s.endpoint
				case classNew:
					var bestPod *endpointState
					bestRoom := 0.0
					for p, room := range rooms {
						if room >= c.tokens && room > bestRoom {
							bestPod, bestRoom = p, room
						}
					}
					if bestPod == nil {
						held++
						return true
					}
					c.fitPod = bestPod
				}
			}
		}
		if best == nil || betterThan(c, best) {
			best = c
		}
		return true
	})

	if best != nil && best.class != classReasoning {
		// Reserve the admitted session's room until PreRequest binds it, so
		// the next cycles do not admit into the same room twice.
		id := best.queue.FlowKey().ID
		s := m.sessions[id]
		if s == nil {
			s = &session{}
			m.sessions[id] = s
		}
		if best.fitPod != nil && s.endpoint != best.fitPod {
			if s.endpoint != nil {
				delete(s.endpoint.sessions, id)
			}
			s.endpoint = best.fitPod
			best.fitPod.sessions[id] = s
		}
		s.reserved = true
		s.reservedTokens = best.tokens
		s.reservedUntil = now.Add(reservationTTL)
	}
	m.mu.Unlock()

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

// betterThan reports whether a outranks b:
// waited past headWaitStarvationMs (oldest first) ->
// class (reasoning -> paused -> new) -> smallest footprint -> oldest.
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
