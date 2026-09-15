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

// programClass ranks a waiting program for dispatch. Lower dispatches first.
type programClass int

const (
	// classReasoning is a program that has been served before and has a
	// request waiting. Its footprint is already counted in the working set
	// and finishing its trajectory is what returns capacity to the pool, so
	// it always dispatches.
	classReasoning programClass = iota
	// classResuming is a shed program whose next turn is waiting. It must
	// fit a pod again (its footprint stopped counting when it was shed), but
	// it outranks never-admitted programs: it is mid-trajectory with sunk
	// work, and finishing it releases capacity permanently, where admitting
	// a new program signs up a whole trajectory of future turns. A new
	// program's smaller arrival size is also transient; it grows to full
	// size after admission, so size ordering across the two would prefer the
	// wrong candidate for a reason that disappears immediately.
	classResuming
	// classNew is a program whose first request is waiting. It is admitted
	// only when its projected footprint fits a pod under the utilThreshold
	// ceiling.
	classNew
)

func (c programClass) String() string {
	switch c {
	case classReasoning:
		return "reasoning"
	case classResuming:
		return "resuming"
	default:
		return "new"
	}
}

// holdWaitFloor separates saturation holds from normal dispatch latency in
// the holds metric: a request that waited at least this long in the
// flow-control queue counts as held.
const holdWaitFloor = time.Second

// Pick selects the queue to service next.
//
// Turns of admitted (REASONING) programs bypass admission: smallest decayed
// footprint first, then the oldest head. New programs are considered only
// after them, smallest projected footprint first, and each must fit a pod:
// working set plus live reservations plus the program's estimate and growth
// buffer, within utilThreshold of the pod's capacity. A fitting new program
// gets a reservation charged to the pod that fit it, released when
// PreRequest binds the program. A head waiting past headWaitStarvationMs
// dispatches regardless of class, size, or fit, oldest first; for a new
// program this is the forced-admission backstop.
//
// Returning nil while non-fitting new programs wait is the hold: the
// processor dispatches nothing and retries next cycle.
//
// The fit view comes from the Saturation hook, so the plugin must be the
// flow control saturation detector. Without a fit view, new programs are
// admitted freely and only the class ordering remains.
//
// There is no ACTING class. Upstream ThunderAgent proactively pauses idle
// programs, so its resume queue can hold a program with no request
// outstanding; that program is ACTING. Flow control here only ever queues
// actual requests, so every candidate has a request pending and is REASONING
// by upstream's definition.
func (a *ThunderAgent) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}

	now := time.Now()
	logger := log.FromContext(ctx)

	type candidate struct {
		queue    fwkfc.FlowQueueAccessor
		class    programClass
		tokens   float64
		waitMs   float64
		starving bool
		fitPod   string
	}

	var best *candidate
	heldNew := 0

	t := a.table
	t.mu.Lock()
	band.IterateQueues(func(queue fwkfc.FlowQueueAccessor) bool {
		// Empty entries appear transiently when a queue drains between
		// iteration and scoring; they carry nothing to service.
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}
		id := queue.FlowKey().ID
		waitMs := float64(now.Sub(head.EnqueueTime()).Milliseconds())
		class, tokens := t.classAndTokens(id, now)
		starving := a.headWaitStarvationMs > 0 && waitMs >= a.headWaitStarvationMs

		fitPod := ""
		if class != classReasoning {
			// A resuming (shed) program carries its known committed
			// footprint; a truly new one is estimated from its request size.
			if tokens == 0 {
				tokens = float64(t.estimateTokens(int(head.OriginalRequest().ByteSize())))
			}
			pod, ok := a.fitPodLocked(tokens+a.bufferTokensPerProgram, now)
			if !ok && !starving {
				heldNew++
				return true
			}
			fitPod = pod
		}

		c := &candidate{queue: queue, class: class, tokens: tokens, waitMs: waitMs, starving: starving, fitPod: fitPod}
		if best == nil || betterThan(c.class, c.tokens, c.waitMs, c.starving,
			best.class, best.tokens, best.waitMs, best.starving) {
			best = c
		}
		return true
	})

	if best != nil && best.class != classReasoning && best.fitPod != "" {
		// Reserve the fitting pod's room until PreRequest binds the program.
		t.pending[best.queue.FlowKey().ID] = pendingAdmission{
			tokens: best.tokens + a.bufferTokensPerProgram, pod: best.fitPod, at: now,
		}
	}
	t.mu.Unlock()

	if best == nil {
		if heldNew > 0 {
			logger.V(logutil.DEBUG).Info("thunderagent.hold", "held_new", heldNew)
		}
		return nil, nil //nolint:nilnil
	}

	a.metrics.releases.WithLabelValues(best.class.String()).Inc()
	held := best.waitMs >= float64(holdWaitFloor.Milliseconds())
	if held {
		a.metrics.holds.WithLabelValues(best.class.String()).Inc()
	}
	if best.starving {
		a.metrics.starvationPromotions.Inc()
	}
	if held || best.starving {
		logger.V(logutil.DEBUG).Info("thunderagent.release",
			"class", best.class.String(), "waited_ms", best.waitMs, "starved", best.starving)
	}
	return best.queue, nil
}

// fitPodLocked returns the pod with the most free admission room that fits
// the required tokens, if any. Room is the utilThreshold share of capacity
// minus the working set and live reservations. With no fit view the check
// fails open: the plugin is then not wired as the saturation detector and
// only class ordering applies. Callers must hold t.mu.
func (a *ThunderAgent) fitPodLocked(required float64, now time.Time) (string, bool) {
	t := a.table
	if len(t.snapshot) == 0 {
		return "", true
	}
	bestPod := ""
	bestRoom := 0.0
	for pod, snap := range t.snapshot {
		room := snap.capacity*a.utilThreshold - snap.tokens - t.pendingOn(pod, now)
		if room >= required && room > bestRoom {
			bestPod, bestRoom = pod, room
		}
	}
	return bestPod, bestPod != ""
}

// betterThan reports whether candidate a outranks the incumbent b.
func betterThan(aClass programClass, aTokens, aWait float64, aStarving bool,
	bClass programClass, bTokens, bWait float64, bStarving bool) bool {
	if aStarving != bStarving {
		return aStarving
	}
	if aStarving && bStarving {
		return aWait > bWait
	}
	if aClass != bClass {
		return aClass < bClass
	}
	if aTokens != bTokens {
		return aTokens < bTokens
	}
	return aWait > bWait
}
