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

package internal

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// InFlightEvictor is the narrow view of the eviction subsystem the reclamation controller needs.
// It is satisfied by *eviction.RequestEvictor.
type InFlightEvictor interface {
	// EvictN evicts up to n requests in victim-policy order, stopping at the first victim whose
	// priority is not strictly below priorityBound, and returns the IDs actually evicted.
	EvictN(ctx context.Context, n int, priorityBound int) ([]string, error)
	// Stats returns the number of tracked in-flight requests and how many are evictable.
	Stats() (inFlight int, evictable int)
	// PeekVictimPriority returns the priority of the next victim, or false if none are evictable.
	PeekVictimPriority() (priority int, ok bool)
	// SetEvictionTerminatedListener registers the confirmation callback, invoked once per evicted
	// request when its stream terminates.
	SetEvictionTerminatedListener(listener func(requestID string))
}

// ReclamationConfig holds the tuning parameters for the ReclamationController.
// See docs/flow-control-eviction.md for the derivation of each parameter.
type ReclamationConfig struct {
	// MaxRevocationsPerDecision caps how many revocations a single decision may issue. It bounds the
	// damage from mean-footprint misestimation.
	MaxRevocationsPerDecision int

	// ConfirmationGrace is the wait after the last confirmation before the next decision may run. It
	// covers the reclaiming stage: engine abort, KV GC, metrics scrape, and staleness window.
	ConfirmationGrace time.Duration

	// ConfirmationTimeout bounds how long an unconfirmed revocation can hold the pacing gate closed.
	// After it expires the revocation is treated as confirmed and a health metric is incremented.
	ConfirmationTimeout time.Duration
}

// revocation is one issued, not-yet-confirmed eviction.
type revocation struct {
	// credit is the pending-reclaim debit taken for this revocation, in saturation-gauge units,
	// retained at its issue-time value until the debit expires.
	credit   float64
	issuedAt time.Time
}

// coolingDebit is a confirmed revocation's debit held through the reclaiming stage: the stream is
// dead, but the freed capacity is not yet visible in the gauge (engine GC, scrape, staleness). The
// debit expires ConfirmationGrace after confirmation.
type coolingDebit struct {
	credit    float64
	expiresAt time.Time
}

// ReclamationController decides when in-flight eviction fires and how many leases it revokes.
// It implements the sizing and pacing rules from docs/flow-control-eviction.md:
//
//   - Sizing: deficit in saturation-gauge units against a mean-footprint credit estimate, capped by
//     MaxRevocationsPerDecision.
//   - Pacing: confirmation-gated. A new decision may run only once every previously issued
//     revocation is confirmed (its ext_proc stream terminated) and ConfirmationGrace has elapsed,
//     so the controller never acts twice on a gauge that has not absorbed its prior actions.
//
// Concurrency: Reclaim and GateOpen are called only from the processor's single-writer run loop.
// Confirm is called from ext_proc handler goroutines. The mutex is held across EvictN inside
// Reclaim so a confirmation arriving mid-decision blocks until its revocation is registered; the
// evictor never calls back into the controller synchronously, so this cannot deadlock.
type ReclamationController struct {
	evictor  InFlightEvictor
	cfg      ReclamationConfig
	clock    clock.PassiveClock
	logger   logr.Logger
	poolName string

	mu             sync.Mutex
	outstanding    map[string]revocation
	cooling        []coolingDebit
	pendingReclaim float64
}

// NewReclamationController constructs a controller. The caller is responsible for registering
// Confirm as the evictor's eviction-terminated listener.
func NewReclamationController(
	cfg ReclamationConfig,
	evictor InFlightEvictor,
	clk clock.PassiveClock,
	logger logr.Logger,
	poolName string,
) *ReclamationController {
	return &ReclamationController{
		evictor:     evictor,
		cfg:         cfg,
		clock:       clk,
		logger:      logger.WithName("reclamation-controller"),
		poolName:    poolName,
		outstanding: make(map[string]revocation),
	}
}

// GateOpen reports whether a new reclamation decision may run: stop-and-wait, open only when
// nothing is outstanding and nothing is cooling, i.e. every prior revocation has confirmed and had
// ConfirmationGrace to become gauge-visible. It first retires timed-out revocations and expired
// cooling debits. This is the cheap early-exit the dispatch cycle calls on every HoL break.
//
// A sliding-window relaxation (issue while outstanding < W) was prototyped and benchmarked with no
// measurable benefit: sizing is deficit-bound at the ceiling boundary, so the epoch rate, not the
// gate, limits reclaim throughput. See docs/flow-control-eviction.md, Alternatives.
func (c *ReclamationController) GateOpen(now time.Time) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sweepTimeoutsLocked(now)
	c.expireCoolingLocked(now)
	return len(c.outstanding) == 0 && len(c.cooling) == 0
}

// VictimPriority returns the priority of the next lease the selector would revoke.
func (c *ReclamationController) VictimPriority() (int, bool) {
	return c.evictor.PeekVictimPriority()
}

// Reclaim runs one sizing decision and issues the resulting revocations. The caller must have
// observed GateOpen() == true in the same dispatch cycle.
//
// All quantities are dimensionless, in the saturation gauge's own units:
//
//	reclaimTarget = saturation - ceiling(blocked band with eligible demand)
//	credit        = saturation / inFlight     (mean lease footprint estimate)
//	n             = max(1, ceil((reclaimTarget - pendingReclaim) / credit)),
//	                capped at MaxRevocationsPerDecision and at the evictable lease count
//
// demandPriority is the eligible demand band's priority, passed to the actuator as a strict upper
// bound on victim priority so that no victim in the decision sits at or above the demand band
// (the victim-head check in the trigger does not cover later victims of a multi-revocation
// decision).
func (c *ReclamationController) Reclaim(ctx context.Context, saturation, ceiling float64, demandPriority int) {
	inFlight, evictable := c.evictor.Stats()
	if inFlight <= 0 || evictable <= 0 {
		return
	}
	credit := saturation / float64(inFlight)
	if credit <= 0 || math.IsNaN(credit) || math.IsInf(credit, 0) {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	reclaimTarget := saturation - ceiling
	netTarget := reclaimTarget - c.pendingReclaim
	metrics.RecordFlowControlReclaimTarget(c.poolName, reclaimTarget)
	// Under stop-and-wait pacing the gate opens only once every debit has expired, so
	// pendingReclaim is zero here and this branch never executes through the dispatch path. The
	// subtraction makes sizing safe independently of pacing: defense in depth for out-of-band
	// calls, and the invariant any relaxed pacing scheme would rely on.
	if netTarget < 0 || (netTarget == 0 && c.pendingReclaim > 0) {
		return
	}

	// The dispatch gate blocks on saturation >= ceiling, so unblocking requires reclaiming strictly
	// more than the deficit. At the boundary (netTarget == 0 with nothing pending) the quotient
	// rounds to zero, which would leave the band blocked with no reclamation; issue at least one
	// revocation instead.
	n := max(1, int(math.Ceil(netTarget/credit)))
	if n > c.cfg.MaxRevocationsPerDecision {
		n = c.cfg.MaxRevocationsPerDecision
	}
	if n > evictable {
		n = evictable
	}

	// The mutex is held across EvictN so that a confirmation racing the registration below blocks in
	// Confirm until the revocation is present in the outstanding set.
	evicted, err := c.evictor.EvictN(ctx, n, demandPriority)
	if err != nil {
		c.logger.Error(err, "Revocation batch failed", "requested", n)
		return
	}
	now := c.clock.Now()
	for _, requestID := range evicted {
		c.outstanding[requestID] = revocation{credit: credit, issuedAt: now}
		c.pendingReclaim += credit
	}
	metrics.RecordFlowControlRevocationsIssued(c.poolName, len(evicted))
	metrics.RecordFlowControlPendingReclaim(c.poolName, c.pendingReclaim)

	c.logger.V(logutil.DEBUG).Info("Revocations issued",
		"saturation", saturation, "ceiling", ceiling,
		"reclaimTarget", reclaimTarget, "credit", credit,
		"requested", n, "issued", len(evicted))
}

// Confirm records that an evicted request's stream has terminated. Safe for concurrent use; called
// from ext_proc handler goroutines via the evictor's eviction-terminated listener. IDs that are
// not outstanding (e.g. already retired by timeout) are ignored.
func (c *ReclamationController) Confirm(requestID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	r, ok := c.outstanding[requestID]
	if !ok {
		return
	}
	delete(c.outstanding, requestID)
	now := c.clock.Now()
	c.startCoolingLocked(r, now)
	metrics.RecordFlowControlRevocations(c.poolName, metrics.RevocationOutcomeConfirmed, 1)
	metrics.RecordFlowControlRevocationConfirmationDuration(c.poolName, now.Sub(r.issuedAt))
}

// sweepTimeoutsLocked retires revocations whose confirmation never arrived within
// ConfirmationTimeout, so a hung stream cannot hold the pacing gate closed forever.
func (c *ReclamationController) sweepTimeoutsLocked(now time.Time) {
	for requestID, r := range c.outstanding {
		if now.Sub(r.issuedAt) < c.cfg.ConfirmationTimeout {
			continue
		}
		delete(c.outstanding, requestID)
		c.startCoolingLocked(r, now)
		metrics.RecordFlowControlRevocations(c.poolName, metrics.RevocationOutcomeTimedOut, 1)
		c.logger.V(logutil.DEFAULT).Info("Revocation confirmation timed out; treating as confirmed",
			"requestID", requestID, "timeout", c.cfg.ConfirmationTimeout)
	}
}

// startCoolingLocked moves a confirmed revocation's debit into the reclaiming (cooling) stage: the
// stream is dead, but the freed capacity is assumed gauge-invisible for another ConfirmationGrace,
// so the debit keeps suppressing re-sizing until then.
func (c *ReclamationController) startCoolingLocked(r revocation, now time.Time) {
	c.cooling = append(c.cooling, coolingDebit{credit: r.credit, expiresAt: now.Add(c.cfg.ConfirmationGrace)})
}

// expireCoolingLocked releases cooling debits whose grace has elapsed.
func (c *ReclamationController) expireCoolingLocked(now time.Time) {
	kept := c.cooling[:0]
	changed := false
	for _, d := range c.cooling {
		if now.Before(d.expiresAt) {
			kept = append(kept, d)
			continue
		}
		changed = true
		c.pendingReclaim -= d.credit
	}
	c.cooling = kept
	if changed {
		if c.pendingReclaim < 0 {
			c.pendingReclaim = 0
		}
		metrics.RecordFlowControlPendingReclaim(c.poolName, c.pendingReclaim)
	}
}

// maybeReclaim evaluates the eviction trigger on a HoL-blocking break at band index breakIdx.
// The pacing gate is checked before any queue scan, so the common saturated-but-gated case costs a
// single comparison. Demand is eligible when a blocked band holds queued requests whose priority
// is strictly greater than the current victim head's priority (no same-band churn).
//
// priorities is ordered highest first, so once a band's priority is not strictly greater than the
// victim's, no later band can be eligible and the scan stops.
func (p *Processor) maybeReclaim(
	ctx context.Context,
	saturation float64,
	priorities []int,
	ceilings []float64,
	breakIdx int,
) {
	if !p.reclamation.GateOpen(p.clock.Now()) {
		return
	}
	victimPriority, ok := p.reclamation.VictimPriority()
	if !ok {
		return // Nothing is evictable.
	}
	for j := breakIdx; j < len(priorities); j++ {
		if priorities[j] <= victimPriority {
			return
		}
		// A zero ceiling is a policy statement that the band must not dispatch regardless of load.
		// No amount of reclamation can unblock it (saturation >= 0 always holds), so treating its
		// queue as demand would revoke leases in a loop for no benefit.
		if ceilings[j] <= 0 {
			continue
		}
		// Each band's ceiling is checked directly; monotonicity across bands is not assumed. An
		// unblocked band needs no eviction because it will dispatch on a subsequent cycle.
		if saturation < ceilings[j] {
			continue
		}
		band, err := p.registry.PriorityBandAccessor(priorities[j])
		if err != nil {
			continue
		}
		if !bandHasQueuedItems(band) {
			continue
		}
		p.reclamation.Reclaim(ctx, saturation, ceilings[j], priorities[j])
		return
	}
}

// bandHasQueuedItems reports whether any flow queue in the band is non-empty.
func bandHasQueuedItems(band flowcontrol.PriorityBandAccessor) bool {
	found := false
	band.IterateQueues(func(q flowcontrol.FlowQueueAccessor) bool {
		if q.Len() > 0 {
			found = true
			return false
		}
		return true
	})
	return found
}
