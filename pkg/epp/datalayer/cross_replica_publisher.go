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

package datalayer

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"golang.org/x/time/rate"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// defaultCrossReplicaSyncInterval is the fallback cadence at which local
	// per-endpoint state is pushed to the syncer when none is configured.
	defaultCrossReplicaSyncInterval   = 200 * time.Millisecond
	defaultCrossReplicaPublishTimeout = time.Second
	publishFailureLogInterval         = 30 * time.Second
)

// crossReplicaPublisher owns cross-replica publishing and endpoint lifecycle
// coordination. One shared ticker publishes every registered endpoint.
type crossReplicaPublisher struct {
	syncer         fwkdl.CrossReplicaSyncer
	contributors   []fwkdl.CrossReplicaContributor
	interval       time.Duration
	publishTimeout time.Duration

	// mu guards endpoints and orders syncer writes with endpoint removal.
	mu        sync.RWMutex
	endpoints sets.Set[types.NamespacedName]

	publishFailureLog rate.Sometimes
}

// newCrossReplicaPublisher collects the opted-in CrossReplicaContributors, or
// returns nil if there is no syncer or none opt in. Non-positive durations
// fall back to their defaults.
func newCrossReplicaPublisher(syncer fwkdl.CrossReplicaSyncer, extractors *extractorMap, interval, publishTimeout time.Duration) *crossReplicaPublisher {
	if syncer == nil {
		return nil
	}
	var contributors []fwkdl.CrossReplicaContributor
	extractors.Range(func(_ string, exts []fwkplugin.Plugin) bool {
		for _, ext := range exts {
			if c, ok := ext.(fwkdl.CrossReplicaContributor); ok && !c.CrossReplicaState().SyncDisabled {
				contributors = append(contributors, c)
			}
		}
		return true
	})
	if len(contributors) == 0 {
		return nil
	}
	if interval <= 0 {
		interval = defaultCrossReplicaSyncInterval
	}
	if publishTimeout <= 0 {
		publishTimeout = defaultCrossReplicaPublishTimeout
	}
	for _, c := range contributors {
		metrics.InitCrossReplicaPublishMetrics(string(c.CrossReplicaState().StateKey))
	}
	return &crossReplicaPublisher{
		syncer:            syncer,
		contributors:      contributors,
		interval:          interval,
		publishTimeout:    publishTimeout,
		publishFailureLog: rate.Sometimes{Interval: publishFailureLogInterval},
	}
}

func (p *crossReplicaPublisher) start(ctx context.Context) {
	go p.run(ctx)
}

func (p *crossReplicaPublisher) registerEndpoint(key types.NamespacedName) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.endpoints == nil {
		p.endpoints = sets.New[types.NamespacedName]()
	}
	if p.endpoints.Has(key) {
		return false
	}
	p.endpoints.Insert(key)
	return true
}

func (p *crossReplicaPublisher) run(ctx context.Context) {
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.publishAll(ctx)
		}
	}
}

func (p *crossReplicaPublisher) publishAll(ctx context.Context) {
	snapshot := p.endpointSnapshot()
	// A cycle with no registered endpoints makes no syncer calls; timing it
	// would add a sub-microsecond sample every interval and skew the quantiles.
	if len(snapshot) == 0 {
		return
	}
	start := time.Now()
	cycle := newPublishCycle()
	defer func() {
		// A cycle cut short by shutdown says nothing about the syncer: its
		// unattempted endpoints would read as failures and its duration as
		// an outlier.
		if ctx.Err() != nil {
			return
		}
		metrics.RecordCrossReplicaPublishCycle(time.Since(start))
		cycle.record(time.Now())
	}()
	if syncer, ok := p.syncer.(fwkdl.CrossReplicaBatchSyncer); ok {
		p.publishBatch(ctx, syncer, snapshot, cycle)
		return
	}
	for _, endpointID := range snapshot {
		dispatchCtx, cancel := context.WithTimeout(ctx, p.publishTimeout)
		p.publish(dispatchCtx, endpointID, cycle)
		cancel()
	}
}

func (p *crossReplicaPublisher) publishBatch(ctx context.Context, syncer fwkdl.CrossReplicaBatchSyncer, snapshot []types.NamespacedName, cycle *publishCycle) {
	specs := make([]fwkdl.CrossReplicaSpec, 0, len(p.contributors))
	for _, c := range p.contributors {
		specs = append(specs, c.CrossReplicaState())
	}
	p.mu.RLock()
	endpointIDs := make([]string, 0, len(snapshot))
	for _, key := range snapshot {
		if p.endpoints.Has(key) {
			endpointIDs = append(endpointIDs, key.String())
		}
	}
	p.mu.RUnlock()
	if len(endpointIDs) == 0 {
		return
	}
	logger := log.FromContext(ctx)
	var wg sync.WaitGroup
	for _, spec := range specs {
		wg.Go(func() {
			failures := syncer.SetBatch(ctx, spec, endpointIDs, p.publishTimeout)
			cycle.addBatch(spec.StateKey, len(endpointIDs), failures)
			if len(failures) == 0 {
				return
			}
			p.publishFailureLog.Do(func() {
				for endpointID, err := range failures {
					logger.Error(err, "cross-replica batch publish failed", "key", spec.StateKey, "endpoint", endpointID, "failedEndpoints", len(failures))
					break
				}
			})
		})
	}
	wg.Wait()
	p.reconcileRemoved(ctx, syncer, specs, snapshot, cycle)
}

// reconcileRemoved deletes shared state left by endpoints removed while the
// batch ran: their delete may have completed before a chunk's write landed,
// resurrecting the field. A missed cleanup still expires within stateTTL.
// Endpoints re-registered during the cleanup lost that field again, so they
// are published once more; a republish that lands after a later delete is the
// same resurrected-field case and is bounded the same way.
func (p *crossReplicaPublisher) reconcileRemoved(ctx context.Context, syncer fwkdl.CrossReplicaBatchSyncer, specs []fwkdl.CrossReplicaSpec, snapshot []types.NamespacedName, cycle *publishCycle) {
	if ctx.Err() != nil {
		return
	}
	logger := log.FromContext(ctx)
	p.mu.RLock()
	var removed []types.NamespacedName
	for _, key := range snapshot {
		if !p.endpoints.Has(key) {
			removed = append(removed, key)
		}
	}
	p.mu.RUnlock()
	if len(removed) == 0 {
		return
	}
	deleteCtx, cancel := context.WithTimeout(ctx, p.publishTimeout)
	for _, key := range removed {
		endpointID := key.String()
		for _, spec := range specs {
			if err := p.syncer.Delete(deleteCtx, spec.StateKey, endpointID); err != nil {
				p.publishFailureLog.Do(func() {
					logger.Error(err, "cross-replica batch delete failed", "key", spec.StateKey, "endpoint", endpointID)
				})
			}
		}
	}
	cancel()
	var revived []string
	p.mu.RLock()
	for _, key := range removed {
		if p.endpoints.Has(key) {
			revived = append(revived, key.String())
		}
	}
	p.mu.RUnlock()
	if len(revived) == 0 {
		return
	}
	var wg sync.WaitGroup
	for _, spec := range specs {
		wg.Go(func() {
			failures := syncer.SetBatch(ctx, spec, revived, p.publishTimeout)
			cycle.addBatch(spec.StateKey, len(revived), failures)
			for endpointID, err := range failures {
				p.publishFailureLog.Do(func() {
					logger.Error(err, "cross-replica batch republish failed", "key", spec.StateKey, "endpoint", endpointID)
				})
			}
		})
	}
	wg.Wait()
}

func (p *crossReplicaPublisher) endpointSnapshot() []types.NamespacedName {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.endpoints.UnsortedList()
}

func (p *crossReplicaPublisher) handleEndpointEvent(ctx context.Context, event fwkdl.EndpointEvent, plugin fwkplugin.Plugin) {
	contributor, ok := plugin.(fwkdl.CrossReplicaContributor)
	if !ok {
		return
	}
	spec := contributor.CrossReplicaState()
	if spec.SyncDisabled {
		return
	}
	if event.Type != fwkdl.EventAddOrUpdate {
		return
	}
	endpointID := event.Endpoint.GetMetadata().GetNamespacedName().String()
	event.Endpoint.GetAttributes().Put(spec.AttributeKey, &fwkdl.DynamicAttribute{
		Get: func() fwkdl.Cloneable {
			value, ok, err := p.get(ctx, spec, endpointID)
			if err != nil || !ok {
				return spec.Read(endpointID)
			}
			if cloneable, ok := value.(fwkdl.Cloneable); ok {
				return cloneable
			}
			return spec.Read(endpointID)
		},
	})
}

// set publishes one endpoint's value for spec. published is false when the
// endpoint was deleted after publishAll took its snapshot and the syncer was
// not called.
func (p *crossReplicaPublisher) set(ctx context.Context, spec fwkdl.CrossReplicaSpec, key types.NamespacedName) (published bool, err error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if !p.endpoints.Has(key) {
		return false, nil
	}
	if err := p.syncer.Set(ctx, spec, key.String()); err != nil {
		return false, err
	}
	return true, nil
}

func (p *crossReplicaPublisher) get(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) (any, bool, error) {
	return p.syncer.Get(ctx, spec, endpointID)
}

func (p *crossReplicaPublisher) delete(ctx context.Context, key types.NamespacedName) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.endpoints.Has(key) {
		return false, nil
	}
	p.endpoints.Delete(key)

	ctx, cancel := context.WithTimeout(ctx, p.publishTimeout)
	defer cancel()
	endpointID := key.String()
	var errs []error
	for _, c := range p.contributors {
		spec := c.CrossReplicaState()
		if err := p.syncer.Delete(ctx, spec.StateKey, endpointID); err != nil {
			errs = append(errs, fmt.Errorf("delete shared state for key %s: %w", spec.StateKey, err))
		}
	}
	return true, errors.Join(errs...)
}

// publish writes one endpoint's value for every contributor and reports each
// outcome to cycle. cycle may be nil when the caller is not timing a cycle.
func (p *crossReplicaPublisher) publish(ctx context.Context, key types.NamespacedName, cycle *publishCycle) {
	endpointID := key.String()
	logger := log.FromContext(ctx).WithValues("endpoint", endpointID)
	var wg sync.WaitGroup
	for _, c := range p.contributors {
		spec := c.CrossReplicaState()
		wg.Go(func() {
			published, err := p.set(ctx, spec, key)
			if err != nil {
				cycle.addFailure(spec.StateKey, err, errors.Is(ctx.Err(), context.DeadlineExceeded))
				p.publishFailureLog.Do(func() {
					logger.Error(err, "cross-replica publish failed", "key", spec.StateKey)
				})
				return
			}
			// An endpoint removed after the snapshot was skipped, not
			// attempted: it is not reported, so it can neither count as a
			// success nor make the cycle look clean.
			if published {
				cycle.addSuccess(spec.StateKey)
			}
		})
	}
	wg.Wait()
}

// publishCycle collects, per state key, how many endpoints one publish cycle
// attempted and how each failed one failed. Every publish path only reports
// facts to it; record turns them into metrics once, at the end of the cycle,
// so what counts as a successful cycle is decided in one place. An endpoint
// skipped because it was removed during the cycle is never reported, so it
// can neither count as a success nor make the cycle look clean. All methods
// are safe for concurrent use and are no-ops on a nil cycle.
type publishCycle struct {
	mu      sync.Mutex
	results map[fwkdl.StateKey]*publishResult
}

type publishResult struct {
	attempted, errored, timedOut int
}

func newPublishCycle() *publishCycle {
	return &publishCycle{results: make(map[fwkdl.StateKey]*publishResult)}
}

// addSuccess reports one endpoint the syncer accepted.
func (c *publishCycle) addSuccess(stateKey fwkdl.StateKey) {
	c.update(stateKey, func(r *publishResult) { r.attempted++ })
}

// addFailure reports one endpoint the syncer did not accept. deadlineExpired
// is whether the publish deadline had expired; it takes precedence over err,
// as an error returned after the deadline is a symptom of it.
func (c *publishCycle) addFailure(stateKey fwkdl.StateKey, err error, deadlineExpired bool) {
	c.update(stateKey, func(r *publishResult) {
		r.attempted++
		r.count(err, deadlineExpired)
	})
}

// addBatch reports one SetBatch call: attempted endpoints and the failure of
// each endpoint that was not published.
func (c *publishCycle) addBatch(stateKey fwkdl.StateKey, attempted int, failures map[string]error) {
	if attempted == 0 {
		return
	}
	c.update(stateKey, func(r *publishResult) {
		r.attempted += attempted
		for _, err := range failures {
			r.count(err, false)
		}
	})
}

func (c *publishCycle) update(stateKey fwkdl.StateKey, f func(*publishResult)) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	r, ok := c.results[stateKey]
	if !ok {
		r = &publishResult{}
		c.results[stateKey] = r
	}
	f(r)
}

func (r *publishResult) count(err error, deadlineExpired bool) {
	if deadlineExpired || isTimeout(err) {
		r.timedOut++
		return
	}
	r.errored++
}

// record emits one cycle's outcome per state key. The failure log is rate
// limited, so these counters are the only complete record of a syncer outage.
func (c *publishCycle) record(now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for stateKey, r := range c.results {
		succeeded := r.attempted - r.errored - r.timedOut
		metrics.RecordCrossReplicaCycleOutcome(string(stateKey), succeeded, r.errored, r.timedOut, now)
	}
}

// isTimeout reports whether err is an expired deadline, either a context
// deadline or a network timeout from the syncer's client.
func isTimeout(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var timeout interface{ Timeout() bool }
	return errors.As(err, &timeout) && timeout.Timeout()
}
