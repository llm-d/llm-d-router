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

package eviction

import (
	"context"
	"net"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// RequestEvictor tracks in-flight requests via RequestControl hooks and provides eviction capability.
// It is a builtin component wired directly by the EPP, not a user-configurable plugin.
type RequestEvictor struct {
	queue            *EvictionQueue
	evictor          Evictor
	evictionRegistry *EvictionRegistry

	// mu guards the fields below.
	mu sync.Mutex
	// pendingEvictions holds the IDs of requests whose eviction has been issued but whose stream has
	// not yet terminated. Populated by EvictN before the eviction signal is sent; consumed exactly
	// once by cleanupRequest when the stream terminates.
	pendingEvictions map[string]struct{}
	// evictionTerminatedListener, if set, is invoked with the request ID each time an evicted
	// request's stream terminates. It may be called from ext_proc handler goroutines and must be
	// safe for concurrent use.
	evictionTerminatedListener func(requestID string)
}

// NewRequestEvictor creates a RequestEvictor with the given policies and evictor.
func NewRequestEvictor(
	ordering flowcontrol.EvictionOrderingPolicy,
	filter flowcontrol.EvictionFilterPolicy,
	evictor Evictor,
) *RequestEvictor {
	registry := NewEvictionRegistry()
	if re, ok := evictor.(EvictorWithRegistry); ok {
		re.SetRegistry(registry)
	}
	return &RequestEvictor{
		queue:            NewEvictionQueue(ordering, filter),
		evictor:          evictor,
		evictionRegistry: registry,
		pendingEvictions: make(map[string]struct{}),
	}
}

// SetEvictionTerminatedListener registers a callback invoked once per evicted request when its
// stream terminates. This is the confirmation signal for eviction pacing: the request is dead from
// the EPP's perspective, even though the engine may not have freed its resources yet.
// Must be called before the RequestEvictor is in use.
func (p *RequestEvictor) SetEvictionTerminatedListener(listener func(requestID string)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.evictionTerminatedListener = listener
}

// PeekVictimPriority returns the priority of the next request that would be evicted, or false if
// no evictable requests exist. It does not modify the queue.
func (p *RequestEvictor) PeekVictimPriority() (priority int, ok bool) {
	item := p.queue.Peek()
	if item == nil {
		return 0, false
	}
	return item.Priority, true
}

// EvictionRegistry returns the shared eviction registry.
// The ext_proc Process() goroutine uses this to look up eviction channels for dispatched requests.
func (p *RequestEvictor) EvictionRegistry() *EvictionRegistry {
	return p.evictionRegistry
}

// PreRequest is called after scheduling, before the request reaches the model server.
// It tracks the request and, if the filter policy accepts it, adds it to the eviction queue.
func (p *RequestEvictor) PreRequest(
	ctx context.Context,
	request *scheduling.InferenceRequest,
	result *scheduling.SchedulingResult,
) {
	if request == nil || result == nil || len(result.ProfileResults) == 0 {
		return
	}

	profileResult := result.ProfileResults[result.PrimaryProfileName]
	if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
		return
	}

	targetEndpoint := profileResult.TargetEndpoints[0]
	metadata := targetEndpoint.GetMetadata()
	requestID := request.Headers[reqcommon.RequestIDHeaderKey]
	if requestID == "" {
		return
	}

	evictCh := make(chan struct{})

	item := &flowcontrol.EvictionItem{
		RequestID:      requestID,
		Priority:       request.Objectives.Priority,
		DispatchTime:   time.Now(),
		TargetURL:      "http://" + net.JoinHostPort(metadata.GetIPAddress(), metadata.GetPort()),
		Request:        request,
		TargetEndpoint: metadata,
		EvictCh:        evictCh,
	}

	p.queue.Track(item)
	p.evictionRegistry.Register(requestID, evictCh)

	// Bind untrack to the request context's lifetime as a safety net.
	// If the client disconnects and ResponseBody(EndOfStream) never fires,
	// ctx.Done() ensures the request is still cleaned up. Untrack is idempotent.
	go func() {
		<-ctx.Done()
		p.cleanupRequest(requestID)
	}()

	log.FromContext(ctx).V(logutil.DEBUG).Info("Tracked in-flight request",
		"requestID", requestID,
		"priority", item.Priority,
		"evictable", p.queue.EvictableLen(),
		"inFlight", p.queue.InFlightLen())
}

// ResponseBody is called for every response data chunk (streaming) or once (non-streaming).
// On the final call (EndOfStream == true), it removes the request from tracking and the eviction queue.
func (p *RequestEvictor) ResponseBody(
	ctx context.Context,
	request *scheduling.InferenceRequest,
	response *requestcontrol.Response,
	targetEndpoint *datalayer.EndpointMetadata,
) {
	if !response.EndOfStream {
		return
	}
	if request == nil {
		return
	}
	requestID := request.Headers[reqcommon.RequestIDHeaderKey]
	if requestID == "" {
		return
	}

	p.cleanupRequest(requestID)

	log.FromContext(ctx).V(logutil.DEBUG).Info("Untracked completed request",
		"requestID", requestID,
		"evictable", p.queue.EvictableLen(),
		"inFlight", p.queue.InFlightLen())
}

// EvictN attempts to evict up to n requests from the eviction queue, in victim-policy order,
// stopping at the first victim whose priority is not strictly below priorityBound. The bound
// enforces strict priority dominance across a whole multi-revocation decision: the victim head
// check alone does not cover later victims, which come off the heap at priorities at or above the
// head's. Each request is only removed from tracking after a successful eviction. If the eviction
// fails, the request remains in the queue for a future eviction attempt.
// Returns the request IDs that were successfully evicted.
func (p *RequestEvictor) EvictN(ctx context.Context, n int, priorityBound int) ([]string, error) {
	logger := log.FromContext(ctx)
	evicted := make([]string, 0, n)

	for range n {
		items := p.queue.PopN(1)
		if len(items) == 0 {
			break
		}
		item := items[0]

		if item.Priority >= priorityBound {
			// The heap orders victims lowest-priority first, so no remaining victim can be below the
			// bound either. Re-tracking races a concurrent completion: if cleanupRequest ran between
			// the pop and this Track, the entry is re-inserted dead and survives until a future
			// decision evicts it, which then stalls the pacing gate for one ConfirmationTimeout. The
			// window is sub-microsecond and the cost is bounded, so it is tolerated.
			p.queue.Track(item)
			break
		}

		// Mark before the eviction signal is sent so that cleanupRequest observes the marker no matter
		// how quickly the stream terminates.
		p.markPendingEviction(item.RequestID)
		if err := p.evictor.Evict(ctx, item); err != nil {
			logger.Error(err, "Failed to evict request, re-tracking", "requestID", item.RequestID, "targetURL", item.TargetURL)
			p.unmarkPendingEviction(item.RequestID)
			// Same tolerated re-track-vs-completion window as the bound branch above.
			p.queue.Track(item)
			continue
		}
		evicted = append(evicted, item.RequestID)
	}

	if len(evicted) > 0 {
		logger.Info("Eviction complete", "requested", n, "evicted", len(evicted), "requestIDs", evicted)
	}
	return evicted, nil
}

// Stats returns the current in-flight and evictable request counts.
func (p *RequestEvictor) Stats() (inFlight int, evictable int) {
	return p.queue.InFlightLen(), p.queue.EvictableLen()
}

// cleanupRequest removes a request from all tracking structures.
// If the evictor supports cleanup (e.g., ImmediateResponseEvictor), it also
// cleans up evictor-internal state to prevent unbounded map growth.
// If the request had been evicted, the eviction-terminated listener is notified exactly once.
func (p *RequestEvictor) cleanupRequest(requestID string) {
	p.queue.Untrack(requestID)
	p.evictionRegistry.Deregister(requestID)
	if c, ok := p.evictor.(EvictorWithCleanup); ok {
		c.Cleanup(requestID)
	}
	if listener := p.consumePendingEviction(requestID); listener != nil {
		listener(requestID)
	}
}

// markPendingEviction records that an eviction has been issued for the request.
func (p *RequestEvictor) markPendingEviction(requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pendingEvictions[requestID] = struct{}{}
}

// unmarkPendingEviction removes the pending-eviction marker after a failed eviction attempt.
func (p *RequestEvictor) unmarkPendingEviction(requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.pendingEvictions, requestID)
}

// consumePendingEviction removes the pending-eviction marker if present and returns the listener
// to notify, or nil if the request was not evicted or was already consumed. Consuming under the
// same lock as marking guarantees at-most-once notification across the idempotent cleanup paths.
func (p *RequestEvictor) consumePendingEviction(requestID string) func(requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.pendingEvictions[requestID]; !ok {
		return nil
	}
	delete(p.pendingEvictions, requestID)
	return p.evictionTerminatedListener
}

// EvictorWithCleanup is an optional interface for evictors that maintain per-request state
// that needs to be cleaned up when a request completes or is untracked.
type EvictorWithCleanup interface {
	Cleanup(requestID string)
}
