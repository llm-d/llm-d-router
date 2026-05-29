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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
)

// DemandDrivenEvictionHandler evicts in-flight requests proportionally to the queued demand
// signaled by the dispatch cycle's HoL blocking detection.
type DemandDrivenEvictionHandler struct {
	evictor *RequestEvictor
}

var _ types.EvictionHandler = (*DemandDrivenEvictionHandler)(nil)

// NewDemandDrivenEvictionHandler creates a handler that evicts in-flight requests when
// higher-priority requests are blocked by saturation.
func NewDemandDrivenEvictionHandler(evictor *RequestEvictor) *DemandDrivenEvictionHandler {
	return &DemandDrivenEvictionHandler{evictor: evictor}
}

// HandleEvictionDemand evicts up to min(demand.QueuedCount, evictable) in-flight requests,
// but only if the evictable requests have strictly lower priority than the blocked band.
func (h *DemandDrivenEvictionHandler) HandleEvictionDemand(ctx context.Context, demand types.EvictionDemand) {
	if ctx.Err() != nil {
		return
	}

	logger := log.FromContext(ctx)

	_, evictable := h.evictor.Stats()
	if evictable == 0 {
		logger.V(logutil.DEBUG).Info("Demand-driven eviction: no evictable in-flight requests",
			"blockedPriority", demand.BlockedPriority,
			"queuedCount", demand.QueuedCount)
		return
	}

	evictablePriority, ok := h.evictor.PeekNextEvictable()
	if !ok {
		return
	}
	if evictablePriority >= demand.BlockedPriority {
		logger.V(logutil.DEBUG).Info("Demand-driven eviction: evictable priority is not lower than blocked priority, skipping",
			"blockedPriority", demand.BlockedPriority,
			"evictablePriority", evictablePriority)
		return
	}

	n := min(demand.QueuedCount, evictable)

	evicted, err := h.evictor.EvictN(ctx, n)
	if err != nil {
		logger.Error(err, "Demand-driven eviction failed",
			"blockedPriority", demand.BlockedPriority,
			"requested", n)
		return
	}

	if len(evicted) > 0 {
		logger.Info("Demand-driven eviction completed",
			"blockedPriority", demand.BlockedPriority,
			"queuedCount", demand.QueuedCount,
			"requested", n,
			"evicted", len(evicted),
			"saturation", demand.Saturation)
	}
}
