package preciseprefixcache

import (
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// KVEventsHealthMonitor tracks per-endpoint KV events pipeline health.
// It records when confirmed (non-speculative) entries were last observed
// in index lookups and when requests were last routed, allowing the scorer
// to distinguish between a broken pipeline (routing but no confirmed events)
// and normal idle (no routing, no events).
//
// This component is data-collection only — it does not modify TTL behavior.
// Dynamic TTL adjustment will be added in a subsequent PR.
type KVEventsHealthMonitor struct {
	state    sync.Map // map[string]*endpointHealth, key: endpoint identifier (e.g. "ip:port")
	hasKVCfg bool     // whether kvEventsConfig is present
}

// endpointHealth holds per-endpoint health data.
type endpointHealth struct {
	mu sync.Mutex

	// lastConfirmedTime is the last time a confirmed (non-speculative) entry
	// was observed in an index lookup for this endpoint. This serves as a
	// proxy for "KV events are arriving" without requiring changes to the
	// kv-cache library.
	lastConfirmedTime time.Time

	// lastRoutedTime is the last time we routed a request to this endpoint
	// via PreRequest.
	lastRoutedTime time.Time
}

// NewKVEventsHealthMonitor creates a new health monitor.
// hasKVEventsConfig indicates whether KV events are configured at all.
func NewKVEventsHealthMonitor(hasKVEventsConfig bool) *KVEventsHealthMonitor {
	return &KVEventsHealthMonitor{
		hasKVCfg: hasKVEventsConfig,
	}
}

// RecordConfirmedEntry is called when a confirmed (non-speculative) entry
// is observed in an index lookup for an endpoint. This indicates that
// KV events are flowing for this endpoint.
func (m *KVEventsHealthMonitor) RecordConfirmedEntry(endpointKey string) {
	h := m.getOrCreate(endpointKey)
	h.mu.Lock()
	h.lastConfirmedTime = time.Now()
	h.mu.Unlock()
}

// RecordRouting is called when a request is routed to an endpoint (PreRequest).
func (m *KVEventsHealthMonitor) RecordRouting(endpointKey string) {
	h := m.getOrCreate(endpointKey)
	h.mu.Lock()
	h.lastRoutedTime = time.Now()
	h.mu.Unlock()
}

// GetHealthStatus returns the health status for an endpoint.
// Returns lastConfirmedTime, lastRoutedTime, and whether the endpoint is known.
func (m *KVEventsHealthMonitor) GetHealthStatus(endpointKey string) (lastConfirmed, lastRouted time.Time, known bool) {
	val, ok := m.state.Load(endpointKey)
	if !ok {
		return time.Time{}, time.Time{}, false
	}
	h := val.(*endpointHealth)
	h.mu.Lock()
	lastConfirmed = h.lastConfirmedTime
	lastRouted = h.lastRoutedTime
	h.mu.Unlock()
	return lastConfirmed, lastRouted, true
}

// HasKVEventsConfig returns whether KV events are configured.
func (m *KVEventsHealthMonitor) HasKVEventsConfig() bool {
	return m.hasKVCfg
}

// RemoveEndpoint cleans up health state for a removed endpoint.
func (m *KVEventsHealthMonitor) RemoveEndpoint(endpointKey string) {
	m.state.Delete(endpointKey)
}

// recordRouting notes that the primary profile's target endpoint was routed to.
func (p *Producer) recordRouting(schedulingResult *scheduling.SchedulingResult) {
	if schedulingResult == nil {
		return
	}
	primary := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return
	}
	meta := primary.TargetEndpoints[0].GetMetadata()
	if meta == nil {
		return
	}
	p.healthMonitor.RecordRouting(endpointIdentifier(meta.Address, meta.Port))
}

// recordConfirmedEndpoints notes the endpoints holding confirmed
// (non-speculative) entries in a lookup result, at most once per endpoint per
// request. The monitor keeps a single timestamp per endpoint, so recording
// every matching block entry would repeat the same write once per block per
// endpoint on the request path. Recording stops as soon as every candidate
// endpoint has been seen; candidates of 0 means the lookup was unfiltered, so
// the endpoint count is unknown and every entry is inspected.
func (p *Producer) recordConfirmedEndpoints(keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	confirmed map[string]struct{}, candidates int,
) {
	for _, pods := range keyToPods {
		if candidates > 0 && len(confirmed) == candidates {
			return
		}
		for _, pod := range pods {
			if pod.Speculative {
				continue
			}
			if _, seen := confirmed[pod.PodIdentifier]; seen {
				continue
			}
			confirmed[pod.PodIdentifier] = struct{}{}
			p.healthMonitor.RecordConfirmedEntry(pod.PodIdentifier)
		}
	}
}

// getOrCreate returns the health state for an endpoint, creating it if needed.
func (m *KVEventsHealthMonitor) getOrCreate(endpointKey string) *endpointHealth {
	val, _ := m.state.LoadOrStore(endpointKey, &endpointHealth{})
	return val.(*endpointHealth)
}
