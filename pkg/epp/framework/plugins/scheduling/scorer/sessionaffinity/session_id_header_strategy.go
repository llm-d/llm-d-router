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

package sessionaffinity

import (
	"context"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// sessionIDHeaderStrategy maps an opaque client-supplied session identifier to
// a pod via a bounded, TTL-evicted binding store. An unbound session is placed
// on the pod with the fewest bound sessions. A session whose bound pod is
// absent from the candidate set stays pinned until it has been absent for
// missThreshold consecutive requests, then migrates to the least-loaded
// present pod; this absorbs a transient candidate-set blip the same way
// program-aware-scorer does.
type sessionIDHeaderStrategy struct {
	sessionHeader string
	profileName   string
	// ttl is how long a binding survives unused.
	ttl time.Duration
	// sweepInterval is how often expired bindings are swept.
	sweepInterval time.Duration
	// missThreshold is the number of consecutive requests a session's bound pod
	// may be absent from the candidate set before the session migrates.
	missThreshold int
	// bindings maps a session identifier to the pod serving it.
	bindings sync.Map
	// podCount maps a pod key to the number of sessions currently bound to it.
	podCount sync.Map
	// misses maps a session identifier to the count of consecutive requests
	// during which its bound pod was absent from the candidate set.
	misses sync.Map
	// rrCursor rotates the least-loaded tie-break so new sessions spread round-robin.
	rrCursor atomic.Int64
	// pluginState carries the score->preRequest handoff of whether the bound
	// pod was present in this request's candidate set.
	pluginState *plugin.PluginState
}

// boundPodPresence records, for one request, whether the session's bound pod
// was present in the candidate set scored. preRequest reads this so it can
// tell a genuine absence from the picker merely choosing a different pod.
type boundPodPresence struct {
	present bool
}

func (b *boundPodPresence) Clone() plugin.StateData {
	return &boundPodPresence{present: b.present}
}

const defaultMissThreshold = 3

// newSessionIDHeaderStrategy builds the session_id_header strategy and starts
// its eviction sweep for the plugin's lifetime.
func newSessionIDHeaderStrategy(params parameters, handle plugin.Handle) strategy {
	missThreshold := defaultMissThreshold
	if params.MissThreshold > 0 {
		missThreshold = params.MissThreshold
	}
	s := &sessionIDHeaderStrategy{
		sessionHeader: sessionutil.NormalizeHeader(params.HeaderName),
		profileName:   params.ProfileName,
		ttl:           time.Duration(params.EvictionTTLSeconds * float64(time.Second)),
		sweepInterval: time.Duration(params.EvictionSweepSeconds * float64(time.Second)),
		missThreshold: missThreshold,
		pluginState:   plugin.NewPluginState(handle.Context()),
	}
	go s.runEviction(handle.Context(), s.sweepInterval)
	return s
}

// binding is the pod a session is pinned to, and when that pin was last used.
type binding struct {
	podName  string
	lastSeen time.Time
}

func (s *sessionIDHeaderStrategy) score(_ context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64)
	if len(endpoints) == 0 {
		return scoredEndpoints
	}

	sessionID := sessionutil.SessionID(request, s.sessionHeader)
	podName := s.boundPod(sessionID)

	var target scheduling.Endpoint
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] = 0.0
		if podName != "" && endpoint.GetMetadata().ID.String() == podName {
			target = endpoint
		}
	}

	if sessionID != "" && podName != "" {
		// Record whether the bound pod was actually a candidate, so PreRequest can
		// tell a genuine absence from the picker merely choosing a different pod.
		s.pluginState.Write(request.RequestID, plugin.StateKey(SessionAffinityType), &boundPodPresence{present: target != nil})
	}

	if target == nil {
		// Unbound, or the bound pod is absent from this request's candidate set.
		// Prefer the least-loaded present pod without changing any binding here.
		target = s.leastLoadedPod(endpoints)
	}
	scoredEndpoints[target] = 1.0

	return scoredEndpoints
}

// leastLoadedPod returns the candidate pod bound by the fewest sessions,
// breaking ties with a rotating cursor so new sessions spread round-robin
// across equally loaded pods.
func (s *sessionIDHeaderStrategy) leastLoadedPod(endpoints []scheduling.Endpoint) scheduling.Endpoint {
	keys := make([]string, len(endpoints))
	byKey := make(map[string]scheduling.Endpoint, len(endpoints))
	for i, endpoint := range endpoints {
		k := endpoint.GetMetadata().ID.String()
		keys[i] = k
		byKey[k] = endpoint
	}
	sort.Strings(keys) // deterministic order independent of candidate-slice ordering

	minCount := -1
	candidates := make([]string, 0, len(keys))
	for _, k := range keys {
		c := s.podSessionCount(k)
		switch {
		case minCount == -1 || c < minCount:
			minCount = c
			candidates = append(candidates[:0], k)
		case c == minCount:
			candidates = append(candidates, k)
		}
	}
	cursor := s.rrCursor.Add(1)
	return byKey[candidates[cursor%int64(len(candidates))]]
}

// podSessionCount returns the number of sessions currently bound to podKey.
func (s *sessionIDHeaderStrategy) podSessionCount(podKey string) int {
	v, ok := s.podCount.Load(podKey)
	if !ok {
		return 0
	}
	c, ok := v.(int)
	if !ok {
		return 0
	}
	return c
}

// boundPod returns the pod bound to sessionID, or "" when unbound. Expiry is
// enforced by the sweeper, not on read.
func (s *sessionIDHeaderStrategy) boundPod(sessionID string) string {
	if sessionID == "" {
		return ""
	}
	value, ok := s.bindings.Load(sessionID)
	if !ok {
		return ""
	}
	b, ok := value.(binding)
	if !ok {
		return ""
	}
	return b.podName
}

// preRequest commits this request's outcome for its session: a first-seen
// session binds to the pod picked; an existing binding is kept sticky until its
// pod has been absent from the candidate set for missThreshold consecutive
// requests, then migrates to the pod picked. podCount is adjusted here, and
// only here, to match the commit.
func (s *sessionIDHeaderStrategy) preRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	sessionID := sessionutil.SessionID(request, s.sessionHeader)
	if sessionID == "" {
		return
	}
	podName := s.pickedPodName(schedulingResult)
	if podName == "" {
		return
	}

	fresh := binding{podName: podName, lastSeen: time.Now()}
	existingValue, loaded := s.bindings.LoadOrStore(sessionID, fresh)
	if !loaded {
		// First time we see this session: bind it to the pod picked.
		s.incrementPodCount(podName)
		s.misses.Delete(sessionID)
		log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - bound session to pod",
			"plugin", SessionAffinityType, "pod", podName)
		return
	}

	existing, ok := existingValue.(binding)
	if !ok {
		return
	}

	if existing.podName == podName {
		// Pin honored (or the picker re-selected it): refresh recency and reset misses.
		s.bindings.CompareAndSwap(sessionID, existingValue, fresh)
		s.misses.Delete(sessionID)
		return
	}

	present, _ := plugin.ReadPluginStateKey[*boundPodPresence](s.pluginState, request.RequestID, plugin.StateKey(SessionAffinityType))
	if present != nil && present.present {
		// The bound pod was a candidate but the picker chose otherwise: leave the
		// pin untouched. Only a genuine absence counts toward migration.
		return
	}

	misses := s.recordMiss(sessionID)
	if misses < s.missThreshold {
		return
	}

	if !s.bindings.CompareAndSwap(sessionID, existingValue, fresh) {
		return
	}
	s.decrementPodCount(existing.podName)
	s.incrementPodCount(podName)
	s.misses.Delete(sessionID)
	log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - session migrated pod",
		"plugin", SessionAffinityType, "from", existing.podName, "to", podName)
}

// recordMiss increments and returns sessionID's consecutive-absence counter.
func (s *sessionIDHeaderStrategy) recordMiss(sessionID string) int {
	for {
		v, _ := s.misses.LoadOrStore(sessionID, 0)
		c, _ := v.(int)
		if s.misses.CompareAndSwap(sessionID, c, c+1) {
			return c + 1
		}
	}
}

// incrementPodCount records one more session bound to podKey.
func (s *sessionIDHeaderStrategy) incrementPodCount(podKey string) {
	for {
		v, _ := s.podCount.LoadOrStore(podKey, 0)
		c, _ := v.(int)
		if s.podCount.CompareAndSwap(podKey, c, c+1) {
			return
		}
	}
}

// decrementPodCount records one fewer session bound to podKey, removing the
// entry once it reaches zero.
func (s *sessionIDHeaderStrategy) decrementPodCount(podKey string) {
	for {
		v, ok := s.podCount.Load(podKey)
		if !ok {
			return
		}
		c, _ := v.(int)
		if c <= 1 {
			if s.podCount.CompareAndDelete(podKey, v) {
				return
			}
			continue
		}
		if s.podCount.CompareAndSwap(podKey, c, c-1) {
			return
		}
	}
}

// pickedPodName returns the pod chosen for this instance's profile, or "" when
// that profile was not scheduled. An empty profileName selects the primary
// (decode) profile, matching ResolvePodToWrite.
func (s *sessionIDHeaderStrategy) pickedPodName(schedulingResult *scheduling.SchedulingResult) string {
	if schedulingResult == nil {
		return ""
	}
	profileName := s.profileName
	if profileName == "" {
		profileName = schedulingResult.PrimaryProfileName
	}
	result, ok := schedulingResult.ProfileResults[profileName]
	if !ok || result == nil || len(result.TargetEndpoints) == 0 || result.TargetEndpoints[0] == nil {
		return ""
	}
	if md := result.TargetEndpoints[0].GetMetadata(); md != nil {
		return md.ID.String()
	}
	return ""
}

// responseHeader is a no-op: under session_id_header the client owns its
// session identifier.
func (s *sessionIDHeaderStrategy) responseHeader(context.Context, *scheduling.InferenceRequest, *requestcontrol.Response, *datalayer.EndpointMetadata) {
}

// expired reports whether a binding last used at lastSeen has outlived the TTL.
func (s *sessionIDHeaderStrategy) expired(lastSeen, now time.Time) bool {
	return s.ttl > 0 && now.Sub(lastSeen) > s.ttl
}

// runEviction drops bindings unused for longer than the TTL until ctx is cancelled.
func (s *sessionIDHeaderStrategy) runEviction(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			s.bindings.Range(func(key, value any) bool {
				b, ok := value.(binding)
				if !ok {
					s.bindings.CompareAndDelete(key, value)
					return true
				}
				if !s.expired(b.lastSeen, now) {
					return true
				}
				// Guard against a concurrent PreRequest refreshing or migrating this
				// binding between Range's snapshot and this delete.
				if s.bindings.CompareAndDelete(key, value) {
					s.misses.Delete(key)
					s.decrementPodCount(b.podName)
				}
				return true
			})
		}
	}
}
