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
	"sync"
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
// absent from the candidate set migrates to the least-loaded present pod
// immediately, no retry wait.
type sessionIDHeaderStrategy struct {
	sessionHeader string
	profileName   string
	// ttl is how long a binding survives unused.
	ttl time.Duration
	// sweepInterval is how often expired bindings are swept.
	sweepInterval time.Duration
	// mu serializes the bind, migrate, and evict sequences so that a binding's
	// insertion or removal and the matching podCount adjustment are atomic with
	// respect to each other. Without it a first-bind's increment can land after
	// a concurrent sweep's decrement, stranding a count with no live binding.
	mu sync.Mutex
	// bindings maps a session identifier to the pod serving it.
	bindings sync.Map
	// podCount maps a pod key to the number of sessions currently bound to it.
	podCount sync.Map
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

// newSessionIDHeaderStrategy builds the session_id_header strategy and starts
// its eviction sweep for the plugin's lifetime.
func newSessionIDHeaderStrategy(params parameters, handle plugin.Handle) strategy {
	s := &sessionIDHeaderStrategy{
		sessionHeader: sessionutil.NormalizeHeader(params.HeaderName),
		profileName:   params.ProfileName,
		ttl:           time.Duration(params.EvictionTTLSeconds * float64(time.Second)),
		sweepInterval: time.Duration(params.EvictionSweepSeconds * float64(time.Second)),
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
		s.pluginState.Write(request.RequestID, plugin.StateKey(SessionAffinityType), &boundPodPresence{present: target != nil})
	}

	if target == nil {
		target = s.leastLoadedPod(endpoints)
	}
	scoredEndpoints[target] = 1.0

	return scoredEndpoints
}

// leastLoadedPod returns the candidate pod bound by the fewest sessions.
func (s *sessionIDHeaderStrategy) leastLoadedPod(endpoints []scheduling.Endpoint) scheduling.Endpoint {
	best := endpoints[0]
	minCount := s.podSessionCount(best.GetMetadata().ID.String())
	for _, endpoint := range endpoints[1:] {
		c := s.podSessionCount(endpoint.GetMetadata().ID.String())
		if c < minCount {
			minCount = c
			best = endpoint
		}
	}
	return best
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

// preRequest is the only place podCount and bindings are mutated. A session
// migrates only when its bound pod is absent from the candidate set.
func (s *sessionIDHeaderStrategy) preRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	sessionID := sessionutil.SessionID(request, s.sessionHeader)
	if sessionID == "" {
		return
	}
	podName := s.pickedPodName(schedulingResult)
	if podName == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	fresh := binding{podName: podName, lastSeen: time.Now()}
	existingValue, loaded := s.bindings.LoadOrStore(sessionID, fresh)
	if !loaded {
		s.incrementPodCount(podName)
		log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - bound session to pod",
			"plugin", SessionAffinityType, "pod", podName)
		return
	}

	existing, ok := existingValue.(binding)
	if !ok {
		return
	}

	if existing.podName == podName {
		s.bindings.CompareAndSwap(sessionID, existingValue, fresh)
		return
	}

	// Migrate only on a confirmed absence recorded by this request's own Score
	// call. A missing record (e.g. a concurrent first-bind race) is not treated
	// as absence, so it never forces a migration.
	present, err := plugin.ReadPluginStateKey[*boundPodPresence](s.pluginState, request.RequestID, plugin.StateKey(SessionAffinityType))
	if err != nil || present == nil || present.present {
		return
	}

	if !s.bindings.CompareAndSwap(sessionID, existingValue, fresh) {
		return
	}
	s.decrementPodCount(existing.podName)
	s.incrementPodCount(podName)
	log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - session migrated pod",
		"plugin", SessionAffinityType, "from", existing.podName, "to", podName)
}

// incrementPodCount records one more session bound to podKey. Callers hold
// s.mu, so the read-modify-write needs no CAS retry; sync.Map is still
// required for the unlocked reader in podSessionCount.
func (s *sessionIDHeaderStrategy) incrementPodCount(podKey string) {
	s.podCount.Store(podKey, s.podSessionCount(podKey)+1)
}

// decrementPodCount records one fewer session bound to podKey, removing the
// entry once it reaches zero. Callers hold s.mu.
func (s *sessionIDHeaderStrategy) decrementPodCount(podKey string) {
	c := s.podSessionCount(podKey)
	if c <= 1 {
		s.podCount.Delete(podKey)
		return
	}
	s.podCount.Store(podKey, c-1)
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
					s.mu.Lock()
					s.bindings.CompareAndDelete(key, value)
					s.mu.Unlock()
					return true
				}
				if !s.expired(b.lastSeen, now) {
					return true
				}
				// Re-check under the lock: a concurrent preRequest may have
				// refreshed this binding since Range snapshotted it, in which
				// case CompareAndDelete correctly declines to remove it.
				s.mu.Lock()
				if s.bindings.CompareAndDelete(key, value) {
					s.decrementPodCount(b.podName)
				}
				s.mu.Unlock()
				return true
			})
		}
	}
}
