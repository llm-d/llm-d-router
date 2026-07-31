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
)

// SessionIDHeader maps an opaque client-supplied session identifier to a pod
// via a bounded, TTL-evicted binding store. An unbound session is placed on the
// pod with the fewest bound sessions. A session whose bound pod is absent from
// the candidate set migrates to the least-loaded present pod immediately, no
// retry wait.
//
// It is the shared implementation behind the session-affinity scorer and
// filter: Choose resolves the target endpoint, PreRequest commits the binding,
// ResponseHeader is a no-op. Each plugin embeds it and adds only its own output
// method.
type SessionIDHeader struct {
	sessionHeader string
	profileName   string
	// pluginKey keys the Choose->PreRequest presence handoff and labels the
	// plugin in logs; the owning scorer/filter passes its own plugin type.
	pluginKey     plugin.StateKey
	ttl           time.Duration
	sweepInterval time.Duration
	// mu makes each bind/migrate/evict sequence atomic with its podCount
	// adjustment: without it a first-bind increment can land after a concurrent
	// sweep decrement, stranding a count with no live binding.
	mu          sync.Mutex
	bindings    sync.Map // session identifier -> binding
	podCount    sync.Map // pod key -> session count
	pluginState *plugin.PluginState
}

// boundPodPresence records, for one request, whether the session's bound pod
// was present in the candidate set scored. PreRequest reads this so it can
// tell a genuine absence from the picker merely choosing a different pod.
type boundPodPresence struct {
	present bool
}

func (b *boundPodPresence) Clone() plugin.StateData {
	return &boundPodPresence{present: b.present}
}

// NewSessionIDHeader builds the shared session_id_header implementation and
// starts its eviction sweep for the plugin's lifetime. pluginKey is the owning
// plugin's type, used for the PluginState handoff key and log labels.
func NewSessionIDHeader(headerName, profileName string, ttlSeconds, sweepSeconds float64, pluginKey plugin.StateKey, handle plugin.Handle) *SessionIDHeader {
	s := &SessionIDHeader{
		sessionHeader: NormalizeHeader(headerName),
		profileName:   profileName,
		pluginKey:     pluginKey,
		ttl:           time.Duration(ttlSeconds * float64(time.Second)),
		sweepInterval: time.Duration(sweepSeconds * float64(time.Second)),
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

// Choose resolves the endpoint a request should be pinned to and records, for
// PreRequest, whether the session's bound pod was present in the candidate set.
// It returns (nil, false) when there are no candidates or the request carries
// no session identifier: no session means no affinity preference. Otherwise it
// returns the bound pod when present, else the least-loaded candidate, with ok
// true. Choose does not mutate the store; the binding is committed in
// PreRequest.
func (s *SessionIDHeader) Choose(request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) (scheduling.Endpoint, bool) {
	if len(endpoints) == 0 {
		return nil, false
	}

	sessionID := SessionID(request, s.sessionHeader)
	if sessionID == "" {
		return nil, false
	}

	podName := s.boundPod(sessionID)
	var target scheduling.Endpoint
	for _, endpoint := range endpoints {
		if podName != "" && endpoint.GetMetadata().ID.String() == podName {
			target = endpoint
		}
	}

	if podName != "" {
		s.pluginState.Write(request.RequestID, s.pluginKey, &boundPodPresence{present: target != nil})
	}

	if target == nil {
		target = s.leastLoadedPod(endpoints)
	}
	return target, true
}

// leastLoadedPod returns the candidate bound by the fewest sessions.
func (s *SessionIDHeader) leastLoadedPod(endpoints []scheduling.Endpoint) scheduling.Endpoint {
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

func (s *SessionIDHeader) podSessionCount(podKey string) int {
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
func (s *SessionIDHeader) boundPod(sessionID string) string {
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

// PreRequest is the only place podCount and bindings are mutated. A session
// migrates only when its bound pod is absent from the candidate set.
func (s *SessionIDHeader) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	sessionID := SessionID(request, s.sessionHeader)
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
			"plugin", string(s.pluginKey), "pod", podName)
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

	// Migrate only on a confirmed absence recorded by this request's own Choose
	// call. A missing record (e.g. a concurrent first-bind race) is not treated
	// as absence, so it never forces a migration.
	present, err := plugin.ReadPluginStateKey[*boundPodPresence](s.pluginState, request.RequestID, s.pluginKey)
	if err != nil || present == nil || present.present {
		return
	}

	if !s.bindings.CompareAndSwap(sessionID, existingValue, fresh) {
		return
	}
	s.decrementPodCount(existing.podName)
	s.incrementPodCount(podName)
	log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - session migrated pod",
		"plugin", string(s.pluginKey), "from", existing.podName, "to", podName)
}

// incrementPodCount records one more session bound to podKey. Callers hold
// s.mu, so the read-modify-write needs no CAS retry; sync.Map is still
// required for the unlocked reader in podSessionCount.
func (s *SessionIDHeader) incrementPodCount(podKey string) {
	s.podCount.Store(podKey, s.podSessionCount(podKey)+1)
}

// decrementPodCount records one fewer session bound to podKey, removing the
// entry once it reaches zero. Callers hold s.mu.
func (s *SessionIDHeader) decrementPodCount(podKey string) {
	c := s.podSessionCount(podKey)
	if c <= 1 {
		s.podCount.Delete(podKey)
		return
	}
	s.podCount.Store(podKey, c-1)
}

// pickedPodName returns the pod chosen for this instance's profile, or "" when
// that profile was not scheduled. An empty profileName selects the primary
// (decode) profile.
func (s *SessionIDHeader) pickedPodName(schedulingResult *scheduling.SchedulingResult) string {
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

// ResponseHeader is a no-op: under session_id_header the client owns its
// session identifier.
func (s *SessionIDHeader) ResponseHeader(context.Context, *scheduling.InferenceRequest, *requestcontrol.Response, *datalayer.EndpointMetadata) {
}

// expired reports whether a binding last used at lastSeen has outlived the TTL.
func (s *SessionIDHeader) expired(lastSeen, now time.Time) bool {
	return s.ttl > 0 && now.Sub(lastSeen) > s.ttl
}

// runEviction drops bindings unused for longer than the TTL until ctx is cancelled.
func (s *SessionIDHeader) runEviction(ctx context.Context, interval time.Duration) {
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
