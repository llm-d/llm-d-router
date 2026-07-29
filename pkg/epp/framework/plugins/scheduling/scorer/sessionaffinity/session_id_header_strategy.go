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
// a pod via a bounded, TTL-evicted binding store.
type sessionIDHeaderStrategy struct {
	sessionHeader string
	profileName   string
	// ttl is how long a binding survives unused.
	ttl time.Duration
	// sweepInterval is how often expired bindings are swept.
	sweepInterval time.Duration
	// bindings maps a session identifier to the pod serving it.
	bindings sync.Map
}

// newSessionIDHeaderStrategy builds the session_id_header strategy and starts
// its eviction sweep for the plugin's lifetime.
func newSessionIDHeaderStrategy(params parameters, handle plugin.Handle) strategy {
	s := &sessionIDHeaderStrategy{
		sessionHeader: sessionutil.NormalizeHeader(params.HeaderName),
		profileName:   params.ProfileName,
		ttl:           time.Duration(params.EvictionTTLSeconds * float64(time.Second)),
		sweepInterval: time.Duration(params.EvictionSweepSeconds * float64(time.Second)),
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
	podName := s.boundPod(sessionutil.SessionID(request, s.sessionHeader))
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] = 0.0 // initial value
		if endpoint.GetMetadata().ID.String() == podName {
			scoredEndpoints[endpoint] = 1.0
		}
	}
	return scoredEndpoints
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

// preRequest binds this request's session to the pod that served it, so subsequent
// requests in the session prefer it. It runs after the pick because only then is
// that pod known.
func (s *sessionIDHeaderStrategy) preRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	sessionID := sessionutil.SessionID(request, s.sessionHeader)
	if sessionID == "" {
		return
	}
	podName := s.pickedPodName(schedulingResult)
	if podName == "" {
		return
	}

	s.bindings.Store(sessionID, binding{podName: podName, lastSeen: time.Now()})
	log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - bound session to pod",
		"plugin", SessionAffinityType, "pod", podName)
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
				if b, ok := value.(binding); !ok || s.expired(b.lastSeen, now) {
					s.bindings.Delete(key)
				}
				return true
			})
		}
	}
}
