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
	"encoding/json"
	"fmt"
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

const (
	// SessionAffinityType is the type of the SessionAffinity scorer.
	SessionAffinityType = "session-affinity-scorer"

	// AlgorithmEncodedEndpointHeader echoes the picked pod back to the client,
	// which resends it on subsequent requests.
	AlgorithmEncodedEndpointHeader = "encoded_endpoint_header"

	// AlgorithmSessionIDHeader maps an opaque client-supplied session identifier
	// to a pod.
	AlgorithmSessionIDHeader = "session_id_header"
)

// parameters configures the SessionAffinity scorer.
type parameters struct {
	// Algorithm is AlgorithmEncodedEndpointHeader (the default) or
	// AlgorithmSessionIDHeader.
	Algorithm string `json:"algorithm"`
	// HeaderName overrides the default x-session-token header used to read and
	// write the session token. When empty the default is used.
	HeaderName string `json:"headerName"`
	// ProfileName is the name of the profile this instance is associated with (optional).
	// When empty, the plugin defaults to the primary (decode) pod.
	// Selects which pod of the SchedulingResult this instance pins.
	ProfileName string `json:"profileName"`
	// EvictionTTLSeconds is how long a session binding survives unused.
	// session_id_header only.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds"`
	// EvictionSweepSeconds is how often expired bindings are swept.
	// session_id_header only.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds"`
}

// defaultParameters returns the parameters used when a field is not set.
func defaultParameters() parameters {
	return parameters{
		Algorithm:            AlgorithmEncodedEndpointHeader,
		EvictionTTLSeconds:   600,
		EvictionSweepSeconds: 30,
	}
}

// compile-time type assertion
var _ scheduling.Scorer = &SessionAffinity{}
var _ requestcontrol.ResponseHeaderProcessor = &SessionAffinity{}
var _ requestcontrol.PreRequest = &SessionAffinity{}

// Factory defines the factory function for SessionAffinity scorer.
func Factory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := defaultParameters()
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", SessionAffinityType, err)
		}
	}
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' scorer - %w", SessionAffinityType, err)
	}

	s := NewSessionAffinity(name, params.HeaderName, params.ProfileName)
	if params.Algorithm == AlgorithmSessionIDHeader {
		s.algorithm = AlgorithmSessionIDHeader
		s.ttl = time.Duration(params.EvictionTTLSeconds * float64(time.Second))
		s.sweepInterval = time.Duration(params.EvictionSweepSeconds * float64(time.Second))
		go s.runEviction(handle.Context(), s.sweepInterval)
	}
	return s, nil
}

// validate rejects only what would change behavior. Fields belonging to the other
// algorithm are ignored, not rejected: they change nothing.
func (p *parameters) validate() error {
	switch p.Algorithm {
	case AlgorithmEncodedEndpointHeader, AlgorithmSessionIDHeader:
	default:
		return fmt.Errorf("algorithm must be %q or %q, got %q", AlgorithmEncodedEndpointHeader, AlgorithmSessionIDHeader, p.Algorithm)
	}
	if p.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", p.EvictionTTLSeconds)
	}
	if p.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", p.EvictionSweepSeconds)
	}
	return nil
}

// NewSessionAffinity returns a scorer. When sessionHeader is empty the default
// x-session-token header is used.
func NewSessionAffinity(name, sessionHeader, profileName string) *SessionAffinity {
	return &SessionAffinity{
		typedName:     plugin.TypedName{Type: SessionAffinityType, Name: name},
		sessionHeader: sessionutil.NormalizeHeader(sessionHeader),
		profileName:   profileName,
	}
}

// SessionAffinity is a routing scorer that routes subsequent
// requests in a session to the same pod as the first request in the
// session was sent to, by giving that pod the specified weight and assigning
// zero score to the rest of the targets
type SessionAffinity struct {
	typedName plugin.TypedName
	// sessionHeader is the request/response header carrying the session token.
	sessionHeader string
	// profileName is the name of the profile this instance is associated with.
	profileName string
	// algorithm selects how a session is pinned to a pod.
	algorithm string
	// ttl is how long a binding survives unused; zero disables expiry.
	// session_id_header only.
	ttl time.Duration
	// sweepInterval is how often expired bindings are swept.
	// session_id_header only.
	sweepInterval time.Duration
	// bindings maps a session identifier to the pod serving it.
	// session_id_header only.
	bindings sync.Map
}

// TypedName returns the typed name of the plugin.
func (s *SessionAffinity) TypedName() plugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *SessionAffinity) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// Score assign a high score to the pod used in previous requests and zero to others
func (s *SessionAffinity) Score(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64)

	var podName string
	if s.algorithm == AlgorithmSessionIDHeader {
		podName = s.boundPod(sessionutil.SessionID(request, s.sessionHeader))
	} else {
		podName = sessionutil.DecodePodName(ctx, request.Headers[s.sessionHeader])
	}

	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] = 0.0 // initial value
		if endpoint.GetMetadata().ID.String() == podName {
			scoredEndpoints[endpoint] = 1.0
		}
	}

	return scoredEndpoints
}

// ResponseHeader sets the session header on the response sent to the client.
// Under session_id_header the client owns its identifier, so nothing is written.
func (s *SessionAffinity) ResponseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	if s.algorithm == AlgorithmSessionIDHeader {
		return
	}
	podToWrite := sessionutil.ResolvePodToWrite(request, s.profileName, targetPod)
	sessionutil.WriteResponseHeader(ctx, SessionAffinityType, s.sessionHeader, response, podToWrite)
}

// binding is the pod a session is pinned to, and when that pin was last used.
type binding struct {
	podName  string
	lastSeen time.Time
}

// boundPod returns the pod bound to sessionID, or "" when unbound. Expiry is
// enforced by the sweeper, not on read.
func (s *SessionAffinity) boundPod(sessionID string) string {
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

// PreRequest binds this request's session to the pod that served it, so subsequent
// requests in the session prefer it. It runs after the pick because only then is
// that pod known.
func (s *SessionAffinity) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	if s.algorithm != AlgorithmSessionIDHeader {
		return
	}
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
		"plugin", s.typedName.String(), "pod", podName)
}

// pickedPodName returns the pod chosen for this instance's profile, or "" when
// that profile was not scheduled. An empty profileName selects the primary
// (decode) profile, matching ResolvePodToWrite.
func (s *SessionAffinity) pickedPodName(schedulingResult *scheduling.SchedulingResult) string {
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

// expired reports whether a binding last used at lastSeen has outlived the TTL.
func (s *SessionAffinity) expired(lastSeen, now time.Time) bool {
	return s.ttl > 0 && now.Sub(lastSeen) > s.ttl
}

// runEviction drops bindings unused for longer than the TTL until ctx is cancelled.
func (s *SessionAffinity) runEviction(ctx context.Context, interval time.Duration) {
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
