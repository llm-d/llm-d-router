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
	// MissThreshold is the number of consecutive requests a session's bound pod
	// may be absent from the candidate set before the session migrates.
	// Defaults to 3 when unset (non-positive). session_id_header only.
	MissThreshold int `json:"missThreshold"`
}

// defaultParameters returns the parameters used when a field is not set.
func defaultParameters() parameters {
	return parameters{
		Algorithm:            AlgorithmEncodedEndpointHeader,
		EvictionTTLSeconds:   600,
		EvictionSweepSeconds: 30,
	}
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
	if p.MissThreshold < 0 {
		return fmt.Errorf("missThreshold must be >= 0, got %v", p.MissThreshold)
	}
	return nil
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

	return &SessionAffinity{
		typedName: plugin.TypedName{Type: SessionAffinityType, Name: name},
		strategy:  newStrategy(params, handle),
	}, nil
}

// NewSessionAffinity returns a scorer using the encoded_endpoint_header algorithm.
// When sessionHeader is empty the default x-session-token header is used.
func NewSessionAffinity(name, sessionHeader, profileName string) *SessionAffinity {
	return &SessionAffinity{
		typedName: plugin.TypedName{Type: SessionAffinityType, Name: name},
		strategy: &encodedEndpointHeaderStrategy{
			sessionHeader: sessionutil.NormalizeHeader(sessionHeader),
			profileName:   profileName,
		},
	}
}

// strategy is the algorithm-specific behavior for session affinity: how a
// session's preferred pod is scored, how a fresh pick is recorded, and what
// (if anything) is written back to the client.
type strategy interface {
	score(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64
	preRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult)
	responseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata)
}

// newStrategy builds the strategy selected by params.Algorithm.
func newStrategy(params parameters, handle plugin.Handle) strategy {
	if params.Algorithm == AlgorithmSessionIDHeader {
		return newSessionIDHeaderStrategy(params, handle)
	}
	return newEncodedEndpointHeaderStrategy(params)
}

// SessionAffinity is a routing scorer that routes subsequent
// requests in a session to the same pod as the first request in the
// session was sent to, by giving that pod the specified weight and assigning
// zero score to the rest of the targets
type SessionAffinity struct {
	typedName plugin.TypedName
	strategy  strategy
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
	return s.strategy.score(ctx, request, endpoints)
}

// PreRequest records the pod chosen for this request so a future request in
// the same session can prefer it.
func (s *SessionAffinity) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	s.strategy.preRequest(ctx, request, schedulingResult)
}

// ResponseHeader sets the session header on the response sent to the client.
func (s *SessionAffinity) ResponseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	s.strategy.responseHeader(ctx, request, response, targetPod)
}