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

	// StrategyEncodedEndpointHeader echoes the picked pod back to the client,
	// which resends it on subsequent requests.
	StrategyEncodedEndpointHeader = "encoded_endpoint_header"

	// StrategySessionIDHeader maps an opaque client-supplied session identifier
	// to a pod.
	StrategySessionIDHeader = "session_id_header"
)

type parameters struct {
	// Strategy is StrategyEncodedEndpointHeader (the default) or
	// StrategySessionIDHeader.
	Strategy string `json:"strategy"`
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

func defaultParameters() parameters {
	return parameters{
		Strategy:             StrategyEncodedEndpointHeader,
		EvictionTTLSeconds:   300,
		EvictionSweepSeconds: 10,
	}
}

// validate rejects only what would change behavior. Fields belonging to the other
// strategy are ignored, not rejected: they change nothing.
func (p *parameters) validate() error {
	switch p.Strategy {
	case StrategyEncodedEndpointHeader, StrategySessionIDHeader:
	default:
		return fmt.Errorf("strategy must be %q or %q, got %q", StrategyEncodedEndpointHeader, StrategySessionIDHeader, p.Strategy)
	}
	if p.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", p.EvictionTTLSeconds)
	}
	if p.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", p.EvictionSweepSeconds)
	}
	return nil
}

var _ scheduling.Scorer = &SessionAffinity{}
var _ requestcontrol.ResponseHeaderProcessor = &SessionAffinity{}
var _ requestcontrol.PreRequest = &SessionAffinity{}

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

func newStrategy(params parameters, handle plugin.Handle) strategy {
	if params.Strategy == StrategySessionIDHeader {
		return newSessionIDHeaderStrategy(params, handle)
	}
	return newEncodedEndpointHeaderStrategy(params)
}

// SessionAffinity is a routing scorer that routes a session's requests to the
// same pod, per whichever algorithm strategy implements.
type SessionAffinity struct {
	typedName plugin.TypedName
	strategy  strategy
}

func (s *SessionAffinity) TypedName() plugin.TypedName {
	return s.typedName
}

func (s *SessionAffinity) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

func (s *SessionAffinity) Score(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	return s.strategy.score(ctx, request, endpoints)
}

func (s *SessionAffinity) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	s.strategy.preRequest(ctx, request, schedulingResult)
}

func (s *SessionAffinity) ResponseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	s.strategy.responseHeader(ctx, request, response, targetPod)
}
