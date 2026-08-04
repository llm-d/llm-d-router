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

	// StrategySessionID maps an opaque client-supplied session identifier,
	// sourced from a header or request attribute, to a pod.
	StrategySessionID = "session_id"
)

type parameters struct {
	// Strategy is StrategyEncodedEndpointHeader (the default) or
	// StrategySessionID. Only the config matching Strategy is used.
	Strategy                    string                                  `json:"strategy"`
	EncodedEndpointHeaderConfig sessionutil.EncodedEndpointHeaderConfig `json:"encodedEndpointHeaderConfig"`
	SessionIDConfig             sessionutil.SessionIDConfig             `json:"sessionIdConfig"`
	// ProfileName selects which pod of the SchedulingResult this instance pins.
	// When empty, the primary (decode) pod is used.
	ProfileName string `json:"profileName"`
}

func defaultParameters() parameters {
	return parameters{Strategy: StrategyEncodedEndpointHeader}
}

// applyDefaults fills the selected strategy's config with its defaults. Run
// after decode, before validate.
func (p *parameters) applyDefaults() {
	switch p.Strategy {
	case StrategyEncodedEndpointHeader:
		p.EncodedEndpointHeaderConfig.ApplyDefaults()
	case StrategySessionID:
		p.SessionIDConfig.ApplyDefaults()
	}
}

// validate dispatches to the selected strategy's config validator.
func (p *parameters) validate() error {
	switch p.Strategy {
	case StrategyEncodedEndpointHeader:
		return nil
	case StrategySessionID:
		return p.SessionIDConfig.Validate()
	default:
		return fmt.Errorf("strategy must be %q or %q, got %q", StrategyEncodedEndpointHeader, StrategySessionID, p.Strategy)
	}
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
	params.applyDefaults()
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
	if params.Strategy == StrategySessionID {
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
