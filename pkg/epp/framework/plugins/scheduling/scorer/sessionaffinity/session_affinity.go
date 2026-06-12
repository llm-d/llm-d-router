package sessionaffinity

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessiontoken "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/sessionaffinity"
)

const (
	// SessionAffinityType is the type of the SessionAffinity scorer.
	SessionAffinityType = "session-affinity-scorer"
)

// parameters configures the SessionAffinity scorer.
type parameters struct {
	// HeaderName overrides the default x-session-token header used to read and
	// write the session token. When empty the default is used.
	HeaderName string `json:"headerName"`
}

// compile-time type assertion
var _ scheduling.Scorer = &SessionAffinity{}
var _ requestcontrol.ResponseHeaderProcessor = &SessionAffinity{}

// Factory defines the factory function for SessionAffinity scorer.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", SessionAffinityType, err)
		}
	}

	return NewSessionAffinity(params.HeaderName).WithName(name), nil
}

// NewSessionAffinity returns a scorer. When sessionHeader is empty the default
// x-session-token header is used.
func NewSessionAffinity(sessionHeader string) *SessionAffinity {
	return &SessionAffinity{
		typedName:     plugin.TypedName{Type: SessionAffinityType},
		sessionHeader: sessiontoken.NormalizeHeader(sessionHeader),
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
}

// TypedName returns the typed name of the plugin.
func (s *SessionAffinity) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *SessionAffinity) WithName(name string) *SessionAffinity {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *SessionAffinity) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// Score assign a high score to the pod used in previous requests and zero to others
func (s *SessionAffinity) Score(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64)
	podName := sessiontoken.DecodePodName(ctx, request.Headers[s.sessionHeader])

	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] = 0.0 // initial value
		if endpoint.GetMetadata().NamespacedName.String() == podName {
			scoredEndpoints[endpoint] = 1.0
		}
	}

	return scoredEndpoints
}

// ResponseHeader sets the session header on the response sent to the client.
func (s *SessionAffinity) ResponseHeader(ctx context.Context, _ *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	sessiontoken.WriteResponseHeader(ctx, SessionAffinityType, s.sessionHeader, response, targetPod)
}
