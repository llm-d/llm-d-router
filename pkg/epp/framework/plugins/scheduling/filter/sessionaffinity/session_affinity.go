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
	// SessionAffinityType is the type of the SessionAffinity filter.
	SessionAffinityType = "session-affinity-filter"
)

// parameters configures the SessionAffinity filter.
type parameters struct {
	// HeaderName overrides the default x-session-token header used to read and
	// write the session token. When empty the default is used.
	HeaderName string `json:"headerName"`
}

// compile-time type assertion
var _ scheduling.Filter = &SessionAffinity{}
var _ requestcontrol.ResponseHeaderProcessor = &SessionAffinity{}

// Factory defines the factory function for the SessionAffinity filter.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", SessionAffinityType, err)
		}
	}

	return NewSessionAffinity(params.HeaderName).WithName(name), nil
}

// NewSessionAffinity returns a filter. When sessionHeader is empty the default
// x-session-token header is used.
func NewSessionAffinity(sessionHeader string) *SessionAffinity {
	return &SessionAffinity{
		typedName:     plugin.TypedName{Type: SessionAffinityType},
		sessionHeader: sessiontoken.NormalizeHeader(sessionHeader),
	}
}

// SessionAffinity is a routing filter that pins subsequent requests in a
// session to the same pod the first request in the session was sent to. When
// the session pod is among the candidates it is returned as the sole endpoint;
// otherwise all candidates are returned so downstream filters and scorers can
// decide.
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

// Filter returns the endpoint running the session when it is among the
// candidates, otherwise all candidate endpoints.
func (s *SessionAffinity) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	podName := sessiontoken.DecodePodName(ctx, request.Headers[s.sessionHeader])
	if podName == "" {
		return endpoints
	}

	for _, endpoint := range endpoints {
		if endpoint.GetMetadata().NamespacedName.String() == podName {
			return []scheduling.Endpoint{endpoint}
		}
	}

	return endpoints
}

// ResponseHeader sets the session header on the response sent to the client.
func (s *SessionAffinity) ResponseHeader(ctx context.Context, _ *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	sessiontoken.WriteResponseHeader(ctx, SessionAffinityType, s.sessionHeader, response, targetPod)
}
