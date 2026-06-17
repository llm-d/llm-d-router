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
	// SessionAffinityType is the type of the SessionAffinity filter.
	SessionAffinityType = "session-affinity-filter"
)

// parameters configures the SessionAffinity filter.
type parameters struct {
	// HeaderName overrides the default x-session-token header used to read and
	// write the session token. When empty the default is used.
	HeaderName string `json:"headerName"`
	// ProfileName is the name of the profile this instance is associated with (optional).
	// When empty, the plugin defaults to the primary (decode) pod.
	// Used in ResponseHeader to look up the correct target pod from SchedulingResult.
	ProfileName string `json:"profileName"`
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

	return NewSessionAffinity(name, params.HeaderName, params.ProfileName), nil
}

// NewSessionAffinity returns a filter. When sessionHeader is empty the default
// x-session-token header is used.
func NewSessionAffinity(name, sessionHeader, profileName string) *SessionAffinity {
	return &SessionAffinity{
		typedName:     plugin.TypedName{Type: SessionAffinityType, Name: name},
		sessionHeader: sessionutil.NormalizeHeader(sessionHeader),
		profileName:   profileName,
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
	// profileName is the name of the profile this instance is associated with.
	profileName string
}

// TypedName returns the typed name of the plugin.
func (s *SessionAffinity) TypedName() plugin.TypedName {
	return s.typedName
}

// Filter returns the endpoint running the session when it is among the
// candidates, otherwise all candidate endpoints.
func (s *SessionAffinity) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	podName := sessionutil.DecodePodName(ctx, request.Headers[s.sessionHeader])
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
func (s *SessionAffinity) ResponseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	podToWrite := sessionutil.ResolvePodToWrite(request, s.profileName, targetPod)
	sessionutil.WriteResponseHeader(ctx, SessionAffinityType, s.sessionHeader, response, podToWrite)
}
