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

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// encodedEndpointHeaderStrategy decodes the picked pod from the client-supplied
// session token and pins to it when present. Stateless.
type encodedEndpointHeaderStrategy struct {
	sessionHeader string
	profileName   string
}

func newEncodedEndpointHeaderStrategy(params parameters) strategy {
	return &encodedEndpointHeaderStrategy{
		sessionHeader: sessionutil.NormalizeHeader(params.EncodedEndpointHeaderConfig.Header),
		profileName:   params.ProfileName,
	}
}

// filter returns the session's pod alone when it is among the candidates,
// otherwise all candidates so downstream filters and scorers can decide.
func (e *encodedEndpointHeaderStrategy) filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	podName := sessionutil.DecodePodName(ctx, request.Headers[e.sessionHeader])
	if podName == "" {
		return endpoints
	}

	for _, endpoint := range endpoints {
		if endpoint.GetMetadata().ID.String() == podName {
			return []scheduling.Endpoint{endpoint}
		}
	}

	return endpoints
}

func (e *encodedEndpointHeaderStrategy) preRequest(context.Context, *scheduling.InferenceRequest, *scheduling.SchedulingResult) {
}

func (e *encodedEndpointHeaderStrategy) responseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	podToWrite := sessionutil.ResolvePodToWrite(request, e.profileName, targetPod)
	sessionutil.WriteResponseHeader(ctx, SessionAffinityType, e.sessionHeader, response, podToWrite)
}
