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

// encodedEndpointHeaderStrategy echoes the picked pod back to the client,
// which resends it on subsequent requests. Stateless.
type encodedEndpointHeaderStrategy struct {
	sessionHeader string
	profileName   string
}

// newEncodedEndpointHeaderStrategy builds the encoded_endpoint_header strategy.
func newEncodedEndpointHeaderStrategy(params parameters) strategy {
	return &encodedEndpointHeaderStrategy{
		sessionHeader: sessionutil.NormalizeHeader(params.EncodedEndpointHeaderConfig.Header),
		profileName:   params.ProfileName,
	}
}

func (e *encodedEndpointHeaderStrategy) score(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64)
	podName := sessionutil.DecodePodName(ctx, request.Headers[e.sessionHeader])
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] = 0.0
		if endpoint.GetMetadata().ID.String() == podName {
			scoredEndpoints[endpoint] = 1.0
		}
	}
	return scoredEndpoints
}

func (e *encodedEndpointHeaderStrategy) preRequest(context.Context, *scheduling.InferenceRequest, *scheduling.SchedulingResult) {
}

func (e *encodedEndpointHeaderStrategy) responseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	podToWrite := sessionutil.ResolvePodToWrite(request, e.profileName, targetPod)
	sessionutil.WriteResponseHeader(ctx, SessionAffinityType, e.sessionHeader, response, podToWrite)
}
