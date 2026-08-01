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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// sessionIDHeaderStrategy filters the candidate set to the session's pinned pod.
// It embeds the shared implementation and adds only the filter output;
// PreRequest and ResponseHeader are promoted from the embed.
type sessionIDHeaderStrategy struct {
	*sessionutil.SessionIDHeader
}

func newSessionIDHeaderStrategy(params parameters, handle plugin.Handle) strategy {
	return &sessionIDHeaderStrategy{
		SessionIDHeader: sessionutil.NewSessionIDHeader(
			params.SessionIDConfig, params.ProfileName,
			plugin.StateKey(SessionAffinityType), handle,
		),
	}
}

// preRequest and responseHeader forward to the embedded shared implementation.
// They exist because the exported promoted methods cannot satisfy the
// package-private strategy interface's lowercase methods directly.
func (s *sessionIDHeaderStrategy) preRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	s.PreRequest(ctx, request, schedulingResult)
}

func (s *sessionIDHeaderStrategy) responseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	s.ResponseHeader(ctx, request, response, targetPod)
}

// filter returns the session's pinned pod alone when it is among the
// candidates, otherwise all candidates. When the request carries no session
// identifier the filter is a pass-through: all candidates are returned. The
// set is never empty, so scheduling never fails for lack of endpoints.
func (s *sessionIDHeaderStrategy) filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	target, ok := s.Choose(ctx, request, endpoints)
	if !ok || target == nil {
		return endpoints
	}
	return []scheduling.Endpoint{target}
}
