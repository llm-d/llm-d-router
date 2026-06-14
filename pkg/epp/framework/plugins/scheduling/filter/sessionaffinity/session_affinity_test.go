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

package sessionaffinity_test

import (
	"context"
	"encoding/base64"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionaffinity "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestSessionAffinity_Filter(t *testing.T) {
	endpointA := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{},
		nil,
	)
	endpointB := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-b"}},
		&fwkdl.Metrics{},
		nil,
	)

	inputEndpoints := []scheduling.Endpoint{endpointA, endpointB}

	// valid session token for endpointB
	validSessionTokenForEndpointB := base64.StdEncoding.EncodeToString([]byte(endpointB.GetMetadata().NamespacedName.String()))
	// valid token whose pod is not among the candidates
	tokenForMissingPod := base64.StdEncoding.EncodeToString([]byte("pod-missing"))

	sessionAffinityFilter := sessionaffinity.NewSessionAffinity("test-filter", "")
	customHeaderFilter := sessionaffinity.NewSessionAffinity("test-filter", "x-custom-session")

	tests := []struct {
		name     string
		filter   *sessionaffinity.SessionAffinity
		req      *scheduling.InferenceRequest
		input    []scheduling.Endpoint
		wantPods []string
	}{
		{
			name:   "selects the session endpoint : endpointB",
			filter: sessionAffinityFilter,
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-session-token": validSessionTokenForEndpointB},
			},
			input:    inputEndpoints,
			wantPods: []string{"pod-b"},
		},
		{
			name:   "custom header selects the session endpoint : endpointB",
			filter: customHeaderFilter,
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-custom-session": validSessionTokenForEndpointB},
			},
			input:    inputEndpoints,
			wantPods: []string{"pod-b"},
		},
		{
			name:   "custom header ignores default header",
			filter: customHeaderFilter,
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-session-token": validSessionTokenForEndpointB},
			},
			input:    inputEndpoints,
			wantPods: []string{"pod-a", "pod-b"},
		},
		{
			name:   "no session token returns all endpoints",
			filter: sessionAffinityFilter,
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{},
			},
			input:    inputEndpoints,
			wantPods: []string{"pod-a", "pod-b"},
		},
		{
			name:   "invalid session token returns all endpoints",
			filter: sessionAffinityFilter,
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-session-token": "garbage-token"},
			},
			input:    inputEndpoints,
			wantPods: []string{"pod-a", "pod-b"},
		},
		{
			name:   "session pod not among candidates returns all endpoints",
			filter: sessionAffinityFilter,
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-session-token": tokenForMissingPod},
			},
			input:    inputEndpoints,
			wantPods: []string{"pod-a", "pod-b"},
		},
		{
			name:     "no endpoints available",
			filter:   sessionAffinityFilter,
			req:      &scheduling.InferenceRequest{Headers: map[string]string{"x-session-token": validSessionTokenForEndpointB}},
			input:    []scheduling.Endpoint{},
			wantPods: []string{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := test.filter.Filter(context.Background(), test.req, test.input)

			gotPods := make([]string, len(got))
			for idx, endpoint := range got {
				gotPods[idx] = endpoint.GetMetadata().NamespacedName.Name
			}

			assert.ElementsMatch(t, test.wantPods, gotPods, "filtered endpoints should match expected endpoints")
			assert.Len(t, got, len(test.wantPods), "filtered endpoints count should match expected count")
		})
	}
}

func TestSessionAffinity_ResponseHeader(t *testing.T) {
	targetEndpoint := &fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
		Address:        "1.2.3.4",
	}

	// expected token to be set in response header
	wantToken := base64.StdEncoding.EncodeToString([]byte(targetEndpoint.NamespacedName.String()))

	tests := []struct {
		name            string
		sessionHeader   string
		initialResponse *requestcontrol.Response
		targetPod       *fwkdl.EndpointMetadata
		wantHeaders     map[string]string
	}{
		{
			name:            "standard case with existing headers map",
			initialResponse: &requestcontrol.Response{RequestID: "req-1", Headers: make(map[string]string)},
			targetPod:       targetEndpoint,
			wantHeaders:     map[string]string{"x-session-token": wantToken},
		},
		{
			name:            "response with nil headers map",
			initialResponse: &requestcontrol.Response{RequestID: "req-2", Headers: nil},
			targetPod:       targetEndpoint,
			wantHeaders:     map[string]string{"x-session-token": wantToken},
		},
		{
			name:            "custom header carries the token",
			sessionHeader:   "x-custom-session",
			initialResponse: &requestcontrol.Response{RequestID: "req-custom", Headers: make(map[string]string)},
			targetPod:       targetEndpoint,
			wantHeaders:     map[string]string{"x-custom-session": wantToken},
		},
		{
			name:            "nil targetPod should do nothing",
			initialResponse: &requestcontrol.Response{RequestID: "req-3", Headers: make(map[string]string)},
			targetPod:       nil,
			wantHeaders:     map[string]string{},
		},
		{
			name:            "nil response should do nothing",
			initialResponse: nil,
			targetPod:       targetEndpoint,
		},
	}

	ctx := utils.NewTestContext(t)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := sessionaffinity.NewSessionAffinity("test-filter", test.sessionHeader)
			s.ResponseHeader(ctx, nil, test.initialResponse, test.targetPod)

			if test.initialResponse == nil {
				return
			}

			if diff := cmp.Diff(test.wantHeaders, test.initialResponse.Headers); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
