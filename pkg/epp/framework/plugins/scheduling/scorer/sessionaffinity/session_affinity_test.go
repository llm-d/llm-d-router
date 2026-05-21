package sessionaffinity_test

import (
	"context"
	"encoding/base64"
	"testing"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionaffinity "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/sessionaffinity"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestSessionAffinity_Score(t *testing.T) {
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

	sessionAffinityScorer := sessionaffinity.NewSessionAffinity()

	tests := []struct {
		name       string
		req        *scheduling.InferenceRequest
		input      []scheduling.Endpoint
		wantScores map[scheduling.Endpoint]float64
	}{
		{
			name: "selects correct endpoint : endpointB",
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-session-token": validSessionTokenForEndpointB},
			},
			input: inputEndpoints,
			wantScores: map[scheduling.Endpoint]float64{
				endpointA: 0.0,
				endpointB: 1.0,
			},
		},
		{
			name: "no session token",
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{},
			},
			// both endpoints get score 0.0
			input: inputEndpoints,
			wantScores: map[scheduling.Endpoint]float64{
				endpointA: 0.0,
				endpointB: 0.0,
			},
		},
		{
			name: "invalid session token",
			req: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-session-token": "garbage-token"},
			},
			// expect same behavior as no session token
			input: inputEndpoints,
			wantScores: map[scheduling.Endpoint]float64{
				endpointA: 0.0,
				endpointB: 0.0,
			},
		},
		{
			name:  "no endpoints available",
			req:   &scheduling.InferenceRequest{},
			input: []scheduling.Endpoint{},
			// returns empty score map
			wantScores: map[scheduling.Endpoint]float64{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotScores := sessionAffinityScorer.Score(context.Background(), nil, test.req, test.input)

			if diff := cmp.Diff(test.wantScores, gotScores); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestSessionAffinity_ResponseBody(t *testing.T) {

	targetEndpoint := &fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
		Address:        "1.2.3.4",
	}

	// expected token to be set in response header
	wantToken := base64.StdEncoding.EncodeToString([]byte(targetEndpoint.NamespacedName.String()))

	tests := []struct {
		name            string
		initialResponse *requestcontrol.Response
		targetPod       *fwkdl.EndpointMetadata
		wantHeaders     map[string]string
	}{
		{
			name:            "standard case with existing headers map",
			initialResponse: &requestcontrol.Response{RequestID: "req-1", Headers: make(map[string]string), EndOfStream: true},
			targetPod:       targetEndpoint,
			wantHeaders:     map[string]string{"x-session-token": wantToken},
		},
		{
			name:            "response with nil headers map",
			initialResponse: &requestcontrol.Response{RequestID: "req-2", Headers: nil, EndOfStream: true},
			targetPod:       targetEndpoint,
			wantHeaders:     map[string]string{"x-session-token": wantToken},
		},
		{
			name:            "nil targetPod should do nothing",
			initialResponse: &requestcontrol.Response{RequestID: "req-3", Headers: make(map[string]string), EndOfStream: true},
			targetPod:       nil,
			wantHeaders:     map[string]string{},
		},
		{
			name:            "incomplete response should do nothing (EndOfStream=false)",
			initialResponse: &requestcontrol.Response{RequestID: "req-4", Headers: make(map[string]string), EndOfStream: false},
			targetPod:       targetEndpoint,
			wantHeaders:     map[string]string{},
		},
	}

	s := sessionaffinity.NewSessionAffinity()
	ctx := utils.NewTestContext(t)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s.ResponseBody(ctx, nil, test.initialResponse, test.targetPod)

			if diff := cmp.Diff(test.wantHeaders, test.initialResponse.Headers); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
