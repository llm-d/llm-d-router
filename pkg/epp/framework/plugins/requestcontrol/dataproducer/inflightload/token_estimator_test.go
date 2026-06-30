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

package inflightload

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// tokenizedRequest builds a request whose body carries a tokenized prompt of n tokens.
func tokenizedRequest(n int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, n)}},
			},
		},
	}
}

func TestSimpleTokenEstimator_Estimate(t *testing.T) {
	estimator := NewSimpleTokenEstimator()

	testCases := []struct {
		name     string
		request  *fwksched.InferenceRequest
		expected int64
	}{
		{
			name:     "Nil request",
			request:  nil,
			expected: 0,
		},
		{
			name:     "Empty request",
			request:  &fwksched.InferenceRequest{},
			expected: 0,
		},
		{
			name: "Body nil",
			request: &fwksched.InferenceRequest{
				Body: nil,
			},
			expected: 0,
		},
		{
			name: "Body without tokenized prompt",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{},
			},
			expected: 0,
		},
		{
			name: "Empty tokenized prompt",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					TokenizedRequest: &fwkrh.TokenizedRequest{},
				},
			},
			expected: 0,
		},
		{
			name:     "Single token",
			request:  tokenizedRequest(1),
			expected: 3, // 1 input + round(1*1.5)=2 output
		},
		{
			name:     "Ten tokens",
			request:  tokenizedRequest(10),
			expected: 25, // 10 input + round(10*1.5)=15 output
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := estimator.Estimate(tc.request)
			require.Equal(t, tc.expected, actual)
		})
	}
}

func TestSimpleTokenEstimator_EstimateInput(t *testing.T) {
	estimator := NewSimpleTokenEstimator()

	testCases := []struct {
		name     string
		request  *fwksched.InferenceRequest
		expected int64
	}{
		{
			name:     "Nil request",
			request:  nil,
			expected: 0,
		},
		{
			name:     "Nil body",
			request:  &fwksched.InferenceRequest{Body: nil},
			expected: 0,
		},
		{
			name:     "Nil tokenized prompt",
			request:  &fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}},
			expected: 0,
		},
		{
			name:     "Empty token IDs",
			request:  tokenizedRequest(0),
			expected: 0,
		},
		{
			name:     "Several tokens",
			request:  tokenizedRequest(7),
			expected: 7,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := estimator.EstimateInput(tc.request)
			require.Equal(t, tc.expected, actual)
		})
	}
}

func TestSimpleTokenEstimator_EstimateOutput(t *testing.T) {
	testCases := []struct {
		name        string
		ratio       float64
		operatorCap *int64
		inputTokens int64
		clientCap   *int64
		expected    int64
	}{
		{name: "Zero input", ratio: 2.0, inputTokens: 0, expected: 0},
		{name: "Negative input", ratio: 2.0, inputTokens: -5, expected: 0},
		{name: "Positive input, no caps", ratio: 2.0, inputTokens: 8, expected: 16},
		{name: "Zero ratio", ratio: 0.0, inputTokens: 100, expected: 0},
		{name: "Client cap binds", ratio: 1.5, inputTokens: 100, clientCap: ptr.To(int64(50)), expected: 50},
		{name: "Client cap looser than estimate", ratio: 1.5, inputTokens: 100, clientCap: ptr.To(int64(500)), expected: 150},
		{name: "Client cap zero binds", ratio: 1.5, inputTokens: 100, clientCap: ptr.To(int64(0)), expected: 0},
		{name: "Negative client cap ignored", ratio: 1.5, inputTokens: 100, clientCap: ptr.To(int64(-10)), expected: 150},
		{name: "Operator cap binds", ratio: 1.5, operatorCap: ptr.To(int64(40)), inputTokens: 100, expected: 40},
		{name: "Both caps, client tighter", ratio: 1.5, operatorCap: ptr.To(int64(80)), inputTokens: 100, clientCap: ptr.To(int64(30)), expected: 30},
		{name: "Both caps, operator tighter", ratio: 1.5, operatorCap: ptr.To(int64(30)), inputTokens: 100, clientCap: ptr.To(int64(80)), expected: 30},
		{name: "Both caps, estimate tightest", ratio: 1.5, operatorCap: ptr.To(int64(500)), inputTokens: 100, clientCap: ptr.To(int64(500)), expected: 150},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			estimator := &SimpleTokenEstimator{OutputRatio: tc.ratio, MaxEstimatedOutputTokens: tc.operatorCap}
			require.Equal(t, tc.expected, estimator.EstimateOutput(tc.inputTokens, tc.clientCap))
		})
	}
}

func TestSimpleTokenEstimator_Estimate_HonorsClientCap(t *testing.T) {
	estimator := NewSimpleTokenEstimator() // ratio 1.5, no operator cap

	// 10 input tokens, client caps output at 3 -> 10 + min(15, 3) = 13.
	req := tokenizedRequest(10)
	req.Body.MaxOutputTokens = ptr.To(int64(3))
	require.Equal(t, int64(13), estimator.Estimate(req))

	// Without a client cap, the full ratio applies -> 10 + 15 = 25.
	require.Equal(t, int64(25), estimator.Estimate(tokenizedRequest(10)))
}

func TestSimpleTokenEstimator_Estimate_CustomConfig(t *testing.T) {
	estimator := &SimpleTokenEstimator{OutputRatio: 2.0}

	testCases := []struct {
		name     string
		request  *fwksched.InferenceRequest
		expected int64
	}{
		{
			name:     "Empty tokenized prompt with custom config",
			request:  tokenizedRequest(0),
			expected: 0,
		},
		{
			name:     "Four tokens with custom config",
			request:  tokenizedRequest(4),
			expected: 12, // 4 input + 4*2.0=8 output
		},
		{
			name:     "Ten tokens with custom config",
			request:  tokenizedRequest(10),
			expected: 30, // 10 input + 10*2.0=20 output
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := estimator.Estimate(tc.request)
			require.Equal(t, tc.expected, actual)
		})
	}
}
