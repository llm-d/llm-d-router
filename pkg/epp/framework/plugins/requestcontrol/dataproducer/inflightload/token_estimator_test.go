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
	estimator := &SimpleTokenEstimator{OutputRatio: 2.0}

	testCases := []struct {
		name        string
		inputTokens int64
		expected    int64
	}{
		{name: "Zero input", inputTokens: 0, expected: 0},
		{name: "Negative input", inputTokens: -5, expected: 0},
		{name: "Positive input", inputTokens: 8, expected: 16},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := estimator.EstimateOutput(tc.inputTokens)
			require.Equal(t, tc.expected, actual)
		})
	}
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
