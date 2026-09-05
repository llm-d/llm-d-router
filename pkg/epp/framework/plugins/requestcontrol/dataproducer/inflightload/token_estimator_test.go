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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/outlenbucket"
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

// requestWithBucket builds a request whose outlen-bucket attribute is set (as the
// outlen-bucket plugin would), plus an optional client output cap.
func requestWithBucket(bucket outlenbucket.Bucket, maxOut *int64) *fwksched.InferenceRequest {
	req := &fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{MaxOutputTokens: maxOut},
	}
	req.PutAttribute(outlenbucket.AttributeKey, bucket)
	return req
}

func TestSimpleTokenEstimator_EstimateInput(t *testing.T) {
	estimator := NewSimpleTokenEstimator(nil)

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

// TestEstimateOutputFromRequest_Buckets verifies bucket-to-estimate mapping and client-cap clamping,
// including that max_output_tokens=0 is treated as "not set" (no cap), not a clamp to 0.
func TestEstimateOutputFromRequest_Buckets(t *testing.T) {
	e := NewSimpleTokenEstimator(nil)

	t.Run("nil request -> 0", func(t *testing.T) {
		require.Equal(t, int64(0), e.EstimateOutputFromRequest(nil))
	})

	t.Run("LONG bucket -> flat 4096", func(t *testing.T) {
		req := requestWithBucket(outlenbucket.Long, nil)
		require.Equal(t, int64(4096), e.EstimateOutputFromRequest(req))
	})

	t.Run("LONG capped by max_output_tokens", func(t *testing.T) {
		req := requestWithBucket(outlenbucket.Long, ptr.To(int64(2000)))
		require.Equal(t, int64(2000), e.EstimateOutputFromRequest(req))
	})

	t.Run("LONG, max_output_tokens=0 -> no cap (0 means unset in API requests)", func(t *testing.T) {
		req := requestWithBucket(outlenbucket.Long, ptr.To(int64(0)))
		require.Equal(t, int64(4096), e.EstimateOutputFromRequest(req))
	})

	t.Run("SHORT bucket -> 100", func(t *testing.T) {
		req := requestWithBucket(outlenbucket.Short, nil)
		require.Equal(t, int64(100), e.EstimateOutputFromRequest(req))
	})

	t.Run("SHORT capped below 100 by max_output_tokens", func(t *testing.T) {
		req := requestWithBucket(outlenbucket.Short, ptr.To(int64(50)))
		require.Equal(t, int64(50), e.EstimateOutputFromRequest(req))
	})

	t.Run("UNKNOWN bucket -> flat 1000", func(t *testing.T) {
		req := requestWithBucket(outlenbucket.Unknown, nil)
		require.Equal(t, int64(1000), e.EstimateOutputFromRequest(req))
	})

	t.Run("UNKNOWN capped by max_output_tokens", func(t *testing.T) {
		req := requestWithBucket(outlenbucket.Unknown, ptr.To(int64(400)))
		require.Equal(t, int64(400), e.EstimateOutputFromRequest(req))
	})

	// A missing attribute reads as the zero value (UNKNOWN): the estimate is the
	// flat UNKNOWN value and does NOT scale with input length -- input tokens carry
	// no output-length signal, which is the whole premise of the outlen-bucket plugin.
	t.Run("missing attribute -> UNKNOWN 1000 (input length ignored)", func(t *testing.T) {
		req := tokenizedRequest(100) // no outlen-bucket attribute
		require.Equal(t, int64(1000), e.EstimateOutputFromRequest(req))
	})

	t.Run("missing attribute, untokenized -> UNKNOWN 1000", func(t *testing.T) {
		req := &fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}}
		require.Equal(t, int64(1000), e.EstimateOutputFromRequest(req))
	})
}

func TestEstimateOutputFromRequest_OperatorCap(t *testing.T) {
	t.Run("LONG capped by operator cap", func(t *testing.T) {
		e := NewSimpleTokenEstimator(ptr.To(int64(200)))
		req := requestWithBucket(outlenbucket.Long, nil)
		require.Equal(t, int64(200), e.EstimateOutputFromRequest(req))
	})

	t.Run("operator cap 0 clamps to 0", func(t *testing.T) {
		e := NewSimpleTokenEstimator(ptr.To(int64(0)))
		req := requestWithBucket(outlenbucket.Long, nil)
		require.Equal(t, int64(0), e.EstimateOutputFromRequest(req))
	})

	t.Run("client cap tighter than operator cap wins", func(t *testing.T) {
		e := NewSimpleTokenEstimator(ptr.To(int64(300)))
		req := requestWithBucket(outlenbucket.Long, ptr.To(int64(150)))
		require.Equal(t, int64(150), e.EstimateOutputFromRequest(req))
	})
}
