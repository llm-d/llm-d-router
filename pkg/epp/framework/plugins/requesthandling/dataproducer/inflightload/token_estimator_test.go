/*
Copyright 2026 The Kubernetes Authors.
Copyright 2026 The llm-d Authors.

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
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/outlenbucket"
)

// ttsRequest builds a request whose body carries a text-to-speech payload.
func ttsRequest(input, instructions, refText string, maxOut *int64) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Body: &fwkrequest.InferenceRequestBody{
			TextToSpeech: &fwkrequest.TextToSpeechRequest{
				Input:        input,
				Instructions: instructions,
				RefText:      refText,
			},
			MaxOutputTokens: maxOut,
		},
	}
}

// imagesRequest builds a request whose body carries an image generation payload.
func imagesRequest(size string, steps, n *int64) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Body: &fwkrequest.InferenceRequestBody{
			Images: &fwkrequest.ImagesGenerationsRequest{
				Size:              size,
				NumInferenceSteps: steps,
				N:                 n,
			},
		},
	}
}

// tokenizedRequest builds a request whose body carries a tokenized prompt of n tokens.
func tokenizedRequest(n int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: make([]uint32, n)}},
			},
		},
	}
}

// requestWithBucket builds a request whose outlen-bucket attribute is set (as the
// outlen-bucket plugin would), plus an optional client output cap.
func requestWithBucket(bucket outlenbucket.Bucket, maxOut *int64) *fwksched.InferenceRequest {
	req := &fwksched.InferenceRequest{
		Body: &fwkrequest.InferenceRequestBody{MaxOutputTokens: maxOut},
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
			request:  &fwksched.InferenceRequest{Body: &fwkrequest.InferenceRequestBody{}},
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
		req := &fwksched.InferenceRequest{Body: &fwkrequest.InferenceRequestBody{}}
		require.Equal(t, int64(1000), e.EstimateOutputFromRequest(req))
	})
}

func TestEstimateOutputFromRequest_Images(t *testing.T) {
	e := NewSimpleTokenEstimator(nil)

	tests := []struct {
		name       string
		size       string
		steps      *int64
		n          *int64
		layers     *int64
		resolution *int64
		want       int64
	}{
		{name: "default size and steps", size: "", want: 5243},
		{name: "512 square twenty steps", size: "512x512", steps: ptr.To(int64(20)), want: 525},
		{name: "1024 square fifty steps", size: "1024x1024", steps: ptr.To(int64(50)), want: 5243},
		{name: "two images", size: "1024x1024", steps: ptr.To(int64(50)), n: ptr.To(int64(2)), want: 10486},
		{name: "malformed size falls back", size: "1024", steps: ptr.To(int64(50)), want: 5243},
		{name: "steps are capped", size: "1024x1024", steps: ptr.To(int64(1000)), want: 20972},
		{name: "images are capped", size: "1024x1024", steps: ptr.To(int64(50)), n: ptr.To(int64(100)), want: 52429},
		{name: "pixels are capped", size: "8192x8192", steps: ptr.To(int64(50)), want: 83887},
		{name: "oversized dimension does not overflow", size: "2000000000x2", steps: ptr.To(int64(50)), want: 83887},
		{name: "uppercase and whitespace size", size: " 1024X1024 ", steps: ptr.To(int64(50)), want: 5243},
		{name: "zero steps and images use defaults", size: "1024x1024", steps: ptr.To(int64(0)), n: ptr.To(int64(0)), want: 5243},
		{name: "four layers", size: "1024x1024", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), want: 26215},
		{name: "two layers", size: "1024x1024", steps: ptr.To(int64(50)), layers: ptr.To(int64(2)), want: 15729},
		{name: "layers use default resolution", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), want: 10240},
		{name: "resolution uses default layers", steps: ptr.To(int64(50)), resolution: ptr.To(int64(640)), want: 10240},
		{name: "resolution and layers", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), resolution: ptr.To(int64(640)), want: 10240},
		{name: "size wins over resolution", size: "1024x1024", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), resolution: ptr.To(int64(640)), want: 26215},
		{name: "layer count is capped", size: "1024x1024", steps: ptr.To(int64(50)), layers: ptr.To(int64(100)), want: 57672},
		{name: "zero layers use default", steps: ptr.To(int64(50)), layers: ptr.To(int64(0)), want: 10240},
		{name: "layered images multiply", size: "1024x1024", steps: ptr.To(int64(50)), n: ptr.To(int64(2)), layers: ptr.To(int64(4)), want: 52429},
		{name: "resolution is capped", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), resolution: ptr.To(int64(8192)), want: 419431},
		{name: "oversized resolution does not overflow", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), resolution: ptr.To(int64(2000000000)), want: 419431},
		{name: "zero resolution uses layered default", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), resolution: ptr.To(int64(0)), want: 10240},
		{name: "negative resolution uses layered default", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), resolution: ptr.To(int64(-1)), want: 10240},
		{name: "auto size uses layered resolution", size: "auto", steps: ptr.To(int64(50)), layers: ptr.To(int64(4)), resolution: ptr.To(int64(640)), want: 10240},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := imagesRequest(tt.size, tt.steps, tt.n)
			req.Body.Images.Layers = tt.layers
			req.Body.Images.Resolution = tt.resolution
			require.Equal(t, tt.want, e.EstimateOutputFromRequest(req))
		})
	}
}

func TestEstimateOutputFromRequest_ImagesSkipOperatorCap(t *testing.T) {
	e := NewSimpleTokenEstimator(ptr.To(int64(10)))
	require.Equal(t, int64(5243), e.EstimateOutputFromRequest(imagesRequest("", nil, nil)))
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

// TestEstimateOutputFromRequest_UnknownRunningAverage verifies that the UNKNOWN estimate
// follows the completion tokens observed for UNKNOWN requests, stays within [SHORT, LONG],
// and is unaffected by SHORT and LONG observations.
func TestEstimateOutputFromRequest_UnknownRunningAverage(t *testing.T) {
	unknown := requestWithBucket(outlenbucket.Unknown, nil)

	t.Run("averages UNKNOWN observations", func(t *testing.T) {
		e := NewSimpleTokenEstimator(nil)
		e.ObserveOutput(outlenbucket.Unknown, 300)
		e.ObserveOutput(outlenbucket.Unknown, 500)
		require.Equal(t, int64(400), e.EstimateOutputFromRequest(unknown))
	})

	t.Run("ignores SHORT, LONG and empty observations", func(t *testing.T) {
		e := NewSimpleTokenEstimator(nil)
		e.ObserveOutput(outlenbucket.Short, 10)
		e.ObserveOutput(outlenbucket.Long, 8000)
		e.ObserveOutput(outlenbucket.Unknown, 0)
		require.Equal(t, UnknownOutputTokens, e.EstimateOutputFromRequest(unknown))
	})

	t.Run("clamped to [SHORT, LONG]", func(t *testing.T) {
		low := NewSimpleTokenEstimator(nil)
		low.ObserveOutput(outlenbucket.Unknown, 1)
		require.Equal(t, ShortOutputTokens, low.EstimateOutputFromRequest(unknown))

		high := NewSimpleTokenEstimator(nil)
		high.ObserveOutput(outlenbucket.Unknown, 10*LongOutputTokens)
		require.Equal(t, LongOutputTokens, high.EstimateOutputFromRequest(unknown))
	})

	t.Run("bounded by the client cap", func(t *testing.T) {
		e := NewSimpleTokenEstimator(nil)
		e.ObserveOutput(outlenbucket.Unknown, 800)
		require.Equal(t, int64(400), e.EstimateOutputFromRequest(requestWithBucket(outlenbucket.Unknown, ptr.To(int64(400)))))
	})

	t.Run("follows a workload shift faster than the cumulative mean", func(t *testing.T) {
		e := NewSimpleTokenEstimator(nil)
		for range unknownOutputWindow {
			e.ObserveOutput(outlenbucket.Unknown, 200)
		}
		for range unknownOutputWindow {
			e.ObserveOutput(outlenbucket.Unknown, 2000)
		}
		require.Greater(t, e.EstimateOutputFromRequest(unknown), int64(1100))
	})
}

func TestEstimateOutputFromRequest_TTS(t *testing.T) {
	e := NewSimpleTokenEstimator(nil)

	t.Run("scales with spoken text", func(t *testing.T) {
		require.Equal(t, int64(5), e.EstimateOutputFromRequest(ttsRequest("Hello", "", "", nil)))
		require.Equal(t, int64(1000), e.EstimateOutputFromRequest(ttsRequest(strings.Repeat("a", 1000), "", "", nil)))
	})

	t.Run("instructions and reference transcript do not extend output", func(t *testing.T) {
		short := e.EstimateOutputFromRequest(ttsRequest("Hello", "", "", nil))
		padded := e.EstimateOutputFromRequest(ttsRequest("Hello", strings.Repeat("style ", 100), strings.Repeat("ref ", 100), nil))
		require.Equal(t, short, padded)
	})

	t.Run("max_new_tokens caps the estimate", func(t *testing.T) {
		require.Equal(t, int64(2), e.EstimateOutputFromRequest(ttsRequest(strings.Repeat("a", 1000), "", "", ptr.To(int64(2)))))
	})

	t.Run("max_new_tokens above the estimate does not raise it", func(t *testing.T) {
		require.Equal(t, int64(5), e.EstimateOutputFromRequest(ttsRequest("Hello", "", "", ptr.To(int64(100)))))
	})

	t.Run("empty input still carries one frame", func(t *testing.T) {
		require.Equal(t, int64(1), e.EstimateOutputFromRequest(ttsRequest("", "", "", nil)))
	})

	t.Run("clamped to the audio frame ceiling", func(t *testing.T) {
		require.Equal(t, int64(4096), e.EstimateOutputFromRequest(ttsRequest(strings.Repeat("a", 5000), "", "", nil)))
	})
}
