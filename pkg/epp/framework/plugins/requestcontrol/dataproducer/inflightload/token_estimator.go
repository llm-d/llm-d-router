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
	"math"

	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

// TokenEstimator estimates the number of tokens for an LLM request.
type TokenEstimator interface {
	// Estimate returns the total estimated token count (input + output) for the request.
	Estimate(request *fwksched.InferenceRequest) int64
	// EstimateInput returns only the estimated input token count for the request.
	EstimateInput(request *fwksched.InferenceRequest) int64
	// EstimateOutput returns the estimated output token count given the input token count.
	EstimateOutput(inputTokens int64) int64
}

// SimpleTokenEstimator estimates tokens from character count. tokens = characters / CharactersPerToken.
type SimpleTokenEstimator struct {
	CharactersPerToken float64
	OutputRatio        float64
}

// NewSimpleTokenEstimator returns a SimpleTokenEstimator with default 4.0 chars per token.
func NewSimpleTokenEstimator() TokenEstimator {
	return &SimpleTokenEstimator{
		CharactersPerToken: 4.0,
		OutputRatio:        1.5,
	}
}

// Estimate returns the total estimated token count (input + output) for the request.
// When RequestSizeBytes is set, input tokens are derived from request size (~4 bytes per token)
// to avoid allocations. Otherwise, input tokens are estimated from prompt/message character count
// using CharactersPerToken; output tokens are estimated as inputTokens * OutputRatio.
func (e *SimpleTokenEstimator) Estimate(request *fwksched.InferenceRequest) int64 {
	inputTokens := e.EstimateInput(request)
	if inputTokens == 0 {
		return 0
	}
	return inputTokens + e.EstimateOutput(inputTokens)
}

// EstimateInput returns only the estimated input token count for the request.
func (e *SimpleTokenEstimator) EstimateInput(request *fwksched.InferenceRequest) int64 {
	if request == nil {
		return 0
	}
	// Prefer request body size when available: avoids PlainText() and reduces GC pressure.
	switch {
	case request.RequestSizeBytes > 0:
		return max(int64(request.RequestSizeBytes)/4, 1)
	case request.Body != nil:
		hint := request.Body.InputTokenCountHint()
		if hint >= 0 {
			return int64(hint)
		}
		// Fallback: character count from prompt text across all API types
		// (completions, chat/completions, responses, conversations).
		chars := len(request.Body.PromptText())
		return int64(math.Max(1, math.Round(float64(chars)/e.CharactersPerToken)))
	default:
		return 0
	}
}

// EstimateOutput returns the estimated output token count given the input token count.
func (e *SimpleTokenEstimator) EstimateOutput(inputTokens int64) int64 {
	if inputTokens <= 0 {
		return 0
	}
	return int64(math.Round(float64(inputTokens) * e.OutputRatio))
}
