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

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// TokenEstimator estimates the number of tokens for an LLM request.
type TokenEstimator interface {
	// Estimate returns the total estimated token count (input + output) for the request.
	Estimate(request *fwksched.InferenceRequest) int64
	// EstimateInput returns only the estimated input token count for the request.
	EstimateInput(request *fwksched.InferenceRequest) int64
	// EstimateOutput returns the estimated output token count given the input token
	// count, bounded by the client-requested cap (maxOutputTokens, nil if unset)
	// and the estimator's configured operator cap.
	EstimateOutput(inputTokens int64, maxOutputTokens *int64) int64
}

// DefaultOutputRatio is the estimated output-to-input token ratio used when no
// ratio is configured.
const DefaultOutputRatio = 1.5

// SimpleTokenEstimator derives input tokens from the tokenized prompt and
// estimates output tokens as inputTokens * OutputRatio, bounded by the
// client-requested cap and an optional operator cap.
type SimpleTokenEstimator struct {
	OutputRatio float64
	// MaxEstimatedOutputTokens optionally caps the estimated output tokens
	// regardless of input length or the client-requested cap. nil means no cap.
	MaxEstimatedOutputTokens *int64
}

// NewSimpleTokenEstimator returns a SimpleTokenEstimator with the default output
// ratio and no operator cap.
func NewSimpleTokenEstimator() TokenEstimator {
	return NewSimpleTokenEstimatorWithRatio(DefaultOutputRatio)
}

// NewSimpleTokenEstimatorWithRatio returns a SimpleTokenEstimator that estimates
// output tokens as round(inputTokens * ratio), with no operator cap.
func NewSimpleTokenEstimatorWithRatio(ratio float64) TokenEstimator {
	return NewSimpleTokenEstimatorWithConfig(ratio, nil)
}

// NewSimpleTokenEstimatorWithConfig returns a SimpleTokenEstimator with the given
// output ratio and an optional operator cap (maxOutput, nil for no cap) on the
// estimated output tokens.
func NewSimpleTokenEstimatorWithConfig(ratio float64, maxOutput *int64) TokenEstimator {
	return &SimpleTokenEstimator{
		OutputRatio:              ratio,
		MaxEstimatedOutputTokens: maxOutput,
	}
}

// Estimate returns the total estimated token count (input + output) for the request.
// Output tokens are estimated as inputTokens * OutputRatio.
func (e *SimpleTokenEstimator) Estimate(request *fwksched.InferenceRequest) int64 {
	inputTokens := e.EstimateInput(request)
	if inputTokens == 0 {
		return 0
	}
	var maxOutputTokens *int64
	if request != nil && request.Body != nil {
		maxOutputTokens = request.Body.MaxOutputTokens
	}
	return inputTokens + e.EstimateOutput(inputTokens, maxOutputTokens)
}

// EstimateInput returns the input token count read from the tokenized prompt,
// or 0 when no tokenization is available.
func (e *SimpleTokenEstimator) EstimateInput(request *fwksched.InferenceRequest) int64 {
	if request == nil || request.Body == nil || request.Body.TokenizedPrompt == nil {
		return 0
	}
	return int64(request.Body.TokenizedPrompt.TokenCount())
}

// EstimateOutput returns the estimated output token count given the input token
// count. The raw estimate (round(inputTokens * OutputRatio)) is bounded by the
// client-requested cap (maxOutputTokens, nil if unset) and the configured
// operator cap (MaxEstimatedOutputTokens), each applied only when non-negative.
func (e *SimpleTokenEstimator) EstimateOutput(inputTokens int64, maxOutputTokens *int64) int64 {
	if inputTokens <= 0 {
		return 0
	}
	est := int64(math.Round(float64(inputTokens) * e.OutputRatio))
	if maxOutputTokens != nil && *maxOutputTokens >= 0 && *maxOutputTokens < est {
		est = *maxOutputTokens
	}
	if e.MaxEstimatedOutputTokens != nil && *e.MaxEstimatedOutputTokens >= 0 && *e.MaxEstimatedOutputTokens < est {
		est = *e.MaxEstimatedOutputTokens
	}
	return est
}
