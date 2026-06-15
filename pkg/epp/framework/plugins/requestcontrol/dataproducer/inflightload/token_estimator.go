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
	// EstimateOutput returns the estimated output token count given the input token count.
	EstimateOutput(inputTokens int64) int64
}

// SimpleTokenEstimator derives input tokens from the tokenized prompt and
// estimates output tokens as inputTokens * OutputRatio.
type SimpleTokenEstimator struct {
	OutputRatio float64
}

// NewSimpleTokenEstimator returns a SimpleTokenEstimator with default output ratio.
func NewSimpleTokenEstimator() TokenEstimator {
	return &SimpleTokenEstimator{
		OutputRatio: 1.5,
	}
}

// Estimate returns the total estimated token count (input + output) for the request.
// Output tokens are estimated as inputTokens * OutputRatio.
func (e *SimpleTokenEstimator) Estimate(request *fwksched.InferenceRequest) int64 {
	inputTokens := e.EstimateInput(request)
	if inputTokens == 0 {
		return 0
	}
	return inputTokens + e.EstimateOutput(inputTokens)
}

// EstimateInput returns the input token count read from the tokenized prompt,
// or 0 when no tokenization is available.
func (e *SimpleTokenEstimator) EstimateInput(request *fwksched.InferenceRequest) int64 {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return 0
	}
	return int64(request.Body.TokenizedRequest.TokenCount())
}

// EstimateOutput returns the estimated output token count given the input token count.
func (e *SimpleTokenEstimator) EstimateOutput(inputTokens int64) int64 {
	if inputTokens <= 0 {
		return 0
	}
	return int64(math.Round(float64(inputTokens) * e.OutputRatio))
}
