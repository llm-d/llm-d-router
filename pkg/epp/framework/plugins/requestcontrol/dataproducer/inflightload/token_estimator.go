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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/oslbucket"
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
	// EstimateOutputFromRequest returns the estimated output token count from the
	// OSL bucket published as a request attribute by the osl-bucket plugin
	// (LONG/SHORT/UNKNOWN mapped to flat estimates), bounded by the
	// client-requested cap.
	EstimateOutputFromRequest(request *fwksched.InferenceRequest) int64
}

// DefaultOutputRatio is the estimated output-to-input token ratio used when no
// ratio is configured.
const DefaultOutputRatio = 1.5

const (
	// longOutputEstimateTokens is the flat output-token estimate for a LONG
	// (reasoning) request. It is deliberately a fixed value rather than the
	// client's thinking_budget: the estimate exists to rank requests by load,
	// where the LONG-vs-SHORT separation dominates, not to predict exact length.
	longOutputEstimateTokens int64 = 4096
	// unknownOutputEstimateTokens is the flat output-token estimate for an
	// UNKNOWN request (no OSL signal). It sits at the midpoint of the UNKNOWN
	// zone (500–1,999 tokens), preserving the ranking invariant
	// SHORT (100) < UNKNOWN (1000) < LONG (4096).
	// TODO(osl): replace with a dynamic estimate (e.g. per-pool running average
	// of observed CompletionTokens) in a follow-up PR.
	unknownOutputEstimateTokens int64 = 1000
	// shortOutputEstimateTokens is the flat output-token estimate for a SHORT
	// (tool-call) request.
	shortOutputEstimateTokens int64 = 100
)

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

// EstimateOutputFromRequest returns the estimated output token count from the OSL
// bucket published by the osl-bucket plugin as a request attribute. LONG requests
// (reasoning mode) use a flat 4096-token estimate, SHORT requests (tool-call) use
// 100, and UNKNOWN (or missing attribute) use 1000 — preserving the ranking
// invariant SHORT < UNKNOWN < LONG. All values are bounded by the client-requested
// cap and the estimator's operator cap.
//
// Requires the osl-bucket RequestHeaderProcessor to have run before any Produce or
// PreRequest call that invokes this method (RequestHeader runs before PreRequest in
// the plugin dispatch order).
func (e *SimpleTokenEstimator) EstimateOutputFromRequest(request *fwksched.InferenceRequest) int64 {
	if request == nil || request.Body == nil {
		return 0
	}

	bucket, _ := fwksched.ReadRequestAttribute[oslbucket.OSLBucket](request, oslbucket.OSLBucketKey)
	var est int64
	switch bucket {
	case oslbucket.Long:
		est = longOutputEstimateTokens
	case oslbucket.Short:
		est = shortOutputEstimateTokens
	default:
		est = unknownOutputEstimateTokens
	}
	return e.clampOutput(est, request.Body.MaxOutputTokens)
}

// clampOutput applies the client-requested cap and the operator cap to est.
func (e *SimpleTokenEstimator) clampOutput(est int64, clientCap *int64) int64 {
	if clientCap != nil && *clientCap > 0 && *clientCap < est {
		est = *clientCap
	}
	if e.MaxEstimatedOutputTokens != nil && *e.MaxEstimatedOutputTokens >= 0 && *e.MaxEstimatedOutputTokens < est {
		est = *e.MaxEstimatedOutputTokens
	}
	return est
}
