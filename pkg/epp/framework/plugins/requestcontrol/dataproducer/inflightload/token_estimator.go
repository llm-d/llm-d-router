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
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/outlenbucket"
)

// TokenEstimator estimates token counts for an LLM request.
type TokenEstimator interface {
	// EstimateInput returns the estimated input token count for the request.
	EstimateInput(request *fwksched.InferenceRequest) int64
	// EstimateOutputFromRequest returns the estimated output token count for a
	// request from the output-length bucket published by the outlen-bucket plugin, bounded by
	// the client-requested cap and the estimator's operator cap.
	EstimateOutputFromRequest(request *fwksched.InferenceRequest) int64
}

const (
	// LongOutputTokens is the flat output-token estimate for a LONG (reasoning)
	// request. It is deliberately a fixed value rather than the client's
	// thinking_budget: the estimate exists to rank requests by load, where the
	// LONG-vs-SHORT separation dominates, not to predict exact length.
	LongOutputTokens int64 = 4096
	// UnknownOutputTokens is the flat output-token estimate for an UNKNOWN
	// request (no output-length signal), preserving the ranking invariant
	// SHORT (100) < UNKNOWN (1000) < LONG (4096).
	// TODO(outlen): replace with a dynamic estimate (e.g. per-pool running average
	// of observed CompletionTokens) in a follow-up PR.
	UnknownOutputTokens int64 = 1000
	// ShortOutputTokens is the flat output-token estimate for a SHORT
	// (tool-call) request.
	ShortOutputTokens int64 = 100
)

// SimpleTokenEstimator reads input tokens from the tokenized prompt and maps the
// output-length bucket published by the outlen-bucket plugin to a flat output-token estimate,
// bounded by the client-requested cap and an optional operator cap.
type SimpleTokenEstimator struct {
	// MaxEstimatedOutputTokens optionally caps the estimated output tokens
	// regardless of the client-requested cap. nil means no cap.
	MaxEstimatedOutputTokens *int64
}

// NewSimpleTokenEstimator returns a SimpleTokenEstimator with an optional operator
// cap (maxOutput, nil for no cap) on the estimated output tokens.
func NewSimpleTokenEstimator(maxOutput *int64) TokenEstimator {
	return &SimpleTokenEstimator{MaxEstimatedOutputTokens: maxOutput}
}

// EstimateInput returns the input token count read from the tokenized prompt,
// or 0 when no tokenization is available.
func (e *SimpleTokenEstimator) EstimateInput(request *fwksched.InferenceRequest) int64 {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return 0
	}
	return int64(request.Body.TokenizedRequest.TokenCount())
}

// EstimateOutputFromRequest returns the estimated output token count for a request
// from the output-length bucket published by the outlen-bucket plugin: LONG (reasoning) maps to
// a flat 4096-token estimate, SHORT (tool-call) to 100, and UNKNOWN to 1000 --
// preserving the ranking invariant SHORT < UNKNOWN < LONG. The estimate is bounded
// by the client-requested cap and the estimator's operator cap.
//
// Ordering dependency: the bucket comes from an attribute that the outlen-bucket
// plugin publishes in its RequestHeader hook, which this method reads. The read is
// correct only because the framework always runs RequestHeader before both Produce
// and PreRequest -- this method's only callers -- so the publish happens first.
// This is a phase guarantee, not a declared Produce/Consume dependency. When the
// outlen-bucket plugin is not enabled, the attribute is absent and reads as its zero
// value, Unknown, so every request is estimated as UNKNOWN; the producer logs a
// one-time warning in that case (see InFlightLoadProducer.warnMissingOutlenBucket).
func (e *SimpleTokenEstimator) EstimateOutputFromRequest(request *fwksched.InferenceRequest) int64 {
	if request == nil || request.Body == nil {
		return 0
	}

	// An absent attribute reads as the zero value (Unknown) and is handled by the
	// switch default; there is no separate fallback path.
	bucket, _ := fwksched.ReadRequestAttribute[outlenbucket.Bucket](request, outlenbucket.AttributeKey)

	var est int64
	switch bucket {
	case outlenbucket.Long:
		est = LongOutputTokens
	case outlenbucket.Short:
		est = ShortOutputTokens
	default:
		est = UnknownOutputTokens
	}
	return e.clampOutput(est, request.Body.MaxOutputTokens)
}

// clampOutput bounds est by the client-requested cap and the operator cap.
// Client cap applies only when positive (> 0): MaxOutputTokens=0 is treated as
// "no cap", matching the outlen-bucket classifier's convention. This intentionally
// diverges from the prior ratio-based estimator (which used >= 0 and clamped to 0)
// and from InferenceRequestBody.MaxOutputTokens's contract elsewhere. Operator cap
// uses >= 0, so MaxEstimatedOutputTokens=0 still clamps to 0.
func (e *SimpleTokenEstimator) clampOutput(est int64, clientCap *int64) int64 {
	if clientCap != nil && *clientCap > 0 && *clientCap < est {
		est = *clientCap
	}
	if e.MaxEstimatedOutputTokens != nil && *e.MaxEstimatedOutputTokens >= 0 && *e.MaxEstimatedOutputTokens < est {
		est = *e.MaxEstimatedOutputTokens
	}
	return est
}
