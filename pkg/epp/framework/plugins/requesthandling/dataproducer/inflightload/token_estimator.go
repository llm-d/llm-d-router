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
	"math"
	"sync"
	"sync/atomic"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/outlenbucket"
)

// TokenEstimator estimates token counts for an LLM request.
type TokenEstimator interface {
	// EstimateInput returns the estimated input token count for the request.
	EstimateInput(request *fwksched.InferenceRequest) int64
	// EstimateOutputFromRequest returns the estimated output work for a request.
	EstimateOutputFromRequest(request *fwksched.InferenceRequest) int64
	// ObserveOutput records the completion tokens a finished request produced.
	ObserveOutput(bucket outlenbucket.Bucket, completionTokens int64)
}

const (
	// LongOutputTokens is the flat output-token estimate for a LONG (reasoning)
	// request. It is deliberately a fixed value rather than the client's
	// thinking_budget: the estimate exists to rank requests by load, where the
	// LONG-vs-SHORT separation dominates, not to predict exact length.
	LongOutputTokens int64 = 4096
	// UnknownOutputTokens is the output-token estimate for an UNKNOWN request
	// (no output-length signal) until the first UNKNOWN completion is observed.
	UnknownOutputTokens int64 = 1000
	// ShortOutputTokens is the flat output-token estimate for a SHORT
	// (tool-call) request.
	ShortOutputTokens int64 = 100

	// unknownOutputWindow bounds the running average's divisor, so each UNKNOWN
	// observation weighs at least 1/unknownOutputWindow and the estimate follows
	// workload shifts.
	unknownOutputWindow = 100

	// TTS output is codec frames at 12.5 frames/s. At a typical speaking rate
	// that is about one frame per spoken character; using request bytes also
	// keeps non-ASCII text proportional to its speech duration.
	ttsFramesPerSpokenByte = 1.0
	ttsMaxFrames           = 4096

	// ASR output is text, but its length follows the audio duration rather than
	// a text prompt. Four tokens per second covers typical speech and
	// punctuation; model-specific rates belong in per-pool configuration.
	asrOutputTokensPerSecond       = 4.0
	asrMaxOutputTokens       int64 = 16_384
	// vLLM chunks long audio at 30s with a 1s split search window and applies
	// max_completion_tokens per chunk. These are its defaults; model-specific
	// values belong in per-pool configuration.
	asrDefaultClipSeconds    = 30.0
	asrDefaultOverlapSeconds = 1.0
)

// SimpleTokenEstimator reads input tokens from the tokenized prompt and maps the
// output-length bucket published by the outlen-bucket plugin or an audio duration
// for transcriptions and translations to an output estimate. The estimate is
// bounded by the client-requested cap and an optional operator cap.
type SimpleTokenEstimator struct {
	// MaxEstimatedOutputTokens optionally caps the estimated output tokens
	// regardless of the client-requested cap. nil means no cap.
	MaxEstimatedOutputTokens *int64

	mu         sync.Mutex // guards unknownAvg and unknownN
	unknownAvg float64
	unknownN   int
	// unknownEst is the published UNKNOWN estimate, read lock-free on the
	// scheduling path. 0 means no UNKNOWN completion observed yet.
	unknownEst atomic.Int64
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

// EstimateOutputFromRequest returns the estimated output work for a request.
// Transcriptions and translations follow the uploaded audio duration; otherwise
// the text estimate comes from the output-length bucket published by the
// outlen-bucket plugin. The text estimate maps LONG (reasoning) to a flat
// 4096-token estimate, SHORT (tool-call) to 100, and UNKNOWN to the running
// average of observed UNKNOWN completion tokens (1000 before the first one),
// kept within [SHORT, LONG] to preserve the ranking SHORT <= UNKNOWN <= LONG.
// The estimate is bounded by the client-requested cap and the estimator's
// operator cap.
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

	// TTS output is codec frames derived from the spoken text, not the text
	// output-length bucket (which the outlen-bucket plugin leaves UNKNOWN for
	// speech).
	if tts := request.Body.TextToSpeech; tts != nil {
		return e.clampOutput(ttsOutputFrames(tts, request.Body.MaxOutputTokens), request.Body.MaxOutputTokens)
	}

	// Transcription and translation output length follows the uploaded audio,
	// not a text prompt.
	if tr := request.Body.Transcriptions; tr != nil {
		return e.clampOutput(transcriptionOutputTokens(tr, request.Body.MaxOutputTokens, request.Headers), nil)
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
		est = e.unknownEst.Load()
		if est == 0 {
			est = UnknownOutputTokens
		}
	}
	return e.clampOutput(est, request.Body.MaxOutputTokens)
}

// ObserveOutput folds the completion tokens of a finished UNKNOWN request into the
// UNKNOWN estimate. SHORT and LONG keep their flat estimates and are not observed.
func (e *SimpleTokenEstimator) ObserveOutput(bucket outlenbucket.Bucket, completionTokens int64) {
	if bucket != outlenbucket.Unknown || completionTokens <= 0 {
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	e.unknownN = min(e.unknownN+1, unknownOutputWindow)
	e.unknownAvg += (float64(completionTokens) - e.unknownAvg) / float64(e.unknownN)
	e.unknownEst.Store(min(max(int64(e.unknownAvg), ShortOutputTokens), LongOutputTokens))
}

// ttsOutputFrames estimates the codec frames a TTS request will generate.
// Only the spoken text contributes to output length; instructions and the
// reference transcript are prefill cost. An explicit max_new_tokens caps the
// estimate.
func ttsOutputFrames(tts *fwkrequest.TextToSpeechRequest, clientCap *int64) int64 {
	expected := int64(math.Ceil(float64(len(tts.Input)) * ttsFramesPerSpokenByte))
	if clientCap != nil && *clientCap > 0 && expected > *clientCap {
		expected = *clientCap
	}
	expected = min(expected, ttsMaxFrames)
	return max(expected, 1)
}

// transcriptionOutputTokens estimates transcript or translation output length
// from the uploaded audio duration. vLLM applies max_completion_tokens per
// audio chunk, so the client cap scales with the estimated chunk count.
func transcriptionOutputTokens(tr *fwkrequest.TranscriptionRequest, maxCompletionTokens *int64, headers map[string]string) int64 {
	duration := tokenproducer.AudioDurationSeconds(tr.Audio.Head, tr.Audio.SizeBytes, headers)

	expected := asrMaxOutputTokens
	if duration <= float64(asrMaxOutputTokens)/asrOutputTokensPerSecond {
		expected = int64(math.Ceil(duration * asrOutputTokensPerSecond))
	}
	if maxCompletionTokens == nil || *maxCompletionTokens <= 0 {
		return max(expected, 1)
	}

	chunks := int64(1)
	if duration > asrDefaultClipSeconds {
		estimatedChunks := math.Ceil(duration / (asrDefaultClipSeconds - asrDefaultOverlapSeconds))
		if estimatedChunks > float64(asrMaxOutputTokens) {
			chunks = asrMaxOutputTokens
		} else {
			chunks = int64(estimatedChunks)
		}
	}
	cap := *maxCompletionTokens
	if chunks > 1 {
		if cap > asrMaxOutputTokens/chunks {
			cap = asrMaxOutputTokens
		} else {
			cap *= chunks
		}
	}
	expected = min(expected, cap)
	return max(expected, 1)
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
