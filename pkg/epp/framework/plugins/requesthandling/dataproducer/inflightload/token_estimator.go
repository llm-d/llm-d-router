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

	// Image output is diffusion work: steps x output pixels x images, scaled
	// to token-equivalent work. The defaults match the common Qwen-Image
	// deployment. A request that omits size, num_inference_steps, or n is
	// estimated with these defaults even when the serving model's own defaults
	// differ (for example a 4-step FLUX-schnell).
	imageDefaultInferenceSteps int64 = 50
	imageDefaultPixels         int64 = 1024 * 1024
	imageMaxInferenceSteps     int64 = 200
	imageMaxImages             int64 = 10
	// Layered image models denoise a sequence of layers plus the origin frame.
	imageDefaultLayers        int64 = 4
	imageDefaultLayeredPixels int64 = 640 * 640
	imageMaxLayers            int64 = 10
	// imageMaxPixels bounds oversized requests and keeps the pixel product
	// inside int64. Raise it if a deployment serves larger outputs.
	imageMaxPixels int64 = 4096 * 4096
	// imageWorkScale converts pixel-steps to token-equivalent work: 10,000
	// pixel-steps per unit, so a 1024x1024, 50-step image is about the size of
	// a LONG text request.
	imageWorkScale = 10_000.0
)

// SimpleTokenEstimator reads input tokens from the tokenized prompt and maps the
// output-length bucket published by the outlen-bucket plugin, or the request's
// diffusion work for images, to an output estimate. Text estimates are bounded
// by the client-requested cap and an optional operator cap; image work is not.
type SimpleTokenEstimator struct {
	// MaxEstimatedOutputTokens optionally caps the text output estimate
	// regardless of the client-requested cap. nil means no cap. Image work is
	// not capped.
	MaxEstimatedOutputTokens *int64

	mu         sync.Mutex // guards unknownAvg and unknownN
	unknownAvg float64
	unknownN   int
	// unknownEst is the published UNKNOWN estimate, read lock-free on the
	// scheduling path. 0 means no UNKNOWN completion observed yet.
	unknownEst atomic.Int64
}

// NewSimpleTokenEstimator returns a SimpleTokenEstimator with an optional operator
// cap (maxOutput, nil for no cap) on the text output estimate.
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

// EstimateOutputFromRequest returns the estimated output work for a request:
// diffusion work for images, otherwise a text estimate from the output-length
// bucket published by the outlen-bucket plugin. The text estimate maps LONG
// (reasoning) to a flat 4096-token estimate, SHORT (tool-call) to 100, and
// UNKNOWN to the running average of observed UNKNOWN completion tokens (1000
// before the first one), kept within [SHORT, LONG] to preserve the ranking
// SHORT <= UNKNOWN <= LONG. Text estimates are bounded by the client-requested
// cap and the estimator's operator cap; image work is not.
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

	// Image output is diffusion work expressed in token-equivalent units, not a
	// text output length, so the text-token client and operator caps do not apply.
	if images := request.Body.Images; images != nil {
		return imageOutputWork(images)
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

// imageOutputWork estimates the diffusion work an image request will perform:
// num_inference_steps x output pixels x image count, with layered requests
// additionally denoising the origin frame plus one frame per requested layer.
// One work unit is 10,000 pixel-steps, so a 1024x1024, 50-step, single-image
// request is about 5,243.
func imageOutputWork(images *fwkrequest.ImagesGenerationsRequest) int64 {
	steps := imageDefaultInferenceSteps
	if images.NumInferenceSteps != nil && *images.NumInferenceSteps > 0 {
		steps = min(*images.NumInferenceSteps, imageMaxInferenceSteps)
	}
	n := int64(1)
	if images.N != nil && *images.N > 0 {
		n = min(*images.N, imageMaxImages)
	}

	layers := int64(1)
	layered := images.Layers != nil || images.Resolution != nil
	if layered {
		requested := imageDefaultLayers
		if images.Layers != nil && *images.Layers > 0 {
			requested = min(*images.Layers, imageMaxLayers)
		}
		// Qwen-Image-Layered denoises the origin frame plus one frame per
		// requested layer.
		layers = requested + 1
	}

	pixels := imageDefaultPixels
	if w, h := fwkrequest.ParseResolution(images.Size); w > 0 && h > 0 {
		// Divide before multiplying so an oversized dimension cannot overflow.
		if w > imageMaxPixels/h {
			pixels = imageMaxPixels
		} else {
			pixels = w * h
		}
	} else if layered {
		pixels = imageDefaultLayeredPixels
		if images.Resolution != nil && *images.Resolution > 0 {
			resolution := *images.Resolution
			if resolution > imageMaxPixels/resolution {
				pixels = imageMaxPixels
			} else {
				pixels = resolution * resolution
			}
		}
	}

	work := int64(math.Ceil(float64(steps*n*layers*pixels) / imageWorkScale))
	return max(work, 1)
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
