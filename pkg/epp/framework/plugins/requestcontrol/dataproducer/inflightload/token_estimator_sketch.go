/*
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
	lru "github.com/hashicorp/golang-lru/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload/outlensketch"
)

// sketchPredictionCacheSize bounds the per-request prediction cache. The first
// EstimateOutputFromRequest call per request embeds the prompt and caches the
// prediction (which carries the embedding); later per-endpoint calls and the
// end-of-stream Observe reuse it, so the prompt is embedded once per request. The
// LRU bound keeps memory flat (each entry holds one embedding) when an entry is
// never explicitly removed, e.g. a request that ends without an end-of-stream.
const sketchPredictionCacheSize = 8192

// OutputObserver is an optional TokenEstimator capability that carries the learning
// feedback: when a response stream ends, the producer calls Observe with the
// request's true output token count. A learning estimator requires this signal, since
// it is the only source of the real output lengths it must predict. The producer
// calls Observe only on estimators that implement it: the sketch estimator folds each
// observed length into its histograms and cannot learn without it; the static
// estimator does not learn, does not implement it, and is unaffected.
type OutputObserver interface {
	Observe(request *fwksched.InferenceRequest, completionTokens int64)
}

// sketchModel is the output-length predictor and learner the estimator drives.
// *outlensketch.Sketch is the production implementation; the seam keeps the
// estimator's caching and fallback logic testable without the embedding asset.
type sketchModel interface {
	Predict(body *fwkrh.InferenceRequestBody) outlensketch.Prediction
	Learn(prediction outlensketch.Prediction, completionTokens int64)
}

var _ sketchModel = (*outlensketch.Sketch)(nil)

// cachedPrediction holds a request's featurized prediction and its clamped output
// estimate. The prediction (which carries the prompt embedding) is reused at
// end-of-stream so the prompt is windowed and embedded once per request, not again
// per candidate endpoint nor again to learn.
type cachedPrediction struct {
	prediction outlensketch.Prediction
	output     int64
}

// SketchTokenEstimator estimates output tokens from the learned output-length
// sketch (outlensketch): a more precise, self-adapting alternative to the static
// signal-rule mapping that an operator selects in place of it. Input-token
// estimation, output clamping, and the cold-key fallback reuse SimpleTokenEstimator.
type SketchTokenEstimator struct {
	*SimpleTokenEstimator
	sketch sketchModel
	cache  *lru.Cache[string, cachedPrediction]
}

var (
	_ TokenEstimator = &SketchTokenEstimator{}
	_ OutputObserver = &SketchTokenEstimator{}
)

// NewSketchTokenEstimator returns a SketchTokenEstimator over the given sketch, with
// an optional operator cap on the estimated output tokens.
func NewSketchTokenEstimator(sketch sketchModel, maxOutput *int64) (*SketchTokenEstimator, error) {
	cache, err := lru.New[string, cachedPrediction](sketchPredictionCacheSize)
	if err != nil {
		return nil, err
	}
	return &SketchTokenEstimator{
		SimpleTokenEstimator: &SimpleTokenEstimator{MaxEstimatedOutputTokens: maxOutput},
		sketch:               sketch,
		cache:                cache,
	}, nil
}

// EstimateOutputFromRequest returns the sketch's output-token estimate for the
// request, cached per request id so repeated per-endpoint calls (and the
// end-of-stream Observe) reuse one embedding. A cold key falls back to the static
// signal-rule estimate, so the estimator is usable from the first request and
// self-heals as keys warm up.
func (e *SketchTokenEstimator) EstimateOutputFromRequest(request *fwksched.InferenceRequest) int64 {
	if request == nil || request.Body == nil {
		return 0
	}
	if request.RequestID != "" {
		if cached, ok := e.cache.Get(request.RequestID); ok {
			return cached.output
		}
	}
	prediction := e.sketch.Predict(request.Body)
	output := e.outputFor(request, prediction)
	if request.RequestID != "" {
		e.cache.Add(request.RequestID, cachedPrediction{prediction: prediction, output: output})
	}
	return output
}

// outputFor maps a prediction to a clamped output estimate, falling back to the
// static signal-rule estimate when the key is cold.
func (e *SketchTokenEstimator) outputFor(request *fwksched.InferenceRequest, prediction outlensketch.Prediction) int64 {
	if prediction.OK {
		return e.clampOutput(prediction.Magnitude, request.Body.MaxOutputTokens)
	}
	return e.SimpleTokenEstimator.EstimateOutputFromRequest(request)
}

// Observe folds the observed output length into the sketch at end-of-stream, reusing
// the prediction cached at scheduling time (recomputing it only on a cache miss),
// and drops the cache entry.
func (e *SketchTokenEstimator) Observe(request *fwksched.InferenceRequest, completionTokens int64) {
	if request == nil || request.Body == nil {
		return
	}
	prediction, ok := e.lookupPrediction(request)
	if !ok {
		prediction = e.sketch.Predict(request.Body)
	}
	e.sketch.Learn(prediction, completionTokens)
	if request.RequestID != "" {
		e.cache.Remove(request.RequestID)
	}
}

func (e *SketchTokenEstimator) lookupPrediction(request *fwksched.InferenceRequest) (outlensketch.Prediction, bool) {
	if request.RequestID == "" {
		return outlensketch.Prediction{}, false
	}
	cached, ok := e.cache.Get(request.RequestID)
	return cached.prediction, ok
}
