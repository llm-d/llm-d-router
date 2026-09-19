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

// Package interturn declares the InterTurnPrediction attribute that carries
// the fitted distribution of a multi-turn workload's inter-turn intervals.
// The value is published once per request on the InferenceRequest attribute
// store, so consumers can predict when the session's next turn is likely to
// arrive.
package interturn

import (
	"math"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	interturnlatencyconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/interturnlatency/constants"
)

// InterTurnPredictionDataKey identifies the inter-turn prediction published
// on the request attribute store. The default producer is the
// session-interturn-latency-producer.
var InterTurnPredictionDataKey = plugin.NewDataKey("InterTurnPredictionDataKey", interturnlatencyconstants.InterTurnLatencyProducerType)

// InterTurnPrediction is the log-normal distribution fitted to a workload's
// inter-turn intervals: the idle time between one turn's response completion
// and the next turn's arrival.
type InterTurnPrediction struct {
	// LogMean and LogStd are the fitted parameters, in log-seconds.
	LogMean float64
	LogStd  float64
	// Observations is the number of intervals the fit has absorbed. Zero
	// means the parameters are still the configured seed.
	Observations int64
}

// Quantile returns the q-quantile of the predicted inter-turn interval, in
// seconds, for q in (0, 1).
func (p InterTurnPrediction) Quantile(q float64) float64 {
	z := math.Sqrt2 * math.Erfinv(2*q-1)
	return math.Exp(p.LogMean + p.LogStd*z)
}

// ReadInterTurnPrediction returns the InterTurnPrediction published by the
// default producer on the request attribute store, or the zero value and
// false if absent.
//
// Consumers should use this helper rather than reading the attribute
// directly: it encapsulates both the key construction and the type assertion,
// so a future change of storage location or value type does not ripple
// through every reader.
func ReadInterTurnPrediction(r *fwksched.InferenceRequest) (InterTurnPrediction, bool) {
	key := InterTurnPredictionDataKey.WithNonEmptyProducerName(interturnlatencyconstants.InterTurnLatencyProducerType)
	return fwksched.ReadRequestAttribute[InterTurnPrediction](r, key)
}
