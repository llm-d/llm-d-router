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

package interturn

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestInterTurnPrediction_Quantile(t *testing.T) {
	t.Parallel()

	prediction := InterTurnPrediction{LogMean: 2.0, LogStd: 1.0}

	// The median of a log-normal is exp(logMean).
	assert.InDelta(t, math.Exp(2.0), prediction.Quantile(0.5), 1e-9)
	// The 90th percentile z-score is about 1.2816.
	assert.InDelta(t, math.Exp(2.0+1.2816), prediction.Quantile(0.9), 1e-3*math.Exp(2.0+1.2816))
	// Quantiles are monotonic in q.
	assert.Less(t, prediction.Quantile(0.5), prediction.Quantile(0.9))
	assert.Less(t, prediction.Quantile(0.1), prediction.Quantile(0.5))
}

func TestReadInterTurnPrediction(t *testing.T) {
	t.Parallel()

	request := &fwksched.InferenceRequest{RequestID: "req-1"}

	_, ok := ReadInterTurnPrediction(request)
	assert.False(t, ok)

	want := InterTurnPrediction{LogMean: 2.28, LogStd: 1.34, Observations: 7}
	request.PutAttribute(InterTurnPredictionDataKey, want)

	got, ok := ReadInterTurnPrediction(request)
	assert.True(t, ok)
	assert.Equal(t, want, got)
}
