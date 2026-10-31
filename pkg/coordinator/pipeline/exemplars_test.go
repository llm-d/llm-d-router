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

package pipeline

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
)

const stepDurationMetric = coordmetrics.LLMDRouterCoordinatorSubsystem + "_step_duration_seconds"

// The exemplar on a step's duration must name that step's own span, not the
// pipeline span it nests under, so opening the exemplar lands on the span that
// produced the observation.
func TestExecute_StepDurationExemplarNamesItsOwnSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)

	steps := []Step{
		&mockStep{name: "decode", fn: func(context.Context, *RequestContext) error { return nil }},
	}
	require.NoError(t, New(steps).Execute(context.Background(), &RequestContext{Model: "m"}))

	exemplars := metricstest.HistogramExemplarLabels(t, reg, stepDurationMetric, map[string]string{"step": "decode"})
	require.Len(t, exemplars, 1, "the step duration must carry one exemplar")

	step := spanNamed(t, recorder, "decode")
	require.Equal(t, map[string]string{
		metricsutil.ExemplarTraceIDLabel: step.SpanContext().TraceID().String(),
		metricsutil.ExemplarSpanIDLabel:  step.SpanContext().SpanID().String(),
	}, exemplars[0])

	require.NotEqual(t, spanNamed(t, recorder, pipelineSpanName).SpanContext().SpanID().String(),
		exemplars[0][metricsutil.ExemplarSpanIDLabel],
		"the exemplar must point at the step span, not the pipeline span")
}
