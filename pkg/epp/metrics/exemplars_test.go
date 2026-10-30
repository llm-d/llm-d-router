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

package metrics

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func ctxWithSpan(t *testing.T, sampled bool) context.Context {
	t.Helper()
	traceID, err := trace.TraceIDFromHex("4bf92f3577b34da6a3ce929d0e0e4736")
	require.NoError(t, err)
	spanID, err := trace.SpanIDFromHex("0102030405060708")
	require.NoError(t, err)

	var flags trace.TraceFlags
	if sampled {
		flags = trace.FlagsSampled
	}
	return trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: flags,
	}))
}

// ctxWithRecordingSpan returns a context with a real SDK span, like the one the
// EPP starts per request.
func ctxWithRecordingSpan(t *testing.T) (context.Context, trace.Span) {
	t.Helper()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	t.Cleanup(func() { require.NoError(t, tp.Shutdown(context.Background())) })
	return tp.Tracer("exemplar-test").Start(context.Background(), "request")
}

// exemplarLabels returns an exemplar's labels as a map.
func exemplarLabels(e *dto.Exemplar) map[string]string {
	labels := map[string]string{}
	for _, l := range e.Label {
		labels[l.GetName()] = l.GetValue()
	}
	return labels
}

// latencyRecorder records one observation through a latency recorder and names
// the series it lands on.
type latencyRecorder struct {
	name   string
	record func(ctx context.Context)
	vec    *prometheus.HistogramVec
	labels []string
}

func latencyRecorders() []latencyRecorder {
	received := time.Now()
	firstToken := received.Add(50 * time.Millisecond)
	complete := received.Add(250 * time.Millisecond)
	return []latencyRecorder{
		{
			name: "ttft",
			record: func(ctx context.Context) {
				RecordRequestTTFT(ctx, "m", "t", reqcommon.APITypeMessages.Surface(), "f", "1", true, received, firstToken)
			},
			vec:    llmdRequestTTFT,
			labels: []string{"m", "t", "f", "1", reqcommon.APITypeMessages.String(), "true"},
		},
		{
			name: "tpot",
			record: func(ctx context.Context) {
				RecordRequestTPOT(ctx, "m", "t", reqcommon.APITypeMessages.Surface(), "f", "1", true, received, firstToken, complete, 5)
			},
			vec:    llmdRequestTPOT,
			labels: []string{"m", "t", "f", "1", reqcommon.APITypeMessages.String()},
		},
		{
			name: "itl",
			record: func(ctx context.Context) {
				RecordInterTokenLatency(ctx, "m", "t", reqcommon.APITypeMessages.Surface(), "f", "1", 0.02)
			},
			vec:    llmdInterTokenLatency,
			labels: []string{"m", "t", "f", "1", reqcommon.APITypeMessages.String()},
		},
		{
			name: "ntpot",
			record: func(ctx context.Context) {
				RecordNormalizedTimePerOutputToken(ctx, "m", "t", reqcommon.APITypeMessages.Surface(), "f", "1", received, complete, 5)
			},
			vec:    llmdNormalizedTimePerOutputToken,
			labels: []string{"m", "t", "f", "1", reqcommon.APITypeMessages.String()},
		},
		{
			name: "scheduler",
			record: func(ctx context.Context) {
				RecordSchedulerE2ELatency(ctx, 3*time.Millisecond)
			},
			vec: llmdSchedulerE2ELatency,
		},
		{
			name: "flow_control_queue",
			record: func(ctx context.Context) {
				RecordFlowControlRequestQueueDuration(ctx, "f", "1", "Dispatched", "pool", "m", "t", 20*time.Millisecond)
			},
			vec:    llmdFlowControlRequestQueueDuration,
			labels: []string{"f", "1", "Dispatched", "pool", "m", "t"},
		},
	}
}

// bucketExemplars returns the exemplars on a histogram's buckets.
func bucketExemplars(h *dto.Histogram) []*dto.Exemplar {
	var found []*dto.Exemplar
	for _, bucket := range h.GetBucket() {
		if bucket.Exemplar != nil {
			found = append(found, bucket.Exemplar)
		}
	}
	return found
}

func TestLatencyRecordersAttachTraceExemplars(t *testing.T) {
	for _, r := range latencyRecorders() {
		t.Run(r.name+"/sampled", func(t *testing.T) {
			Reset()
			ctx, span := ctxWithRecordingSpan(t)
			defer span.End()
			r.record(ctx)

			h, err := getHistogramVecLabelValues(t, r.vec, r.labels...)
			require.NoError(t, err)
			exemplars := bucketExemplars(h)
			require.Len(t, exemplars, 1)
			require.Equal(t, span.SpanContext().TraceID().String(), exemplarLabels(exemplars[0])[metricsutil.ExemplarTraceIDLabel])
		})

		t.Run(r.name+"/unsampled", func(t *testing.T) {
			Reset()
			r.record(ctxWithSpan(t, false))

			h, err := getHistogramVecLabelValues(t, r.vec, r.labels...)
			require.NoError(t, err)
			require.Empty(t, bucketExemplars(h))
			require.Equal(t, uint64(1), h.GetSampleCount())
		})
	}
}

// BenchmarkRecordInterTokenLatency measures the exemplar cost on the ITL path,
// which records once per streaming chunk.
func BenchmarkRecordInterTokenLatency(b *testing.B) {
	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	b.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	sampledCtx, span := tp.Tracer("exemplar-bench").Start(context.Background(), "request")
	defer span.End()

	unsampledCtx := trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1},
		SpanID:  trace.SpanID{1},
	}))

	for _, bc := range []struct {
		name string
		ctx  context.Context
	}{
		{"no-span", context.Background()},
		{"unsampled", unsampledCtx},
		{"sampled", sampledCtx},
	} {
		b.Run(bc.name, func(b *testing.B) {
			Reset()
			b.ReportAllocs()
			for b.Loop() {
				RecordInterTokenLatency(bc.ctx, "m", "t", reqcommon.APITypeMessages.Surface(), "f", "1", 0.02)
			}
		})
	}
}
