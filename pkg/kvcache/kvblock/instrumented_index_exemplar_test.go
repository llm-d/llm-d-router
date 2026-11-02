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

package kvblock_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// lookupExemplars returns the lookup latency histogram's sample count and the
// labels of every bucket exemplar.
func lookupExemplars(t *testing.T) (uint64, []map[string]string) {
	t.Helper()
	metric := &dto.Metric{}
	require.NoError(t, metrics.LookupLatency.Write(metric))
	var exemplars []map[string]string
	for _, bucket := range metric.GetHistogram().GetBucket() {
		if e := bucket.GetExemplar(); e != nil {
			labels := map[string]string{}
			for _, l := range e.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			exemplars = append(exemplars, labels)
		}
	}
	return metric.GetHistogram().GetSampleCount(), exemplars
}

func TestInstrumentedIndexLookupLatencyTraceExemplars(t *testing.T) {
	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	t.Cleanup(func() { require.NoError(t, tp.Shutdown(context.Background())) })

	remote := func(id byte, flags trace.TraceFlags) context.Context {
		return trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
			TraceID: trace.TraceID{id}, SpanID: trace.SpanID{id}, TraceFlags: flags, Remote: true,
		}))
	}
	index := createInstrumentedIndexForTesting(t)
	walker, ok := index.(KeyWalker)
	require.True(t, ok, "the in-memory index walks keys, so its instrumented wrapper must too")
	calls := map[string]func(context.Context) error{
		"Lookup": func(ctx context.Context) error {
			_, err := index.Lookup(ctx, []BlockHash{1, 2}, nil)
			return err
		},
		"WalkKeys": func(ctx context.Context) error {
			return walker.WalkKeys(ctx, []BlockHash{1, 2}, func(int, bool, []EntryRef) bool { return true })
		},
	}

	var id byte
	for name, call := range calls {
		recordingCtx, span := tp.Tracer("kvblock-test").Start(context.Background(), "index_"+name)
		sampledID, unsampledID := id+1, id+2
		id += 2
		for _, tc := range []struct {
			name                string
			ctx                 context.Context
			traceID, spanID     string
			wantTrace, wantSpan bool
		}{
			{"recording", recordingCtx, span.SpanContext().TraceID().String(), span.SpanContext().SpanID().String(), true, true},
			{"propagated", remote(sampledID, trace.FlagsSampled), trace.TraceID{sampledID}.String(), "", true, false},
			{"unsampled", remote(unsampledID, 0), trace.TraceID{unsampledID}.String(), "", false, false},
			{"no span", context.Background(), "", "", false, false},
		} {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				before, _ := lookupExemplars(t)
				require.NoError(t, call(tc.ctx))
				after, exemplars := lookupExemplars(t)
				require.Equal(t, before+1, after, "one observation per call")
				if tc.traceID == "" {
					return
				}
				var found map[string]string
				for _, labels := range exemplars {
					if labels[metricsutil.ExemplarTraceIDLabel] == tc.traceID {
						found = labels
					}
				}
				if !tc.wantTrace {
					require.Nil(t, found, "an unsampled trace must not be attached")
					return
				}
				require.NotNil(t, found, "no exemplar carries the request's trace")
				if tc.wantSpan {
					require.Equal(t, tc.spanID, found[metricsutil.ExemplarSpanIDLabel])
				} else {
					require.NotContains(t, found, metricsutil.ExemplarSpanIDLabel)
				}
			})
		}
		span.End()
	}

	// The exemplar reaches the wire over OpenMetrics.
	recordingCtx, span := tp.Tracer("kvblock-test").Start(context.Background(), "index_lookup")
	defer span.End()
	_, err := index.Lookup(recordingCtx, []BlockHash{1}, nil)
	require.NoError(t, err)
	registry := prometheus.NewRegistry()
	require.NoError(t, registry.Register(metrics.LookupLatency))
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.Header.Set("Accept", "application/openmetrics-text;version=1.0.0")
	rec := httptest.NewRecorder()
	promhttp.HandlerFor(registry, promhttp.HandlerOpts{EnableOpenMetrics: true}).ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Contains(t, rec.Body.String(), `trace_id="`+span.SpanContext().TraceID().String()+`"`)
}
