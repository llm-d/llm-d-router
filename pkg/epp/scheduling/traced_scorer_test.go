/*
Copyright 2025 The Kubernetes Authors.

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

package scheduling

import (
	"context"
	"math"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestSchedulerProfileCreatesScheduleAndScorerSpans(t *testing.T) {
	exporter := installTraceExporter(t)

	scorerA := &recordingScorer{
		typedName: fwkplugin.TypedName{Type: "type-a", Name: "scorer-a"},
		scores:    []float64{0.25, 0.75},
	}
	scorerB := &recordingScorer{
		typedName: fwkplugin.TypedName{Type: "type-b", Name: "scorer-b"},
		scores:    []float64{0.5, 0.4},
	}
	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(scorerA, 2), NewWeightedScorer(scorerB, 3)).
		WithPicker(&firstEndpointPicker{typedName: fwkplugin.TypedName{Type: "test-picker", Name: "picker"}})

	ctx, rootSpan := schedulerTracer().Start(context.Background(), "gateway.request_orchestration")
	_, err := profile.Run(ctx, &fwksched.InferenceRequest{RequestID: "request-1", TargetModel: "model-a"}, testEndpoints())
	rootSpan.End()
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	spans := exporter.GetSpans()
	root := findSpan(t, spans, "gateway.request_orchestration")
	schedule := findSpan(t, spans, scheduleSpanName)
	scorerASpan := findSpan(t, spans, scorerSpanNamePrefix+"type-a")
	scorerBSpan := findSpan(t, spans, scorerSpanNamePrefix+"type-b")

	if schedule.Parent.SpanID() != root.SpanContext.SpanID() {
		t.Fatalf("schedule span parent = %s, want %s", schedule.Parent.SpanID(), root.SpanContext.SpanID())
	}
	if scorerASpan.Parent.SpanID() != schedule.SpanContext.SpanID() {
		t.Fatalf("scorer A span parent = %s, want %s", scorerASpan.Parent.SpanID(), schedule.SpanContext.SpanID())
	}
	if scorerBSpan.Parent.SpanID() != schedule.SpanContext.SpanID() {
		t.Fatalf("scorer B span parent = %s, want %s", scorerBSpan.Parent.SpanID(), schedule.SpanContext.SpanID())
	}
	if scorerA.activeSpan.SpanID() != scorerASpan.SpanContext.SpanID() {
		t.Fatalf("scorer A active span = %s, want %s", scorerA.activeSpan.SpanID(), scorerASpan.SpanContext.SpanID())
	}
	if scorerB.activeSpan.SpanID() != scorerBSpan.SpanContext.SpanID() {
		t.Fatalf("scorer B active span = %s, want %s", scorerB.activeSpan.SpanID(), scorerBSpan.SpanContext.SpanID())
	}

	attrsA := spanAttributes(scorerASpan)
	assertStringAttribute(t, attrsA, scorerTypeAttribute, "type-a")
	assertStringAttribute(t, attrsA, scorerNameAttribute, "scorer-a")
	assertFloatAttribute(t, attrsA, scorerWeightAttribute, 2)

	attrsB := spanAttributes(scorerBSpan)
	assertStringAttribute(t, attrsB, scorerTypeAttribute, "type-b")
	assertStringAttribute(t, attrsB, scorerNameAttribute, "scorer-b")
	assertFloatAttribute(t, attrsB, scorerWeightAttribute, 3)
}

func TestTracedScorerRecordsScoreAttributes(t *testing.T) {
	exporter := installTraceExporter(t)

	scorer := &recordingScorer{
		typedName: fwkplugin.TypedName{Type: "score-type", Name: "score-name"},
		scores:    []float64{0.1, 0.5, 0.9},
	}
	scores := NewTracedScorer(scorer, 4).Score(
		context.Background(),
		&fwksched.InferenceRequest{RequestID: "request-2", TargetModel: "model-b"},
		testEndpoints(),
	)
	if len(scores) != 3 {
		t.Fatalf("Score() returned %d scores, want 3", len(scores))
	}

	span := findSpan(t, exporter.GetSpans(), scorerSpanNamePrefix+"score-type")
	attrs := spanAttributes(span)

	assertOnlyAttributes(t, span, []attribute.Key{
		scorerTypeAttribute,
		scorerNameAttribute,
		scorerWeightAttribute,
		scorerCandidateAttribute,
		scorerEndpointsAttribute,
		scorerMaxScoreAttribute,
		scorerAverageScoreAttribute,
	})
	assertStringAttribute(t, attrs, scorerTypeAttribute, "score-type")
	assertStringAttribute(t, attrs, scorerNameAttribute, "score-name")
	assertFloatAttribute(t, attrs, scorerWeightAttribute, 4)
	assertIntAttribute(t, attrs, scorerCandidateAttribute, 3)
	assertIntAttribute(t, attrs, scorerEndpointsAttribute, 3)
	assertFloatAttribute(t, attrs, scorerMaxScoreAttribute, 0.9)
	assertFloatAttribute(t, attrs, scorerAverageScoreAttribute, 0.5)
}

func TestSchedulerProfileWithZeroScorersCreatesOnlyScheduleSpan(t *testing.T) {
	exporter := installTraceExporter(t)

	profile := NewSchedulerProfile().
		WithPicker(&firstEndpointPicker{typedName: fwkplugin.TypedName{Type: "test-picker", Name: "picker"}})
	endpoints := testEndpoints()[:2]

	weightedScores := profile.runScorerPlugins(context.Background(), &fwksched.InferenceRequest{RequestID: "request-3"}, endpoints)
	if len(weightedScores) != len(endpoints) {
		t.Fatalf("runScorerPlugins() returned %d endpoints, want %d", len(weightedScores), len(endpoints))
	}
	for _, endpoint := range endpoints {
		if weightedScores[endpoint] != 0 {
			t.Fatalf("runScorerPlugins() score = %v, want 0", weightedScores[endpoint])
		}
	}

	_, err := profile.Run(context.Background(), &fwksched.InferenceRequest{RequestID: "request-3"}, endpoints)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	spans := exporter.GetSpans()
	findSpan(t, spans, scheduleSpanName)
	for _, span := range spans {
		if strings.HasPrefix(span.Name, scorerSpanNamePrefix) {
			t.Fatalf("unexpected scorer span %q", span.Name)
		}
	}
}

func TestTracedScorerNoopTracerProvider(t *testing.T) {
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(noop.NewTracerProvider())
	t.Cleanup(func() {
		otel.SetTracerProvider(previous)
	})

	scorer := &recordingScorer{
		typedName: fwkplugin.TypedName{Type: "noop-type", Name: "noop-name"},
		scores:    []float64{0.4},
	}
	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(scorer, 2))
	endpoints := testEndpoints()[:1]

	weightedScores := profile.runScorerPlugins(context.Background(), &fwksched.InferenceRequest{RequestID: "request-4"}, endpoints)
	if scorer.callCount != 1 {
		t.Fatalf("Score() call count = %d, want 1", scorer.callCount)
	}
	if math.Abs(weightedScores[endpoints[0]]-0.8) > 1e-9 {
		t.Fatalf("weighted score = %v, want 0.8", weightedScores[endpoints[0]])
	}
}

type recordingScorer struct {
	typedName  fwkplugin.TypedName
	scores     []float64
	callCount  int
	activeSpan trace.SpanContext
}

func (s *recordingScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *recordingScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

func (s *recordingScorer) Score(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	s.callCount++
	s.activeSpan = trace.SpanFromContext(ctx).SpanContext()
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for i, endpoint := range endpoints {
		if i < len(s.scores) {
			scores[endpoint] = s.scores[i]
		}
	}
	return scores
}

type firstEndpointPicker struct {
	typedName fwkplugin.TypedName
}

func (p *firstEndpointPicker) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *firstEndpointPicker) Pick(_ context.Context, scoredEndpoints []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	if len(scoredEndpoints) == 0 {
		return &fwksched.ProfileRunResult{}
	}
	return &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{scoredEndpoints[0]}}
}

func testEndpoints() []fwksched.Endpoint {
	return []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-b"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-c"}}, nil, nil),
	}
}

func installTraceExporter(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
	})
	return exporter
}

func findSpan(t *testing.T, spans tracetest.SpanStubs, name string) tracetest.SpanStub {
	t.Helper()
	for _, span := range spans {
		if span.Name == name {
			return span
		}
	}
	t.Fatalf("span %q not found in %v", name, spanNames(spans))
	return tracetest.SpanStub{}
}

func spanNames(spans tracetest.SpanStubs) []string {
	names := make([]string, 0, len(spans))
	for _, span := range spans {
		names = append(names, span.Name)
	}
	return names
}

func spanAttributes(span tracetest.SpanStub) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value, len(span.Attributes))
	for _, attr := range span.Attributes {
		attrs[attr.Key] = attr.Value
	}
	return attrs
}

func assertOnlyAttributes(t *testing.T, span tracetest.SpanStub, keys []attribute.Key) {
	t.Helper()
	expected := make(map[attribute.Key]struct{}, len(keys))
	for _, key := range keys {
		expected[key] = struct{}{}
	}
	if len(span.Attributes) != len(expected) {
		t.Fatalf("span %q has %d attributes, want %d: %v", span.Name, len(span.Attributes), len(expected), span.Attributes)
	}
	for _, attr := range span.Attributes {
		if _, ok := expected[attr.Key]; !ok {
			t.Fatalf("span %q has unexpected attribute %q", span.Name, attr.Key)
		}
	}
}

func assertStringAttribute(t *testing.T, attrs map[attribute.Key]attribute.Value, key attribute.Key, want string) {
	t.Helper()
	value, ok := attrs[key]
	if !ok {
		t.Fatalf("attribute %q missing", key)
	}
	if got := value.AsString(); got != want {
		t.Fatalf("attribute %q = %q, want %q", key, got, want)
	}
}

func assertIntAttribute(t *testing.T, attrs map[attribute.Key]attribute.Value, key attribute.Key, want int64) {
	t.Helper()
	value, ok := attrs[key]
	if !ok {
		t.Fatalf("attribute %q missing", key)
	}
	if got := value.AsInt64(); got != want {
		t.Fatalf("attribute %q = %d, want %d", key, got, want)
	}
}

func assertFloatAttribute(t *testing.T, attrs map[attribute.Key]attribute.Value, key attribute.Key, want float64) {
	t.Helper()
	value, ok := attrs[key]
	if !ok {
		t.Fatalf("attribute %q missing", key)
	}
	if got := value.AsFloat64(); math.Abs(got-want) > 1e-9 {
		t.Fatalf("attribute %q = %f, want %f", key, got, want)
	}
}
