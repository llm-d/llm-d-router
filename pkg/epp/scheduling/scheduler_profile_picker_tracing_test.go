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
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return recorder
}

func spansByName(spans []sdktrace.ReadOnlySpan, name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

func spanAttrs(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	m := make(map[attribute.Key]attribute.Value)
	for _, kv := range span.Attributes() {
		m[kv.Key] = kv.Value
	}
	return m
}

func makeWeightedScores(names ...string) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(names))
	for i, name := range names {
		ep := fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}}, nil, nil)
		scores[ep] = float64(i + 1)
	}
	return scores
}

// stubPicker is a minimal Picker for tracing tests.
type stubPicker struct {
	typedName fwkplugin.TypedName
	numPick   int
	nilResult bool
	emitSpan  bool
}

var _ fwksched.Picker = &stubPicker{}

func (s *stubPicker) TypedName() fwkplugin.TypedName { return s.typedName }

func (s *stubPicker) Pick(ctx context.Context, scored []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	if s.emitSpan {
		_, child := otel.Tracer("test").Start(ctx, "child_picker_op")
		child.End()
	}
	if s.nilResult {
		return nil
	}
	n := s.numPick
	if n > len(scored) {
		n = len(scored)
	}
	targets := make([]fwksched.Endpoint, n)
	for i := 0; i < n; i++ {
		targets[i] = scored[i].Endpoint
	}
	return &fwksched.ProfileRunResult{TargetEndpoints: targets}
}

func TestPickerSpan_Basic(t *testing.T) {
	recorder := setupSpanRecorder(t)

	profile := NewSchedulerProfile().WithPicker(&stubPicker{
		typedName: fwkplugin.TypedName{Type: "max-score-picker", Name: "default"},
		numPick:   1,
	})
	scores := makeWeightedScores("pod1", "pod2", "pod3")
	req := &fwksched.InferenceRequest{TargetModel: "llama-7b", RequestID: "req-123"}

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, req, scores)
	root.End()

	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected 1 target endpoint, got %v", result)
	}

	spans := spansByName(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("expected 1 pick_endpoints span, got %d", len(spans))
	}

	span := spans[0]
	if span.SpanKind() != trace.SpanKindInternal {
		t.Errorf("span kind = %v, want Internal", span.SpanKind())
	}
	if span.Parent().SpanID() != root.SpanContext().SpanID() {
		t.Errorf("span parent = %v, want root %v", span.Parent().SpanID(), root.SpanContext().SpanID())
	}

	attrs := spanAttrs(span)
	if got := attrs["llm_d.epp.picker.candidate_endpoints"].AsInt64(); got != 3 {
		t.Errorf("candidate_endpoints = %d, want 3", got)
	}
	if got := attrs["llm_d.epp.picker.selected_endpoints"].AsInt64(); got != 1 {
		t.Errorf("selected_endpoints = %d, want 1", got)
	}
	if got := attrs["gen_ai.request.model"].AsString(); got != "llama-7b" {
		t.Errorf("gen_ai.request.model = %q, want %q", got, "llama-7b")
	}
	if got := attrs["gen_ai.request.id"].AsString(); got != "req-123" {
		t.Errorf("gen_ai.request.id = %q, want %q", got, "req-123")
	}
}

func TestPickerSpan_MultipleSelected(t *testing.T) {
	recorder := setupSpanRecorder(t)

	profile := NewSchedulerProfile().WithPicker(&stubPicker{
		typedName: fwkplugin.TypedName{Type: "max-score-picker", Name: "multi"},
		numPick:   2,
	})
	scores := makeWeightedScores("pod1", "pod2", "pod3")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{TargetModel: "m1", RequestID: "r1"}, scores)
	root.End()

	if result == nil || len(result.TargetEndpoints) != 2 {
		t.Fatalf("expected 2 target endpoints, got %v", result)
	}

	spans := spansByName(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if got := spanAttrs(spans[0])["llm_d.epp.picker.selected_endpoints"].AsInt64(); got != 2 {
		t.Errorf("selected_endpoints = %d, want 2", got)
	}
}

func TestPickerSpan_NilResult(t *testing.T) {
	recorder := setupSpanRecorder(t)

	profile := NewSchedulerProfile().WithPicker(&stubPicker{
		typedName: fwkplugin.TypedName{Type: "noop", Name: "noop"},
		nilResult: true,
	})
	scores := makeWeightedScores("pod1", "pod2")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{RequestID: "r1"}, scores)
	root.End()

	if result != nil {
		t.Fatalf("expected nil result, got %v", result)
	}

	spans := spansByName(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("expected 1 span even on nil result, got %d", len(spans))
	}
	if got := spanAttrs(spans[0])["llm_d.epp.picker.selected_endpoints"].AsInt64(); got != 0 {
		t.Errorf("selected_endpoints = %d, want 0 on nil result", got)
	}
}

func TestPickerSpan_OmitsEmptyGenAIFields(t *testing.T) {
	recorder := setupSpanRecorder(t)

	profile := NewSchedulerProfile().WithPicker(&stubPicker{
		typedName: fwkplugin.TypedName{Type: "p", Name: "n"},
		numPick:   1,
	})
	scores := makeWeightedScores("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	spans := spansByName(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := spanAttrs(spans[0])
	if _, ok := attrs["gen_ai.request.model"]; ok {
		t.Error("gen_ai.request.model should not be set when TargetModel is empty")
	}
	if _, ok := attrs["gen_ai.request.id"]; ok {
		t.Error("gen_ai.request.id should not be set when RequestID is empty")
	}
}

func TestPickerSpan_NilRequest(t *testing.T) {
	recorder := setupSpanRecorder(t)

	profile := NewSchedulerProfile().WithPicker(&stubPicker{
		typedName: fwkplugin.TypedName{Type: "p", Name: "n"},
		numPick:   1,
	})
	scores := makeWeightedScores("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, nil, scores)
	root.End()

	spans := spansByName(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := spanAttrs(spans[0])
	if _, ok := attrs["gen_ai.request.model"]; ok {
		t.Error("gen_ai.request.model should not be set when request is nil")
	}
	if _, ok := attrs["gen_ai.request.id"]; ok {
		t.Error("gen_ai.request.id should not be set when request is nil")
	}
	if got := attrs["llm_d.epp.picker.candidate_endpoints"].AsInt64(); got != 1 {
		t.Errorf("candidate_endpoints = %d, want 1", got)
	}
}

func TestPickerSpan_ChildSpanNests(t *testing.T) {
	recorder := setupSpanRecorder(t)

	profile := NewSchedulerProfile().WithPicker(&stubPicker{
		typedName: fwkplugin.TypedName{Type: "nested", Name: "nested"},
		numPick:   1,
		emitSpan:  true,
	})
	scores := makeWeightedScores("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	pickerSpans := spansByName(recorder.Ended(), "pick_endpoints")
	childSpans := spansByName(recorder.Ended(), "child_picker_op")
	if len(pickerSpans) != 1 || len(childSpans) != 1 {
		t.Fatalf("expected 1 pick_endpoints and 1 child span, got %d and %d", len(pickerSpans), len(childSpans))
	}
	if childSpans[0].Parent().SpanID() != pickerSpans[0].SpanContext().SpanID() {
		t.Errorf("child span parent = %v, want pick_endpoints span %v",
			childSpans[0].Parent().SpanID(), pickerSpans[0].SpanContext().SpanID())
	}
}
