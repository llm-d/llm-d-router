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

// setupPickerSpanRecorder installs an in-memory span recorder as the global
// tracer provider and returns it, restoring the previous provider on cleanup.
func setupPickerSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })
	return recorder
}

func findPickerSpans(spans []sdktrace.ReadOnlySpan, name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

func pickerSpanAttributes(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value)
	for _, kv := range span.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	return attrs
}

func newWeightedScores(names ...string) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(names))
	for i, name := range names {
		endpoint := fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}}, nil, nil)
		scores[endpoint] = float64(i)
	}
	return scores
}

// fakePicker returns the first selected candidates as targets, or a nil result
// when nilResult is set, and optionally emits a child span from its context.
type fakePicker struct {
	typedName fwkplugin.TypedName
	selected  int
	nilResult bool
	emitChild bool
}

var _ fwksched.Picker = &fakePicker{}

func (f *fakePicker) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *fakePicker) Pick(ctx context.Context, scoredPods []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	if f.emitChild {
		_, span := otel.Tracer("test").Start(ctx, "inner_picker_span")
		span.End()
	}
	if f.nilResult {
		return nil
	}
	targets := make([]fwksched.Endpoint, 0, f.selected)
	for i := 0; i < f.selected && i < len(scoredPods); i++ {
		targets = append(targets, scoredPods[i].Endpoint)
	}
	return &fwksched.ProfileRunResult{TargetEndpoints: targets}
}

func TestRunPickerPluginSingleSpan(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "max-score", Name: "instance-a"}, selected: 1}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1", "pod2", "pod3")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{TargetModel: "m1", RequestID: "r1"}, scores)
	root.End()

	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatalf("runPickerPlugin returned %v, want 1 target", result)
	}

	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	span := spans[0]
	if span.SpanKind() != trace.SpanKindInternal {
		t.Errorf("span kind = %v, want Internal", span.SpanKind())
	}
	if span.Parent().SpanID() != root.SpanContext().SpanID() {
		t.Errorf("parent span ID = %v, want root %v", span.Parent().SpanID(), root.SpanContext().SpanID())
	}

	attrs := pickerSpanAttributes(span)
	for _, tc := range []struct {
		key  attribute.Key
		want string
	}{
		{"gen_ai.request.model", "m1"},
		{"gen_ai.request.id", "r1"},
	} {
		if got := attrs[tc.key].AsString(); got != tc.want {
			t.Errorf("%s = %q, want %q", tc.key, got, tc.want)
		}
	}
	if got := attrs["llm_d.epp.picker.candidate_endpoints"].AsInt64(); got != 3 {
		t.Errorf("candidate_endpoints = %d, want 3", got)
	}
	if got := attrs["llm_d.epp.picker.selected_endpoints"].AsInt64(); got != 1 {
		t.Errorf("selected_endpoints = %d, want 1", got)
	}
}

func TestRunPickerPluginMultipleSelected(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "max-score", Name: "multi"}, selected: 2}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1", "pod2", "pod3")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	if result == nil || len(result.TargetEndpoints) != 2 {
		t.Fatalf("runPickerPlugin returned %v, want 2 targets", result)
	}
	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	if got := pickerSpanAttributes(spans[0])["llm_d.epp.picker.selected_endpoints"].AsInt64(); got != 2 {
		t.Errorf("selected_endpoints = %d, want 2 (len(TargetEndpoints))", got)
	}
}

func TestRunPickerPluginNilResult(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "noop-picker", Name: "noop"}, nilResult: true}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1", "pod2")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	if result != nil {
		t.Fatalf("runPickerPlugin returned %v, want nil", result)
	}
	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1 (span must end on nil result)", len(spans))
	}
	if got := pickerSpanAttributes(spans[0])["llm_d.epp.picker.selected_endpoints"].AsInt64(); got != 0 {
		t.Errorf("selected_endpoints = %d, want 0", got)
	}
}

func TestRunPickerPluginNestsDelegateSpan(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "child", Name: "c"}, selected: 1, emitChild: true}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	outer := findPickerSpans(recorder.Ended(), "pick_endpoints")
	inner := findPickerSpans(recorder.Ended(), "inner_picker_span")
	if len(outer) != 1 || len(inner) != 1 {
		t.Fatalf("got %d pick_endpoints and %d inner spans, want 1 each", len(outer), len(inner))
	}
	if inner[0].Parent().SpanID() != outer[0].SpanContext().SpanID() {
		t.Errorf("inner span parent = %v, want pick_endpoints span %v",
			inner[0].Parent().SpanID(), outer[0].SpanContext().SpanID())
	}
}

func TestRunPickerPluginOmitsEmptyGenAI(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "p", Name: "n"}, selected: 1}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	attrs := pickerSpanAttributes(spans[0])
	if _, ok := attrs["gen_ai.request.model"]; ok {
		t.Error("gen_ai.request.model set for empty TargetModel")
	}
	if _, ok := attrs["gen_ai.request.id"]; ok {
		t.Error("gen_ai.request.id set for empty RequestID")
	}
}
