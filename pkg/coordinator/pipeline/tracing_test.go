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
	"net/http"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
)

// setupSpanRecorder installs an in-memory recorder as the process tracer
// provider for the duration of the test.
func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()

	recorder := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	return recorder
}

// spanNamed returns the single ended span with the given name.
func spanNamed(t *testing.T, recorder *tracetest.SpanRecorder, name string) sdktrace.ReadOnlySpan {
	t.Helper()

	var found sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.Name() != name {
			continue
		}
		if found != nil {
			t.Fatalf("expected one span named %q, got more than one", name)
		}
		found = span
	}
	if found == nil {
		t.Fatalf("no span named %q, got %v", name, spanNames(recorder))
	}
	return found
}

func spanNames(recorder *tracetest.SpanRecorder) []string {
	names := make([]string, 0, len(recorder.Ended()))
	for _, span := range recorder.Ended() {
		names = append(names, span.Name())
	}
	return names
}

func attrValue(t *testing.T, span sdktrace.ReadOnlySpan, key attribute.Key) attribute.Value {
	t.Helper()

	for _, attr := range span.Attributes() {
		if attr.Key == key {
			return attr.Value
		}
	}
	t.Fatalf("span %q carries no %s attribute", span.Name(), key)
	return attribute.Value{}
}

// Each step runs under its own span, and those nest under the pipeline span so
// a trace shows where a request spent its time.
func TestExecute_StepSpansNestUnderPipelineSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
		&mockStep{name: "decode", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
	}

	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parent := spanNamed(t, recorder, pipelineSpanName)
	for _, name := range []string{"render", "decode"} {
		step := spanNamed(t, recorder, name)
		if got := step.Parent().SpanID(); got != parent.SpanContext().SpanID() {
			t.Errorf("step %q parent = %s, want the pipeline span %s", name, got, parent.SpanContext().SpanID())
		}
		if got := step.SpanKind(); got != trace.SpanKindInternal {
			t.Errorf("step %q kind = %v, want %v", name, got, trace.SpanKindInternal)
		}
	}
}

// The pipeline span carries the request-level attributes, including the path
// that is only known once the steps have run.
func TestExecute_PipelineSpanAttributes(t *testing.T) {
	recorder := setupSpanRecorder(t)
	steps := []Step{
		&mockStep{name: "prefill", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
		&mockStep{name: "decode", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
	}

	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	span := spanNamed(t, recorder, pipelineSpanName)
	if got := attrValue(t, span, semconv.GenAIRequestModelKey).AsString(); got != "m" {
		t.Errorf("model = %q, want %q", got, "m")
	}
	if got := attrValue(t, span, semconv.LLMDCoordinatorPipelineStepCountKey).AsInt64(); got != 2 {
		t.Errorf("step_count = %d, want 2", got)
	}
	if got := attrValue(t, span, semconv.LLMDCoordinatorPipelineExecutionPathKey).AsString(); got != coordmetrics.PathPrefillDecode {
		t.Errorf("execution_path = %q, want %q", got, coordmetrics.PathPrefillDecode)
	}
}

// A failing step marks both its own span and the pipeline span, so a search
// for failed traces finds the request and identifies the step that failed.
func TestExecute_FailureMarksStepAndPipelineSpans(t *testing.T) {
	recorder := setupSpanRecorder(t)
	steps := []Step{
		&mockStep{name: "prefill", fn: func(_ context.Context, _ *RequestContext) error {
			return &UpstreamError{Step: "prefill", StatusCode: http.StatusServiceUnavailable}
		}},
	}

	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err == nil {
		t.Fatal("expected error")
	}

	if got := spanNamed(t, recorder, "prefill").Status().Code; got != codes.Error {
		t.Errorf("step span status = %v, want %v", got, codes.Error)
	}
	if got := spanNamed(t, recorder, pipelineSpanName).Status().Code; got != codes.Error {
		t.Errorf("pipeline span status = %v, want %v", got, codes.Error)
	}
}

// The response body of an upstream failure can hold prompt data. UpstreamError
// keeps it out of Error(), and the span status must not reintroduce it.
func TestExecute_SpanStatusOmitsUpstreamBody(t *testing.T) {
	recorder := setupSpanRecorder(t)
	const secret = "prompt-text-that-must-not-leak"
	steps := []Step{
		&mockStep{name: "prefill", fn: func(_ context.Context, _ *RequestContext) error {
			return &UpstreamError{Step: "prefill", StatusCode: http.StatusBadRequest, Body: secret}
		}},
	}

	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err == nil {
		t.Fatal("expected error")
	}

	for _, name := range []string{"prefill", pipelineSpanName} {
		if desc := spanNamed(t, recorder, name).Status().Description; strings.Contains(desc, secret) {
			t.Errorf("span %q status description leaked the upstream body: %q", name, desc)
		}
	}
}

// A conditional-decode cache hit ends the pipeline early and succeeds, so
// neither span may be marked as failed.
func TestExecute_ErrPipelineDoneLeavesSpansUnset(t *testing.T) {
	recorder := setupSpanRecorder(t)
	steps := []Step{
		&mockStep{name: "conditional-decode", fn: func(_ context.Context, _ *RequestContext) error {
			return ErrPipelineDone
		}},
	}

	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
		t.Fatalf("expected nil (clean early exit), got %v", err)
	}

	for _, name := range []string{"conditional-decode", pipelineSpanName} {
		if got := spanNamed(t, recorder, name).Status().Code; got != codes.Unset {
			t.Errorf("span %q status = %v, want %v", name, got, codes.Unset)
		}
	}
}

// A panicking step still has to end its span, otherwise the step is missing
// from the trace of the request that crashed. The pipeline span is marked as
// well, matching the error-return path.
func TestExecute_PanicMarksStepAndPipelineSpans(t *testing.T) {
	recorder := setupSpanRecorder(t)
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, _ *RequestContext) error {
			panic("boom")
		}},
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected the panic to propagate")
		}
		span := spanNamed(t, recorder, "render")
		if got := span.Status().Code; got != codes.Error {
			t.Errorf("step span status = %v, want %v", got, codes.Error)
		}
		if !span.EndTime().After(span.StartTime()) {
			t.Error("step span was not ended")
		}
		if got := spanNamed(t, recorder, pipelineSpanName).Status().Code; got != codes.Error {
			t.Errorf("pipeline span status = %v, want %v", got, codes.Error)
		}
	}()

	_ = New(steps).Execute(context.Background(), &RequestContext{Model: "m"})
}

// Steps must see their span in their context, since that is what nests their
// outbound calls into the request's trace.
func TestExecute_StepContextCarriesItsSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)

	var seen trace.SpanContext
	steps := []Step{
		&mockStep{name: "decode", fn: func(ctx context.Context, _ *RequestContext) error {
			seen = trace.SpanContextFromContext(ctx)
			return nil
		}},
	}

	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := seen.SpanID(); got != spanNamed(t, recorder, "decode").SpanContext().SpanID() {
		t.Errorf("step saw span %s, want its own span %s", got, spanNamed(t, recorder, "decode").SpanContext().SpanID())
	}
}

// An empty pipeline still opens its span, and opens nothing else.
func TestExecute_EmptyPipelineEmitsOnlyItsOwnSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)

	if err := New(nil).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if names := spanNames(recorder); len(names) != 1 || names[0] != pipelineSpanName {
		t.Errorf("spans = %v, want just the pipeline span", names)
	}
}

// An unset model is left off the span, as the EPP does, so traces filtered by
// model do not match an empty value.
func TestExecute_EmptyModelOmitsModelAttribute(t *testing.T) {
	recorder := setupSpanRecorder(t)

	if err := New(nil).Execute(context.Background(), &RequestContext{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, attr := range spanNamed(t, recorder, pipelineSpanName).Attributes() {
		if attr.Key == semconv.GenAIRequestModelKey {
			t.Errorf("%s set for an empty model", semconv.GenAIRequestModelKey)
		}
	}
}
