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

// setupSpanRecorder installs an in-memory span recorder as the global tracer
// provider and returns it, restoring the previous provider on cleanup.
func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })
	return recorder
}

func findSpans(spans []sdktrace.ReadOnlySpan, name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

func spanAttributes(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value)
	for _, kv := range span.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	return attrs
}

func newTestEndpoints(names ...string) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, len(names))
	for i, name := range names {
		endpoints[i] = fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}}, nil, nil)
	}
	return endpoints
}

type allProfilesTraceHandler struct{}

func (*allProfilesTraceHandler) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "all-profiles-trace-handler"}
}

func (*allProfilesTraceHandler) Pick(_ context.Context, _ *fwksched.InferenceRequest,
	profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult,
) map[string]fwksched.SchedulerProfile {
	if len(profileResults) == 0 {
		return profiles
	}
	return nil
}

func (*allProfilesTraceHandler) ProcessResults(_ context.Context, _ *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult,
) (*fwksched.SchedulingResult, error) {
	return &fwksched.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: "decode",
	}, nil
}

type panicTraceFilter struct{}

func (*panicTraceFilter) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "panic-trace-filter"}
}

func (*panicTraceFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest,
	_ []fwksched.Endpoint,
) []fwksched.Endpoint {
	panic("filter panic")
}

func TestScheduleNestsFilterSpansUnderNamedProfiles(t *testing.T) {
	recorder := setupSpanRecorder(t)
	endpoints := newTestEndpoints("pod1")

	newProfile := func(name string) *SchedulerProfile {
		plugin := &testPlugin{
			typedName: fwkplugin.TypedName{Type: "test-plugin", Name: name},
			FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
			PickRes:   k8stypes.NamespacedName{Name: "pod1"},
		}
		return NewSchedulerProfile().WithFilters(plugin).WithPicker(plugin)
	}

	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(
		&allProfilesTraceHandler{},
		map[string]fwksched.SchedulerProfile{
			"decode":  newProfile("decode"),
			"prefill": newProfile("prefill"),
		},
	))

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	_, err := scheduler.Schedule(ctx, &fwksched.InferenceRequest{TargetModel: "m1", RequestID: "r1"}, endpoints)
	root.End()
	if err != nil {
		t.Fatalf("Schedule returned error: %v", err)
	}

	profileSpans := findSpans(recorder.Ended(), "run_scheduler_profile")
	if len(profileSpans) != 2 {
		t.Fatalf("got %d run_scheduler_profile spans, want 2", len(profileSpans))
	}
	profileNameBySpanID := make(map[trace.SpanID]string, len(profileSpans))
	for _, profileSpan := range profileSpans {
		profileName := spanAttributes(profileSpan)["llm_d.epp.scheduling.profile.name"].AsString()
		profileNameBySpanID[profileSpan.SpanContext().SpanID()] = profileName
	}

	filterSpans := findSpans(recorder.Ended(), "filter_endpoints")
	if len(filterSpans) != 2 {
		t.Fatalf("got %d filter_endpoints spans, want 2", len(filterSpans))
	}
	seenProfiles := make(map[string]bool, len(filterSpans))
	for _, filterSpan := range filterSpans {
		profileName, ok := profileNameBySpanID[filterSpan.Parent().SpanID()]
		if !ok {
			t.Errorf("filter_endpoints parent %v is not a run_scheduler_profile span", filterSpan.Parent().SpanID())
			continue
		}
		seenProfiles[profileName] = true
	}
	for _, profileName := range []string{"decode", "prefill"} {
		if !seenProfiles[profileName] {
			t.Errorf("no filter_endpoints span attributed to profile %q", profileName)
		}
	}
}

func TestScheduleEndsProfileSpanWhenFilterPanics(t *testing.T) {
	recorder := setupSpanRecorder(t)
	endpoints := newTestEndpoints("pod1")
	picker := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-picker"},
		PickRes:   k8stypes.NamespacedName{Name: "pod1"},
	}
	profile := NewSchedulerProfile().WithFilters(&panicTraceFilter{}).WithPicker(picker)
	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(
		&allProfilesTraceHandler{},
		map[string]fwksched.SchedulerProfile{"decode": profile},
	))

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	recovered := func() (value any) {
		defer func() { value = recover() }()
		_, _ = scheduler.Schedule(ctx, &fwksched.InferenceRequest{TargetModel: "m1", RequestID: "r1"}, endpoints)
		return nil
	}()
	root.End()
	if recovered != "filter panic" {
		t.Fatalf("recovered panic = %v, want filter panic", recovered)
	}

	profileSpans := findSpans(recorder.Ended(), "run_scheduler_profile")
	if len(profileSpans) != 1 {
		t.Fatalf("got %d run_scheduler_profile spans after panic, want 1", len(profileSpans))
	}
	filterSpans := findSpans(recorder.Ended(), "filter_endpoints")
	if len(filterSpans) != 1 {
		t.Fatalf("got %d filter_endpoints spans after panic, want 1", len(filterSpans))
	}
	if filterSpans[0].Parent().SpanID() != profileSpans[0].SpanContext().SpanID() {
		t.Errorf("filter_endpoints parent = %v, want profile span %v",
			filterSpans[0].Parent().SpanID(), profileSpans[0].SpanContext().SpanID())
	}
	if got := spanAttributes(profileSpans[0])["llm_d.epp.scheduling.profile.name"].AsString(); got != "decode" {
		t.Errorf("profile name = %q, want decode", got)
	}
}

func TestRunFilterPluginsSingleSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)

	filter := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-filter", Name: "instance-a"},
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	profile := NewSchedulerProfile().WithFilters(filter)
	endpoints := newTestEndpoints("pod1", "pod2", "pod3")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runFilterPlugins(ctx, &fwksched.InferenceRequest{TargetModel: "m1", RequestID: "r1"}, endpoints)
	root.End()

	if len(result) != 2 {
		t.Fatalf("runFilterPlugins returned %d endpoints, want 2", len(result))
	}

	spans := findSpans(recorder.Ended(), "filter_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d filter_endpoints spans, want 1", len(spans))
	}
	span := spans[0]
	if span.SpanKind() != trace.SpanKindInternal {
		t.Errorf("span kind = %v, want Internal", span.SpanKind())
	}
	if span.Parent().SpanID() != root.SpanContext().SpanID() {
		t.Errorf("parent span ID = %v, want root %v", span.Parent().SpanID(), root.SpanContext().SpanID())
	}

	attrs := spanAttributes(span)
	if got := attrs["llm_d.epp.filter.candidate_endpoints"].AsInt64(); got != 3 {
		t.Errorf("candidate_endpoints = %d, want 3", got)
	}
	if got := attrs["llm_d.epp.filter.filtered_endpoints"].AsInt64(); got != 2 {
		t.Errorf("filtered_endpoints = %d, want 2", got)
	}
	if got := attrs["gen_ai.request.model"].AsString(); got != "m1" {
		t.Errorf("gen_ai.request.model = %q, want %q", got, "m1")
	}
	if got := attrs["gen_ai.request.id"].AsString(); got != "r1" {
		t.Errorf("gen_ai.request.id = %q, want %q", got, "r1")
	}
}

// A filter chain emits one span over the whole stage: candidate is the input to
// the first filter, filtered is the output of the last filter that ran.
func TestRunFilterPluginsChainEmitsOneSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)

	filterA := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "filter-a", Name: "a"},
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	filterB := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "filter-b", Name: "b"},
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	profile := NewSchedulerProfile().WithFilters(filterA, filterB)
	endpoints := newTestEndpoints("pod1", "pod2", "pod3")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runFilterPlugins(ctx, &fwksched.InferenceRequest{}, endpoints)
	root.End()

	if len(result) != 1 {
		t.Fatalf("runFilterPlugins returned %d endpoints, want 1", len(result))
	}
	spans := findSpans(recorder.Ended(), "filter_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d filter_endpoints spans, want 1 for the whole chain", len(spans))
	}
	attrs := spanAttributes(spans[0])
	if got := attrs["llm_d.epp.filter.candidate_endpoints"].AsInt64(); got != 3 {
		t.Errorf("candidate_endpoints = %d, want 3", got)
	}
	if got := attrs["llm_d.epp.filter.filtered_endpoints"].AsInt64(); got != 1 {
		t.Errorf("filtered_endpoints = %d, want 1", got)
	}
}

func TestRunFilterPluginsDrainBreakStillEndsSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)

	drain := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "drain-filter", Name: "drain"},
		FilterRes: []k8stypes.NamespacedName{},
	}
	never := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "never-filter", Name: "never"},
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	profile := NewSchedulerProfile().WithFilters(drain, never)
	endpoints := newTestEndpoints("pod1", "pod2")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runFilterPlugins(ctx, &fwksched.InferenceRequest{}, endpoints)
	root.End()

	if len(result) != 0 {
		t.Fatalf("runFilterPlugins returned %d endpoints, want 0", len(result))
	}
	spans := findSpans(recorder.Ended(), "filter_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d filter_endpoints spans, want 1 (span must end on drain)", len(spans))
	}
	if got := spanAttributes(spans[0])["llm_d.epp.filter.filtered_endpoints"].AsInt64(); got != 0 {
		t.Errorf("filtered_endpoints = %d, want 0", got)
	}
	if never.FilterCallCount != 0 {
		t.Errorf("second filter ran %d times after drain, want 0", never.FilterCallCount)
	}
}

// childSpanFilter starts a child span from the context it is given, so a test
// can assert the delegate runs inside the filter span.
type childSpanFilter struct{ typedName fwkplugin.TypedName }

func (f *childSpanFilter) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *childSpanFilter) Filter(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	_, span := otel.Tracer("test").Start(ctx, "inner_filter_span")
	span.End()
	return endpoints
}

func TestRunFilterPluginsNestsDelegateSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)

	filter := &childSpanFilter{typedName: fwkplugin.TypedName{Type: "child", Name: "c"}}
	profile := NewSchedulerProfile().WithFilters(filter)
	endpoints := newTestEndpoints("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runFilterPlugins(ctx, &fwksched.InferenceRequest{}, endpoints)
	root.End()

	outer := findSpans(recorder.Ended(), "filter_endpoints")
	inner := findSpans(recorder.Ended(), "inner_filter_span")
	if len(outer) != 1 || len(inner) != 1 {
		t.Fatalf("got %d filter_endpoints and %d inner spans, want 1 each", len(outer), len(inner))
	}
	if inner[0].Parent().SpanID() != outer[0].SpanContext().SpanID() {
		t.Errorf("inner span parent = %v, want filter_endpoints span %v",
			inner[0].Parent().SpanID(), outer[0].SpanContext().SpanID())
	}
}

func TestRunFilterPluginsOmitsEmptyGenAI(t *testing.T) {
	recorder := setupSpanRecorder(t)

	filter := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "f", Name: "n"},
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	profile := NewSchedulerProfile().WithFilters(filter)
	endpoints := newTestEndpoints("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runFilterPlugins(ctx, &fwksched.InferenceRequest{}, endpoints)
	root.End()

	spans := findSpans(recorder.Ended(), "filter_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d filter_endpoints spans, want 1", len(spans))
	}
	attrs := spanAttributes(spans[0])
	if _, ok := attrs["gen_ai.request.model"]; ok {
		t.Error("gen_ai.request.model set for empty TargetModel")
	}
	if _, ok := attrs["gen_ai.request.id"]; ok {
		t.Error("gen_ai.request.id set for empty RequestID")
	}
}
