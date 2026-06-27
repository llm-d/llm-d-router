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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	tracenoop "go.opentelemetry.io/otel/trace/noop"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestSchedulePlugins(t *testing.T) {
	tp1 := &testPlugin{
		TypeRes:   "test1",
		ScoreRes:  0.3,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}, {Name: "pod3"}},
	}
	tp2 := &testPlugin{
		TypeRes:   "test2",
		ScoreRes:  0.8,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	tpFilterAll := &testPlugin{
		TypeRes:   "filter all",
		FilterRes: []k8stypes.NamespacedName{},
	}
	pickerPlugin := &testPlugin{
		TypeRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	tests := []struct {
		name                string
		profile             *SchedulerProfile
		input               []fwksched.Endpoint
		wantTargetEndpoint  k8stypes.NamespacedName
		targetEndpointScore float64
		// Number of expected endpoints to score (after filter)
		numEndpointsToScore int
		err                 bool
	}{
		{
			name: "all plugins executed successfully, all scorers with same weight",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp2).
				WithScorers(NewWeightedScorer(tp1, 1), NewWeightedScorer(tp2, 1)).
				WithPicker(pickerPlugin),
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil),
			},
			wantTargetEndpoint:  k8stypes.NamespacedName{Name: "pod1"},
			targetEndpointScore: 1.1,
			numEndpointsToScore: 2,
			err:                 false,
		},
		{
			name: "all plugins executed successfully, different scorers weights",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp2).
				WithScorers(NewWeightedScorer(tp1, 60), NewWeightedScorer(tp2, 40)).
				WithPicker(pickerPlugin),
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil),
			},
			wantTargetEndpoint:  k8stypes.NamespacedName{Name: "pod1"},
			targetEndpointScore: 50,
			numEndpointsToScore: 2,
			err:                 false,
		},
		{
			name: "filter all",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tpFilterAll).
				WithScorers(NewWeightedScorer(tp1, 1), NewWeightedScorer(tp2, 1)).
				WithPicker(pickerPlugin),
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil),
			},
			numEndpointsToScore: 0,
			err:                 true, // no available endpoints to server after filter all
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Reset all plugins before each new test case.
			for _, plugin := range test.profile.filters {
				plugin.(*testPlugin).reset()
			}
			for _, plugin := range test.profile.scorers {
				plugin.Scorer.(*testPlugin).reset()
			}
			test.profile.picker.(*testPlugin).reset()

			// Initialize the scheduling context
			request := &fwksched.InferenceRequest{
				TargetModel: "test-model",
				RequestID:   uuid.NewString(),
			}
			// Run profile cycle
			got, err := test.profile.Run(context.Background(), request, test.input)

			// Validate error state
			if test.err != (err != nil) {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.err)
			}

			if err != nil {
				return
			}

			// Validate output
			wantRes := &fwksched.ProfileRunResult{
				TargetEndpoints: []fwksched.Endpoint{
					fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: test.wantTargetEndpoint}, nil, nil),
				},
			}

			if diff := cmp.Diff(wantRes, got, cmp.Comparer(fwksched.EndpointComparer)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
			// Validate plugin execution counts dynamically
			for _, plugin := range test.profile.filters {
				tp, _ := plugin.(*testPlugin)
				if tp.FilterCallCount != 1 {
					t.Errorf("Plugin '%s' Filter() called %d times, expected 1", plugin.TypedName(), tp.FilterCallCount)
				}
			}
			for _, plugin := range test.profile.scorers {
				tp, _ := plugin.Scorer.(*testPlugin)
				if tp.ScoreCallCount != 1 {
					t.Errorf("Plugin '%s' Score() called %d times, expected 1", plugin.TypedName(), tp.ScoreCallCount)
				}
				if test.numEndpointsToScore != tp.NumOfScoredEndpoints {
					t.Errorf("Plugin '%s' Score() called with %d pods, expected %d", plugin.TypedName(), tp.NumOfScoredEndpoints, test.numEndpointsToScore)
				}
			}
			tp, _ := test.profile.picker.(*testPlugin)
			if tp.NumOfPickerCandidates != test.numEndpointsToScore {
				t.Errorf("Picker plugin '%s' Pick() called with %d candidates, expected %d", tp.TypedName(), tp.NumOfPickerCandidates, tp.NumOfScoredEndpoints)
			}
			if tp.PickCallCount != 1 {
				t.Errorf("Picker plugin '%s' Pick() called %d times, expected 1", tp.TypedName(), tp.PickCallCount)
			}
			if tp.WinnerEndpointScore != test.targetEndpointScore {
				t.Errorf("winner pod score %v, expected %v", tp.WinnerEndpointScore, test.targetEndpointScore)
			}
		})
	}
}

// compile-time type assertion
var _ fwksched.Filter = &testPlugin{}
var _ fwksched.Scorer = &testPlugin{}
var _ fwksched.Picker = &testPlugin{}

// testPlugin is an implementation useful in unit tests.
type testPlugin struct {
	typedName             fwkplugin.TypedName
	TypeRes               string
	ScoreCallCount        int
	NumOfScoredEndpoints  int
	ScoreRes              float64
	FilterCallCount       int
	FilterRes             []k8stypes.NamespacedName
	PickCallCount         int
	NumOfPickerCandidates int
	PickRes               k8stypes.NamespacedName
	WinnerEndpointScore   float64
}

func (tp *testPlugin) TypedName() fwkplugin.TypedName {
	return tp.typedName
}

func (tp *testPlugin) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

func (tp *testPlugin) Filter(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	tp.FilterCallCount++
	return findEndpoints(endpoints, tp.FilterRes...)

}

func (tp *testPlugin) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	tp.ScoreCallCount++
	scoredEndpoints := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] += tp.ScoreRes
	}
	tp.NumOfScoredEndpoints = len(scoredEndpoints)
	return scoredEndpoints
}

func (tp *testPlugin) Pick(_ context.Context, scoredEndpoints []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	tp.PickCallCount++
	tp.NumOfPickerCandidates = len(scoredEndpoints)

	winnerEndpoints := []fwksched.Endpoint{}
	for _, scoredEndpoint := range scoredEndpoints {
		if scoredEndpoint.GetMetadata().NamespacedName.String() == tp.PickRes.String() {
			winnerEndpoints = append(winnerEndpoints, scoredEndpoint.Endpoint)
			tp.WinnerEndpointScore = scoredEndpoint.Score
		}
	}

	return &fwksched.ProfileRunResult{TargetEndpoints: winnerEndpoints}
}

func (tp *testPlugin) reset() {
	tp.FilterCallCount = 0
	tp.ScoreCallCount = 0
	tp.NumOfScoredEndpoints = 0
	tp.PickCallCount = 0
	tp.NumOfPickerCandidates = 0
}

func TestAddPlugins(t *testing.T) {
	tests := []struct {
		name        string
		plugins     []fwkplugin.Plugin
		wantFilters int
		wantScorers int
		wantPicker  bool
		wantErr     bool
		errContains string
	}{
		{
			name: "add WeightedScorer that also implements Filter and Picker",
			plugins: []fwkplugin.Plugin{
				NewWeightedScorer(&testPlugin{TypeRes: "multi"}, 1.0),
			},
			wantFilters: 1,
			wantScorers: 1,
			wantPicker:  true,
		},
		{
			name: "add plugin that only implements Filter",
			plugins: []fwkplugin.Plugin{
				&filterOnlyPlugin{typedName: fwkplugin.TypedName{Name: "filter-only"}},
			},
			wantFilters: 1,
			wantScorers: 0,
			wantPicker:  false,
		},
		{
			name: "error when adding Scorer without weight",
			plugins: []fwkplugin.Plugin{
				&testPlugin{TypeRes: "bare-scorer"},
			},
			wantErr:     true,
			errContains: "without a weight",
		},
		{
			name: "error when adding duplicate picker",
			plugins: []fwkplugin.Plugin{
				NewWeightedScorer(&testPlugin{TypeRes: "picker1"}, 1.0),
				NewWeightedScorer(&testPlugin{TypeRes: "picker2"}, 1.0),
			},
			wantErr:     true,
			errContains: "already have a registered picker",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile := NewSchedulerProfile()
			err := profile.AddPlugins(test.plugins...)

			if test.wantErr {
				if err == nil {
					t.Fatalf("expected error but got nil")
				}
				if test.errContains != "" && !strings.Contains(err.Error(), test.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), test.errContains)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(profile.filters) != test.wantFilters {
				t.Errorf("got %d filters, want %d", len(profile.filters), test.wantFilters)
			}
			if len(profile.scorers) != test.wantScorers {
				t.Errorf("got %d scorers, want %d", len(profile.scorers), test.wantScorers)
			}
			if test.wantPicker && profile.picker == nil {
				t.Errorf("expected picker to be set")
			}
			if !test.wantPicker && profile.picker != nil {
				t.Errorf("expected picker to be nil")
			}
		})
	}
}

func TestSchedulerProfileString(t *testing.T) {
	tp1 := &testPlugin{TypeRes: "test1"}
	tp2 := &testPlugin{TypeRes: "test2"}
	pickerPlugin := &testPlugin{TypeRes: "picker"}

	profile := NewSchedulerProfile().
		WithFilters(tp1, tp2).
		WithScorers(NewWeightedScorer(tp1, 1.5), NewWeightedScorer(tp2, 2.0)).
		WithPicker(pickerPlugin)

	result := profile.String()

	// Verify the string contains filter, scorer, and picker info
	if !strings.Contains(result, "Filters:") {
		t.Errorf("String() missing Filters section: %s", result)
	}
	if !strings.Contains(result, "Scorers:") {
		t.Errorf("String() missing Scorers section: %s", result)
	}
	if !strings.Contains(result, "Picker:") {
		t.Errorf("String() missing Picker section: %s", result)
	}
}

func TestEnforceScoreRange(t *testing.T) {
	tests := []struct {
		name  string
		score float64
		want  float64
	}{
		{name: "negative score clamped to 0", score: -0.5, want: 0},
		{name: "score above 1 clamped to 1", score: 1.5, want: 1},
		{name: "score at 0 stays 0", score: 0, want: 0},
		{name: "score at 1 stays 1", score: 1, want: 1},
		{name: "score in range stays unchanged", score: 0.5, want: 0.5},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := enforceScoreRange(test.score)
			if got != test.want {
				t.Errorf("enforceScoreRange(%v) = %v, want %v", test.score, got, test.want)
			}
		})
	}
}

func TestRunWithOutOfRangeScores(t *testing.T) {
	// Scorer that returns negative score
	negativeScorer := &testPlugin{
		TypeRes:   "negative",
		ScoreRes:  -0.5,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	// Scorer that returns score > 1
	overScorer := &testPlugin{
		TypeRes:   "over",
		ScoreRes:  1.5,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	pickerPlugin := &testPlugin{
		TypeRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	profile := NewSchedulerProfile().
		WithFilters().
		WithScorers(NewWeightedScorer(negativeScorer, 1), NewWeightedScorer(overScorer, 1)).
		WithPicker(pickerPlugin)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
	}

	request := &fwksched.InferenceRequest{
		TargetModel: "test-model",
		RequestID:   uuid.NewString(),
	}

	_, err := profile.Run(context.Background(), request, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// negative score clamped to 0, over score clamped to 1: total = 0*1 + 1*1 = 1.0
	if pickerPlugin.WinnerEndpointScore != 1.0 {
		t.Errorf("expected winner score 1.0, got %v", pickerPlugin.WinnerEndpointScore)
	}
}

// TestFilterExecutionOrder verifies that filters execute in the order they are
// registered in the scheduling profile. See also TestFilterExecutionOrderFromYAML
// in pkg/epp/config/loader which verifies that YAML declaration order is preserved
// during deserialization.
func TestFilterExecutionOrder(t *testing.T) {
	executionOrder := []string{}

	// Create three order-tracking filters.
	filterA := &orderTrackingFilter{
		name:           "filter-A",
		executionOrder: &executionOrder,
	}
	filterB := &orderTrackingFilter{
		name:           "filter-B",
		executionOrder: &executionOrder,
	}
	filterC := &orderTrackingFilter{
		name:           "filter-C",
		executionOrder: &executionOrder,
	}

	pickerPlugin := &testPlugin{
		TypeRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	// Declare filters in order A, B, C.
	profile := NewSchedulerProfile().
		WithFilters(filterA, filterB, filterC).
		WithPicker(pickerPlugin)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
	}

	request := &fwksched.InferenceRequest{
		TargetModel: "test-model",
		RequestID:   uuid.NewString(),
	}

	_, err := profile.Run(context.Background(), request, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify execution order matches declaration order.
	wantOrder := []string{"filter-A", "filter-B", "filter-C"}
	if diff := cmp.Diff(wantOrder, executionOrder); diff != "" {
		t.Errorf("Filter execution order mismatch (-want +got):\n%s", diff)
	}
}

// TestFilterExecutionOrderViaAddPlugins verifies that filters added via
// AddPlugins (the path used by the config loader) execute in registration order.
func TestFilterExecutionOrderViaAddPlugins(t *testing.T) {
	executionOrder := []string{}

	filterA := &orderTrackingFilter{name: "filter-A", executionOrder: &executionOrder}
	filterB := &orderTrackingFilter{name: "filter-B", executionOrder: &executionOrder}
	filterC := &orderTrackingFilter{name: "filter-C", executionOrder: &executionOrder}

	pickerPlugin := &testPlugin{
		TypeRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	// Use AddPlugins sequentially, as the config loader does.
	profile := NewSchedulerProfile()
	for _, p := range []fwkplugin.Plugin{filterA, filterB, filterC} {
		if err := profile.AddPlugins(p); err != nil {
			t.Fatalf("unexpected error adding plugin: %v", err)
		}
	}
	profile.WithPicker(pickerPlugin)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
	}

	request := &fwksched.InferenceRequest{
		TargetModel: "test-model",
		RequestID:   uuid.NewString(),
	}

	_, err := profile.Run(context.Background(), request, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	wantOrder := []string{"filter-A", "filter-B", "filter-C"}
	if diff := cmp.Diff(wantOrder, executionOrder); diff != "" {
		t.Errorf("Filter execution order mismatch (-want +got):\n%s", diff)
	}
}

// TestFilterChainReceivesPreviousOutput verifies that each filter in the chain
// receives the filtered output of the previous filter, not the original input.
// This confirms filters execute as a sequential pipeline.
func TestFilterChainReceivesPreviousOutput(t *testing.T) {
	// First filter keeps pod1 and pod2 (removes pod3).
	filter1 := &testPlugin{
		TypeRes:   "filter1",
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	// Second filter keeps only pod1 (removes pod2).
	filter2 := &testPlugin{
		TypeRes:   "filter2",
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	// Third filter is a pass-through that records what it received.
	receivedCount := 0
	filter3 := &countingFilter{
		name:          "filter3",
		receivedCount: &receivedCount,
	}

	pickerPlugin := &testPlugin{
		TypeRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	profile := NewSchedulerProfile().
		WithFilters(filter1, filter2, filter3).
		WithPicker(pickerPlugin)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil),
	}

	request := &fwksched.InferenceRequest{
		TargetModel: "test-model",
		RequestID:   uuid.NewString(),
	}

	_, err := profile.Run(context.Background(), request, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// filter3 should have received only 1 endpoint (pod1) — the output of filter2.
	if receivedCount != 1 {
		t.Errorf("third filter received %d endpoints, want 1 (chained output of previous filters)", receivedCount)
	}
}

// orderTrackingFilter records its name into a shared slice when Filter is called.
type orderTrackingFilter struct {
	name           string
	executionOrder *[]string
}

func (f *orderTrackingFilter) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: f.name, Type: f.name}
}

func (f *orderTrackingFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	*f.executionOrder = append(*f.executionOrder, f.name)
	return endpoints // pass-through
}

// countingFilter records how many endpoints it received.
type countingFilter struct {
	name          string
	receivedCount *int
}

func (f *countingFilter) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: f.name, Type: f.name}
}

func (f *countingFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	*f.receivedCount = len(endpoints)
	return endpoints // pass-through
}

// filterOnlyPlugin implements only the Filter interface (not Scorer or Picker).
type filterOnlyPlugin struct {
	typedName fwkplugin.TypedName
}

func (p *filterOnlyPlugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *filterOnlyPlugin) Filter(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	return endpoints
}

// fixedScoresScorer returns a caller-supplied score per endpoint name, letting
// tests assert aggregate span attributes (max/avg) over a non-uniform map.
type fixedScoresScorer struct {
	typedName fwkplugin.TypedName
	scores    map[string]float64
}

func (s *fixedScoresScorer) TypedName() fwkplugin.TypedName { return s.typedName }

func (s *fixedScoresScorer) Category() fwksched.ScorerCategory { return fwksched.Distribution }

func (s *fixedScoresScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	// A nil score table models a scorer that declines to score any endpoint,
	// exercising the empty-map aggregate guard in runScorer.
	if s.scores == nil {
		return map[fwksched.Endpoint]float64{}
	}
	out := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, e := range endpoints {
		out[e] = s.scores[e.GetMetadata().NamespacedName.Name]
	}
	return out
}

// installSpanRecorder routes spans to an in-memory recorder for the duration of
// the test and restores an explicit no-op provider afterward. Restoring the
// no-op provider (rather than the global proxy returned by GetTracerProvider)
// ensures later tests do not inherit this test's recording SDK provider.
func installSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(tracenoop.NewTracerProvider()) })
	return recorder
}

// findSpan returns the first recorded span with the given name, or nil.
func findSpan(spans tracetest.SpanStubs, name string) *tracetest.SpanStub {
	for i := range spans {
		if spans[i].Name == name {
			return &spans[i]
		}
	}
	return nil
}

func spanFloat(t *testing.T, span *tracetest.SpanStub, key string) float64 {
	t.Helper()
	for _, a := range span.Attributes {
		if string(a.Key) == key {
			return a.Value.AsFloat64()
		}
	}
	t.Fatalf("span %q missing attribute %q", span.Name, key)
	return 0
}

func spanInt(t *testing.T, span *tracetest.SpanStub, key string) int64 {
	t.Helper()
	for _, a := range span.Attributes {
		if string(a.Key) == key {
			return a.Value.AsInt64()
		}
	}
	t.Fatalf("span %q missing attribute %q", span.Name, key)
	return 0
}

func spanHasAttr(span *tracetest.SpanStub, key string) bool {
	for _, a := range span.Attributes {
		if string(a.Key) == key {
			return true
		}
	}
	return false
}

// TestRunScorerPluginsTracing verifies the scheduler scoring path emits a parent
// llm_d.epp.scoring span with one llm_d.epp.scorer.<type> child per scorer,
// carrying the documented identity, weight, candidate-count, and aggregate
// score attributes, and no per-endpoint attribute keys.
func TestRunScorerPluginsTracing(t *testing.T) {
	recorder := installSpanRecorder(t)

	scorerA := &fixedScoresScorer{
		typedName: fwkplugin.TypedName{Type: "scorer-a", Name: "a"},
		scores:    map[string]float64{"pod1": 0.2, "pod2": 0.8},
	}
	scorerB := &fixedScoresScorer{
		typedName: fwkplugin.TypedName{Type: "scorer-b", Name: "b"},
		scores:    map[string]float64{"pod1": 0.5, "pod2": 0.5},
	}
	picker := &testPlugin{TypeRes: "picker", PickRes: k8stypes.NamespacedName{Name: "pod1"}}

	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(scorerA, 0.25), NewWeightedScorer(scorerB, 0.75)).
		WithPicker(picker)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
	}
	request := &fwksched.InferenceRequest{TargetModel: "test-model", RequestID: "req-123"}

	if _, err := profile.Run(context.Background(), request, input); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	spans := tracetest.SpanStubsFromReadOnlySpans(recorder.Ended())
	parent := findSpan(spans, "llm_d.epp.scoring")
	if parent == nil {
		t.Fatalf("missing parent span llm_d.epp.scoring; got %d spans", len(spans))
	}
	if got := spanInt(t, parent, "llm_d.epp.scorer.count"); got != 2 {
		t.Errorf("parent llm_d.epp.scorer.count = %d, want 2", got)
	}
	if got := spanInt(t, parent, "llm_d.epp.scoring.candidate_endpoints"); got != 2 {
		t.Errorf("parent candidate_endpoints = %d, want 2", got)
	}

	childA := findSpan(spans, "llm_d.epp.scorer.scorer-a")
	childB := findSpan(spans, "llm_d.epp.scorer.scorer-b")
	if childA == nil || childB == nil {
		t.Fatalf("missing per-scorer child spans (a=%v b=%v)", childA != nil, childB != nil)
	}

	// Both children must nest under the parent scoring span.
	if childA.Parent.SpanID() != parent.SpanContext.SpanID() {
		t.Errorf("scorer-a span is not a child of the scoring span")
	}

	if got := spanFloat(t, childA, "llm_d.epp.scorer.weight"); got != 0.25 {
		t.Errorf("scorer-a weight = %v, want 0.25", got)
	}
	if got := spanInt(t, childA, "llm_d.epp.scorer.candidate_endpoints"); got != 2 {
		t.Errorf("scorer-a candidate_endpoints = %d, want 2", got)
	}
	// scorer-a scores {0.2, 0.8}: max 0.8, avg 0.5.
	if got := spanFloat(t, childA, "llm_d.epp.scorer.score.max"); got != 0.8 {
		t.Errorf("scorer-a score.max = %v, want 0.8", got)
	}
	if got := spanFloat(t, childA, "llm_d.epp.scorer.score.avg"); got != 0.5 {
		t.Errorf("scorer-a score.avg = %v, want 0.5", got)
	}
	if got := spanInt(t, childA, "llm_d.epp.scorer.endpoints_scored"); got != 2 {
		t.Errorf("scorer-a endpoints_scored = %d, want 2", got)
	}

	// Cardinality guard: no per-pod/per-endpoint identifier attribute keys.
	for _, span := range []*tracetest.SpanStub{parent, childA, childB} {
		for _, a := range span.Attributes {
			key := string(a.Key)
			if strings.Contains(key, "pod") || strings.Contains(key, "endpoint.") || strings.Contains(key, "namespacedname") {
				t.Errorf("span %q carries per-endpoint attribute key %q", span.Name, key)
			}
		}
	}
}

// TestRunScorerPluginsTracingDisabled verifies scoring works and records no
// spans when the global provider is the default no-op, and that an empty
// candidate set still emits the parent span without dividing by zero.
func TestRunScorerPluginsTracingDisabled(t *testing.T) {
	// Explicitly install a no-op provider so this test exercises the
	// tracing-disabled path regardless of provider state left by other tests.
	otel.SetTracerProvider(tracenoop.NewTracerProvider())
	t.Cleanup(func() { otel.SetTracerProvider(tracenoop.NewTracerProvider()) })

	scorer := &testPlugin{TypeRes: "noop", ScoreRes: 0.5}
	picker := &testPlugin{TypeRes: "picker", PickRes: k8stypes.NamespacedName{Name: "pod1"}}

	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(scorer, 1)).
		WithPicker(picker)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
	}
	request := &fwksched.InferenceRequest{TargetModel: "test-model", RequestID: uuid.NewString()}

	if _, err := profile.Run(context.Background(), request, input); err != nil {
		t.Fatalf("unexpected error with no-op tracer: %v", err)
	}
	if scorer.ScoreCallCount != 1 {
		t.Errorf("scorer called %d times, want 1", scorer.ScoreCallCount)
	}
}

// TestRunScorerEmptyCandidateAvg verifies a scorer that returns an empty score
// map does not trigger a divide-by-zero when computing the average attribute.
func TestRunScorerEmptyCandidateAvg(t *testing.T) {
	recorder := installSpanRecorder(t)

	empty := &fixedScoresScorer{typedName: fwkplugin.TypedName{Type: "empty", Name: "e"}}
	picker := &testPlugin{TypeRes: "picker", PickRes: k8stypes.NamespacedName{Name: "pod1"}}
	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(empty, 1)).
		WithPicker(picker)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
	}
	request := &fwksched.InferenceRequest{TargetModel: "test-model", RequestID: "req-1"}

	if _, err := profile.Run(context.Background(), request, input); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	child := findSpan(tracetest.SpanStubsFromReadOnlySpans(recorder.Ended()), "llm_d.epp.scorer.empty")
	if child == nil {
		t.Fatal("missing scorer span for empty scorer")
	}
	// With no scored endpoints, aggregate attributes are omitted entirely.
	if spanHasAttr(child, "llm_d.epp.scorer.score.avg") {
		t.Error("empty scorer span should not carry a score.avg attribute")
	}
}

func findEndpoints(endpoints []fwksched.Endpoint, names ...k8stypes.NamespacedName) []fwksched.Endpoint {
	res := []fwksched.Endpoint{}
	for _, endpoint := range endpoints {
		for _, name := range names {
			if endpoint.GetMetadata().NamespacedName.String() == name.String() {
				res = append(res, endpoint)
			}
		}
	}
	return res
}
