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

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/loraaffinity"
	schedprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
)

// Tests the default scheduler configuration and expected behavior.
func TestSchedule(t *testing.T) {
	kvCacheUtilizationScorer := kvcacheutilization.NewKVCacheUtilizationScorer()
	queueingScorer := queuedepth.NewQueueScorer()
	prefixCacheScorer, err := schedprefix.New(context.Background())
	assert.NoError(t, err)
	loraAffinityScorer := loraaffinity.NewLoraAffinityScorer()

	defaultProfile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(kvCacheUtilizationScorer, 1),
			NewWeightedScorer(queueingScorer, 1),
			NewWeightedScorer(prefixCacheScorer, 1),
			NewWeightedScorer(loraAffinityScorer, 1),
		).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	profileHandler := single.NewSingleProfileHandler()

	schedulerConfig := NewSchedulerConfig(profileHandler, map[string]fwksched.SchedulerProfile{"default": defaultProfile})

	tests := []struct {
		name    string
		req     *fwksched.InferenceRequest
		input   []fwksched.Endpoint
		wantRes *fwksched.SchedulingResult
		err     bool
	}{
		{
			name: "no candidate endpoints",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "any-model",
			},
			input:   []fwksched.Endpoint{},
			wantRes: nil,
			err:     true,
		},
		{
			name: "finds optimal endpoint",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
			},
			// pod2 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					&fwkdl.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					}, nil),
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					&fwkdl.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
					}, nil),
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
					&fwkdl.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.8,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					}, nil),
			},
			wantRes: &fwksched.SchedulingResult{
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"default": {
						TargetEndpoints: []fwksched.Endpoint{
							&fwksched.ScoredEndpoint{
								Endpoint: fwksched.NewEndpoint(
									&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
									&fwkdl.Metrics{
										WaitingQueueSize:    0,
										KVCacheUsagePercent: 0.2,
										MaxActiveModels:     2,
										ActiveModels: map[string]int{
											"foo":      1,
											"critical": 1,
										},
									}, nil),
								Score: 2.8,
							},
						},
					},
				},
				PrimaryProfileName: "default",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheduler := NewSchedulerWithConfig(schedulerConfig)
			got, err := scheduler.Schedule(context.Background(), test.req, test.input)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.wantRes, got, cmp.Comparer(fwksched.ScoredEndpointComparer)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

// TestSchedule_FallbackFiresWhenPrimaryHasNoSignal verifies that when a primary
// profile's Run reports NoSignal=true, the scheduler runs the configured fallback
// profile and substitutes its result before ProfileHandler.ProcessResults is
// called. See #1139.
func TestSchedule_FallbackFiresWhenPrimaryHasNoSignal(t *testing.T) {
	zeroScorer := &testPlugin{
		TypeRes:   "zero-scorer",
		ScoreRes:  0,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	primaryPicker := &testPlugin{
		TypeRes: "primary-picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}
	primary := NewSchedulerProfile().
		WithFilters(zeroScorer).
		WithScorers(NewWeightedScorer(zeroScorer, 1)).
		WithPicker(primaryPicker)

	fallbackScorer := &testPlugin{
		TypeRes:   "fallback-scorer",
		ScoreRes:  0.5,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	fallbackPicker := &testPlugin{
		TypeRes: "fallback-picker",
		PickRes: k8stypes.NamespacedName{Name: "pod2"},
	}
	fallback := NewSchedulerProfile().
		WithFilters(fallbackScorer).
		WithScorers(NewWeightedScorer(fallbackScorer, 1)).
		WithPicker(fallbackPicker)

	cfg := NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": primary},
	).WithFallbacks(map[string]fwksched.SchedulerProfile{
		"default": fallback,
	})
	scheduler := NewSchedulerWithConfig(cfg)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
	}
	req := &fwksched.InferenceRequest{TargetModel: "test", RequestID: uuid.NewString()}

	result, err := scheduler.Schedule(context.Background(), req, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if fallbackPicker.PickCallCount != 1 {
		t.Errorf("fallback picker should run exactly once; got %d calls", fallbackPicker.PickCallCount)
	}

	got := result.ProfileResults["default"].TargetEndpoints[0].GetMetadata().NamespacedName.Name
	if got != "pod2" {
		t.Errorf("expected fallback's pick (pod2) under primary name, got %s", got)
	}
}

// TestSchedule_FallbackDoesNotFireWhenPrimaryHasSignal verifies that the
// fallback is not invoked when the primary profile produces non-zero scores,
// even if a fallback is configured.
func TestSchedule_FallbackDoesNotFireWhenPrimaryHasSignal(t *testing.T) {
	primaryScorer := &testPlugin{
		TypeRes:   "primary-scorer",
		ScoreRes:  0.5,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	primaryPicker := &testPlugin{
		TypeRes: "primary-picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}
	primary := NewSchedulerProfile().
		WithFilters(primaryScorer).
		WithScorers(NewWeightedScorer(primaryScorer, 1)).
		WithPicker(primaryPicker)

	fallbackPicker := &testPlugin{
		TypeRes: "fallback-picker",
		PickRes: k8stypes.NamespacedName{Name: "pod2"},
	}
	fallback := NewSchedulerProfile().
		WithFilters(primaryScorer).
		WithScorers(NewWeightedScorer(primaryScorer, 1)).
		WithPicker(fallbackPicker)

	cfg := NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": primary},
	).WithFallbacks(map[string]fwksched.SchedulerProfile{
		"default": fallback,
	})
	scheduler := NewSchedulerWithConfig(cfg)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
	}
	req := &fwksched.InferenceRequest{TargetModel: "test", RequestID: uuid.NewString()}

	result, err := scheduler.Schedule(context.Background(), req, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if fallbackPicker.PickCallCount != 0 {
		t.Errorf("fallback picker should NOT run when primary has signal; got %d calls", fallbackPicker.PickCallCount)
	}
	if primaryPicker.PickCallCount != 1 {
		t.Errorf("primary picker should run; got %d calls", primaryPicker.PickCallCount)
	}

	got := result.ProfileResults["default"].TargetEndpoints[0].GetMetadata().NamespacedName.Name
	if got != "pod1" {
		t.Errorf("expected primary's pick (pod1), got %s", got)
	}
}

// TestSchedule_NoFallbackConfigured verifies that when no fallback is registered
// for a primary, the scheduler returns the primary's result even with NoSignal=true
// (preserves pre-fallback behavior).
func TestSchedule_NoFallbackConfigured(t *testing.T) {
	zeroScorer := &testPlugin{
		TypeRes:   "zero-scorer",
		ScoreRes:  0,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	primaryPicker := &testPlugin{
		TypeRes: "primary-picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}
	primary := NewSchedulerProfile().
		WithFilters(zeroScorer).
		WithScorers(NewWeightedScorer(zeroScorer, 1)).
		WithPicker(primaryPicker)

	cfg := NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": primary},
	) // no WithFallbacks
	scheduler := NewSchedulerWithConfig(cfg)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
	}
	req := &fwksched.InferenceRequest{TargetModel: "test", RequestID: uuid.NewString()}

	result, err := scheduler.Schedule(context.Background(), req, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if primaryPicker.PickCallCount != 1 {
		t.Errorf("primary picker should run; got %d calls", primaryPicker.PickCallCount)
	}

	got := result.ProfileResults["default"].TargetEndpoints[0].GetMetadata().NamespacedName.Name
	if got != "pod1" {
		t.Errorf("expected primary's pick (pod1) when no fallback configured, got %s", got)
	}
}
