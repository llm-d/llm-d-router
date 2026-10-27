/*
Copyright 2025 The Kubernetes Authors.
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

package scheduling

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraaffinity"
	schedprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
)

// Tests the default scheduler configuration and expected behavior.
func TestSchedule(t *testing.T) {
	kvCacheUtilizationScorer := kvcacheutilization.NewKVCacheUtilizationScorer()
	queueingScorer := queuedepth.NewQueueScorer()
	prefixCacheScorer, err := schedprefix.New(context.Background(), schedprefix.PrefixCacheScorerPluginType, "approx-prefix-cache-producer")
	assert.NoError(t, err)
	loraAffinityScorer := loraaffinity.NewLoraAffinityScorer()
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{
		kvCacheUtilizationScorer, queueingScorer, prefixCacheScorer, loraAffinityScorer,
	})

	defaultProfile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(kvCacheUtilizationScorer, 1),
			NewWeightedScorer(queueingScorer, 1),
			NewWeightedScorer(prefixCacheScorer, 1),
			NewWeightedScorer(loraAffinityScorer, 1),
		).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	profileHandler := single.NewSingleProfileHandler()

	schedulerConfig := NewSchedulerConfig(profileHandler, map[string]fwksched.SchedulerProfile{"default": defaultProfile})
	scrapedAt := time.Now()

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
					&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}},
					&fwkdl.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
						UpdateTime: scrapedAt,
					}, nil),
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}},
					&fwkdl.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
						UpdateTime: scrapedAt,
					}, nil),
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod3"}},
					&fwkdl.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.8,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
						UpdateTime: scrapedAt,
					}, nil),
			},
			wantRes: &fwksched.SchedulingResult{
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"default": {
						TargetEndpoints: []fwksched.Endpoint{
							&fwksched.ScoredEndpoint{
								Endpoint: fwksched.NewEndpoint(
									&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}},
									&fwkdl.Metrics{
										WaitingQueueSize:    0,
										KVCacheUsagePercent: 0.2,
										MaxActiveModels:     2,
										ActiveModels: map[string]int{
											"foo":      1,
											"critical": 1,
										},
										UpdateTime: scrapedAt,
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

			// ScoredCandidates covers the whole candidate set in unspecified order and
			// is asserted in TestSchedulerProfileScoredCandidates.
			if diff := cmp.Diff(test.wantRes, got, cmp.Comparer(fwksched.ScoredEndpointComparer),
				cmpopts.IgnoreFields(fwksched.ProfileRunResult{}, "ScoredCandidates")); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestScheduleFilterDrainReturnsTypedError(t *testing.T) {
	drainingFilter := &testPlugin{typedName: fwkplugin.TypedName{Type: "drain-filter", Name: "drain-filter"}} // empty FilterRes drops every endpoint

	profile := NewSchedulerProfile().
		WithFilters(drainingFilter).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	schedulerConfig := NewSchedulerConfig(single.NewSingleProfileHandler(), map[string]fwksched.SchedulerProfile{"default": profile})
	scheduler := NewSchedulerWithConfig(schedulerConfig)

	req := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "any-model",
	}
	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, nil),
	}

	result, err := scheduler.Schedule(context.Background(), req, input)
	assert.Nil(t, result)
	assert.Error(t, err)

	var typedErr errcommon.Error
	if !errors.As(err, &typedErr) {
		t.Fatalf("Schedule error is not an errcommon.Error: %v", err)
	}
	assert.Equal(t, errcommon.ServiceUnavailable, typedErr.Code)
	assert.Equal(t, string(errcommon.RequestDroppedReasonNoEndpoints), typedErr.Headers[errcommon.RequestDroppedReasonHeaderKey])
}

// messageCaptureSink records the message of every log call at every verbosity.
type messageCaptureSink struct {
	messages []string
}

func (s *messageCaptureSink) Init(_ logr.RuntimeInfo) {}
func (s *messageCaptureSink) Enabled(_ int) bool      { return true }
func (s *messageCaptureSink) Info(_ int, msg string, _ ...any) {
	s.messages = append(s.messages, msg)
}
func (s *messageCaptureSink) Error(_ error, msg string, _ ...any) {
	s.messages = append(s.messages, msg)
}
func (s *messageCaptureSink) WithValues(_ ...any) logr.LogSink { return s }
func (s *messageCaptureSink) WithName(_ string) logr.LogSink   { return s }

func TestScheduleLogsProcessResultsSuccessOnlyWithoutError(t *testing.T) {
	const successMsg = "Completed running profile handler ProcessResults successfully"

	tests := []struct {
		name       string
		filters    []fwksched.Filter
		wantErr    bool
		wantLogged bool
	}{
		{
			name:       "ProcessResults succeeds",
			wantLogged: true,
		},
		{
			name:    "ProcessResults fails",
			filters: []fwksched.Filter{&testPlugin{typedName: fwkplugin.TypedName{Type: "drain-filter", Name: "drain-filter"}}},
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile := NewSchedulerProfile().
				WithFilters(test.filters...).
				WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
			scheduler := NewSchedulerWithConfig(NewSchedulerConfig(single.NewSingleProfileHandler(),
				map[string]fwksched.SchedulerProfile{"default": profile}))

			sink := &messageCaptureSink{}
			ctx := log.IntoContext(context.Background(), logr.New(sink))
			req := &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "any-model",
			}
			input := []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, nil),
			}

			_, err := scheduler.Schedule(ctx, req, input)
			assert.Equal(t, test.wantErr, err != nil, "Schedule error: %v", err)
			assert.Equal(t, test.wantLogged, slices.Contains(sink.messages, successMsg),
				"success message logged; captured messages: %q", sink.messages)
		})
	}
}

// TestScheduleDefaultProfileFilterWithPrefixAffinity runs the default scheduling
// profile end to end: the real utilization detector filters candidates before the
// queue, KV cache, and prefix cache scorers run with the default weights. It pins
// the interaction between the filter and prefix affinity, which the per-plugin
// tests do not cover.
func TestScheduleDefaultProfileFilterWithPrefixAffinity(t *testing.T) {
	const (
		producerName = "approx-prefix-cache-producer"
		queueWeight  = 2.0
		kvWeight     = 2.0
		prefixWeight = 3.0
	)

	handle := fwkplugin.NewEppHandle(t.Context(), func() []k8stypes.NamespacedName { return nil })
	detectorPlugin, err := utilization.UtilizationDetectorFactory(utilization.UtilizationDetectorType, nil, handle)
	require.NoError(t, err)
	detector, ok := detectorPlugin.(fwksched.Filter)
	require.True(t, ok, "the default utilization detector must implement Filter")

	prefixScorer, err := schedprefix.New(t.Context(), schedprefix.PrefixCacheScorerPluginType, producerName)
	require.NoError(t, err)

	profile := NewSchedulerProfile().
		WithFilters(detector).
		WithScorers(
			NewWeightedScorer(queuedepth.NewQueueScorer(), queueWeight),
			NewWeightedScorer(kvcacheutilization.NewKVCacheUtilizationScorer(), kvWeight),
			NewWeightedScorer(prefixScorer, prefixWeight),
		).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": profile}))

	type endpointSpec struct {
		name       string
		queue      int
		kvCache    float64
		prefixFull bool
	}

	tests := []struct {
		name       string
		endpoints  []endpointSpec
		wantPicked string
		wantScores map[string]float64
	}{
		{
			// The full prefix match scores 3.0 and outweighs the queue advantage
			// (weight 2.0) of the idle endpoint while the queue is below the
			// default threshold of 5.
			name: "full prefix match wins below the queue threshold",
			endpoints: []endpointSpec{
				{name: "pod1", queue: 4, kvCache: 0.1, prefixFull: true},
				{name: "pod2", queue: 0, kvCache: 0.1},
			},
			wantPicked: "pod1",
			wantScores: map[string]float64{"pod1": 4.8, "pod2": 3.8},
		},
		{
			// pod1 is removed before scoring, so the queue scorer normalizes over
			// pod2 (queue 0) and pod3 (queue 2) only: pod3 scores 0, not the 0.6
			// it would score if the filtered endpoint still set the range.
			name: "at the queue threshold the prefix match is filtered before scoring",
			endpoints: []endpointSpec{
				{name: "pod1", queue: 5, kvCache: 0.1, prefixFull: true},
				{name: "pod2", queue: 0, kvCache: 0.1},
				{name: "pod3", queue: 2, kvCache: 0.1},
			},
			wantPicked: "pod2",
			wantScores: map[string]float64{"pod2": 3.8, "pod3": 1.8},
		},
		{
			// Every endpoint is at or over the threshold, so the filter fails open
			// and all of them are scored and ranked as usual.
			name: "all endpoints over the threshold are kept and scored",
			endpoints: []endpointSpec{
				{name: "pod1", queue: 5, kvCache: 0.1, prefixFull: true},
				{name: "pod2", queue: 6, kvCache: 0.1},
			},
			wantPicked: "pod1",
			wantScores: map[string]float64{"pod1": 6.8, "pod2": 1.8},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			prefixKey := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName)
			scrapedAt := time.Now()
			endpoints := make([]fwksched.Endpoint, 0, len(test.endpoints))
			for _, spec := range test.endpoints {
				endpoint := fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: spec.name}},
					&fwkdl.Metrics{
						WaitingQueueSize:    spec.queue,
						KVCacheUsagePercent: spec.kvCache,
						UpdateTime:          scrapedAt,
					}, nil)
				matchedBlocks := 0
				if spec.prefixFull {
					matchedBlocks = 10
				}
				endpoint.Put(prefixKey, attrprefix.NewPrefixCacheMatchInfo(matchedBlocks, 10, 1))
				endpoints = append(endpoints, endpoint)
			}

			result, err := scheduler.Schedule(context.Background(), &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "any-model",
			}, endpoints)
			require.NoError(t, err)

			profileResult := result.ProfileResults["default"]
			require.NotNil(t, profileResult)
			require.Len(t, profileResult.TargetEndpoints, 1)
			assert.Equal(t, test.wantPicked, profileResult.TargetEndpoints[0].GetMetadata().ID.Name)

			gotScores := make(map[string]float64, len(profileResult.ScoredCandidates))
			for _, candidate := range profileResult.ScoredCandidates {
				gotScores[candidate.GetMetadata().ID.Name] = candidate.Score
			}
			require.Len(t, gotScores, len(test.wantScores))
			for name, want := range test.wantScores {
				assert.InDelta(t, want, gotScores[name], 1e-9, "score of %s", name)
			}
		})
	}
}
