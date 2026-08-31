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
	"encoding/json"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/utilization"
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

// Tests that a filter draining the candidate set surfaces a typed capacity
// rejection from Schedule.
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
	assert.Equal(t, errcommon.ResourceExhausted, typedErr.Code)
	assert.Equal(t, string(errcommon.RequestDroppedReasonSaturated), typedErr.Headers[errcommon.RequestDroppedReasonHeaderKey])
}

func newDocumentedDecodeSelector(t *testing.T) fwksched.Filter {
	t.Helper()
	return newLabelSelector(t, "decode-filter", json.RawMessage(`{
		"matchExpressions": [
			{
				"key": "llm-d.ai/role",
				"operator": "In",
				"values": ["decode", "prefill-decode", "encode-prefill-decode"]
			}
		]
	}`))
}

func newLabelSelector(t *testing.T, name string, rawParams json.RawMessage) fwksched.Filter {
	t.Helper()
	plugin, err := bylabel.SelectorFactory(name, fwkplugin.StrictDecoder(rawParams), nil)
	require.NoError(t, err)
	filter, ok := plugin.(fwksched.Filter)
	require.True(t, ok, "label selector factory should return a scheduling filter")
	return filter
}

func newWaitingQueueCapacityFilter(t *testing.T) fwksched.Filter {
	t.Helper()
	rawParams := json.RawMessage(`{"conditions":[{"metric":"waiting-queue","maxValue":0}]}`)
	plugin, err := utilization.Factory("capacity-filter", fwkplugin.StrictDecoder(rawParams), nil)
	require.NoError(t, err)
	filter, ok := plugin.(fwksched.Filter)
	require.True(t, ok, "utilization factory should return a scheduling filter")
	return filter
}

func assertTypedScheduleError(t *testing.T, err error, code string, reason errcommon.RequestDroppedReason) {
	t.Helper()
	require.Error(t, err)
	var typedErr errcommon.Error
	require.True(t, errors.As(err, &typedErr), "Schedule error is not an errcommon.Error: %v", err)
	assert.Equal(t, code, typedErr.Code)
	assert.Equal(t, string(reason), typedErr.Headers[errcommon.RequestDroppedReasonHeaderKey])
}

func TestScheduleEndpointEligibilityClassification(t *testing.T) {
	prefillEndpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     k8stypes.NamespacedName{Name: "prefill-pod"},
		Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
	}, &fwkdl.Metrics{}, nil)
	saturatedDecodeEndpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     k8stypes.NamespacedName{Name: "saturated-decode-pod"},
		Labels: map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
	}, &fwkdl.Metrics{WaitingQueueSize: 1}, nil)

	tests := []struct {
		name    string
		filters func(*testing.T) []fwksched.Filter
		input   []fwksched.Endpoint
		code    string
		reason  errcommon.RequestDroppedReason
	}{
		{
			name: "label selector finds no decode endpoint",
			filters: func(t *testing.T) []fwksched.Filter {
				return []fwksched.Filter{newDocumentedDecodeSelector(t)}
			},
			input:  []fwksched.Endpoint{prefillEndpoint},
			code:   errcommon.ServiceUnavailable,
			reason: errcommon.RequestDroppedReasonNoEndpoints,
		},
		{
			name: "capacity before eligibility",
			filters: func(t *testing.T) []fwksched.Filter {
				return []fwksched.Filter{newWaitingQueueCapacityFilter(t), bylabel.NewDecodeRole()}
			},
			input:  []fwksched.Endpoint{saturatedDecodeEndpoint, prefillEndpoint},
			code:   errcommon.ResourceExhausted,
			reason: errcommon.RequestDroppedReasonSaturated,
		},
		{
			name: "capacity drains the only eligible endpoint",
			filters: func(t *testing.T) []fwksched.Filter {
				return []fwksched.Filter{newWaitingQueueCapacityFilter(t), bylabel.NewDecodeRole()}
			},
			input:  []fwksched.Endpoint{saturatedDecodeEndpoint},
			code:   errcommon.ResourceExhausted,
			reason: errcommon.RequestDroppedReasonSaturated,
		},
		{
			name: "capacity after eligibility",
			filters: func(t *testing.T) []fwksched.Filter {
				return []fwksched.Filter{bylabel.NewDecodeRole(), newWaitingQueueCapacityFilter(t)}
			},
			input:  []fwksched.Endpoint{saturatedDecodeEndpoint, prefillEndpoint},
			code:   errcommon.ResourceExhausted,
			reason: errcommon.RequestDroppedReasonSaturated,
		},
		{
			name: "capacity drains candidates with no original decode endpoint",
			filters: func(t *testing.T) []fwksched.Filter {
				return []fwksched.Filter{newWaitingQueueCapacityFilter(t), bylabel.NewDecodeRole()}
			},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
					ID:     k8stypes.NamespacedName{Name: "saturated-prefill-pod"},
					Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
				}, &fwkdl.Metrics{WaitingQueueSize: 1}, nil),
			},
			code:   errcommon.ServiceUnavailable,
			reason: errcommon.RequestDroppedReasonNoEndpoints,
		},
		{
			name: "eligibility filters have no common endpoint",
			filters: func(t *testing.T) []fwksched.Filter {
				return []fwksched.Filter{
					newDocumentedDecodeSelector(t),
					newLabelSelector(t, "zone-filter", json.RawMessage(`{"matchLabels":{"zone":"b"}}`)),
				}
			},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
					ID:     k8stypes.NamespacedName{Name: "decode-zone-a"},
					Labels: map[string]string{bylabel.RoleLabel: bylabel.RoleDecode, "zone": "a"},
				}, &fwkdl.Metrics{}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
					ID:     k8stypes.NamespacedName{Name: "prefill-zone-b"},
					Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill, "zone": "b"},
				}, &fwkdl.Metrics{}, nil),
			},
			code:   errcommon.ServiceUnavailable,
			reason: errcommon.RequestDroppedReasonNoEndpoints,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile := NewSchedulerProfile().
				WithFilters(test.filters(t)...).
				WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
			schedulerConfig := NewSchedulerConfig(single.NewSingleProfileHandler(), map[string]fwksched.SchedulerProfile{"decode": profile})
			scheduler := NewSchedulerWithConfig(schedulerConfig)
			req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "any-model"}

			result, err := scheduler.Schedule(context.Background(), req, test.input)
			assert.Nil(t, result)
			assertTypedScheduleError(t, err, test.code, test.reason)
		})
	}
}
