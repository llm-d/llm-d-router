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

package scheduling

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	filterEliminatedMetric = "llm_d_epp_scheduler_filter_eliminated_total"
	failuresMetric         = "llm_d_epp_scheduler_failures_total"
	attemptsMetric         = "llm_d_epp_scheduler_attempts_total"
)

func TestRunFilterPluginsRecordsEliminatedEndpoints(t *testing.T) {
	tests := []struct {
		name    string
		filters []fwksched.Filter
		want    map[string]float64
	}{
		{
			name: "each filter counts the endpoints it removed",
			filters: []fwksched.Filter{
				keepFilter("keep-3", "pod1", "pod2", "pod3"),
				&filterOnlyPlugin{typedName: fwkplugin.TypedName{Type: "pass", Name: "pass"}},
				keepFilter("keep-1", "pod1"),
			},
			want: map[string]float64{
				"plugin_name=keep-3,plugin_type=keep,profile=decode": 2,
				"plugin_name=keep-1,plugin_type=keep,profile=decode": 2,
			},
		},
		{
			name: "the filter that empties the set is the last one counted",
			filters: []fwksched.Filter{
				keepFilter("keep-3", "pod1", "pod2", "pod3"),
				keepFilter("drain"),
				keepFilter("keep-1", "pod1"),
			},
			want: map[string]float64{
				"plugin_name=keep-3,plugin_type=keep,profile=decode": 2,
				"plugin_name=drain,plugin_type=keep,profile=decode":  3,
			},
		},
		{
			name: "a filter that returns more endpoints than it received is not counted",
			filters: []fwksched.Filter{
				keepFilter("keep-1", "pod1"),
				&fixedFilter{
					typedName: fwkplugin.TypedName{Type: "fixed", Name: "fixed"},
					endpoints: newMetricsTestEndpoints("pod1", "pod2"),
				},
			},
			want: map[string]float64{
				"plugin_name=keep-1,plugin_type=keep,profile=decode": 4,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetSchedulerMetrics(t)
			profile := NewSchedulerProfile().
				WithName("decode").
				WithFilters(test.filters...).
				WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
			req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "any-model"}

			require.NotPanics(t, func() {
				_, _ = profile.Run(context.Background(), req, newMetricsTestEndpoints("pod1", "pod2", "pod3", "pod4", "pod5"))
			})
			assert.Equal(t, test.want, counterSeries(t, filterEliminatedMetric))
		})
	}
}

func TestScheduleRecordsFailureReason(t *testing.T) {
	passProfile := NewSchedulerProfile().
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	drainProfile := NewSchedulerProfile().
		WithFilters(keepFilter("drain")).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	tests := []struct {
		name         string
		handler      fwksched.ProfileHandler
		profiles     map[string]fwksched.SchedulerProfile
		wantErr      bool
		wantFailures map[string]float64
	}{
		{
			name:         "a successful attempt records no failure",
			handler:      single.NewSingleProfileHandler(),
			profiles:     map[string]fwksched.SchedulerProfile{"default": passProfile},
			wantFailures: map[string]float64{},
		},
		{
			name:         "filters remove every candidate",
			handler:      single.NewSingleProfileHandler(),
			profiles:     map[string]fwksched.SchedulerProfile{"default": drainProfile},
			wantErr:      true,
			wantFailures: map[string]float64{"reason=all_filtered": 1},
		},
		{
			name:         "the profile handler picks no profile",
			handler:      &stubProfileHandler{pickNone: true},
			profiles:     map[string]fwksched.SchedulerProfile{"default": passProfile},
			wantErr:      true,
			wantFailures: map[string]float64{"reason=no_profile_selected": 1},
		},
		{
			name:         "the profile handler fails after every profile ran",
			handler:      &stubProfileHandler{primary: "default", processErr: errors.New("handler failure")},
			profiles:     map[string]fwksched.SchedulerProfile{"default": passProfile},
			wantErr:      true,
			wantFailures: map[string]float64{"reason=profile_handler_error": 1},
		},
		{
			name:         "a drained profile the handler does not need records no failure",
			handler:      &stubProfileHandler{primary: "decode"},
			profiles:     map[string]fwksched.SchedulerProfile{"decode": passProfile, "prefill": drainProfile},
			wantFailures: map[string]float64{},
		},
		{
			name:         "an undeclared handler write is a handler error even with a drained profile",
			handler:      &stubProfileHandler{primary: "decode", writeUndeclared: true},
			profiles:     map[string]fwksched.SchedulerProfile{"decode": passProfile, "prefill": drainProfile},
			wantErr:      true,
			wantFailures: map[string]float64{"reason=profile_handler_error": 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetSchedulerMetrics(t)
			scheduler := NewSchedulerWithConfig(NewSchedulerConfig(test.handler, test.profiles))
			req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "any-model"}

			_, err := scheduler.Schedule(context.Background(), req, newMetricsTestEndpoints("pod1"))
			assert.Equal(t, test.wantErr, err != nil, "Schedule error: %v", err)

			failures := counterSeries(t, failuresMetric)
			assert.Equal(t, test.wantFailures, failures)
			var failedAttempts, recordedFailures float64
			for labels, value := range counterSeries(t, attemptsMetric) {
				if strings.Contains(labels, "status="+metrics.SchedulerStatusFailure) {
					failedAttempts += value
				}
			}
			for _, value := range failures {
				recordedFailures += value
			}
			assert.InDelta(t, failedAttempts, recordedFailures, 0, "each failed attempt is counted under one reason")
		})
	}
}

func resetSchedulerMetrics(t *testing.T) {
	t.Helper()
	metrics.Register()
	metrics.Reset()
	t.Cleanup(metrics.Reset)
}

// counterSeries returns every series of the named counter, keyed by its label
// pairs in the registry's label-name order.
func counterSeries(t *testing.T, name string) map[string]float64 {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	series := map[string]float64{}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			pairs := make([]string, 0, len(metric.GetLabel()))
			for _, label := range metric.GetLabel() {
				pairs = append(pairs, label.GetName()+"="+label.GetValue())
			}
			series[strings.Join(pairs, ",")] = metric.GetCounter().GetValue()
		}
	}
	return series
}

func newMetricsTestEndpoints(names ...string) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, len(names))
	for i, name := range names {
		endpoints[i] = fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, &fwkdl.Metrics{}, nil)
	}
	return endpoints
}

// keepFilter keeps the named endpoints; with no names it removes every endpoint.
func keepFilter(name string, keep ...string) *testPlugin {
	filterRes := make([]k8stypes.NamespacedName, len(keep))
	for i, endpoint := range keep {
		filterRes[i] = k8stypes.NamespacedName{Name: endpoint}
	}
	return &testPlugin{typedName: fwkplugin.TypedName{Type: "keep", Name: name}, FilterRes: filterRes}
}

// fixedFilter returns the same endpoints whatever it receives.
type fixedFilter struct {
	typedName fwkplugin.TypedName
	endpoints []fwksched.Endpoint
}

func (f *fixedFilter) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *fixedFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) []fwksched.Endpoint {
	return f.endpoints
}

// stubProfileHandler runs every profile once and reports primary as the
// primary profile.
type stubProfileHandler struct {
	pickNone        bool
	primary         string
	processErr      error
	writeUndeclared bool
}

func (h *stubProfileHandler) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "stub-profile-handler", Name: "stub-profile-handler"}
}

func (h *stubProfileHandler) Pick(_ context.Context, _ *fwksched.InferenceRequest, profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult) map[string]fwksched.SchedulerProfile {
	if h.pickNone || len(profileResults) > 0 {
		return nil
	}
	return profiles
}

func (h *stubProfileHandler) ProcessResults(_ context.Context, request *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult) (*fwksched.SchedulingResult, error) {
	if h.processErr != nil {
		return nil, h.processErr
	}
	if profileResults[h.primary] == nil {
		return nil, errors.New("primary profile failed")
	}
	if h.writeUndeclared {
		request.PutAttribute(fwkplugin.NewDataKey("undeclared", "stub-profile-handler"), true)
	}
	return &fwksched.SchedulingResult{ProfileResults: profileResults, PrimaryProfileName: h.primary}, nil
}
