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
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling/decisionhistory"
)

func newDecisionHistoryEndpoints() []fwksched.Endpoint {
	return []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, &fwkdl.Metrics{}, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod3"}}, &fwkdl.Metrics{}, nil),
	}
}

// Verifies the decision recorded for a successful run: filter counts, scorer
// identity and weight, per-endpoint scorer scores for the top endpoints, and
// the picked target.
func TestScheduleRecordsDecisionHistory(t *testing.T) {
	keepTwo := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-filter", Name: "keep-two"},
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	scorer := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-scorer", Name: "my-scorer"},
		ScoreRes:  0.5,
	}

	profile := NewSchedulerProfile().
		WithFilters(keepTwo).
		WithScorers(NewWeightedScorer(scorer, 2)).
		WithPicker(maxscore.NewMaxScorePicker(1))

	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": profile},
	))
	recorder := decisionhistory.NewRecorder(4, 2)
	scheduler.WithDecisionRecorder(recorder)

	req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "model-a"}
	result, err := scheduler.Schedule(context.Background(), req, newDecisionHistoryEndpoints())
	require.NoError(t, err)
	require.Equal(t, "pod1", result.PrimaryEndpoint().GetMetadata().ID.Name)

	decisions := recorder.Snapshot()
	require.Len(t, decisions, 1)
	d := decisions[0]
	assert.Equal(t, req.RequestID, d.RequestID)
	assert.Equal(t, "model-a", d.Model)
	assert.Equal(t, "/pod1", d.TargetEndpoint)
	assert.Empty(t, d.FailureReason)

	require.Len(t, d.Profiles, 1)
	p := d.Profiles[0]
	assert.Equal(t, "default", p.Profile)
	assert.Equal(t, 3, p.CandidatesBeforeFilters)
	assert.Equal(t, []decisionhistory.FilterStage{
		{Type: "test-filter", Name: "keep-two", Before: 3, After: 2},
	}, p.Filters)
	assert.Equal(t, []decisionhistory.ScorerEntry{
		{Type: "test-scorer", Name: "my-scorer", Weight: 2},
	}, p.Scorers)
	assert.Equal(t, []string{"/pod1"}, p.TargetEndpoints)
	assert.Empty(t, p.Error)

	// pod1 and pod2 tie at weighted score 1 (0.5 raw x weight 2); the name is
	// the tiebreaker and the cap keeps both.
	require.Len(t, p.TopEndpoints, 2)
	for _, top := range p.TopEndpoints {
		assert.InDelta(t, 1.0, top.WeightedScore, 1e-9)
		assert.Equal(t, map[string]float64{"my-scorer/test-scorer": 0.5}, top.ScorerScores)
	}
	assert.ElementsMatch(t, []string{"/pod1", "/pod2"},
		[]string{p.TopEndpoints[0].Endpoint, p.TopEndpoints[1].Endpoint})
}

// Verifies the decision recorded when a profile's filters drain the
// candidate set: the filter stage shows the drain and the failure reason
// matches the scheduler failure metric.
func TestScheduleRecordsDecisionHistoryFilterDrain(t *testing.T) {
	drainingFilter := &testPlugin{typedName: fwkplugin.TypedName{Type: "test-filter", Name: "drain"}}

	profile := NewSchedulerProfile().
		WithFilters(drainingFilter).
		WithPicker(maxscore.NewMaxScorePicker(1))

	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": profile},
	))
	recorder := decisionhistory.NewRecorder(4, 2)
	scheduler.WithDecisionRecorder(recorder)

	req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "model-a"}
	result, err := scheduler.Schedule(context.Background(), req, newDecisionHistoryEndpoints())
	assert.Nil(t, result)
	require.Error(t, err)

	decisions := recorder.Snapshot()
	require.Len(t, decisions, 1)
	d := decisions[0]
	assert.Equal(t, "all_filtered", d.FailureReason)
	assert.Empty(t, d.TargetEndpoint)

	require.Len(t, d.Profiles, 1)
	p := d.Profiles[0]
	assert.Equal(t, []decisionhistory.FilterStage{
		{Type: "test-filter", Name: "drain", Before: 3, After: 0},
	}, p.Filters)
	assert.Contains(t, p.Error, "no endpoints available")
	assert.Empty(t, p.TopEndpoints)
	assert.Empty(t, p.TargetEndpoints)
}

// A scheduler without a recorder must behave exactly as before: no collector
// reaches the plugin call sites.
func TestScheduleWithoutDecisionRecorder(t *testing.T) {
	scorer := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-scorer", Name: "my-scorer"},
		ScoreRes:  1,
	}
	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(scorer, 1)).
		WithPicker(maxscore.NewMaxScorePicker(1))

	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": profile},
	))

	req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "model-a"}
	result, err := scheduler.Schedule(context.Background(), req, newDecisionHistoryEndpoints())
	require.NoError(t, err)
	assert.NotNil(t, result)
}

// An endpoint that carries no metadata must not panic the decision commit.
func TestPrimaryEndpointIDNilMetadata(t *testing.T) {
	nilMeta := fwksched.NewEndpoint(nil, &fwkdl.Metrics{}, nil)
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{nilMeta}},
		},
	}
	require.Equal(t, "", primaryEndpointID(result))
	require.Equal(t, "", primaryEndpointID(nil))
}
