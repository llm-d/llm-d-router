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

package decisionhistory

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestRecorderRingRetainsNewestFirst(t *testing.T) {
	r := NewRecorder(3, 5)
	for i := 1; i <= 5; i++ {
		c := r.Begin(fmt.Sprintf("req-%d", i), "model")
		r.Commit(c, time.Unix(int64(i), 0), fmt.Sprintf("pod-%d", i), "")
	}

	got := r.Snapshot()
	require.Len(t, got, 3)
	for i, want := range []string{"req-5", "req-4", "req-3"} {
		require.Equal(t, want, got[i].RequestID)
		require.Equal(t, fmt.Sprintf("pod-%d", 5-i), got[i].TargetEndpoint)
	}
}

func TestCollectorRecordsProfileRun(t *testing.T) {
	r := NewRecorder(10, 2)
	ts := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

	c := r.Begin("req-1", "model-a")
	c.ProfileStarted("default", 3)
	c.FilterRan("test-filter", "my-filter", 3, 2)
	c.ScorerRan(fwkplugin.TypedName{Type: "test-scorer", Name: "scorer-a"}, 0.25, map[string]float64{"pod1": 1, "pod2": 0.5})
	c.ScorerRan(fwkplugin.TypedName{Type: "test-scorer", Name: "scorer-b"}, 0.75, map[string]float64{"pod1": 0, "pod2": 1})
	c.Picked([]string{"pod2", "pod1"}, []float64{0.875, 0.25}, []string{"pod2"})
	r.Commit(c, ts, "pod2", "")

	got := r.Snapshot()
	require.Len(t, got, 1)
	d := got[0]
	require.Equal(t, ts.UTC(), d.Timestamp)
	require.Equal(t, "req-1", d.RequestID)
	require.Equal(t, "model-a", d.Model)
	require.Equal(t, "pod2", d.TargetEndpoint)
	require.Empty(t, d.FailureReason)
	require.Len(t, d.Profiles, 1)

	p := d.Profiles[0]
	require.Equal(t, "default", p.Profile)
	require.Equal(t, 3, p.CandidatesBeforeFilters)
	require.Equal(t, []FilterStage{{Type: "test-filter", Name: "my-filter", Before: 3, After: 2}}, p.Filters)
	require.Equal(t, []ScorerEntry{
		{Type: "test-scorer", Name: "scorer-a", Weight: 0.25},
		{Type: "test-scorer", Name: "scorer-b", Weight: 0.75},
	}, p.Scorers)
	require.Equal(t, []string{"pod2"}, p.TargetEndpoints)
	require.Empty(t, p.Error)

	require.Len(t, p.TopEndpoints, 2)
	top := p.TopEndpoints[0]
	require.Equal(t, "pod2", top.Endpoint)
	require.InDelta(t, 0.875, top.WeightedScore, 1e-9)
	require.Equal(t, map[string]float64{"scorer-a/test-scorer": 0.5, "scorer-b/test-scorer": 1}, top.ScorerScores)
	second := p.TopEndpoints[1]
	require.Equal(t, "pod1", second.Endpoint)
	require.Equal(t, map[string]float64{"scorer-a/test-scorer": 1, "scorer-b/test-scorer": 0}, second.ScorerScores)
}

func TestCollectorTrimsToTopK(t *testing.T) {
	r := NewRecorder(10, 1)
	c := r.Begin("req-1", "model-a")
	c.ProfileStarted("default", 3)
	c.ScorerRan(fwkplugin.TypedName{Type: "test-scorer", Name: "scorer-a"}, 1, map[string]float64{"pod1": 1, "pod2": 0.5, "pod3": 0.25})
	c.Picked([]string{"pod1"}, []float64{1}, []string{"pod1"})
	r.Commit(c, time.Now(), "pod1", "")

	p := r.Snapshot()[0].Profiles[0]
	require.Len(t, p.TopEndpoints, 1)
	require.Equal(t, "pod1", p.TopEndpoints[0].Endpoint)
	require.Equal(t, map[string]float64{"scorer-a/test-scorer": 1}, p.TopEndpoints[0].ScorerScores)
}

func TestCollectorProfileFailed(t *testing.T) {
	r := NewRecorder(10, 5)
	c := r.Begin("req-1", "model-a")
	c.ProfileStarted("default", 2)
	c.FilterRan("test-filter", "my-filter", 2, 0)
	c.ProfileFailed("no endpoints available for the given request")
	r.Commit(c, time.Now(), "", "all_filtered")

	d := r.Snapshot()[0]
	require.Equal(t, "all_filtered", d.FailureReason)
	require.Empty(t, d.TargetEndpoint)
	require.Len(t, d.Profiles, 1)
	require.Equal(t, "no endpoints available for the given request", d.Profiles[0].Error)
	require.Empty(t, d.Profiles[0].TopEndpoints)
}

func TestNilRecorderAndCollectorAreNoOps(t *testing.T) {
	var r *Recorder
	require.Nil(t, r.Begin("req-1", "model-a"))
	require.Nil(t, r.Snapshot())
	r.Commit(nil, time.Now(), "pod1", "")

	var c *Collector
	require.Equal(t, 0, c.TopK())
	c.ProfileStarted("default", 1)
	c.FilterRan("test-filter", "my-filter", 1, 0)
	c.ScorerRan(fwkplugin.TypedName{Type: "test-scorer", Name: "scorer-a"}, 1, nil)
	c.Picked([]string{"pod1"}, []float64{1}, []string{"pod1"})
	c.ProfileFailed("boom")
}

func TestNewRecorderZeroCapacityAndTopKClamp(t *testing.T) {
	require.Nil(t, NewRecorder(0, 5).Begin("req-1", "model-a"))

	c := NewRecorder(1, 0).Begin("req-1", "model-a")
	require.Equal(t, 1, c.TopK())
}

func TestContextCarriesCollector(t *testing.T) {
	r := NewRecorder(1, 5)
	c := r.Begin("req-1", "model-a")
	require.Nil(t, CollectorFromContext(t.Context()))

	ctx := ContextWithCollector(t.Context(), c)
	require.Equal(t, c, CollectorFromContext(ctx))

	// A nil collector leaves the context unchanged.
	require.Nil(t, CollectorFromContext(ContextWithCollector(t.Context(), nil)))
}

func TestCollectorSanitizesNonFiniteScores(t *testing.T) {
	r := NewRecorder(10, 5)
	c := r.Begin("req-1", "model-a")
	c.ProfileStarted("default", 3)
	c.ScorerRan(fwkplugin.TypedName{Type: "test-scorer", Name: "scorer-a"}, 1, map[string]float64{
		"pod1": math.NaN(),
		"pod2": math.Inf(1),
		"pod3": 0.5,
	})
	c.Picked([]string{"pod1", "pod3"}, []float64{math.NaN(), 0.5}, []string{"pod3"})
	r.Commit(c, time.Now(), "pod3", "")

	d := r.Snapshot()[0]
	require.Len(t, d.Profiles, 1)
	p := d.Profiles[0]
	require.Len(t, p.TopEndpoints, 2)

	// A non-finite weighted score is recorded as 0, and the non-finite raw
	// scores are dropped, so the entry stays marshalable.
	require.InDelta(t, 0, p.TopEndpoints[0].WeightedScore, 0)
	require.Empty(t, p.TopEndpoints[0].ScorerScores)
	require.InDelta(t, 0.5, p.TopEndpoints[1].WeightedScore, 1e-9)
	require.Equal(t, map[string]float64{"scorer-a/test-scorer": 0.5}, p.TopEndpoints[1].ScorerScores)

	_, err := json.Marshal(d)
	require.NoError(t, err)
}

func TestBeginTruncatesModel(t *testing.T) {
	r := NewRecorder(2, 5)

	c := r.Begin("req-1", strings.Repeat("a", maxModelLength+10))
	r.Commit(c, time.Now(), "", "")
	require.Len(t, r.Snapshot()[0].Model, maxModelLength)

	// Truncation lands on a rune boundary so the stored string stays valid
	// UTF-8.
	c = r.Begin("req-2", strings.Repeat("é", maxModelLength))
	r.Commit(c, time.Now(), "", "")
	model := r.Snapshot()[0].Model
	require.LessOrEqual(t, len(model), maxModelLength)
	require.True(t, utf8.ValidString(model))
}

func TestEmptyFieldsSerializeAsArrays(t *testing.T) {
	r := NewRecorder(2, 5)

	// A request the profile handler rejected before any profile ran.
	c := r.Begin("req-1", "model-a")
	r.Commit(c, time.Now(), "", "no_profile_selected")

	// A profile that failed before any plugin ran.
	c = r.Begin("req-2", "model-a")
	c.ProfileStarted("default", 2)
	c.ProfileFailed("no endpoints available")
	r.Commit(c, time.Now(), "", "all_filtered")

	payload, err := json.Marshal(r.Snapshot())
	require.NoError(t, err)
	for _, field := range []string{`"profiles":[]`, `"filters":[]`, `"scorers":[]`, `"topEndpoints":[]`, `"targetEndpoints":[]`} {
		require.Contains(t, string(payload), field)
	}
	require.NotContains(t, string(payload), "null")
}
