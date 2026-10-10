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

package servedmodel

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry), "re-registering the same collectors is tolerated")
	require.NoError(t, registerMetrics(nil), "a nil registerer is a no-op")

	_, err := NewServedModelFilter("test", parameters{})
	require.NoError(t, err)
	families, err := registry.Gather()
	require.NoError(t, err)
	series := map[string]int{}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			assert.Zero(t, metric.GetCounter().GetValue(), family.GetName())
			series[family.GetName()]++
		}
	}
	assert.Equal(t, map[string]int{
		"llm_d_epp_served_model_filter_decisions_total":           len(outcomes),
		"llm_d_epp_served_model_filter_candidates_total":          1,
		"llm_d_epp_served_model_filter_unlisted_candidates_total": 1,
	}, series, "a new filter exposes every series at zero")
}

func TestFilterRecordsDecision(t *testing.T) {
	listed := newEndpoint("p1", "llama", "sql-v3")
	noList := newEndpointWithoutModels("p2")
	noList2 := newEndpointWithoutModels("p3")

	tests := []struct {
		name           string
		params         parameters
		model          string
		candidates     []scheduling.Endpoint
		wantOutcome    string
		wantCandidates float64
		wantUnlisted   float64
	}{
		{name: "listed", model: "sql-v3", candidates: []scheduling.Endpoint{listed, noList},
			wantOutcome: outcomeListed, wantCandidates: 2, wantUnlisted: 1},
		{name: "unlisted", model: "other", candidates: []scheduling.Endpoint{listed, noList, noList2},
			wantOutcome: outcomeUnlisted, wantCandidates: 3, wantUnlisted: 2},
		{name: "fallback", params: parameters{OnMissing: onMissingFail, FallbackOnEmpty: true}, model: "other",
			candidates: []scheduling.Endpoint{listed, noList}, wantOutcome: outcomeFallback, wantCandidates: 2, wantUnlisted: 1},
		{name: "empty", params: parameters{OnMissing: onMissingFail}, model: "other",
			candidates: []scheduling.Endpoint{listed}, wantOutcome: outcomeEmpty, wantCandidates: 1},
		{name: "not applicable", model: "", candidates: []scheduling.Endpoint{listed}, wantOutcome: outcomeNotApplicable},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)

			f, err := NewServedModelFilter("test", test.params)
			require.NoError(t, err)
			f.Filter(t.Context(), &scheduling.InferenceRequest{TargetModel: test.model}, test.candidates)

			for _, outcome := range outcomes {
				want := float64(0)
				if outcome == test.wantOutcome {
					want = 1
				}
				assert.Equalf(t, want, testutil.ToFloat64(filterDecisions.WithLabelValues("test", outcome)), "outcome %q", outcome)
			}
			assert.Equal(t, test.wantCandidates, testutil.ToFloat64(candidates.WithLabelValues("test")))
			assert.Equal(t, test.wantUnlisted, testutil.ToFloat64(unlistedCandidates.WithLabelValues("test")))
		})
	}
}
