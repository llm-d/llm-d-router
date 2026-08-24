/*
Copyright 2026 The Kubernetes Authors.

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

package prefixcacheaffinity

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry), "re-registering the same collector is tolerated")
	require.NoError(t, registerMetrics(nil), "a nil registerer is a no-op")
}

// Each Filter branch records exactly one decision outcome, so the counter labels
// distinguish an affinity-driven route from one where load reopened the set.
func TestFilterRecordsDecisionOutcome(t *testing.T) {
	tests := []struct {
		name    string
		config  Config
		input   []fwksched.Endpoint
		outcome string
	}{
		{
			name:   "single candidate is not applicable",
			config: Config{AffinityThreshold: 0.80},
			input:  []fwksched.Endpoint{makeEndpoint("a", 90, 10, 0)},
			// A single candidate has no set to narrow.
			outcome: outcomeNotApplicable,
		},
		{
			name:    "disabled threshold is not applicable",
			config:  Config{AffinityThreshold: 0},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 10, 0), makeEndpoint("b", 10, 10, 0)},
			outcome: outcomeNotApplicable,
		},
		{
			name:    "exploration skip",
			config:  Config{AffinityThreshold: 0.80, ExplorationProbability: 1.0},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 100, 0), makeEndpoint("b", 10, 50, 0)},
			outcome: outcomeExploration,
		},
		{
			name:    "no endpoint meets the threshold",
			config:  Config{AffinityThreshold: 0.80},
			input:   []fwksched.Endpoint{makeEndpoint("a", 10, 10, 0), makeEndpoint("b", 20, 20, 0)},
			outcome: outcomeNoMatch,
		},
		{
			name:    "sticky set kept",
			config:  Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 5000, TTFTSource: TTFTSourceLatencyPredictor},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 100, 0), makeEndpoint("b", 85, 120, 0), makeEndpoint("c", 10, 50, 0)},
			outcome: outcomeSticky,
		},
		{
			name:    "load gate reopens the set",
			config:  Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 100, TTFTSource: TTFTSourceLatencyPredictor},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 500, 0), makeEndpoint("b", 10, 50, 0)},
			outcome: outcomeLoadOverride,
		},
	}

	allOutcomes := []string{outcomeSticky, outcomeNoMatch, outcomeLoadOverride, outcomeExploration, outcomeNotApplicable}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)

			p := newTestPlugin(test.config)
			p.Filter(context.Background(), nil, test.input)

			for _, outcome := range allOutcomes {
				want := float64(0)
				if outcome == test.outcome {
					want = 1
				}
				got := testutil.ToFloat64(filterDecisions.WithLabelValues("test", outcome))
				assert.Equalf(t, want, got, "outcome %q", outcome)
			}
		})
	}
}
