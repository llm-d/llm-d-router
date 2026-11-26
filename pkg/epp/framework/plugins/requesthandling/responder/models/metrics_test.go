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

package models

import (
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestRegisterMetrics(t *testing.T) {
	t.Parallel()

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry), "re-registering the same collector is tolerated")
	require.NoError(t, registerMetrics(nil), "a nil registerer is a no-op")
}

func TestFactoryRegistersMetricOnHandle(t *testing.T) {
	t.Parallel()

	registry := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(t.Context(), nil, fwkplugin.WithMetricsRecorder(registry))
	p, err := Factory("registered", nil, handle)
	require.NoError(t, err)
	p.(*Responder).Screen(t.Context(), &fwksched.InferenceRequest{TargetModel: "base"}, nil)

	got, err := testutil.GatherAndCount(registry, "llm_d_epp_models_responder_screen_decisions_total")
	require.NoError(t, err)
	assert.Positive(t, got)
}

// Not parallel: it swaps the package-level limiter.
func TestRecordDecisionBoundsModelLabel(t *testing.T) {
	const pluginName = "bounded"
	old := boundModelName
	boundModelName = metricsutil.NewBoundedLabel(3).Bound
	t.Cleanup(func() {
		boundModelName = old
		screenDecisions.DeletePartialMatch(prometheus.Labels{"plugin_name": pluginName})
	})

	for i := range 100 {
		recordDecision(pluginName, fmt.Sprintf("model-%d", i), outcomeNotFound)
	}

	assert.Equal(t, 97.0, testutil.ToFloat64(screenDecisions.WithLabelValues(pluginName, metricsutil.OverflowValue, outcomeNotFound)),
		"names past the cap fold into one series")
}
