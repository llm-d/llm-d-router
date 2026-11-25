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

package heavyhitter

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func TestCollector(t *testing.T) {
	reg := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(context.Background(),
		func() []types.NamespacedName { return nil }, fwkplugin.WithMetricsRecorder(reg))
	got, err := PluginFactory("hh", fwkplugin.StrictDecoder(json.RawMessage(
		`{"metricTopK": 1, "windowSeconds": 60, "sliceSeconds": 1}`)), handle)
	require.NoError(t, err)
	p := got.(*Plugin)

	charge(p, "tenant-a", "user-heavy", 100, 10) // 100 + 2*10 = 120, the heaviest
	charge(p, "tenant-a", "user-light", 20, 10)  // 40, outside metricTopK

	expected := `
# HELP llm_d_epp_heavy_hitter_evictions_total [ALPHA] Cumulative slot takeovers. A nonzero rate means the reported loads are approximate and slotsPerSlice is below the active fairness-ID count per slice.
# TYPE llm_d_epp_heavy_hitter_evictions_total counter
llm_d_epp_heavy_hitter_evictions_total{plugin_name="hh"} 0
# HELP llm_d_epp_heavy_hitter_weight [ALPHA] Estimated windowed load of each of the heaviest fairness IDs.
# TYPE llm_d_epp_heavy_hitter_weight gauge
llm_d_epp_heavy_hitter_weight{fairness_id="tenant-a/user-heavy",plugin_name="hh"} 120
# HELP llm_d_epp_heavy_hitter_weight_error [ALPHA] Lower-bound correction of the fairness ID's windowed load: the true load is at least weight minus this value.
# TYPE llm_d_epp_heavy_hitter_weight_error gauge
llm_d_epp_heavy_hitter_weight_error{fairness_id="tenant-a/user-heavy",plugin_name="hh"} 0
# HELP llm_d_epp_heavy_hitter_window_weight [ALPHA] Exact windowed load of all fairness IDs.
# TYPE llm_d_epp_heavy_hitter_window_weight gauge
llm_d_epp_heavy_hitter_window_weight{plugin_name="hh"} 160
`
	require.NoError(t, promtestutil.GatherAndCompare(reg, strings.NewReader(expected),
		"llm_d_epp_heavy_hitter_weight",
		"llm_d_epp_heavy_hitter_weight_error",
		"llm_d_epp_heavy_hitter_window_weight",
		"llm_d_epp_heavy_hitter_evictions_total"))
}

// With a threshold configured, only the IDs at or above it get series; the lighter IDs stay
// counted in the window total.
func TestCollector_ThresholdFiltersSeries(t *testing.T) {
	reg := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(context.Background(),
		func() []types.NamespacedName { return nil }, fwkplugin.WithMetricsRecorder(reg))
	got, err := PluginFactory("hh", fwkplugin.StrictDecoder(json.RawMessage(
		`{"heavyThreshold": 100, "expectedWindowLoad": 1000, "windowSeconds": 60, "sliceSeconds": 1}`)), handle)
	require.NoError(t, err)
	p := got.(*Plugin)

	charge(p, "tenant-a", "user-heavy", 60, 20) // 100, at the threshold
	charge(p, "tenant-a", "user-light", 10, 10) // 30, below it

	expected := `
# HELP llm_d_epp_heavy_hitter_weight [ALPHA] Estimated windowed load of each of the heaviest fairness IDs.
# TYPE llm_d_epp_heavy_hitter_weight gauge
llm_d_epp_heavy_hitter_weight{fairness_id="tenant-a/user-heavy",plugin_name="hh"} 100
`
	require.NoError(t, promtestutil.GatherAndCompare(reg, strings.NewReader(expected), "llm_d_epp_heavy_hitter_weight"))
	assert.Equal(t, 1, promtestutil.CollectAndCount(newCollector("test", p), "llm_d_epp_heavy_hitter_weight"))
}

// The per-ID series do not use the shared fairness-ID label limiter: they are rebuilt on
// every scrape and bounded by metricTopK, so a changing top-k must not consume the
// process-wide cap that other fairness_id metrics share.
func TestCollector_DoesNotConsumeSharedLabelCap(t *testing.T) {
	metricsutil.SetFairnessIDLabelLimit(2)
	t.Cleanup(func() { metricsutil.SetFairnessIDLabelLimit(metricsutil.DefaultFairnessIDLabelLimit) })

	p := newPlugin(t, `{"metricTopK": 10}`)
	for i := range 5 {
		charge(p, fmt.Sprintf("tenant-%d", i), "user-1", 10, 0)
	}

	assert.Equal(t, 5, promtestutil.CollectAndCount(newCollector("test", p), "llm_d_epp_heavy_hitter_weight"))
}

// A key that is not valid UTF-8 gets no series (MustNewConstMetric would panic), while its
// load stays counted in the totals.
func TestCollector_SkipsInvalidUTF8Key(t *testing.T) {
	p := newPlugin(t, `{"metricTopK": 10, "windowSeconds": 60, "sliceSeconds": 1}`)
	chargeWithHeaders(p, map[string]string{
		metadata.FlowFairnessIDKey: "tenant-\x80\x81",
		metadata.FlowUserIDKey:     "user-1",
	}, 10, 0)
	charge(p, "tenant-ok", "user-1", 5, 0)

	c := newCollector("test", p)
	assert.Equal(t, 1, promtestutil.CollectAndCount(c, "llm_d_epp_heavy_hitter_weight"))
	reg := prometheus.NewRegistry()
	require.NoError(t, reg.Register(c))
	require.NoError(t, promtestutil.GatherAndCompare(reg, strings.NewReader(`
# HELP llm_d_epp_heavy_hitter_window_weight [ALPHA] Exact windowed load of all fairness IDs.
# TYPE llm_d_epp_heavy_hitter_window_weight gauge
llm_d_epp_heavy_hitter_window_weight{plugin_name="test"} 15
`), "llm_d_epp_heavy_hitter_window_weight"))
}

// One registry can hold several instances, for example a 1h and a 5m window.
func TestFactory_TwoInstancesShareRegistry(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(),
		func() []types.NamespacedName { return nil }, fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	_, err := PluginFactory("window-1h", nil, handle)
	require.NoError(t, err)
	_, err = PluginFactory("window-5m", fwkplugin.StrictDecoder(json.RawMessage(`{"windowSeconds": 300}`)), handle)
	require.NoError(t, err)
}
