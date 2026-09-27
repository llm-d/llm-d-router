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

package thunderagent

import (
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// The ledger gauges are computed at scrape time: with no request after the
// idle TTL, a scrape evicts the session and drops the stale endpoint.
func TestLedgerGaugesWithoutTraffic(t *testing.T) {
	a := newTestAgent(testConfig())
	reg := prometheus.NewPedanticRegistry()
	require.NoError(t, a.metrics.register(reg))
	runTurn(t, a, "s1", schedEndpoint("pod-a", 16, 100), 400, 300)

	require.Equal(t, map[string]float64{
		`endpoint_capacity_tokens{endpoint="default/pod-a"}`:                     1600,
		`endpoint_working_set_tokens{endpoint="default/pod-a",view="undecayed"}`: 300,
		`sessions{state="idle"}`:    1,
		`sessions{state="running"}`: 0,
	}, gather(t, reg))

	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.endpoints["default/pod-a"].updatedAt = time.Now().Add(-2 * endpointStaleAfter)
	a.mgr.lastMaintenance = time.Time{}
	a.mgr.mu.Unlock()

	require.Equal(t, map[string]float64{
		`sessions{state="idle"}`:    0,
		`sessions{state="running"}`: 0,
	}, gather(t, reg))
}

// A rebuilt plugin (config reload) replaces the previous ledger collector, so
// scrapes report the live ledger.
func TestLedgerCollectorReplacedOnReload(t *testing.T) {
	reg := prometheus.NewPedanticRegistry()
	old := newTestAgent(testConfig())
	require.NoError(t, old.metrics.register(reg))

	live := newTestAgent(testConfig())
	require.NoError(t, live.metrics.register(reg))
	runTurn(t, live, "s1", schedEndpoint("pod-a", 0, 0), 400, 300)

	require.Equal(t, float64(1), gather(t, reg)[`sessions{state="idle"}`])
}

// gather scrapes reg and returns the thunder_agent gauges keyed by name
// (without the common prefix) and labels.
func gather(t *testing.T, reg *prometheus.Registry) map[string]float64 {
	t.Helper()
	families, err := reg.Gather()
	require.NoError(t, err)
	prefix := eppmetrics.LLMDRouterEndpointPickerSubsystem + "_thunder_agent_"
	out := map[string]float64{}
	for _, f := range families {
		name, ok := strings.CutPrefix(f.GetName(), prefix)
		if !ok || f.GetType().String() != "GAUGE" {
			continue
		}
		for _, m := range f.GetMetric() {
			labels := make([]string, 0, len(m.GetLabel()))
			for _, l := range m.GetLabel() {
				labels = append(labels, l.GetName()+`="`+l.GetValue()+`"`)
			}
			sort.Strings(labels)
			out[name+"{"+strings.Join(labels, ",")+"}"] = m.GetGauge().GetValue()
		}
	}
	return out
}
