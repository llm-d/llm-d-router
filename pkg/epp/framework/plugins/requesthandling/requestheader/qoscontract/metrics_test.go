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

package qoscontract

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// TestTeamLabelIgnoresFairnessIDLimit verifies that the team label is the configured team ID,
// not the fairness ID bounded by the shared label limiter. Past the limiter's cap, teams would
// fold onto one label and the in-flight gauge would flip between them.
func TestTeamLabelIgnoresFairnessIDLimit(t *testing.T) {
	metricsutil.SetFairnessIDLabelLimit(1)
	t.Cleanup(func() { metricsutil.SetFairnessIDLabelLimit(metricsutil.DefaultFairnessIDLabelLimit) })

	p, _ := newTestPlugin(t, nil)
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-a", 100, 50)))
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-b", 100, 50)))

	assert.InDelta(t, 1, promtestutil.ToFloat64(inflightRequests.WithLabelValues("team-a")), 1e-9)
	assert.InDelta(t, 1, promtestutil.ToFloat64(inflightRequests.WithLabelValues("team-b")), 1e-9)
}

// TestMetricNames verifies that every qos-contract metric carries the qos_contract segment.
func TestMetricNames(t *testing.T) {
	want := map[prometheus.Collector]string{
		contractDecisions: "llm_d_epp_qos_contract_decisions_total",
		chargedTokens:     "llm_d_epp_qos_contract_charged_tokens_total",
		inflightRequests:  "llm_d_epp_qos_contract_inflight_requests",
	}
	for collector, name := range want {
		descs := make(chan *prometheus.Desc, 1)
		collector.Describe(descs)
		close(descs)
		desc := <-descs
		assert.Contains(t, desc.String(), `fqName: "`+name+`"`)
	}
}
