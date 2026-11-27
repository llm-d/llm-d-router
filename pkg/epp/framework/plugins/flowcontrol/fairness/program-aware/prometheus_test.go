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

package programaware

import (
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

func TestFactory_TwoInstancesShareRegistry(t *testing.T) {
	for _, strategies := range [][2]string{{"las", "las"}, {"turn-priority", "turn-priority"}, {"turn-priority", "las"}, {"las", "turn-priority"}} {
		t.Run(strategies[0]+"_"+strategies[1], func(t *testing.T) {
			handle := plugin.NewEppHandle(t.Context(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
			for i, strategy := range strategies {
				require.NotPanics(t, func() {
					_, err := ProgramAwarePluginFactory(fmt.Sprintf("band-%d", i),
						decoder(fmt.Sprintf(`{"strategy":%q,"evictionTtlSeconds":0}`, strategy)), handle)
					require.NoError(t, err)
				})
			}
		})
	}
}

func TestFactory_MetricRegistrationError(t *testing.T) {
	handle := plugin.NewEppHandle(t.Context(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
	_, err := ProgramAwarePluginFactory("duplicate", decoder(`{"evictionTtlSeconds":0}`), handle)
	require.NoError(t, err)
	require.NotPanics(t, func() {
		p, err := ProgramAwarePluginFactory("duplicate", decoder(`{"evictionTtlSeconds":0}`), handle)
		require.ErrorContains(t, err, "registering metrics")
		require.Nil(t, p)
	})
}

func TestFactory_InitialFairnessIndex(t *testing.T) {
	for _, strategy := range []string{"las", "turn-priority"} {
		t.Run(strategy, func(t *testing.T) {
			registry := prometheus.NewRegistry()
			handle := plugin.NewEppHandle(t.Context(), nil, plugin.WithMetricsRecorder(registry))
			for _, name := range []string{"interactive", "batch"} {
				_, err := ProgramAwarePluginFactory(name,
					decoder(fmt.Sprintf(`{"strategy":%q,"evictionTtlSeconds":0}`, strategy)), handle)
				require.NoError(t, err)
			}

			families, err := registry.Gather()
			require.NoError(t, err)
			require.Len(t, families, 1)
			require.Equal(t, eppmetrics.LLMDRouterEndpointPickerSubsystem+"_program_aware_jains_fairness_index", families[0].GetName())
			require.Len(t, families[0].Metric, 2)
			values := make(map[string]float64)
			for _, sample := range families[0].Metric {
				require.Len(t, sample.Label, 1)
				require.Equal(t, "policy", sample.Label[0].GetName())
				values[sample.Label[0].GetValue()] = sample.GetGauge().GetValue()
			}
			require.Equal(t, map[string]float64{"interactive": 1, "batch": 1}, values)
		})
	}
}

func TestMetrics_InstancesAreIndependent(t *testing.T) {
	registry := prometheus.NewRegistry()
	handle := plugin.NewEppHandle(t.Context(), nil, plugin.WithMetricsRecorder(registry))
	policies := make([]*ProgramAwarePlugin, 2)
	for i, name := range []string{"interactive", "batch"} {
		created, err := ProgramAwarePluginFactory(name, decoder(`{"strategy":"las","lasHalfLifeSeconds":0,"evictionTtlSeconds":0}`), handle)
		require.NoError(t, err)
		policies[i] = created.(*ProgramAwarePlugin)
		req := &fwksched.InferenceRequest{FairnessID: "shared-program"}
		req.PutAttribute(enqueueTimeAttributeKey, time.Now().Add(-time.Duration(i+1)*time.Second))
		require.NoError(t, policies[i].PreRequest(t.Context(), req, nil))
		response := &fwkrh.Response{EndOfStream: true}
		response.Usage.PromptTokens = 100 * (i + 1)
		policies[i].ResponseBody(t.Context(), req, response, nil)
		if i == 1 {
			policies[i].getOrCreateMetrics("other-program").RecordDispatched(time.Now().Add(-10 * time.Second))
		}
		_, err = policies[i].Pick(t.Context(), &fwkfcmocks.MockPriorityBandAccessor{})
		require.NoError(t, err)
	}

	values := func(metric string) map[string]float64 {
		t.Helper()
		families, err := registry.Gather()
		require.NoError(t, err)
		result := make(map[string]float64)
		for _, family := range families {
			if family.GetName() != eppmetrics.LLMDRouterEndpointPickerSubsystem+"_"+metric {
				continue
			}
			for _, sample := range family.Metric {
				for _, label := range sample.Label {
					if label.GetName() == "policy" {
						result[label.GetValue()] = sample.GetGauge().GetValue()
					}
				}
			}
		}
		return result
	}

	waits := map[string]float64{
		"interactive": policies[0].getOrCreateMetrics("shared-program").AverageWaitTime(),
		"batch":       policies[1].getOrCreateMetrics("shared-program").AverageWaitTime(),
	}
	require.Equal(t, waits, values("program_aware_avg_wait_time_milliseconds"))
	require.Equal(t, map[string]float64{"interactive": 100, "batch": 200}, values("program_aware_attained_service_tokens"))
	fairness := map[string]float64{"interactive": policies[0].computeFairnessIndex(), "batch": policies[1].computeFairnessIndex()}
	require.Less(t, fairness["batch"], fairness["interactive"])
	require.Equal(t, fairness, values("program_aware_jains_fairness_index"))

	policies[0].evictKey("shared-program")
	require.Equal(t, map[string]float64{"batch": waits["batch"]}, values("program_aware_avg_wait_time_milliseconds"))
	require.Equal(t, map[string]float64{"batch": 200}, values("program_aware_attained_service_tokens"))
	_, exists := policies[1].programMetrics.Load("shared-program")
	require.True(t, exists)
	_, err := policies[0].Pick(t.Context(), &fwkfcmocks.MockPriorityBandAccessor{})
	require.NoError(t, err)
	require.Equal(t, fairness, values("program_aware_jains_fairness_index"))
}

func TestMetrics_TurnPriorityHasNoLASCollector(t *testing.T) {
	registry := prometheus.NewRegistry()
	handle := plugin.NewEppHandle(t.Context(), nil, plugin.WithMetricsRecorder(registry))
	created, err := ProgramAwarePluginFactory("", decoder(`{"strategy":"turn-priority","evictionTtlSeconds":0}`), handle)
	require.NoError(t, err)
	require.Equal(t, ProgramAwarePluginType, created.TypedName().Name)
	p := created.(*ProgramAwarePlugin)
	require.NoError(t, p.PreRequest(t.Context(), &fwksched.InferenceRequest{FairnessID: "program"}, nil))
	families, err := registry.Gather()
	require.NoError(t, err)
	require.Len(t, families, 2)
	for _, family := range families {
		require.NotEqual(t, eppmetrics.LLMDRouterEndpointPickerSubsystem+"_program_aware_attained_service_tokens", family.GetName())
		for _, sample := range family.Metric {
			labels := make(map[string]string)
			for _, label := range sample.Label {
				labels[label.GetName()] = label.GetValue()
			}
			require.Equal(t, ProgramAwarePluginType, labels["policy"])
		}
	}
}
