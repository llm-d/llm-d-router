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

package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
)

func TestRecordToolCallingFieldStatuses(t *testing.T) {
	Reset()
	RecordToolCallingFieldStatuses(toolcalling.ComponentEPP, toolcalling.DirectionRequest, []toolcalling.FieldStatus{
		{Field: toolcalling.FieldTools, Status: toolcalling.FieldStatusPreserved, Observed: true},
		{Field: toolcalling.FieldToolChoice, Status: toolcalling.FieldStatusPreserved, Observed: false},
		{Field: toolcalling.FieldResponseFormat, Status: toolcalling.FieldStatusChanged, Observed: true},
	})

	require.Equal(t, float64(1), promtestutil.ToFloat64(llmdToolCallingFieldStatusTotal.WithLabelValues(
		toolcalling.ComponentEPP,
		toolcalling.DirectionRequest,
		string(toolcalling.FieldTools),
		string(toolcalling.FieldStatusPreserved),
	)))
	require.Equal(t, float64(0), promtestutil.ToFloat64(llmdToolCallingFieldStatusTotal.WithLabelValues(
		toolcalling.ComponentEPP,
		toolcalling.DirectionRequest,
		string(toolcalling.FieldToolChoice),
		string(toolcalling.FieldStatusPreserved),
	)))
	require.Equal(t, float64(1), promtestutil.ToFloat64(llmdToolCallingFieldStatusTotal.WithLabelValues(
		toolcalling.ComponentEPP,
		toolcalling.DirectionRequest,
		string(toolcalling.FieldResponseFormat),
		string(toolcalling.FieldStatusChanged),
	)))
}

func TestToolCallingMetricSchema(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	registry := prometheus.NewRegistry()
	require.NoError(t, registry.Register(llmdToolCallingFieldStatusTotal))
	RecordToolCallingFieldStatuses(toolcalling.ComponentEPP, toolcalling.DirectionRequest, []toolcalling.FieldStatus{
		{Field: toolcalling.FieldTools, Status: toolcalling.FieldStatusPreserved, Observed: true},
	})
	families, err := registry.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)
	require.Equal(t, "llm_d_epp_tool_calling_field_status_total", families[0].GetName())
	require.Len(t, families[0].GetMetric(), 1)
	metric := families[0].GetMetric()[0]
	labels := make(map[string]string)
	for _, label := range metric.GetLabel() {
		labels[label.GetName()] = label.GetValue()
	}
	require.Equal(t, map[string]string{
		"component": "epp", "direction": "request", "field": "tools", "status": "preserved",
	}, labels)
	require.Equal(t, float64(1), metric.GetCounter().GetValue())
}
