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

package proxy

import (
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

const (
	metricDisaggRequests  = "llm_d_disagg_sidecar_disagg_requests_total"
	metricRequestErrors   = "llm_d_disagg_sidecar_request_errors_total"
	metricEncodeDuration  = "llm_d_disagg_sidecar_encode_duration_seconds"
	metricPrefillDuration = "llm_d_disagg_sidecar_prefill_duration_seconds"
	metricDecodeDuration  = "llm_d_disagg_sidecar_decode_duration_seconds"
)

// gatherFamily returns the named metric family from the controller-runtime
// registry, or nil if it has no samples yet.
func gatherFamily(t *testing.T, name string) *dto.MetricFamily {
	t.Helper()
	metrics.Register()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, mf := range families {
		if mf.GetName() == name {
			return mf
		}
	}
	return nil
}

// counterValue returns the value of the counter series with the given label,
// or 0 if the series has not been created.
func counterValue(t *testing.T, name, labelName, labelValue string) float64 {
	t.Helper()
	mf := gatherFamily(t, name)
	if mf == nil {
		return 0
	}
	for _, m := range mf.GetMetric() {
		for _, lp := range m.GetLabel() {
			if lp.GetName() == labelName && lp.GetValue() == labelValue {
				return m.GetCounter().GetValue()
			}
		}
	}
	return 0
}

// histogramCount returns the sample count of an unlabeled histogram.
func histogramCount(t *testing.T, name string) uint64 {
	t.Helper()
	mf := gatherFamily(t, name)
	if mf == nil || len(mf.GetMetric()) == 0 {
		return 0
	}
	return mf.GetMetric()[0].GetHistogram().GetSampleCount()
}

func stageErrors(t *testing.T, stage string) float64 {
	t.Helper()
	return counterValue(t, metricRequestErrors, "stage", stage)
}

func disaggCount(t *testing.T, disaggType string) float64 {
	t.Helper()
	return counterValue(t, metricDisaggRequests, "disagg_type", disaggType)
}
