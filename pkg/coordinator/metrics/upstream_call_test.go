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
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

func TestUpstreamCall_StartIncrementsCounter(t *testing.T) {
	Reset()
	_ = StartUpstreamCall(UpstreamRender)
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(upstreamRequestTotal.WithLabelValues(UpstreamRender)),
		1e-9, "counter must fire at StartUpstreamCall, before Done")
}

func TestUpstreamCall_DoneObservesDuration(t *testing.T) {
	Reset()
	call := StartUpstreamCall(UpstreamRender)
	call.Done()

	obs, err := upstreamRequestDuration.GetMetricWithLabelValues(UpstreamRender)
	require.NoError(t, err)
	m := &dto.Metric{}
	require.NoError(t, obs.(prometheus.Metric).Write(m))
	require.Equal(t, uint64(1), m.GetHistogram().GetSampleCount(),
		"Done must observe on upstream_request_duration_seconds")
}

func TestUpstreamCall_RepeatedUseCompounds(t *testing.T) {
	Reset()
	for i := 0; i < 3; i++ {
		StartUpstreamCall(UpstreamRender).Done()
	}
	require.InDelta(t, 3.0,
		promtestutil.ToFloat64(upstreamRequestTotal.WithLabelValues(UpstreamRender)),
		1e-9)

	obs, err := upstreamRequestDuration.GetMetricWithLabelValues(UpstreamRender)
	require.NoError(t, err)
	m := &dto.Metric{}
	require.NoError(t, obs.(prometheus.Metric).Write(m))
	require.Equal(t, uint64(3), m.GetHistogram().GetSampleCount())
}
