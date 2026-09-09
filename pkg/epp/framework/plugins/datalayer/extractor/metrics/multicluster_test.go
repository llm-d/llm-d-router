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

package metrics

import (
	"context"
	"net/url"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// TestMultiClusterMetricsPipeline is the end-to-end path: a peer cluster's pool
// aggregates, served over httptest, flow through the multicluster source and
// extractor into the pool attributes the multicluster scorers read. Turning those
// attributes into scores is covered by the scorers' own tests.
func TestMultiClusterMetricsPipeline(t *testing.T) {
	srv := createMockServer([]MetricMock{
		{Name: defaultKVCacheUtilizationMetric, Value: 0.62},
		{Name: defaultQueueSizeMetric, Value: 4},
	})
	defer srv.Close()

	u, err := url.Parse(srv.URL)
	require.NoError(t, err)
	source, err := sourcemetrics.NewHTTPMultiClusterMetricsDataSource(u.Scheme, u.Path, "mc-source")
	require.NoError(t, err)
	ext, err := NewMultiClusterMetricsExtractor("mc-extractor", nil)
	require.NoError(t, err)
	require.NoError(t, source.AppendExtractor(ext))

	ep := newEndpointAt(mustHost(t, srv.URL), nil)
	require.NoError(t, source.Dispatch(context.Background(), ep))

	kv, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), attrmetrics.MultiClusterKVCacheUtilizationDataKey)
	require.True(t, ok, "pool KV-cache utilization attribute")
	require.InDelta(t, 0.62, float64(kv), 1e-9)
	q, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), attrmetrics.MultiClusterQueueSizeDataKey)
	require.True(t, ok, "pool queue size attribute")
	require.InDelta(t, 4, float64(q), 1e-9)
}

func gauge(v float64) *dto.MetricFamily {
	return &dto.MetricFamily{
		Type:   dto.MetricType_GAUGE.Enum(),
		Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(v)}}},
	}
}

func TestMultiClusterMetricsExtractor(t *testing.T) {
	tests := []struct {
		name    string
		data    sourcemetrics.PrometheusMetricMap
		wantErr bool
		wantKV  *float64 // nil = attribute expected absent
		wantQ   *float64
	}{
		{
			name: "both aggregates present",
			data: sourcemetrics.PrometheusMetricMap{
				defaultKVCacheUtilizationMetric: gauge(0.42),
				defaultQueueSizeMetric:          gauge(7),
			},
			wantKV: ptr.To(0.42),
			wantQ:  ptr.To(7.0),
		},
		{
			name: "missing queue still writes kv and errors",
			data: sourcemetrics.PrometheusMetricMap{
				defaultKVCacheUtilizationMetric: gauge(0.9),
			},
			wantErr: true,
			wantKV:  ptr.To(0.9),
		},
		{
			name:    "empty map errors and writes nothing",
			data:    sourcemetrics.PrometheusMetricMap{},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ext, err := NewMultiClusterMetricsExtractor("multicluster-metrics", nil)
			require.NoError(t, err)
			require.Equal(t, MultiClusterMetricsExtractorType, ext.TypedName().Type)

			ep := fwkdl.NewEndpoint(nil, nil)
			err = ext.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tc.data, Endpoint: ep})
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			assertAttr(t, ep, attrmetrics.MultiClusterKVCacheUtilizationDataKey, tc.wantKV)
			assertAttr(t, ep, attrmetrics.MultiClusterQueueSizeDataKey, tc.wantQ)
		})
	}
}

func assertAttr(t *testing.T, ep fwkdl.Endpoint, key fwkplugin.DataKey, want *float64) {
	t.Helper()
	got, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), key)
	if want == nil {
		require.False(t, ok, "expected %s absent", key)
		return
	}
	require.True(t, ok, "expected %s present", key)
	require.InDelta(t, *want, float64(got), 1e-9)
}

// A custom metric name resolves to the same attribute key.
func TestMultiClusterMetricsExtractorCustomNames(t *testing.T) {
	ext, err := NewMultiClusterMetricsExtractor("multicluster-metrics", &multiClusterMetricsExtractorParams{
		KVCacheUtilizationMetric: "custom_kv",
		QueueSizeMetric:          "custom_queue",
	})
	require.NoError(t, err)

	ep := fwkdl.NewEndpoint(nil, nil)
	data := sourcemetrics.PrometheusMetricMap{"custom_kv": gauge(0.3), "custom_queue": gauge(2)}
	require.NoError(t, ext.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep}))

	kv, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), attrmetrics.MultiClusterKVCacheUtilizationDataKey)
	require.True(t, ok)
	require.InDelta(t, 0.3, float64(kv), 1e-9)
}

func TestMultiClusterMetricsExtractorInvalidMetricName(t *testing.T) {
	_, err := NewMultiClusterMetricsExtractor("multicluster-metrics", &multiClusterMetricsExtractorParams{KVCacheUtilizationMetric: "1_bad"})
	require.Error(t, err)
}
