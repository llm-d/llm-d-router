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

package utilization

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func float64Ptr(v float64) *float64 { return &v }

func condCap(metric string, maxValue float64) condition {
	return condition{Metric: metric, MaxValue: float64Ptr(maxValue)}
}

type stubEndpoint struct {
	metadata *datalayer.EndpointMetadata
	metrics  *datalayer.Metrics
	attr     datalayer.AttributeMap
}

func newStubEndpoint(name string) *stubEndpoint {
	return &stubEndpoint{
		metadata: &datalayer.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		metrics:  &datalayer.Metrics{},
		attr:     datalayer.NewAttributes(),
	}
}

func (f *stubEndpoint) GetMetadata() *datalayer.EndpointMetadata   { return f.metadata }
func (f *stubEndpoint) UpdateMetadata(*datalayer.EndpointMetadata) {}
func (f *stubEndpoint) GetMetrics() *datalayer.Metrics             { return f.metrics }
func (f *stubEndpoint) UpdateMetrics(*datalayer.Metrics)           {}
func (f *stubEndpoint) GetAttributes() datalayer.AttributeMap      { return f.attr }
func (f *stubEndpoint) String() string                             { return f.metadata.ID.String() }
func (f *stubEndpoint) Put(key string, val datalayer.Cloneable)    { f.attr.Put(key, val) }
func (f *stubEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	return f.attr.Get(key)
}
func (f *stubEndpoint) Keys() []string                { return f.attr.Keys() }
func (f *stubEndpoint) Clone() datalayer.AttributeMap { return f.attr.Clone() }

func newTestEndpoint(name string) scheduling.Endpoint {
	return newStubEndpoint(name)
}

func newTestEndpointWithLoad(name string, requests int64) scheduling.Endpoint {
	ep := newStubEndpoint(name)
	ep.Put(attrconcurrency.InFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Requests: requests})
	return ep
}

func newTestEndpointWithMetrics(name string, metrics datalayer.Metrics) scheduling.Endpoint {
	ep := newStubEndpoint(name)
	ep.metrics = &metrics
	return ep
}

func endpointNames(endpoints []scheduling.Endpoint) []string {
	names := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		names = append(names, ep.GetMetadata().ID.Name)
	}
	return names
}

func TestUtilizationFilter_Filter(t *testing.T) {
	tests := []struct {
		name      string
		params    parameters
		endpoints func() []scheduling.Endpoint
		want      []string
	}{
		{
			name:   "active-requests keeps endpoints at or below maxValue",
			params: parameters{Conditions: []condition{condCap(MetricActiveRequests, 2)}},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 1),
					newTestEndpointWithLoad("pod-b", 2),
					newTestEndpointWithLoad("pod-c", 3),
				}
			},
			want: []string{"pod-a", "pod-b"},
		},
		{
			name:   "active-requests endpoints without load attribute are kept",
			params: parameters{Conditions: []condition{condCap(MetricActiveRequests, 1)}},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpoint("pod-a"),
					newTestEndpointWithLoad("pod-b", 5),
				}
			},
			want: []string{"pod-a"},
		},
		{
			name:   "running-requests keeps endpoints at or below maxValue",
			params: parameters{Conditions: []condition{condCap(MetricRunningRequests, 4)}},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithMetrics("pod-a", datalayer.Metrics{RunningRequestsSize: 4}),
					newTestEndpointWithMetrics("pod-b", datalayer.Metrics{RunningRequestsSize: 5}),
				}
			},
			want: []string{"pod-a"},
		},
		{
			name:   "waiting-queue keeps endpoints at or below maxValue",
			params: parameters{Conditions: []condition{condCap(MetricWaitingQueue, 0)}},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithMetrics("pod-a", datalayer.Metrics{WaitingQueueSize: 0}),
					newTestEndpointWithMetrics("pod-b", datalayer.Metrics{WaitingQueueSize: 3}),
				}
			},
			want: []string{"pod-a"},
		},
		{
			name:   "kv-cache-utilization keeps endpoints at or below maxValue",
			params: parameters{Conditions: []condition{condCap(MetricKVCacheUtilization, 0.8)}},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithMetrics("pod-a", datalayer.Metrics{KVCacheUsagePercent: 0.5}),
					newTestEndpointWithMetrics("pod-b", datalayer.Metrics{KVCacheUsagePercent: 0.9}),
				}
			},
			want: []string{"pod-a"},
		},
		{
			name: "multiple conditions drop endpoints exceeding any cap",
			params: parameters{Conditions: []condition{
				condCap(MetricRunningRequests, 40),
				condCap(MetricWaitingQueue, 0),
			}},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithMetrics("pod-a", datalayer.Metrics{RunningRequestsSize: 10, WaitingQueueSize: 0}),
					newTestEndpointWithMetrics("pod-b", datalayer.Metrics{RunningRequestsSize: 41, WaitingQueueSize: 0}),
					newTestEndpointWithMetrics("pod-c", datalayer.Metrics{RunningRequestsSize: 10, WaitingQueueSize: 1}),
				}
			},
			want: []string{"pod-a"},
		},
		{
			name: "conditions mix EPP-tracked and model-server metrics",
			params: parameters{Conditions: []condition{
				condCap(MetricActiveRequests, 10),
				condCap(MetricKVCacheUtilization, 0.8),
			}},
			endpoints: func() []scheduling.Endpoint {
				busy := newStubEndpoint("pod-b")
				busy.Put(attrconcurrency.InFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Requests: 11})
				full := newStubEndpoint("pod-c")
				full.metrics = &datalayer.Metrics{KVCacheUsagePercent: 0.9}
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 5),
					busy,
					full,
				}
			},
			want: []string{"pod-a"},
		},
		{
			name:   "all filtered out without fallback returns empty",
			params: parameters{Conditions: []condition{condCap(MetricActiveRequests, 1)}},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 2),
					newTestEndpointWithLoad("pod-b", 3),
				}
			},
			want: []string{},
		},
		{
			name: "all filtered out with fallback returns original candidates",
			params: parameters{
				Conditions:      []condition{condCap(MetricActiveRequests, 1)},
				FallbackOnEmpty: true,
			},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 2),
					newTestEndpointWithLoad("pod-b", 3),
				}
			},
			want: []string{"pod-a", "pod-b"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := testutils.NewTestContext(t)
			filter, err := NewUtilizationFilter("", test.params)
			require.NoError(t, err)

			got := filter.Filter(ctx, nil, test.endpoints())

			assert.Equal(t, test.want, endpointNames(got))
		})
	}
}

func TestNewUtilizationFilter_InvalidParameters(t *testing.T) {
	tests := []struct {
		name   string
		params parameters
	}{
		{name: "no conditions", params: parameters{}},
		{name: "missing metric", params: parameters{Conditions: []condition{{MaxValue: float64Ptr(1)}}}},
		{name: "unknown metric", params: parameters{Conditions: []condition{condCap("unknown", 1)}}},
		{name: "missing maxValue", params: parameters{Conditions: []condition{{Metric: MetricActiveRequests}}}},
		{name: "negative maxValue", params: parameters{Conditions: []condition{condCap(MetricActiveRequests, -1)}}},
		{name: "kv-cache maxValue above 1", params: parameters{Conditions: []condition{condCap(MetricKVCacheUtilization, 80)}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewUtilizationFilter("", test.params)
			assert.Error(t, err)
		})
	}
}

func TestUtilizationFilter_Factory(t *testing.T) {
	ctx := testutils.NewTestContext(t)

	decoder := json.NewDecoder(bytes.NewBufferString(
		`{"conditions": [{"metric": "running-requests", "maxValue": 40}, {"metric": "waiting-queue", "maxValue": 0}]}`))
	p, err := Factory("test-filter", decoder, testutils.NewTestHandle(ctx))
	require.NoError(t, err)

	filter, ok := p.(*UtilizationFilter)
	require.True(t, ok)
	assert.Equal(t, UtilizationFilterType, filter.TypedName().Type)
	assert.Equal(t, "test-filter", filter.TypedName().Name)
	require.Len(t, filter.conditions, 2)
	assert.Equal(t, MetricRunningRequests, filter.conditions[0].Metric)
	assert.Equal(t, 40.0, *filter.conditions[0].MaxValue)
}

func TestUtilizationFilter_FactoryRejectsMissingParameters(t *testing.T) {
	ctx := testutils.NewTestContext(t)

	_, err := Factory("test-filter", nil, testutils.NewTestHandle(ctx))
	assert.Error(t, err)
}

func TestUtilizationFilter_Consumes(t *testing.T) {
	t.Run("active-requests requires the in-flight load attribute", func(t *testing.T) {
		filter, err := NewUtilizationFilter("", parameters{Conditions: []condition{
			condCap(MetricWaitingQueue, 0),
			condCap(MetricActiveRequests, 1),
		}})
		require.NoError(t, err)

		consumes := filter.Consumes()

		require.Len(t, consumes.Required, 1)
		assert.Equal(t, attrconcurrency.InFlightLoad{}, consumes.Required[attrconcurrency.InFlightLoadDataKey])
	})

	t.Run("model-server metrics require no producer", func(t *testing.T) {
		for _, metric := range []string{MetricRunningRequests, MetricWaitingQueue, MetricKVCacheUtilization} {
			filter, err := NewUtilizationFilter("", parameters{Conditions: []condition{condCap(metric, 1)}})
			require.NoError(t, err)

			consumes := filter.Consumes()

			assert.Empty(t, consumes.Required, "metric %s", metric)
			assert.Empty(t, consumes.Optional, "metric %s", metric)
		}
	})
}
