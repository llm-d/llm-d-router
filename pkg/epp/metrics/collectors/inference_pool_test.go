/*
Copyright 2025 The Kubernetes Authors.
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

package collectors

import (
	"context"
	"strings"
	"testing"
	"time"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-base/metrics/testutil"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
)

var (
	pod1 = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
		},
	}
	pod1NamespacedName = types.NamespacedName{Name: pod1.Name + "-rank-0", Namespace: pod1.Namespace}
	pod1Metrics        = &fwkdl.Metrics{
		WaitingQueueSize:    100,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
	}
)

func TestNoMetricsCollected(t *testing.T) {
	period := time.Second
	factories := []datalayer.EndpointFactory{
		datalayer.NewTestRuntime(t, period),
	}
	for _, epf := range factories {
		ds := datastore.NewDatastore(context.Background(), epf)

		collector := &inferencePoolMetricsCollector{
			ds: ds,
		}

		if err := testutil.CollectAndCompare(collector, strings.NewReader(""), ""); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMetricsCollected(t *testing.T) {
	metrics := map[types.NamespacedName]*fwkdl.Metrics{
		pod1NamespacedName: pod1Metrics,
	}
	period := time.Millisecond
	mockDS := &mocks.MetricsDataSource{}
	mockDS.SetMetrics(metrics)
	factories := []datalayer.EndpointFactory{
		datalayer.NewTestRuntimeWithConfig(t, period, &datalayer.Config{
			Sources: []datalayer.DataSourceConfig{
				{Plugin: mockDS},
			},
		}),
	}
	for _, epf := range factories {
		inferencePool := &v1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-pool",
			},
			Spec: v1.InferencePoolSpec{
				TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
			},
		}
		ds := datastore.NewDatastore(context.Background(), epf)

		scheme := runtime.NewScheme()
		fakeClient := fake.NewClientBuilder().
			WithScheme(scheme).
			Build()

		_ = ds.PoolSet(context.Background(), fakeClient, poolutil.InferencePoolToEndpointPool(inferencePool))
		_ = ds.PodUpdateOrAddIfNotExist(context.Background(), pod1)

		time.Sleep(1 * time.Second)

		collector := &inferencePoolMetricsCollector{
			ds: ds,
		}
		err := promtestutil.CollectAndCompare(collector, strings.NewReader(`
		# HELP llm_d_epp_per_endpoint_queue_size [ALPHA] The total number of requests pending in the model server queue for each underlying endpoint.
		# TYPE llm_d_epp_per_endpoint_queue_size gauge
		llm_d_epp_per_endpoint_queue_size{model_server_endpoint="pod1-rank-0",name="test-pool"} 100
`), "llm_d_epp_per_endpoint_queue_size")
		if err != nil {
			t.Fatal(err)
		}
	}
}

// TestNixlFailureCountersCollected verifies that the collector exposes the
// NIXL failure counters for an endpoint carrying them, and no NIXL series for
// an endpoint that does not.
func TestNixlFailureCountersCollected(t *testing.T) {
	pod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod2",
		},
	}
	inferencePool := &v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pool",
		},
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
		},
	}
	ds := datastore.NewDatastore(context.Background(), datalayer.NewTestRuntime(t, time.Second))
	fakeClient := fake.NewClientBuilder().
		WithScheme(runtime.NewScheme()).
		Build()

	_ = ds.PoolSet(context.Background(), fakeClient, poolutil.InferencePoolToEndpointPool(inferencePool))
	_ = ds.PodUpdateOrAddIfNotExist(context.Background(), pod1)
	_ = ds.PodUpdateOrAddIfNotExist(context.Background(), pod2)

	reporting := ds.PodList(func(ep fwkdl.Endpoint) bool {
		return ep.GetMetadata().ID == pod1NamespacedName
	})
	if len(reporting) != 1 {
		t.Fatalf("expected one endpoint for %v, got %d", pod1NamespacedName, len(reporting))
	}
	attrs := reporting[0].GetAttributes()
	attrs.Put(attrmetrics.NixlFailedTransfersDataKey, attrmetrics.ScalarMetricValue(44))
	attrs.Put(attrmetrics.NixlFailedNotificationsDataKey, attrmetrics.ScalarMetricValue(22))
	attrs.Put(attrmetrics.NixlKVExpiredRequestsDataKey, attrmetrics.ScalarMetricValue(3))

	collector := &inferencePoolMetricsCollector{
		ds: ds,
	}
	err := promtestutil.CollectAndCompare(collector, strings.NewReader(`
		# HELP llm_d_epp_per_endpoint_nixl_failed_notifications_total [ALPHA] The number of failed NIXL KV cache notifications reported by each underlying endpoint.
		# TYPE llm_d_epp_per_endpoint_nixl_failed_notifications_total counter
		llm_d_epp_per_endpoint_nixl_failed_notifications_total{model_server_endpoint="pod1-rank-0",name="test-pool"} 22
		# HELP llm_d_epp_per_endpoint_nixl_failed_transfers_total [ALPHA] The number of failed NIXL KV cache transfers reported by each underlying endpoint.
		# TYPE llm_d_epp_per_endpoint_nixl_failed_transfers_total counter
		llm_d_epp_per_endpoint_nixl_failed_transfers_total{model_server_endpoint="pod1-rank-0",name="test-pool"} 44
		# HELP llm_d_epp_per_endpoint_nixl_kv_expired_requests_total [ALPHA] The number of requests whose KV cache expired before it was read, reported by each underlying endpoint.
		# TYPE llm_d_epp_per_endpoint_nixl_kv_expired_requests_total counter
		llm_d_epp_per_endpoint_nixl_kv_expired_requests_total{model_server_endpoint="pod1-rank-0",name="test-pool"} 3
`),
		"llm_d_epp_per_endpoint_nixl_failed_transfers_total",
		"llm_d_epp_per_endpoint_nixl_failed_notifications_total",
		"llm_d_epp_per_endpoint_nixl_kv_expired_requests_total",
	)
	if err != nil {
		t.Fatal(err)
	}
}
