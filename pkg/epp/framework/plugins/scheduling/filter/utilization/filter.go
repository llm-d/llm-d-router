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
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	// UtilizationFilterType is the type of the UtilizationFilter.
	UtilizationFilterType = "utilization-filter"

	// MetricActiveRequests is the in-flight request count tracked by the EPP:
	// every request routed to the endpoint and not yet seen complete, covering
	// both running and queued requests.
	MetricActiveRequests = "active-requests"
	// MetricRunningRequests is the running request count reported by the model server.
	MetricRunningRequests = "running-requests"
	// MetricWaitingQueue is the waiting queue size reported by the model server.
	MetricWaitingQueue = "waiting-queue"
	// MetricKVCacheUtilization is the KV cache usage fraction (0 to 1) reported
	// by the model server.
	MetricKVCacheUtilization = "kv-cache-utilization"
)

// condition caps a single utilization metric. An endpoint passes the
// condition when its metric value is at most MaxValue.
type condition struct {
	// Metric selects the utilization signal compared against MaxValue.
	Metric string `json:"metric"`

	// MaxValue is the maximum metric value an endpoint may have and still
	// pass the condition.
	MaxValue *float64 `json:"maxValue"`
}

// parameters defines the parameters for UtilizationFilter.
type parameters struct {
	// Conditions lists the metric caps an endpoint must all satisfy to be
	// kept. An endpoint is dropped when any condition's metric exceeds its
	// MaxValue.
	Conditions []condition `json:"conditions"`

	// FallbackOnEmpty returns the unfiltered candidates when every endpoint
	// was filtered out, so the request can still be routed somewhere.
	FallbackOnEmpty bool `json:"fallbackOnEmpty"`

	// InFlightLoadProducerName applies to the active-requests metric only.
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`
}

// compile-time type assertion
var _ scheduling.Filter = &UtilizationFilter{}
var _ plugin.ConsumerPlugin = &UtilizationFilter{}

// Factory defines the factory function for UtilizationFilter.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", UtilizationFilterType, err)
		}
	}

	return NewUtilizationFilter(name, params)
}

// NewUtilizationFilter validates the given parameters and returns a new
// UtilizationFilter with the given name.
func NewUtilizationFilter(name string, params parameters) (*UtilizationFilter, error) {
	if name == "" {
		name = UtilizationFilterType
	}
	if len(params.Conditions) == 0 {
		return nil, errors.New("utilization filter requires at least one condition")
	}
	for _, cond := range params.Conditions {
		switch cond.Metric {
		case MetricActiveRequests, MetricRunningRequests, MetricWaitingQueue, MetricKVCacheUtilization:
		default:
			return nil, fmt.Errorf("utilization filter condition metric must be one of %q, %q, %q, %q, got %q",
				MetricActiveRequests, MetricRunningRequests, MetricWaitingQueue, MetricKVCacheUtilization, cond.Metric)
		}
		if cond.MaxValue == nil {
			return nil, fmt.Errorf("utilization filter condition for %q requires maxValue", cond.Metric)
		}
		if *cond.MaxValue < 0 {
			return nil, fmt.Errorf("utilization filter condition for %q requires a non-negative maxValue, got %v",
				cond.Metric, *cond.MaxValue)
		}
		if cond.Metric == MetricKVCacheUtilization && *cond.MaxValue > 1 {
			return nil, fmt.Errorf("utilization filter maxValue for %q is a fraction and must be at most 1, got %v",
				MetricKVCacheUtilization, *cond.MaxValue)
		}
	}

	return &UtilizationFilter{
		typedName:           plugin.TypedName{Type: UtilizationFilterType, Name: name},
		conditions:          params.Conditions,
		fallbackOnEmpty:     params.FallbackOnEmpty,
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(params.InFlightLoadProducerName),
	}, nil
}

// UtilizationFilter filters candidate endpoints by utilization metrics. An
// endpoint is kept only when it satisfies every configured condition; it is
// dropped when any condition's metric exceeds its maximum. The metrics are
// the in-flight request count tracked by the EPP (produced by the
// inflight-load-producer data producer) and the model-server-reported running
// requests, waiting queue size, and KV cache utilization. Endpoints missing a
// metric are treated as having a zero value for it.
type UtilizationFilter struct {
	typedName           plugin.TypedName
	inFlightLoadDataKey plugin.DataKey

	// conditions lists the metric caps an endpoint must all satisfy to be kept.
	conditions []condition
	// fallbackOnEmpty returns the unfiltered candidates when every endpoint
	// was filtered out.
	fallbackOnEmpty bool
}

// TypedName returns the typed name of the plugin.
func (f *UtilizationFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Consumes returns the in-flight load attribute when a condition uses the
// active-requests metric. Model-server metrics are read directly from the
// endpoint and need no producer.
func (f *UtilizationFilter) Consumes() plugin.DataDependencies {
	for _, cond := range f.conditions {
		if cond.Metric == MetricActiveRequests {
			return plugin.DataDependencies{
				Required: map[plugin.DataKey]any{f.inFlightLoadDataKey: attrconcurrency.InFlightLoad{}},
			}
		}
	}
	return plugin.DataDependencies{}
}

// Filter keeps the endpoints that satisfy every condition. When all endpoints
// are filtered out and fallbackOnEmpty is set, the original candidates are
// returned.
func (f *UtilizationFilter) Filter(ctx context.Context, _ *scheduling.InferenceRequest,
	endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filtered := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if f.satisfiesAll(ctx, endpoint) {
			filtered = append(filtered, endpoint)
		}
	}

	log.FromContext(ctx).V(logutil.TRACE).Info("Filtered endpoints by utilization",
		"conditions", len(f.conditions), "candidates", len(endpoints), "kept", len(filtered))

	if len(filtered) == 0 && f.fallbackOnEmpty {
		return endpoints
	}
	return filtered
}

func (f *UtilizationFilter) satisfiesAll(ctx context.Context, endpoint scheduling.Endpoint) bool {
	for _, cond := range f.conditions {
		if f.metricValue(ctx, endpoint, cond.Metric) > *cond.MaxValue {
			return false
		}
	}
	return true
}

func (f *UtilizationFilter) metricValue(ctx context.Context, endpoint scheduling.Endpoint, metric string) float64 {
	if metric == MetricActiveRequests {
		return float64(f.activeRequestCount(ctx, endpoint))
	}

	metrics := endpoint.GetMetrics()
	if metrics == nil {
		return 0
	}
	switch metric {
	case MetricRunningRequests:
		return float64(metrics.RunningRequestsSize)
	case MetricWaitingQueue:
		return float64(metrics.WaitingQueueSize)
	case MetricKVCacheUtilization:
		return metrics.KVCacheUsagePercent
	default:
		// Unreachable: the metric is validated at construction time.
		return 0
	}
}

func (f *UtilizationFilter) activeRequestCount(ctx context.Context, endpoint scheduling.Endpoint) int64 {
	val, ok := endpoint.Get(f.inFlightLoadDataKey)
	if !ok {
		return 0
	}

	switch load := val.(type) {
	case *attrconcurrency.InFlightLoad:
		if load == nil {
			log.FromContext(ctx).V(logutil.TRACE).Info("Ignoring nil in-flight load attribute",
				"endpoint", endpoint.GetMetadata().ID.String())
			return 0
		}
		return load.Requests
	default:
		log.FromContext(ctx).V(logutil.TRACE).Info("Ignoring in-flight load attribute with unexpected type",
			"endpoint", endpoint.GetMetadata().ID.String(),
			"attributeType", fmt.Sprintf("%T", val))
		return 0
	}
}
