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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const (
	// MultiClusterMetricsExtractorType reads only a pool's aggregate metrics, not engine metrics.
	MultiClusterMetricsExtractorType = "multicluster-metrics-extractor"

	defaultKVCacheUtilizationMetric = "llm_d_epp_average_kv_cache_utilization"
	defaultQueueSizeMetric          = "llm_d_epp_average_queue_size"
)

type multiClusterMetricsExtractorParams struct {
	KVCacheUtilizationMetric string `json:"kvCacheUtilizationMetric"`
	QueueSizeMetric          string `json:"queueSizeMetric"`
}

type poolMetric struct {
	spec *Spec
	key  string
}

// MultiClusterMetricsExtractor writes a pool's aggregate KV-cache utilization and queue
// size to the attributes the multicluster scorers read. The stock per-pod fields are empty
// on a cluster-endpoint.
type MultiClusterMetricsExtractor struct {
	typedName fwkplugin.TypedName
	metrics   []poolMetric
}

func NewMultiClusterMetricsExtractor(name string, params *multiClusterMetricsExtractorParams) (*MultiClusterMetricsExtractor, error) {
	if params == nil {
		params = &multiClusterMetricsExtractorParams{}
	}
	kvSpec, err := parseStringToSpec(cmp.Or(params.KVCacheUtilizationMetric, defaultKVCacheUtilizationMetric))
	if err != nil {
		return nil, fmt.Errorf("kv-cache utilization metric: %w", err)
	}
	queueSpec, err := parseStringToSpec(cmp.Or(params.QueueSizeMetric, defaultQueueSizeMetric))
	if err != nil {
		return nil, fmt.Errorf("queue size metric: %w", err)
	}
	return &MultiClusterMetricsExtractor{
		typedName: fwkplugin.TypedName{Type: MultiClusterMetricsExtractorType, Name: name},
		metrics: []poolMetric{
			{spec: kvSpec, key: attrmetrics.MultiClusterKVCacheUtilizationKey},
			{spec: queueSpec, key: attrmetrics.MultiClusterQueueSizeKey},
		},
	}, nil
}

func (e *MultiClusterMetricsExtractor) TypedName() fwkplugin.TypedName {
	return e.typedName
}

// Extract writes each pool aggregate to its attribute. A missing metric is reported
// but does not stop the others.
func (e *MultiClusterMetricsExtractor) Extract(_ context.Context, in fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]) error {
	ep := in.Endpoint
	var errs []error
	for _, m := range e.metrics {
		metric, err := m.spec.getLatestMetric(in.Payload)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", m.key, err))
			continue
		}
		ep.GetAttributes().Put(m.key, attrmetrics.ScalarMetricValue(extractValue(metric)))
	}
	return errors.Join(errs...)
}

var _ fwkplugin.ProducerPlugin = &MultiClusterMetricsExtractor{}

// Produces advertises the pool attributes so the dependency graph can verify a
// scorer's required key has a producer.
func (e *MultiClusterMetricsExtractor) Produces() map[fwkplugin.DataKey]any {
	out := make(map[fwkplugin.DataKey]any, len(e.metrics))
	for _, m := range e.metrics {
		out[fwkplugin.NewDataKey(m.key, "")] = attrmetrics.ScalarMetricValue(0)
	}
	return out
}

// MultiClusterMetricsExtractorFactory instantiates the extractor from configuration.
func MultiClusterMetricsExtractorFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := &multiClusterMetricsExtractorParams{}
	if parameters != nil {
		if err := parameters.Decode(params); err != nil {
			return nil, err
		}
	}
	return NewMultiClusterMetricsExtractor(name, params)
}
