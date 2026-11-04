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

package metrics

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const (

	// --- Internal Keys (for Legacy/Gauge Usage) ---
	KVCacheUsagePercentKey = "KVCacheUsagePercent"
	WaitingQueueSizeKey    = "WaitingQueueSize"
	RunningRequestsSizeKey = "RunningRequestsSize"
	ActiveModelsKey        = "ActiveModels"
	WaitingModelsKey       = "WaitingModels"

	// LoRA metrics based on MSP
	LoraInfoRunningAdaptersMetricName = "running_lora_adapters"
	LoraInfoWaitingAdaptersMetricName = "waiting_lora_adapters"
	LoraInfoMaxAdaptersMetricName     = "max_lora"

	CacheConfigBlockSizeInfoMetricName   = "block_size"
	CacheConfigNumGPUBlocksMetricName    = "num_gpu_blocks"
	CacheConfigPrefixMatchUnitMetricName = "prefix_match_unit"
)

// Extractor implements the metrics extraction based on the model
// server protocol standard.
type Extractor struct {
	typedName      fwkplugin.TypedName
	registry       *MappingRegistry
	engineLabelKey string
}

// NewCoreMetricsExtractor returns a new model server protocol (MSP) metrics extractor,
// configured with the given metrics' registry.
func NewCoreMetricsExtractor(registry *MappingRegistry, engineLabelKey string) (*Extractor, error) {
	if registry == nil {
		return nil, errors.New("mapping registry cannot be nil")
	}
	if engineLabelKey == "" {
		engineLabelKey = DefaultEngineTypeLabelKey
	}
	return &Extractor{
		typedName: fwkplugin.TypedName{
			Type: MetricsExtractorType,
			Name: MetricsExtractorType,
		},
		registry:       registry,
		engineLabelKey: engineLabelKey,
	}, nil
}

// TypedName returns the type and name of the metrics.Extractor.
func (ext *Extractor) TypedName() fwkplugin.TypedName {
	return ext.typedName
}

var _ fwkplugin.ProducerPlugin = &Extractor{}

// Produces declares the built-in and custom scalar metric attributes stored in
// the endpoint's AttributeMap as ScalarMetricValue, plus the LoRA adapter maps
// written into the endpoint's Metrics struct that scorers declare as Consumes()
// dependencies. Declaring all of them lets the data-attribute registry and the
// DAG builder validate the scorers and filters that read them.
func (ext *Extractor) Produces() map[fwkplugin.DataKey]any {
	produced := map[fwkplugin.DataKey]any{
		attrmetrics.ScalarMetricDataKey(KVCacheUsagePercentKey):      attrmetrics.ScalarMetricValue(0),
		attrmetrics.ScalarMetricDataKey(WaitingQueueSizeKey):         attrmetrics.ScalarMetricValue(0),
		attrmetrics.ScalarMetricDataKey(RunningRequestsSizeKey):      attrmetrics.ScalarMetricValue(0),
		fwkplugin.NewDataKey(ActiveModelsKey, MetricsExtractorType):  map[string]int{},
		fwkplugin.NewDataKey(WaitingModelsKey, MetricsExtractorType): map[string]int{},
	}
	for _, mapping := range ext.registry.Mappings() {
		for _, custom := range mapping.CustomMetrics {
			produced[attrmetrics.ScalarMetricDataKey(custom.AttributeKey)] = attrmetrics.ScalarMetricValue(0)
		}
	}
	return produced
}

var _ sourcemetrics.FamilyReader = &Extractor{}

// MetricFamilies lists the families of every registered engine mapping. An endpoint's engine
// is known only at extraction time.
func (ext *Extractor) MetricFamilies() []string {
	var names []string
	for _, mapping := range ext.registry.Mappings() {
		names = append(names, mapping.MetricNames()...)
		for _, spec := range []*Spec{mapping.CacheBlockSize, mapping.CacheNumBlocks} {
			if spec != nil {
				names = append(names, spec.Name)
			}
		}
	}
	return names
}

// Extract transforms the typed metrics payload into endpoint attributes.
func (ext *Extractor) Extract(ctx context.Context, in fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]) error {
	families := in.Payload
	ep := in.Endpoint

	engineType := getEngineTypeFromEndpoint(ep, ext.engineLabelKey)
	mapping, ok := ext.registry.Get(engineType)
	if !ok {
		return fmt.Errorf("no mapping found for engine type %q and no default mapping registered", engineType)
	}

	var errs []error
	current := ep.GetMetrics()
	clone := current.Clone()
	updated := false

	if spec := mapping.TotalQueuedRequests; spec != nil { // extract queued requests
		if value, err := spec.aggregateMetric(families, aggregateSum); err != nil {
			errs = append(errs, err)
		} else {
			clone.WaitingQueueSize = int(value)
			putBuiltinScalarAttribute(ep, mapping, WaitingQueueSizeKey, value)
			updated = true
		}
	}

	if spec := mapping.TotalRunningRequests; spec != nil { // extract running requests
		if value, err := spec.aggregateMetric(families, aggregateSum); err != nil {
			errs = append(errs, err)
		} else {
			clone.RunningRequestsSize = int(value)
			putBuiltinScalarAttribute(ep, mapping, RunningRequestsSizeKey, value)
			updated = true
		}
	}

	if spec := mapping.KVCacheUtilization; spec != nil { // extract KV cache usage
		if value, err := spec.aggregateMetric(families, aggregateMax); err != nil {
			errs = append(errs, err)
		} else {
			clone.KVCacheUsagePercent = value
			putBuiltinScalarAttribute(ep, mapping, KVCacheUsagePercentKey, value)
			updated = true
		}
	}

	if spec := mapping.LoraRequestInfo; spec != nil { // extract LoRA-specific metrics
		if metric := spec.getLatestMetric(families); metric != nil {
			populateLoRAMetrics(clone, metric, &errs)
			updated = true
		}
	}

	if spec := mapping.CacheInfo; spec != nil { // extract CacheInfo-specific metrics (labels)
		metric, err := spec.getLatestMetric(families)
		if err != nil {
			errs = append(errs, err)
		} else if metric != nil {
			blockSizeLabel := mapping.CacheBlockSizeLabel
			if blockSizeLabel == "" {
				blockSizeLabel = CacheConfigBlockSizeInfoMetricName
			}
			numBlocksLabel := mapping.CacheNumBlocksLabel
			if numBlocksLabel == "" {
				numBlocksLabel = CacheConfigNumGPUBlocksMetricName
			}
			populateCacheInfoMetrics(clone, metric, blockSizeLabel, numBlocksLabel, &errs)
			updated = true
		}
	}

	if spec := mapping.CacheBlockSize; spec != nil { // extract block size as direct gauge value
		if metric, err := spec.getLatestMetric(families); err != nil {
			errs = append(errs, err)
		} else {
			clone.CacheBlockSize = int(extractValue(metric))
			updated = true
		}
	}

	if spec := mapping.CacheNumBlocks; spec != nil { // extract num GPU blocks as direct gauge value
		if metric, err := spec.getLatestMetric(families); err != nil {
			errs = append(errs, err)
		} else {
			clone.CacheNumBlocks = int(extractValue(metric))
			updated = true
		}
	}

	trace := log.FromContext(ctx).V(logutil.TRACE)
	for _, custom := range mapping.CustomMetrics {
		var value float64
		var err error
		if custom.Aggregation == metricAggregationSum {
			value, err = custom.Spec.aggregateMetric(families, aggregateSum)
		} else {
			var metric *dto.Metric
			metric, err = custom.Spec.getLatestMetric(families)
			if err == nil {
				value = extractValue(metric)
			}
		}
		if err != nil {
			if !custom.Optional {
				errs = append(errs, fmt.Errorf("customMetrics entry %q: %w", custom.AttributeKey, err))
			} else if trace.Enabled() {
				trace.Info("Skipped optional customMetrics entry", "attributeKey", custom.AttributeKey, "reason", err.Error())
			}
			continue
		}
		ep.GetAttributes().Put(attrmetrics.ScalarMetricDataKey(custom.AttributeKey), attrmetrics.ScalarMetricValue(value))
		updated = true
	}

	if updated {
		clone.UpdateTime = time.Now()
		// Guarded: this runs per endpoint per scrape tick, and the logger
		// construction, MetricNames slice, and boxed args all allocate even
		// when TRACE is off.
		if trace.Enabled() {
			trace.Info("Refreshed metrics",
				"endpoint", ep.GetMetadata().ID,
				"metrics", mapping.MetricNames(),
				"updated", clone,
			)
		}
		ep.UpdateMetrics(clone)
	}

	if len(errs) != 0 {
		return errors.Join(errs...)
	}
	return nil
}

// putBuiltinScalarAttribute writes a built-in scalar metric to the endpoint's
// attribute map unless a customMetrics entry owns that attributeKey. Skipping
// the built-in write keeps a failed custom scrape from falling back to the
// built-in value and overwriting the custom metric's last-known value.
func putBuiltinScalarAttribute(ep fwkdl.Endpoint, mapping *Mapping, key string, value float64) {
	if mapping.hasCustomMetric(key) {
		return
	}
	ep.GetAttributes().Put(attrmetrics.ScalarMetricDataKey(key), attrmetrics.ScalarMetricValue(value))
}

// getEngineTypeFromEndpoint extracts the engine type from endpoint metadata labels.
func getEngineTypeFromEndpoint(ep fwkdl.Endpoint, labelKey string) string {
	meta := ep.GetMetadata()
	if meta == nil || meta.Labels == nil {
		return DefaultEngineType
	}

	engineType, ok := meta.Labels[labelKey]
	if !ok || engineType == "" {
		return DefaultEngineType
	}

	return engineType
}

// populateLoRAMetrics updates the metrics with LoRA adapter info from the metric labels.
func populateLoRAMetrics(clone *fwkdl.Metrics, metric *dto.Metric, errs *[]error) {
	clone.ActiveModels = map[string]int{}
	clone.WaitingModels = map[string]int{}

	for _, label := range metric.GetLabel() {
		switch label.GetName() {
		case LoraInfoRunningAdaptersMetricName:
			addAdapters(clone.ActiveModels, label.GetValue())
		case LoraInfoWaitingAdaptersMetricName:
			addAdapters(clone.WaitingModels, label.GetValue())
		case LoraInfoMaxAdaptersMetricName:
			if label.GetValue() != "" {
				if val, err := strconv.Atoi(label.GetValue()); err == nil {
					clone.MaxActiveModels = val
				} else {
					*errs = append(*errs, err)
				}
			}
		}
	}
}

// populateCacheInfoMetrics updates the metrics with cache info from the metric labels.
// blockSizeLabelName and numBlocksLabelName allow engines to carry the values under
// label names other than "block_size" and "num_gpu_blocks".
func populateCacheInfoMetrics(clone *fwkdl.Metrics, metric *dto.Metric, blockSizeLabelName, numBlocksLabelName string, errs *[]error) {
	clone.CacheBlockSize = 0
	clone.CachePrefixMatchUnit = 0
	for _, label := range metric.GetLabel() {
		switch label.GetName() {
		case blockSizeLabelName:
			if label.GetValue() != "" {
				if val, err := strconv.Atoi(label.GetValue()); err == nil {
					clone.CacheBlockSize = val
				} else {
					*errs = append(*errs, err)
				}
			}
		case numBlocksLabelName:
			if label.GetValue() != "" {
				if val, err := strconv.Atoi(label.GetValue()); err == nil {
					clone.CacheNumBlocks = val
				} else {
					*errs = append(*errs, err)
				}
			}
		case CacheConfigPrefixMatchUnitMetricName:
			if label.GetValue() != "" && label.GetValue() != "None" {
				if val, err := strconv.Atoi(label.GetValue()); err == nil {
					clone.CachePrefixMatchUnit = val
				} else {
					*errs = append(*errs, err)
				}
			}
		}
	}
}

// addAdapters splits a comma-separated adapter list and stores keys with default value 0.
func addAdapters(m map[string]int, csv string) {
	for name := range strings.SplitSeq(csv, ",") {
		if trimmed := strings.TrimSpace(name); trimmed != "" {
			m[trimmed] = 0
		}
	}
}
