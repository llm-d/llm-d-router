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
// Package utilization implements a reactive saturation detector and scheduling filter for LLM
// routing. It evaluates endpoint telemetry (queue depth and KV cache memory utilization) using a
// roofline model to determine physical system saturation and apply proportional backpressure.
//
// For detailed architectural trade-offs and configuration, see the package README.
package utilization

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// UtilizationDetectorType is the unique identifier for this plugin.
	UtilizationDetectorType = "utilization-detector"

	// staleWarnInterval bounds how often the detector logs about endpoints with missing or stale
	// metrics. Saturation is evaluated every dispatch cycle (~1ms), so the condition must be logged
	// on a time budget, never per evaluation.
	staleWarnInterval = 30 * time.Second
)

// UtilizationDetectorFactory instantiates the detector plugin using the provided JSON parameters.
func UtilizationDetectorFactory(
	name string,
	params *json.Decoder,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	var apiCfg apiConfig
	if params != nil {
		if err := params.Decode(&apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal utilization detector config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	logger := log.FromContext(handle.Context())
	if refresh := handle.RefreshMetricsInterval(); refresh > cfg.MetricsStalenessThreshold {
		logger.Info("Warning: refresh-metrics-interval exceeds metricsStalenessThreshold; "+
			"metrics will be stale between polls",
			"detector", name,
			"refreshMetricsInterval", refresh.String(),
			"metricsStalenessThreshold", cfg.MetricsStalenessThreshold.String())
	}
	return NewDetector(name, *cfg, logger), nil
}

var (
	_ fwksched.Filter                = &Detector{}
	_ flowcontrol.SaturationDetector = &Detector{}
	_ fwkplugin.ConsumerPlugin       = &Detector{}
)

// Detector determines system saturation based on metrics of the given candidate pods.
type Detector struct {
	config              Config
	typedName           fwkplugin.TypedName
	inFlightLoadDataKey fwkplugin.DataKey
	logger              logr.Logger

	// lastStaleWarnNanos is the wall-clock time (UnixNano) of the most recent stale-metrics log,
	// accessed atomically because Saturation may be called concurrently.
	lastStaleWarnNanos atomic.Int64

	// lastKVCapacityWarnNanos is the wall-clock time (UnixNano) of the most recent log about a KV
	// cache growth projection that cannot be applied because an endpoint reports no cache capacity.
	lastKVCapacityWarnNanos atomic.Int64
}

// NewDetector creates a new instance of the Utilization Detector.
// The config provides the thresholds for determining saturation.
func NewDetector(name string, cfg Config, logger logr.Logger) *Detector {
	typedName := fwkplugin.TypedName{
		Type: UtilizationDetectorType,
		Name: name,
	}

	pluginLogger := logger.WithName(typedName.String())
	pluginLogger.V(logutil.DEFAULT).Info("Creating new UtilizationDetector",
		"queueDepthThreshold", cfg.QueueDepthThreshold,
		"kvCacheUtilThreshold", cfg.KVCacheUtilThreshold,
		"metricsStalenessThreshold", cfg.MetricsStalenessThreshold.String(),
		"stalenessPolicy", cfg.StalenessPolicy,
		"headroom", cfg.Headroom)

	if cfg.Headroom > 1.0 {
		pluginLogger.Info("Unusually high headroom configured; verify value is a fraction, not a percentage",
			"headroom", cfg.Headroom,
			"effectiveBurst", fmt.Sprintf("%.0f%%", cfg.Headroom*100))
	}

	return &Detector{
		config:              cfg,
		typedName:           typedName,
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
		logger:              pluginLogger,
	}
}

// Consumes declares InFlightLoad as an optional dependency: it supplies the
// scrape-lag compensation and the projected KV cache growth in Saturation and Filter, but the
// detector still functions on scraped metrics alone when no inflight-load-producer is configured.
// DeferredQueueSize is optional too: only engine configs with a deferredRequestsSpec publish it.
func (d *Detector) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{
			d.inFlightLoadDataKey:    attrconcurrency.InFlightLoad{},
			deferredQueueSizeDataKey: attrmetrics.ScalarMetricValue(0),
		},
	}
}

var deferredQueueSizeDataKey = attrmetrics.ScalarMetricDataKey(extractormetrics.DeferredQueueSizeKey)

// inFlightLoad returns the load the inflight-load-producer published for the endpoint, or nil.
func (d *Detector) inFlightLoad(attrs datalayer.AttributeMap) *attrconcurrency.InFlightLoad {
	if attrs == nil {
		return nil
	}
	if val, ok := attrs.Get(d.inFlightLoadDataKey); ok {
		if load, ok := val.(*attrconcurrency.InFlightLoad); ok {
			return load
		}
	}
	return nil
}

// queueDepth returns the scraped waiting queue less DeferredQueueSize, plus
// the in-flight requests the scrape does not yet reflect (see Saturation). A
// deferred count above the waiting queue comes from series scraped out of
// step and leaves the queue term unchanged.
func queueDepth(m *datalayer.Metrics, attrs datalayer.AttributeMap, load *attrconcurrency.InFlightLoad) int64 {
	waiting := int64(m.WaitingQueueSize)
	if attrs == nil {
		return waiting
	}
	queued := waiting
	if deferred, ok := attrmetrics.ReadScalarMetricValue(attrs, deferredQueueSizeDataKey); ok && deferred >= 0 && float64(deferred) <= float64(waiting) {
		queued = waiting - int64(deferred)
	}
	if load != nil {
		queued += max(0, load.Requests-waiting-int64(m.RunningRequestsSize))
	}
	return queued
}

// projectedKVCacheUsage returns the share of the KV cache the endpoint is projected to need at its
// peak (see Saturation), or zero when the inflight-load-producer publishes no projection for the
// endpoint or the endpoint does not report its cache capacity.
func (d *Detector) projectedKVCacheUsage(
	metadata *datalayer.EndpointMetadata, m *datalayer.Metrics, load *attrconcurrency.InFlightLoad,
) float64 {
	if load == nil || load.KVGrowthTokens <= 0 {
		return 0
	}
	if m.CacheBlockSize <= 0 || m.CacheNumBlocks <= 0 {
		d.maybeLogMissingKVCapacity(metadata)
		return 0
	}
	return m.KVCacheUsagePercent + float64(load.KVGrowthTokens)/(float64(m.CacheBlockSize)*float64(m.CacheNumBlocks))
}

// maybeLogMissingKVCapacity logs at most once per staleWarnInterval that an endpoint's KV cache
// growth projection is not applied.
func (d *Detector) maybeLogMissingKVCapacity(metadata *datalayer.EndpointMetadata) {
	now := time.Now().UnixNano()
	last := d.lastKVCapacityWarnNanos.Load()
	if now-last < int64(staleWarnInterval) || !d.lastKVCapacityWarnNanos.CompareAndSwap(last, now) {
		return
	}
	var endpoint string
	if metadata != nil {
		endpoint = metadata.ID.String()
	}
	d.logger.V(logutil.DEFAULT).Info(
		"KV cache growth projection is not applied: the endpoint reports no KV cache capacity; "+
			"check that the metrics mapping scrapes the cache block size and block count",
		"endpoint", endpoint)
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *Detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

// Saturation calculates the saturation level of the pool.
//
// It returns an aggregate saturation signal where:
//
//	Saturation = Average(PodSaturationScore)
//
// For each pod, the score is determined by the most constrained resource (Compute or Memory):
//
//	PodScore = Max(WaitingQueue / QueueThreshold, KVCacheUsage / KVCacheThreshold, ProjectedKVCacheUsage)
//
// WaitingQueue is WaitingQueueSize less the DeferredQueueSize attribute when the extractor
// publishes one.
//
// Pods with missing or stale metrics contribute 1.0 under the saturated policy, or are excluded
// from the average under the ignore policy.
//
// # Scrape-lag compensation
//
// WaitingQueueSize is scraped on a poller (default 50ms) while the flow
// controller's dispatch loop runs at ~1ms, so between two scrapes the queue term
// is stale-low and the gate would let the controller over-dispatch. When an
// inflight-load-producer is configured, the detector corrects for this using the
// producer's per-endpoint in-flight request count, which is incremented at
// dispatch and decremented on request completion (so it already accounts for
// requests that returned since the last scrape). The lag is the tracked
// in-flight requests the scrape does not yet reflect:
//
//	credit = max(0, InFlightRequests - (WaitingQueueSize + RunningRequestsSize))
//
// The credit corrects each endpoint's score before aggregation. It is added
// only to the queue-depth term (the fast, controllable signal); the KV-cache
// term is left as measured. Endpoints without the attribute contribute zero
// credit and fall back to the scraped queue depth. See the package README for
// the assumptions the compensation rests on.
//
// # Projected KV cache growth
//
// KVCacheUsage is the usage at the time of the scrape, and a running sequence
// keeps allocating blocks while it decodes. When the inflight-load-producer
// publishes a growth projection (its kvGrowthProjection parameter), the score
// also covers what the endpoint is projected to need:
//
//	ProjectedKVCacheUsage = KVCacheUsage + KVGrowthTokens / KVCacheCapacity
//
// KVGrowthTokens is the producer's projection of the tokens the endpoint will
// need beyond its scraped usage, and KVCacheCapacity is the scraped block count
// times the block size. The term reaches 1.0 when the projected need equals the
// capacity; KVCacheThreshold applies to the measured usage only.
func (d *Detector) Saturation(ctx context.Context, candidates []datalayer.Endpoint) float64 {
	stage := flowcontrol.SaturationStageFromContext(ctx)
	if len(candidates) == 0 {
		// No candidates means no stale endpoints. Keeping the gauge current here prevents a stale
		// reading from a previous evaluation misattributing an empty-pool stall to a metrics
		// collection failure.
		metrics.RecordFlowControlStaleEndpoints(d.typedName.Name, stage, 0)
		return 1.0
	}

	var totalScore float64
	staleCount := 0
	for _, e := range candidates {
		podMetrics := e.GetMetrics()

		if podMetrics == nil || time.Since(podMetrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			if d.config.StalenessPolicy != StalenessIgnore {
				// Saturated: an endpoint whose metrics are missing or stale scores as fully
				// saturated. A fleet-wide metrics collection failure therefore halts dispatch
				// entirely rather than admitting blind; the gauge and the rate-limited log below
				// exist so operators can tell that stall apart from genuine overload (which
				// typically scores above 1.0).
				totalScore += 1.0
			}
			staleCount++
			continue
		}

		load := d.inFlightLoad(e.GetAttributes())
		qRatio := float64(queueDepth(podMetrics, e.GetAttributes(), load)) / float64(d.config.QueueDepthThreshold)
		kvRatio := podMetrics.KVCacheUsagePercent / d.config.KVCacheUtilThreshold

		// Roofline Analysis: The pod is saturated if either resource is exhausted.
		totalScore += max(qRatio, kvRatio, d.projectedKVCacheUsage(e.GetMetadata(), podMetrics, load))
	}

	metrics.RecordFlowControlStaleEndpoints(d.typedName.Name, stage, staleCount)
	if staleCount > 0 {
		d.maybeLogStaleEndpoints(staleCount, len(candidates))
	}

	// Under ignore, stale endpoints are out of the average; if every candidate is stale the pool
	// scores 0.0 and dispatch continues, consistent with Filter's fail-open fallback.
	if d.config.StalenessPolicy == StalenessIgnore {
		fresh := len(candidates) - staleCount
		if fresh == 0 {
			return 0.0
		}
		return totalScore / float64(fresh)
	}

	return totalScore / float64(len(candidates))
}

// maybeLogStaleEndpoints logs the stale-metrics condition at most once per staleWarnInterval.
func (d *Detector) maybeLogStaleEndpoints(staleCount, total int) {
	now := time.Now().UnixNano()
	last := d.lastStaleWarnNanos.Load()
	if now-last < int64(staleWarnInterval) {
		return
	}
	if !d.lastStaleWarnNanos.CompareAndSwap(last, now) {
		return // Another goroutine logged concurrently.
	}
	d.logger.V(logutil.DEFAULT).Info(
		"Endpoints with missing or stale metrics observed; "+
			"if dispatch is stalled, check model-server metrics collection (scrape path, port, TLS, auth)",
		"staleEndpoints", staleCount,
		"totalEndpoints", total,
		"stalenessPolicy", d.config.StalenessPolicy,
		"metricsStalenessThreshold", d.config.MetricsStalenessThreshold.String())
}

// Filter blocks traffic to specific pods that are physically saturated or exceeding their safety limits.
//
// It applies a relaxed limit (Threshold * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
// The KV cache limit is capped at 1.0.
// The queue depth includes the same scrape-lag compensation as Saturation, and an endpoint whose
// projected KV cache usage reaches its capacity is blocked.
func (d *Detector) Filter(
	_ context.Context,
	_ *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint,
) []fwksched.Endpoint {
	qLimit := float64(d.config.QueueDepthThreshold) * (1.0 + d.config.Headroom)
	kvLimit := min(1.0, d.config.KVCacheUtilThreshold*(1.0+d.config.Headroom))

	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]fwksched.Endpoint, 0, len(endpoints))

	for _, endpoint := range endpoints {
		podMetrics := endpoint.GetMetrics()
		if podMetrics == nil || time.Since(podMetrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			continue
		}

		load := d.inFlightLoad(endpoint)
		if float64(queueDepth(podMetrics, endpoint, load)) < qLimit && podMetrics.KVCacheUsagePercent < kvLimit &&
			d.projectedKVCacheUsage(endpoint.GetMetadata(), podMetrics, load) < 1.0 {
			filtered = append(filtered, endpoint)
		}
	}
	if len(filtered) == 0 {
		return endpoints
	}
	return filtered
}
