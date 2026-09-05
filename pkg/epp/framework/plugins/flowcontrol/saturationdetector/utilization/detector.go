/*
Copyright 2025 The Kubernetes Authors.

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
	return NewDetector(name, *cfg, log.FromContext(handle.Context())), nil
}

var (
	_ fwksched.Filter                = &Detector{}
	_ flowcontrol.SaturationDetector = &Detector{}
)

// Detector determines system saturation based on metrics of the given candidate pods.
type Detector struct {
	config    Config
	typedName fwkplugin.TypedName
	logger    logr.Logger

	// lastStaleWarnNanos is the wall-clock time (UnixNano) of the most recent stale-metrics log,
	// accessed atomically because Saturation may be called concurrently.
	lastStaleWarnNanos atomic.Int64
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
		"headroom", cfg.Headroom)

	if cfg.Headroom > 1.0 {
		pluginLogger.Info("Unusually high headroom configured; verify value is a fraction, not a percentage",
			"headroom", cfg.Headroom,
			"effectiveBurst", fmt.Sprintf("%.0f%%", cfg.Headroom*100))
	}

	return &Detector{
		config:    cfg,
		typedName: typedName,
		logger:    pluginLogger,
	}
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
//	PodScore = Max(WaitingQueue / QueueThreshold, KVCacheUsage / KVCacheThreshold)
func (d *Detector) Saturation(_ context.Context, candidates []datalayer.Endpoint) float64 {
	if len(candidates) == 0 {
		// No candidates means no stale endpoints. Keeping the gauge current here prevents a stale
		// reading from a previous evaluation misattributing an empty-pool stall to a metrics
		// collection failure.
		metrics.RecordFlowControlStaleEndpoints(d.typedName.Name, 0)
		return 1.0
	}

	var totalScore float64
	staleCount := 0
	for _, e := range candidates {
		podMetrics := e.GetMetrics()

		if podMetrics == nil || time.Since(podMetrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			// Fail closed: an endpoint whose metrics are missing or stale scores as fully saturated. A
			// fleet-wide metrics collection failure therefore halts dispatch entirely rather than
			// admitting blind; the gauge and the rate-limited log below exist so operators can tell that
			// stall apart from genuine overload (which typically scores above 1.0).
			totalScore += 1.0
			staleCount++
			continue
		}

		qRatio := float64(podMetrics.WaitingQueueSize) / float64(d.config.QueueDepthThreshold)
		kvRatio := podMetrics.KVCacheUsagePercent / d.config.KVCacheUtilThreshold

		// Roofline Analysis: The pod is saturated if either resource is exhausted.
		totalScore += max(qRatio, kvRatio)
	}

	metrics.RecordFlowControlStaleEndpoints(d.typedName.Name, staleCount)
	if staleCount > 0 {
		d.maybeLogStaleEndpoints(staleCount, len(candidates))
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
		"Endpoints with missing or stale metrics are scored as fully saturated (fail-closed); "+
			"if dispatch is stalled, check model-server metrics collection (scrape path, port, TLS, auth)",
		"staleEndpoints", staleCount,
		"totalEndpoints", total,
		"metricsStalenessThreshold", d.config.MetricsStalenessThreshold.String())
}

// Filter blocks traffic to specific pods that are physically saturated or exceeding their safety limits.
//
// It applies a relaxed limit (Threshold * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
func (d *Detector) Filter(
	_ context.Context,
	_ *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint,
) []fwksched.Endpoint {
	qLimit := float64(d.config.QueueDepthThreshold) * (1.0 + d.config.Headroom)
	kvLimit := d.config.KVCacheUtilThreshold * (1.0 + d.config.Headroom)

	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]fwksched.Endpoint, 0, len(endpoints))

	for _, endpoint := range endpoints {
		podMetrics := endpoint.GetMetrics()
		if podMetrics == nil || time.Since(podMetrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			continue
		}

		if float64(podMetrics.WaitingQueueSize) < qLimit && podMetrics.KVCacheUsagePercent < kvLimit {
			filtered = append(filtered, endpoint)
		}
	}
	if len(filtered) == 0 {
		return endpoints
	}
	return filtered
}
