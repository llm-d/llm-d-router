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
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// UtilizationDetectorType is the unique identifier for this plugin.
	UtilizationDetectorType = "utilization-detector"
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
//	Saturation = Max(PodSaturationScore)   // hottest endpoint drives the pool
//
// For each pod, the score is determined by the most constrained resource (Compute or Memory):
//
//	PodScore = Max(WaitingQueue / QueueThreshold, KVCacheUsage / KVCacheThreshold)
//
// Aggregation is the MAX across endpoints (roofline at the pool level): the pool
// is saturated as soon as its hottest endpoint is, so a single overloaded
// endpoint is not diluted by idle ones. (Previously an unweighted average, which
// let one hot endpoint be averaged away by idle/prefill pods.)
func (d *Detector) Saturation(ctx context.Context, candidates []datalayer.Endpoint) float64 {
	return d.SaturationWithInFlight(ctx, candidates, 0)
}

// SaturationWithInFlight is Saturation, but with `inFlight` extra requests
// attributed to the pool's waiting queues. `inFlight` is the number of
// gate-passing dispatches the flow controller has issued that are not yet
// reflected in the scraped WaitingQueueSize (the metrics are refreshed on a
// poller, default 50ms, while the dispatch loop runs at ~1ms). Without this
// credit the controller bursts many dispatches against a stale-low waiting
// count before the next scrape catches up, overshooting the gate; crediting the
// in-flight count into the queue term makes the gate hold within one scrape
// window. The credit is spread evenly across the non-stale endpoints and only
// affects the queue-depth term (the fast, controllable signal); the KV-cache
// term is left as measured (it lags physically and is already conservative).
//
// Saturation(ctx, c) == SaturationWithInFlight(ctx, c, 0).
func (d *Detector) SaturationWithInFlight(_ context.Context, candidates []datalayer.Endpoint, inFlight int) float64 {
	if len(candidates) == 0 {
		return 1.0
	}

	// Count non-stale endpoints so the in-flight credit is spread only across
	// endpoints that actually contribute a measured score.
	nonStale := 0
	for _, e := range candidates {
		m := e.GetMetrics()
		if m != nil && time.Since(m.UpdateTime) <= d.config.MetricsStalenessThreshold {
			nonStale++
		}
	}
	var perEndpointCredit float64
	if inFlight > 0 && nonStale > 0 {
		perEndpointCredit = float64(inFlight) / float64(nonStale)
	}

	var maxScore float64
	for _, e := range candidates {
		metrics := e.GetMetrics()

		if metrics == nil || time.Since(metrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			// Stale/missing metrics are treated as fully saturated (conservative).
			maxScore = max(maxScore, 1.0)
			continue
		}

		qRatio := (float64(metrics.WaitingQueueSize) + perEndpointCredit) / float64(d.config.QueueDepthThreshold)
		kvRatio := metrics.KVCacheUsagePercent / d.config.KVCacheUtilThreshold

		// Roofline Analysis: The pod is saturated if either resource is exhausted.
		maxScore = max(maxScore, max(qRatio, kvRatio))
	}

	return maxScore
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
		metrics := endpoint.GetMetrics()
		if metrics == nil || time.Since(metrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			continue
		}

		if float64(metrics.WaitingQueueSize) < qLimit && metrics.KVCacheUsagePercent < kvLimit {
			filtered = append(filtered, endpoint)
		}
	}
	if len(filtered) == 0 {
		return endpoints
	}
	return filtered
}
