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
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
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
	_ fwkplugin.ConsumerPlugin       = &Detector{}
)

// Detector determines system saturation based on metrics of the given candidate pods.
type Detector struct {
	config              Config
	typedName           fwkplugin.TypedName
	inFlightLoadDataKey fwkplugin.DataKey
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
		config:              cfg,
		typedName:           typedName,
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
	}
}

// Consumes declares InFlightLoad as an optional dependency: it supplies the
// scrape-lag compensation in Saturation, but the detector still functions on
// scraped metrics alone when no inflight-load-producer is configured.
func (d *Detector) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{d.inFlightLoadDataKey: attrconcurrency.InFlightLoad{}},
	}
}

// inFlightRequests returns the endpoint's live in-flight request count as tracked
// by the InFlightLoadProducer, or 0 when the attribute is absent.
func (d *Detector) inFlightRequests(m datalayer.AttributeMap) int64 {
	if m == nil {
		return 0
	}
	if val, ok := m.Get(d.inFlightLoadDataKey.String()); ok {
		if load, ok := val.(*attrconcurrency.InFlightLoad); ok {
			return load.Requests
		}
	}
	return 0
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
// The credit corrects each endpoint's score before aggregation, so it holds
// under either aggregation shape. It is added only to the queue-depth term (the
// fast, controllable signal); the KV-cache term is left as measured. Endpoints
// without the attribute contribute zero credit and fall back to the scraped
// queue depth.
//
// The compensation assumes aggregated (non-P/D) pools. Under P/D disaggregation
// with the default producer config, a prefill endpoint's in-flight request count
// stays elevated for the whole decode duration of every request whose prefill it
// served, so its credit is inflated; see the package README.
func (d *Detector) Saturation(_ context.Context, candidates []datalayer.Endpoint) float64 {
	if len(candidates) == 0 {
		return 1.0
	}

	var totalScore float64
	for _, e := range candidates {
		metrics := e.GetMetrics()

		if metrics == nil || time.Since(metrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			totalScore += 1.0
			continue
		}

		// In-flight requests the scrape has not yet observed (see doc comment).
		credit := d.inFlightRequests(e.GetAttributes()) -
			int64(metrics.WaitingQueueSize) - int64(metrics.RunningRequestsSize)
		if credit < 0 {
			credit = 0
		}

		qRatio := float64(int64(metrics.WaitingQueueSize)+credit) / float64(d.config.QueueDepthThreshold)
		kvRatio := metrics.KVCacheUsagePercent / d.config.KVCacheUtilThreshold

		// Roofline Analysis: The pod is saturated if either resource is exhausted.
		totalScore += max(qRatio, kvRatio)
	}

	return totalScore / float64(len(candidates))
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
