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

package logger

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const debugPrintInterval = 5 * time.Second

// StartMetricsLogger starts background goroutines for:
// 1. Refreshing Prometheus metrics periodically
// 2. Debug logging (if DEBUG level enabled)
func StartMetricsLogger(ctx context.Context, datastore datalayer.PoolInfo, refreshInterval, stalenessThreshold time.Duration) {
	logger := log.FromContext(ctx)

	go runPrometheusRefresher(ctx, logger, datastore, refreshInterval, stalenessThreshold)

	if logger.V(logutil.DEBUG).Enabled() {
		go runDebugLogger(ctx, logger, datastore, stalenessThreshold)
	}
}

func runPrometheusRefresher(ctx context.Context, logger logr.Logger, datastore datalayer.PoolInfo, interval, stalenessThreshold time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.V(logutil.DEFAULT).Info("Shutting down prometheus metrics thread")
			return
		case <-ticker.C:
			refreshPrometheusMetrics(logger, datastore, stalenessThreshold)
		}
	}
}

func runDebugLogger(ctx context.Context, logger logr.Logger, datastore datalayer.PoolInfo, stalenessThreshold time.Duration) {
	ticker := time.NewTicker(debugPrintInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.V(logutil.DEFAULT).Info("Shutting down metrics logger thread")
			return
		case <-ticker.C:
			printDebugMetrics(logger, datastore, stalenessThreshold)
		}
	}
}

func podsWithFreshMetrics(stalenessThreshold time.Duration) func(fwkdl.Endpoint) bool {
	return func(ep fwkdl.Endpoint) bool {
		if ep == nil {
			return false // Skip nil pods
		}
		return time.Since(ep.GetMetrics().UpdateTime) <= stalenessThreshold
	}
}

func podsWithStaleMetrics(stalenessThreshold time.Duration) func(fwkdl.Endpoint) bool {
	return func(ep fwkdl.Endpoint) bool {
		if ep == nil {
			return false // Skip nil pods
		}
		return time.Since(ep.GetMetrics().UpdateTime) > stalenessThreshold
	}
}

func printDebugMetrics(logger logr.Logger, datastore datalayer.PoolInfo, stalenessThreshold time.Duration) {
	freshPods := datastore.PodList(podsWithFreshMetrics(stalenessThreshold))
	stalePods := datastore.PodList(podsWithStaleMetrics(stalenessThreshold))

	logger.V(logutil.TRACE).Info("Current Pods and metrics gathered",
		"Fresh metrics", fmt.Sprintf("%+v", freshPods), "Stale metrics", fmt.Sprintf("%+v", stalePods))
}

func refreshPrometheusMetrics(logger logr.Logger, datastore datalayer.PoolInfo, stalenessThreshold time.Duration) {
	pool, err := datastore.PoolGet()
	if err != nil {
		logger.V(logutil.DEFAULT).Info("Pool is not initialized, skipping refreshing metrics")
		return
	}

	podMetrics := datastore.PodList(podsWithFreshMetrics(stalenessThreshold))
	logger.V(logutil.TRACE).Info("Refreshing Prometheus Metrics", "ReadyPods", len(podMetrics))
	podCount := len(podMetrics)
	metrics.RecordInferencePoolReadyPods(pool.Name, float64(podCount))

	if podCount == 0 {
		return
	}

	summary := calculateSummary(podMetrics)

	metrics.RecordInferencePoolAvgKVCache(pool.Name, summary.kvCache.mean)
	metrics.RecordInferencePoolAvgQueueSize(pool.Name, summary.queueSize.mean)
	metrics.RecordInferencePoolAvgRunningRequests(pool.Name, summary.runningRequests.mean)
	metrics.RecordInferencePoolStdDevKVCache(pool.Name, summary.kvCache.stdv)
	metrics.RecordInferencePoolStdDevQueueSize(pool.Name, summary.queueSize.stdv)
	metrics.RecordInferencePoolStdDevRunningRequests(pool.Name, summary.runningRequests.stdv)
}

// stats holds aggregated metric values
type stats struct {
	mean float64 // average
	stdv float64 // standard deviation
}

type summary struct {
	kvCache         stats
	queueSize       stats
	runningRequests stats
}

func calculateSummary(endpoints []fwkdl.Endpoint) (result summary) {
	if len(endpoints) == 0 {
		return result
	}

	var kvSum, queueSum, reqSum float64

	for _, pod := range endpoints {
		metrics := pod.GetMetrics()
		kvSum += float64(metrics.KVCacheUsagePercent)
		queueSum += float64(metrics.WaitingQueueSize)
		reqSum += float64(metrics.RunningRequestsSize)
	}

	size := float64(len(endpoints))

	result.kvCache.mean = kvSum / size
	result.queueSize.mean = queueSum / size
	result.runningRequests.mean = reqSum / size

	var kvSS, queueSS, reqSS float64

	for _, pod := range endpoints {
		metrics := pod.GetMetrics()
		kvSS += (metrics.KVCacheUsagePercent - result.kvCache.mean) * (metrics.KVCacheUsagePercent - result.kvCache.mean)
		queueSS += (float64(metrics.WaitingQueueSize) - result.queueSize.mean) * (float64(metrics.WaitingQueueSize) - result.queueSize.mean)
		reqSS += (float64(metrics.RunningRequestsSize) - result.runningRequests.mean) * (float64(metrics.RunningRequestsSize) - result.runningRequests.mean)
	}

	sampleSize := math.Max(1.0, size-1)

	// Round stats to two decimal places
	result.kvCache.mean = round2(result.kvCache.mean)
	result.queueSize.mean = round2(result.queueSize.mean)
	result.runningRequests.mean = round2(result.runningRequests.mean)

	result.kvCache.stdv = round2(math.Sqrt(kvSS / sampleSize))
	result.queueSize.stdv = round2(math.Sqrt(queueSS / sampleSize))
	result.runningRequests.stdv = round2(math.Sqrt(reqSS / sampleSize))

	return result
}

func round2(x float64) float64 { return math.Round(x*100) / 100 }
