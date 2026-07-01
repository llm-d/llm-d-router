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

// totals holds aggregated metric values
type totals struct {
	kvCache         float64
	queueSize       int
	runningRequests int
}

func calculateTotals(endpoints []fwkdl.Endpoint) totals {
	var result totals
	for _, pod := range endpoints {
		metrics := pod.GetMetrics()
		result.kvCache += metrics.KVCacheUsagePercent
		result.queueSize += metrics.WaitingQueueSize
		result.runningRequests += metrics.RunningRequestsSize
	}
	return result
}

// stats holds aggregated metric values
type stats struct {
	mean float64 // average
	stdv float64 // standard deviation
	vrce float64 // variance
}

type summary struct {
	kvCache         stats
	queueSize       stats
	runningRequests stats
}

func calculateSummary(endpoints []fwkdl.Endpoint) summary {
	var result summary
	size := float64(len(endpoints))
	totals := calculateTotals(endpoints)

	result.kvCache.mean = totals.kvCache / size
	result.queueSize.mean = float64(totals.queueSize) / size
	result.runningRequests.mean = float64(totals.runningRequests) / size

	for _, pod := range endpoints {
		metrics := pod.GetMetrics()
		result.kvCache.vrce += (metrics.KVCacheUsagePercent - result.kvCache.mean) * (metrics.KVCacheUsagePercent - result.kvCache.mean)
		result.queueSize.vrce += (float64(metrics.WaitingQueueSize) - result.queueSize.mean) * (float64(metrics.WaitingQueueSize) - result.queueSize.mean)
		result.runningRequests.vrce += (float64(metrics.RunningRequestsSize) - result.runningRequests.mean) * (float64(metrics.RunningRequestsSize) - result.runningRequests.mean)
	}

	var num = 1.0
	if size > 2 {
		num = size - 1
	}
	result.kvCache.vrce /= num
	result.queueSize.vrce /= num
	result.runningRequests.vrce /= num

	result.kvCache.stdv = math.Sqrt(result.kvCache.vrce)
	result.queueSize.stdv = math.Sqrt(result.queueSize.vrce)
	result.runningRequests.stdv = math.Sqrt(result.runningRequests.vrce)

	return result
}
