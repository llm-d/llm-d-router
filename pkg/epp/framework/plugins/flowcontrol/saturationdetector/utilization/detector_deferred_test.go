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

package utilization

import (
	"context"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

// TestDetector_Saturation_QueueTermLeavesDeferredOut: requests the engine
// reports as deferred (parked on a KV transfer, a LoRA slot or a grammar
// compile) are part of WaitingQueueSize but not of the queue-depth term; the
// scrape-lag credit still counts them as observed.
func TestDetector_Saturation_QueueTermLeavesDeferredOut(t *testing.T) {
	t.Parallel()

	baseTime := time.Now()
	config := &Config{
		QueueDepthThreshold:       3,
		KVCacheUtilThreshold:      1.0,
		MetricsStalenessThreshold: 100 * time.Millisecond,
	}
	detector := NewDetector("test-detector", *config, logr.Discard())
	ctx := context.Background()

	pod := func(waiting, running int, deferred float64) fwkdl.Endpoint {
		meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "pod", Namespace: "ns1"}}
		metrics := fwkdl.NewMetrics()
		metrics.WaitingQueueSize = waiting
		metrics.RunningRequestsSize = running
		metrics.KVCacheUsagePercent = 0.1
		metrics.UpdateTime = baseTime
		e := fwkdl.NewEndpoint(meta, metrics)
		e.GetAttributes().Put(attrmetrics.ScalarMetricDataKey("DeferredQueueSize"), attrmetrics.ScalarMetricValue(deferred))
		return e
	}

	// Waiting 2 of which 1 deferred: the queue term is 1 -> qRatio 1/3.
	require.InDelta(t, 1.0/3.0, detector.Saturation(ctx, []fwkdl.Endpoint{pod(2, 0, 1)}), 1e-4,
		"deferred requests leave the queue term")

	// In-flight 4 against waiting 2 (1 deferred) and running 2: every request
	// is observed, credit 0 -> qRatio 1/3.
	require.InDelta(t, 1.0/3.0, detector.Saturation(ctx, []fwkdl.Endpoint{withInFlightLoad(pod(2, 2, 1), 4)}), 1e-4,
		"deferred requests are observed by the scrape, not credited again")

	// One more in flight than the scrape shows: credit 1 -> qRatio 2/3.
	require.InDelta(t, 2.0/3.0, detector.Saturation(ctx, []fwkdl.Endpoint{withInFlightLoad(pod(2, 2, 1), 5)}), 1e-4,
		"only the request the scrape has not seen is credited")

	// A deferred count above the waiting queue comes from series scraped out
	// of step and leaves the queue term at the waiting queue: qRatio 1/3.
	require.InDelta(t, 1.0/3.0, detector.Saturation(ctx, []fwkdl.Endpoint{pod(1, 0, 3)}), 1e-4,
		"deferred above waiting leaves the queue term unchanged")
}
