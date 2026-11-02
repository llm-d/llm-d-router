/*
Copyright 2026 The Kubernetes Authors.
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
	"maps"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Pool-aggregate attribute keys, read by the multicluster scorers and written by the
// multicluster metrics extractor. A cluster-endpoint has no per-pod metrics, so it
// carries the pool's aggregate instead.
const (
	// MultiClusterKVCacheUtilizationKey holds a pool's aggregate KV-cache utilization.
	MultiClusterKVCacheUtilizationKey = "llm-d.ai/multicluster-kv-cache-utilization"
	// MultiClusterQueueSizeKey holds a pool's aggregate waiting-queue size.
	MultiClusterQueueSizeKey = "llm-d.ai/multicluster-queue-size"
	// MultiClusterMetricsUpdatedAtKey holds the Unix time (s) of the last scrape that read a pool aggregate; its age marks a down cluster.
	MultiClusterMetricsUpdatedAtKey = "llm-d.ai/multicluster-metrics-updated-at"
	// MultiClusterFlowControlKey holds a pool's flow-control bands by priority; empty once the pool stops reporting them.
	MultiClusterFlowControlKey = "llm-d.ai/multicluster-flow-control"
	// MultiClusterReadyEndpointsKey holds a pool's ready pods, its weight on the hub; -1 once the pool stops reporting it.
	MultiClusterReadyEndpointsKey = "llm-d.ai/multicluster-ready-endpoints"
)

// The pool aggregates are addressed without a producer name: any extractor that
// can supply the cluster-level aggregate is an acceptable source.
var (
	MultiClusterKVCacheUtilizationDataKey = plugin.NewDataKey(MultiClusterKVCacheUtilizationKey, "")
	MultiClusterQueueSizeDataKey          = plugin.NewDataKey(MultiClusterQueueSizeKey, "")
	MultiClusterMetricsUpdatedAtDataKey   = plugin.NewDataKey(MultiClusterMetricsUpdatedAtKey, "")
	MultiClusterFlowControlDataKey        = plugin.NewDataKey(MultiClusterFlowControlKey, "")
	MultiClusterReadyEndpointsDataKey     = plugin.NewDataKey(MultiClusterReadyEndpointsKey, "")
)

// FlowControlBand is one priority band of a pool's flow control at the last scrape.
type FlowControlBand struct {
	// Held is the number of requests the band holds.
	Held float64
	// RecentWait is the mean wait (s) of the band's requests dispatched in the last 30 s; 0 if none.
	RecentWait float64
	// LastProgress is the Unix time (s) of the last scrape at which a request of the band was dispatched,
	// or at which the band started holding requests.
	LastProgress float64
}

// MultiClusterFlowControl maps a priority to its flow-control band.
type MultiClusterFlowControl map[int]FlowControlBand

// Clone returns an independent copy.
func (m MultiClusterFlowControl) Clone() fwkdl.Cloneable {
	return MultiClusterFlowControl(maps.Clone(map[int]FlowControlBand(m)))
}
