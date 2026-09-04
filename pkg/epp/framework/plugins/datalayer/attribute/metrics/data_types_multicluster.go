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

import "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"

// Pool-aggregate attribute keys, read by the multicluster scorers and written by the
// multicluster metrics extractor. A cluster-endpoint has no per-pod metrics, so it
// carries the pool's aggregate instead.
const (
	// MultiClusterKVCacheUtilizationKey holds a pool's aggregate KV-cache utilization.
	MultiClusterKVCacheUtilizationKey = "llm-d.ai/multicluster-kv-cache-utilization"
	// MultiClusterQueueSizeKey holds a pool's aggregate waiting-queue size.
	MultiClusterQueueSizeKey = "llm-d.ai/multicluster-queue-size"
)

// The pool aggregates are addressed without a producer name: any extractor that
// can supply the cluster-level aggregate is an acceptable source.
var (
	MultiClusterKVCacheUtilizationDataKey = plugin.NewDataKey(MultiClusterKVCacheUtilizationKey, "")
	MultiClusterQueueSizeDataKey          = plugin.NewDataKey(MultiClusterQueueSizeKey, "")
)
