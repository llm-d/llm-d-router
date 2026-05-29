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

package types

import "context"

// EvictionDemand describes a demand for in-flight eviction, emitted by the ShardProcessor
// when HoL blocking is detected with queued requests that cannot dispatch.
type EvictionDemand struct {
	// BlockedPriority is the priority level of the band that cannot dispatch due to saturation.
	BlockedPriority int

	// QueuedCount is the number of requests waiting in the blocked priority band.
	QueuedCount int

	// Saturation is the pool-wide saturation level at the time of the demand.
	Saturation float64

	// UsageLimit is the dispatch ceiling for the blocked priority band.
	UsageLimit float64
}

// EvictionHandler handles demand-driven eviction of in-flight requests.
// Implementations decide how many and which requests to evict based on the demand.
// HandleEvictionDemand is called asynchronously from the dispatch cycle and must be
// safe for concurrent use.
type EvictionHandler interface {
	HandleEvictionDemand(ctx context.Context, demand EvictionDemand)
}
