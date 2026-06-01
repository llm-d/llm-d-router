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

package concurrency

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	inflightloadconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload/constants"
)

// InFlightLoadDataKey carries the per-endpoint in-flight load snapshot used by
// load-aware scorers. Populated by InFlightLoadProducer.Produce on each
// candidate endpoint at the start of every scheduling cycle.
//
// The snapshot is per-cycle (stored on the per-cycle endpoint clone produced
// by fwksched.NewEndpoint) and combines two semantically distinct sources:
//   - Tokens / Requests: accumulated load from already in-flight requests,
//     read from the producer's persistent trackers.
//   - UncachedRequestTokens: the projected work this request would add to
//     the endpoint if scheduled here, computed fresh from the request being
//     scored and the endpoint's prefix-cache state. Not persisted.
var InFlightLoadDataKey = plugin.NewDataKey("InFlightLoadDataKey", inflightloadconstants.InFlightLoadProducerType)

// InFlightLoad captures the current real-time load of an endpoint as tracked
// by the EPP, plus the projected impact of the request currently being scheduled.
type InFlightLoad struct {
	// Tokens is the in-flight token count this endpoint has committed to,
	// accumulated from past scheduling decisions. Updated by PreRequest (when
	// an endpoint is chosen) and OnEvicted (when its request stream ends);
	// snapshotted into this struct each cycle by InFlightLoadProducer.Produce.
	Tokens int64

	// Requests is the in-flight request count this endpoint has committed to,
	// maintained with the same lifecycle as Tokens.
	Requests int64

	// UncachedRequestTokens is a speculative projection: the work the request
	// currently being scheduled would add to this endpoint if it landed here.
	// Includes the uncached input portion (accounting for prefix-cache hits)
	// plus the estimated output when the producer is configured with
	// AddEstimatedOutputTokens=true. Computed fresh by Produce on every cycle
	// from the request being scored and this endpoint's prefix-cache state;
	// never committed to endpoint state and not decremented on stream end.
	// Zero when no request is in scope (e.g., background snapshots).
	//
	// Could equivalently be computed inside any scorer that needs it, but
	// it's produced here so every consumer reads the same value — single
	// source of truth for the current request's per-endpoint impact.
	UncachedRequestTokens int64
}

// Clone returns an independent copy of the InFlightLoad. The value-copy
// idiom (cp := *l) covers every field automatically; new fields added to
// InFlightLoad do not require updating Clone, as long as they remain value
// types (no slices, maps, or pointers requiring deep copy).
func (l *InFlightLoad) Clone() fwkdl.Cloneable {
	if l == nil {
		return nil
	}
	cp := *l
	return &cp
}
