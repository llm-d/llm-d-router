/*
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

package sojourntime

import (
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	observerconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sojourntimeobserver/constants"
)

// InFlightRequestsDataKey carries the per-endpoint list of dispatched-not-completed
// requests published by the sojourn-time-observer-hub. Read by the mrl-scorer-hub
// at scoring time to age each request under the two-term MRL residual.
var InFlightRequestsDataKey = plugin.NewDataKey(
	"InFlightRequestsDataKey",
	observerconstants.SojournTimeObserverProducerType,
)

// InFlightRequest is one dispatched-not-completed request on an endpoint.
// FirstChunkAt is the zero Time when no chunk has arrived yet; callers branch
// on IsZero() to pick the pre- or post-first-chunk term of the residual.
type InFlightRequest struct {
	DispatchedAt time.Time
	FirstChunkAt time.Time
}

// InFlightRequestsSnapshot is the value type behind InFlightRequestsDataKey.
// A nil or empty Requests slice means the endpoint has no dispatched-not-completed
// requests; the scorer treats that as residual 0.
type InFlightRequestsSnapshot struct {
	Requests []InFlightRequest
}

// Clone returns an independent copy. The Requests slice is copied so a
// subsequent publisher-side update does not race with in-flight consumer reads.
func (s *InFlightRequestsSnapshot) Clone() fwkdl.Cloneable {
	if s == nil {
		return nil
	}
	cp := &InFlightRequestsSnapshot{}
	if s.Requests != nil {
		cp.Requests = make([]InFlightRequest, len(s.Requests))
		copy(cp.Requests, s.Requests)
	}
	return cp
}
