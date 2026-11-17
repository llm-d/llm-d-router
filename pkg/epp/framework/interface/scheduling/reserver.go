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

package scheduling

// EndpointReserver is implemented by data producers that track per-endpoint
// in-flight load. A picker calls ReserveFromEndpoint for the endpoint it
// selects, at pick time, so that concurrent scheduling cycles observe the
// decision immediately instead of waiting for the post-dispatch (PreRequest)
// accounting to land.
//
// This closes the thundering-herd race described in issue #1598: without
// reservations, a burst of requests scheduled in parallel all score against the
// same pre-pick snapshot and converge on the same (least-loaded) endpoint.
//
// The reservation is converted into the real in-flight counters when the
// request is dispatched (PreRequest), and swept if the request never dispatches.
type EndpointReserver interface {
	// ReserveFromEndpoint reserves ep for the request currently being
	// scheduled: +1 request and the request's uncached-token estimate for that
	// endpoint (the UncachedRequestTokens attribute the producer's Produce
	// phase already recorded on it). Reading the estimate through the reserver
	// keeps data keys out of the picker.
	ReserveFromEndpoint(ep Endpoint)

	// Reserve records a reservation directly by endpoint ID, for callers that
	// already know the token estimate (or accept zero for a request-count-only
	// reservation).
	Reserve(endpointID string, uncachedTokens int64)
}
