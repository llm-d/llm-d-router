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

package requesthandling

import (
	"context"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// EndpointReserver is implemented by data producers that track per-endpoint
// in-flight load. The director calls ReserveResult as soon as scheduling
// completes, so that concurrently-scheduled requests observe each pick
// immediately instead of waiting for the post-dispatch (PreRequest)
// accounting to land. Without reservations, a burst of requests scheduled
// in parallel is scored against the same pre-pick snapshot and converges on
// the same endpoint (the thundering herd).
//
// Reservations are request-scoped: each is converted into the real in-flight
// counters when the request is dispatched (PreRequest) and reclaimed by the
// plugin-state janitor when a request is abandoned after scheduling.
type EndpointReserver interface {
	// ReserveResult reserves the primary target of every profile in result
	// on behalf of the request identified by requestID. ctx must be the
	// request's lifetime context so the reservation is reclaimed if the
	// request never dispatches.
	ReserveResult(ctx context.Context, requestID string, result *fwksched.SchedulingResult)
}
