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

package metrics

import "time"

// UpstreamCall packages the "count + start + record" triplet that every step
// wraps around a synchronous outbound call. StartUpstreamCall increments
// upstream_request_total and captures the start time; Done records the
// elapsed time on upstream_request_duration_seconds. Call Done immediately
// after the outbound call returns (not deferred to function exit) so the
// duration observation covers only the upstream call, not later work like
// body decoding.
type UpstreamCall struct {
	upstream string
	start    time.Time
}

// StartUpstreamCall increments upstream_request_total for upstream and
// captures a start timestamp. The returned UpstreamCall's Done method
// records the duration on upstream_request_duration_seconds.
func StartUpstreamCall(upstream string) UpstreamCall {
	IncUpstreamRequestTotal(upstream)
	return UpstreamCall{upstream: upstream, start: time.Now()}
}

// Done records time.Since(start) on upstream_request_duration_seconds. On
// the zero value it does not crash but records a very large positive
// duration (time.Since clamps at math.MaxInt64, ~292 years) that lands in
// the histogram's +Inf bucket. Always pair Done with StartUpstreamCall.
func (c UpstreamCall) Done() {
	RecordUpstreamRequestDuration(c.upstream, time.Since(c.start))
}
