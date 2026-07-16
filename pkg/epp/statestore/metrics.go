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

package statestore

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

// remoteCallTotal counts every attempted Remote/FailOpen State API call, by
// capability, regardless of outcome. This exists specifically to disambiguate
// remoteFallbackTotal's blind spot: a CounterVec label that has never been
// incremented emits no line at all on /metrics, so "remoteFallbackTotal absent"
// is consistent with BOTH "the remote path ran and never once failed" and
// "the remote path was never invoked in the first place" -- the two have
// opposite implications for whether a run's numbers say anything about the
// State API's cost. remoteCallTotal being present and > 0 for a capability is
// the only way to confirm the remote path actually ran; comparing it against
// remoteFallbackTotal gives the real fallback rate (fallback / call).
var remoteCallTotal = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: "statestore",
		Name:      "remote_call_total",
		Help:      "Count of Remote/FailOpen State API calls attempted, by capability, regardless of outcome.",
	},
	[]string{"capability"},
)

// remoteFallbackTotal counts how many times a Remote/FailOpen call fell back
// to the Local provider (inflight/prefix read timeout or error, or a
// concurrency-lease Admit/Release call failing). The e2e performance harness
// asserts this stays ~0 relative to remoteCallTotal during a remote-path run:
// a nonzero rate means the run wasn't actually exercising the remote path
// cleanly, which would invalidate any latency numbers collected alongside it.
var remoteFallbackTotal = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: "statestore",
		Name:      "remote_fallback_total",
		Help:      "Count of Remote/FailOpen State API calls that fell back to the Local provider, by capability.",
	},
	[]string{"capability"},
)

var registerOnce sync.Once

// RegisterMetrics registers this package's metrics against the given
// registerer. Safe to call multiple times; registration happens once.
func RegisterMetrics(registerer prometheus.Registerer) {
	registerOnce.Do(func() {
		registerer.MustRegister(remoteCallTotal)
		registerer.MustRegister(remoteFallbackTotal)
	})
}

func recordRemoteCall(capability string) {
	remoteCallTotal.WithLabelValues(capability).Inc()
}

func recordRemoteFallback(capability string) {
	remoteFallbackTotal.WithLabelValues(capability).Inc()
}
