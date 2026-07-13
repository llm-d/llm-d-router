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
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

// remoteFallbackTotal counts how many times a Remote/FailOpen call fell back
// to the Local provider (inflight/prefix read timeout or error, or a
// concurrency-lease Admit/Release call failing). The e2e performance harness
// asserts this stays ~0 during a remote-path run: a nonzero rate means the
// run wasn't actually exercising the remote path, which would invalidate any
// latency numbers collected alongside it.
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
		registerer.MustRegister(remoteFallbackTotal)
	})
}

func recordRemoteFallback(capability string) {
	remoteFallbackTotal.WithLabelValues(capability).Inc()
}

// defaultRegisterer is the controller-runtime global registry, matching the
// registry the EPP's /metrics endpoint already serves (see
// cmd/epp/runner/runner.go). Exposed as a var so tests can avoid touching the
// process-global registry.
var defaultRegisterer prometheus.Registerer = metrics.Registry
