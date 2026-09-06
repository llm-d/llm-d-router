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

// Package metrics exposes the coordinator's Prometheus metric families. All
// metrics live under the llm_d_coordinator subsystem and describe requests the
// coordinator accepts and the pipeline it runs to serve them; see
// docs/metrics.coord.md.
package metrics

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
)

// LLMDRouterCoordinatorSubsystem is the Prometheus subsystem for coordinator
// metrics. Every metric declared in this package uses it.
const LLMDRouterCoordinatorSubsystem = "llm_d_coordinator"

// ModelUnknown is the model_name label value for requests that carried no
// model in the body (empty or absent). It is distinct from the cardinality
// overflow value returned by boundModel once the cap fills.
const ModelUnknown = "unknown"

// Error-code label values for request_error_total and step_errors_total.
// BadRequest is invalid client input. Upstream4xx/Upstream5xx are HTTP
// responses from an upstream worker in those bands. UpstreamTransport is
// a round-trip failure before headers arrive (connection refused, timeout,
// TCP reset); no response was received. Internal is the fallthrough for
// coordinator-owned code paths, never for reachability issues.
const (
	ErrorCodeBadRequest        = "bad_request"
	ErrorCodeUpstream4xx       = "upstream_4xx"
	ErrorCodeUpstream5xx       = "upstream_5xx"
	ErrorCodeUpstreamTransport = "upstream_transport"
	ErrorCodeInternal          = "internal"
)

// Upstream label values for the upstream_request_* metrics. Step names come
// from each step file's own StepName constant (pkg/coordinator/steps/*.go).
const (
	UpstreamRender            = "render"
	UpstreamReplaceMediaURLs  = "replace-media-urls"
	UpstreamEncode            = "encode"
	UpstreamPrefill           = "prefill"
	UpstreamConditionalDecode = "conditional-decode"
	UpstreamDecode            = "decode"
)

// Path label values for execution_path_total. Encode always implies prefill,
// so encode-decode is not a reachable path.
const (
	PathDecodeOnly          = "decode-only"
	PathPrefillDecode       = "prefill-decode"
	PathEncodePrefillDecode = "encode-prefill-decode"
)

// Result label values for conditional_decode_probes_total. Served covers 2xx/3xx
// (the worker answered the request inline). Deferred is exactly HTTP 412 (cache
// miss, pipeline continues). Error covers any other 4xx/5xx: the worker's
// response is still streamed to the client, but the outcome is not a hit.
// TransportError covers a round-trip failure before headers arrive
// (connection refused, timeout, TCP reset); no worker response was received
// and the proxy answered the client 502.
const (
	ProbeResultServed         = "served"
	ProbeResultDeferred       = "deferred"
	ProbeResultError          = "error"
	ProbeResultTransportError = "transport_error"
)

var (
	modelLabel    = []string{"model_name"}
	stepLabel     = []string{"step"}
	upstreamLabel = []string{"upstream"}
)

// withLabel returns a fresh slice of base followed by extra. It allocates a
// new backing array unconditionally, so metric declarations that share a base
// label slice (modelLabel, stepLabel, upstreamLabel) cannot alias each other's
// storage — even if the base slice is later given extra capacity.
func withLabel(base []string, extra string) []string {
	out := make([]string, 0, len(base)+1)
	out = append(out, base...)
	return append(out, extra)
}

// resettableCollector is a prometheus.Collector that supports Reset. All
// vector metrics this package uses (CounterVec, HistogramVec, GaugeVec)
// satisfy it. Adding a collector that does not implement Reset causes a
// compile-time failure in allCollectors, not a silent skip in Reset().
type resettableCollector interface {
	prometheus.Collector
	Reset()
}

// allCollectors returns the ordered list of every metric this package owns.
// Register and Reset both iterate it so the two stay in sync.
func allCollectors() []resettableCollector {
	return []resettableCollector{
		requestTotal,
		requestErrorTotal,
		requestDuration,
		requestSize,
		requestInputTokens,
		requestRunning,
		stepDuration,
		stepErrorTotal,
		stepRunning,
		upstreamRequestTotal,
		upstreamRequestDuration,
		executionPathTotal,
		conditionalDecodeProbesTotal,
	}
}

// Register wires every coordinator metric onto reg. A collector already
// present on reg is treated as success, so calling Register more than once
// (e.g. across tests using a fresh prometheus.NewRegistry() each time) is
// safe.
func Register(reg prometheus.Registerer) error {
	if reg == nil {
		return errors.New("coordinator metrics registerer is required")
	}
	for _, c := range allCollectors() {
		if err := reg.Register(c); err != nil {
			var already prometheus.AlreadyRegisteredError
			if errors.As(err, &already) && already.ExistingCollector == c {
				continue
			}
			return fmt.Errorf("register coordinator metric: %w", err)
		}
	}
	return nil
}

// Reset clears every metric back to its initial state. For integration tests
// only. Every collector in allCollectors implements Reset by construction of
// the resettableCollector interface.
func Reset() {
	for _, c := range allCollectors() {
		c.Reset()
	}
}
