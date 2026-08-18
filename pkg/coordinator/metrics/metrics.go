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
const (
	ErrorCodeBadRequest  = "bad_request"
	ErrorCodeUpstream4xx = "upstream_4xx"
	ErrorCodeUpstream5xx = "upstream_5xx"
	ErrorCodeInternal    = "internal"
)

// Upstream label values for the upstream_request_* metrics. Step names come
// from each step file's own StepName constant (pkg/coordinator/steps/*.go).
const (
	UpstreamRender            = "render"
	UpstreamMediaFetch        = "media-fetch"
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

// Result label values for conditional_decode_probes_total.
const (
	ProbeResultServed   = "served"
	ProbeResultDeferred = "deferred"
)

var (
	modelLabel    = []string{"model_name"}
	stepLabel     = []string{"step"}
	upstreamLabel = []string{"upstream"}

	// generalLatencyBuckets covers durations from 5ms to 1 hour; identical to
	// the EPP request-duration ladder so PromQL translates cleanly between the
	// two components.
	generalLatencyBuckets = []float64{
		0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6,
		8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200,
		1800, 2700, 3600,
	}

	// requestSizeBuckets ranges from 64 bytes to 1 GiB, matching the EPP
	// request-size ladder. Wide enough for multimodal bodies with inlined
	// image data.
	requestSizeBuckets = []float64{
		64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
		131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
		16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
	}

	// inputTokensBuckets matches the EPP request-input-tokens ladder (1..1M);
	// most models have input context windows below 1 million tokens.
	inputTokensBuckets = []float64{
		1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
		32778, 65536, 131072, 262144, 524288, 1048576,
	}
)

// allCollectors returns the ordered list of every metric this package owns.
// Register and Reset both iterate it so the two stay in sync.
func allCollectors() []prometheus.Collector {
	return []prometheus.Collector{
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
// only.
func Reset() {
	for _, c := range allCollectors() {
		if r, ok := c.(interface{ Reset() }); ok {
			r.Reset()
		}
	}
}
