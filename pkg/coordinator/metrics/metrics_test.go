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

import (
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

func TestWithLabel_AppendsAndDoesNotAliasBase(t *testing.T) {
	base := []string{"model_name"}
	out := withLabel(base, "error_code")
	require.Equal(t, []string{"model_name", "error_code"}, out)
	require.Equal(t, []string{"model_name"}, base, "base must not be mutated")

	// Even when base has extra capacity, two withLabel results must own
	// separate backing arrays. A shared base would let the second call
	// overwrite the first's extra element.
	baseWithCap := make([]string, 1, 8)
	baseWithCap[0] = "model_name"
	o1 := withLabel(baseWithCap, "error_code")
	o2 := withLabel(baseWithCap, "path")
	require.Equal(t, "error_code", o1[1])
	require.Equal(t, "path", o2[1], "o2 must not overwrite o1 via shared backing array")
}

func TestRegister_Idempotent(t *testing.T) {
	reg := prometheus.NewRegistry()
	require.NoError(t, Register(reg))
	require.NoError(t, Register(reg))
}

func TestRegister_NilRegisterer(t *testing.T) {
	require.Error(t, Register(nil))
}

func TestRegister_RejectsCollidingCollector(t *testing.T) {
	reg := prometheus.NewRegistry()
	// A collector with the same fqName as one of ours but a different
	// declaration collides; Register must surface the error rather than
	// swallow it as AlreadyRegistered.
	colliding := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: LLMDRouterCoordinatorSubsystem,
		Name:      "request_total",
		Help:      "collides with request_total",
	})
	require.NoError(t, reg.Register(colliding))
	require.Error(t, Register(reg))
}

func TestRequestFamily_Records(t *testing.T) {
	Reset()
	IncRequestTotal("m1")
	IncRequestTotal("m1")
	IncRequestErrorTotal("m1", ErrorCodeUpstream4xx)
	RecordRequestDuration("m1", 250*time.Millisecond)
	RecordRequestSize("m1", 128)
	IncRequestRunning("m1")

	require.InDelta(t, 2.0, promtestutil.ToFloat64(requestTotal.WithLabelValues("m1")), 1e-9)
	require.InDelta(t, 1.0, promtestutil.ToFloat64(requestErrorTotal.WithLabelValues("m1", ErrorCodeUpstream4xx)), 1e-9)
	require.InDelta(t, 1.0, promtestutil.ToFloat64(requestRunning.WithLabelValues("m1")), 1e-9)

	DecRequestRunning("m1")
	require.InDelta(t, 0.0, promtestutil.ToFloat64(requestRunning.WithLabelValues("m1")), 1e-9)

	expected := `
# HELP llm_d_coordinator_request_size_bytes [ALPHA] Incoming request body size distribution in bytes.
# TYPE llm_d_coordinator_request_size_bytes histogram
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="64"} 0
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="128"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="256"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="512"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="1024"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="2048"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="4096"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="8192"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="16384"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="32768"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="65536"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="131072"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="262144"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="524288"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="1.048576e+06"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="2.097152e+06"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="4.194304e+06"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="8.388608e+06"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="1.6777216e+07"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="3.3554432e+07"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="6.7108864e+07"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="1.34217728e+08"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="2.68435456e+08"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="5.36870912e+08"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="1.073741824e+09"} 1
llm_d_coordinator_request_size_bytes_bucket{model_name="m1",le="+Inf"} 1
llm_d_coordinator_request_size_bytes_sum{model_name="m1"} 128
llm_d_coordinator_request_size_bytes_count{model_name="m1"} 1
`
	require.NoError(t, promtestutil.CollectAndCompare(requestSize, strings.NewReader(expected), "llm_d_coordinator_request_size_bytes"))
}

func TestReset_ClearsAllVectors(t *testing.T) {
	IncRequestTotal("m2")
	require.InDelta(t, 1.0, promtestutil.ToFloat64(requestTotal.WithLabelValues("m2")), 1e-9)
	Reset()
	require.InDelta(t, 0.0, promtestutil.ToFloat64(requestTotal.WithLabelValues("m2")), 1e-9)
}

func TestStepFamily_Records(t *testing.T) {
	Reset()
	IncStepRunning("render")
	require.InDelta(t, 1.0, promtestutil.ToFloat64(stepRunning.WithLabelValues("render")), 1e-9)
	DecStepRunning("render")
	require.InDelta(t, 0.0, promtestutil.ToFloat64(stepRunning.WithLabelValues("render")), 1e-9)

	RecordStepDuration("render", 50*time.Millisecond)
	IncStepErrorTotal("prefill", ErrorCodeUpstream5xx)

	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(stepErrorTotal.WithLabelValues("prefill", ErrorCodeUpstream5xx)), 1e-9,
	)
}

func TestUpstreamFamily_Records(t *testing.T) {
	Reset()
	IncUpstreamRequestTotal(UpstreamEncode)
	IncUpstreamRequestTotal(UpstreamEncode)
	IncUpstreamRequestTotal(UpstreamReplaceMediaURLs)
	RecordUpstreamRequestDuration(UpstreamEncode, 10*time.Millisecond)

	require.InDelta(t, 2.0, promtestutil.ToFloat64(upstreamRequestTotal.WithLabelValues(UpstreamEncode)), 1e-9)
	require.InDelta(t, 1.0, promtestutil.ToFloat64(upstreamRequestTotal.WithLabelValues(UpstreamReplaceMediaURLs)), 1e-9)
}

func TestExecutionPathAndProbes_Records(t *testing.T) {
	Reset()
	IncExecutionPath("m", PathEncodePrefillDecode)
	IncExecutionPath("m", PathEncodePrefillDecode)
	IncExecutionPath("m", PathDecodeOnly)
	IncConditionalDecodeProbes(ProbeResultServed)
	IncConditionalDecodeProbes(ProbeResultDeferred)
	IncConditionalDecodeProbes(ProbeResultDeferred)
	IncConditionalDecodeProbes(ProbeResultError)
	IncConditionalDecodeProbes(ProbeResultTransportError)
	RecordRequestInputTokens("m", 512)

	require.InDelta(t, 2.0,
		promtestutil.ToFloat64(executionPathTotal.WithLabelValues("m", PathEncodePrefillDecode)), 1e-9,
	)
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(executionPathTotal.WithLabelValues("m", PathDecodeOnly)), 1e-9,
	)
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(conditionalDecodeProbesTotal.WithLabelValues(ProbeResultServed)), 1e-9,
	)
	require.InDelta(t, 2.0,
		promtestutil.ToFloat64(conditionalDecodeProbesTotal.WithLabelValues(ProbeResultDeferred)), 1e-9,
	)
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(conditionalDecodeProbesTotal.WithLabelValues(ProbeResultError)), 1e-9,
	)
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(conditionalDecodeProbesTotal.WithLabelValues(ProbeResultTransportError)), 1e-9,
	)
}
