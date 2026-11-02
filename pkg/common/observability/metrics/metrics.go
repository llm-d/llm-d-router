/*
Copyright 2025 The Kubernetes Authors.
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
	"fmt"

	compbasemetrics "k8s.io/component-base/metrics"
)

const (
	// LLMDRouterEndpointPickerSubsystem is the subsystem for llm-d router endpoint picker metrics.
	LLMDRouterEndpointPickerSubsystem = "llm_d_epp"
)

// Disaggregation path label values, naming the stages a request is split into
// across pods. The EPP, coordinator, and P/D sidecar label their metrics with
// these values so dashboards can join the components on them.
const (
	// DisaggPathDecodeOnly is a request served without disaggregation.
	DisaggPathDecodeOnly = "decode-only"
	// DisaggPathPrefillDecode is a request split into prefill and decode (P/D or EP/D).
	DisaggPathPrefillDecode = "prefill-decode"
	// DisaggPathEncodeDecode is a request with remote encode and local prefill and decode (E/PD).
	DisaggPathEncodeDecode = "encode-decode"
	// DisaggPathEncodePrefillDecode is a request split into encode, prefill, and decode (E/P/D).
	DisaggPathEncodePrefillDecode = "encode-prefill-decode"
)

// HelpMsgWithStability is a helper function to create a help message with stability level.
func HelpMsgWithStability(msg string, stability compbasemetrics.StabilityLevel) string {
	return fmt.Sprintf("[%v] %v", stability, msg)
}

// GeneralLatencyBuckets is a request-duration histogram ladder from 5ms to
// 1 hour. Every llm-d component that emits a request-duration histogram
// reuses it so PromQL translates cleanly across components.
var GeneralLatencyBuckets = []float64{
	0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6,
	8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200,
	1800, 2700, 3600,
}

// RequestSizeBuckets is a request-body-size histogram ladder from 64 bytes
// to 1 GiB. Every llm-d component that emits a request-size histogram
// reuses it. Wide enough for multimodal bodies with inlined image data.
var RequestSizeBuckets = []float64{
	64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
	131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
	16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
}

// TokenCountBuckets is a token-count histogram ladder from 1 to ~1M in
// powers of two (with 1 as the low end and 8 as the second entry). Input
// and cached-prompt token histograms across llm-d components share this
// shape.
var TokenCountBuckets = []float64{
	1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
	32768, 65536, 131072, 262144, 524288, 1048576,
}

// ResponseSizeBuckets is a response-body-size histogram ladder from 16 bytes
// to 64 MiB. LLM responses span a wider range than request bodies: short
// error messages (~100 bytes), non-streaming completions (hundreds of bytes
// to a few KiB), and long streaming responses that accumulate token-by-token
// (tens of KiB to MiB). The lower floor and tighter mid-range give better
// resolution than RequestSizeBuckets for the typical response distribution,
// and the 64 MiB ceiling covers very long streamed completions.
var ResponseSizeBuckets = []float64{
	16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
	65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
	16777216, 33554432, 67108864,
}
