/*
Copyright 2025 The Kubernetes Authors.

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
