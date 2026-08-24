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
