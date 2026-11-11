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

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

const (
	// PriorityClassUnknownReasonInvalid is the only reason label for an empty or unrecognized priority-class header.
	PriorityClassUnknownReasonInvalid = "invalid"
)

var (
	// LlmdPriorityClassRequestsTotal counts requests by the resolved priority class.
	// Exposed as llm_d_epp_requests_total{priority_class,fairness_id}.
	LlmdPriorityClassRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "requests_total",
			Help:      metricsutil.HelpMsgWithStability("Requests by resolved priority class and tenant.", compbasemetrics.ALPHA),
		},
		[]string{"priority_class", "fairness_id"},
	)

	// LlmdPriorityClassUnknownTotal counts empty or unrecognized priority-class headers.
	// The reason label is a bounded category. The raw header value is never a label.
	LlmdPriorityClassUnknownTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "priority_class_unknown_total",
			Help:      metricsutil.HelpMsgWithStability("Priority-class headers that were empty or unrecognized.", compbasemetrics.ALPHA),
		},
		[]string{"reason"},
	)
)

// RecordPriorityClassRequest increments the resolved priority-class request counter.
func RecordPriorityClassRequest(fairnessID, priorityClass string) {
	LlmdPriorityClassRequestsTotal.WithLabelValues(priorityClass, fairnessID).Inc()
}

// RecordPriorityClassUnknown increments the bounded unknown-priority-class counter.
func RecordPriorityClassUnknown(reason string) {
	LlmdPriorityClassUnknownTotal.WithLabelValues(reason).Inc()
}
