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

package inflightload

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// requestInflight tracks requests admitted to the endpoint picker but not yet
// completed, aggregated per model. The producer owns the gauge: it increments
// in PreRequest and decrements via the per-request state entry's OnEvicted, so
// the count stays balanced across completion, error, and reaper paths.
var requestInflight = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "request_inflight",
		Help:      metricsutil.HelpMsgWithStability("Current number of in-flight requests in the endpoint picker (admitted, not yet completed).", compbasemetrics.ALPHA),
	},
	[]string{"model_name", "target_model_name", "fairness_id", "priority"},
)

// inflightRequests and inflightTokens track per-endpoint in-flight load. The
// producer's concurrency trackers set them from the live counter value on
// every counter update and delete the series when the endpoint is removed.
var inflightRequests = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "inflight_requests",
		Help:      metricsutil.HelpMsgWithStability("Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.", compbasemetrics.ALPHA),
	},
	[]string{"endpoint_name", "namespace", "producer_name"},
)

var inflightTokens = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "inflight_tokens",
		Help:      metricsutil.HelpMsgWithStability("Current number of in-flight tokens per endpoint (uncached prompt tokens, optionally plus estimated output), as tracked by the in-flight load producer.", compbasemetrics.ALPHA),
	},
	[]string{"endpoint_name", "namespace", "producer_name"},
)

// registerMetrics registers the producer-owned metrics with the plugin's metrics
// recorder. All metrics are shared package-level vectors, so a second producer
// instance re-registering them is a benign no-op; the producer_name label keeps
// each instance's per-endpoint series distinct.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("inflight load metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{requestInflight, inflightRequests, inflightTokens} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register inflight load metric: %w", err)
		}
	}
	return nil
}

// splitNamespacedName splits a "namespace/name" key (the string form of a
// k8s types.NamespacedName) into its name and namespace. A key with no
// separator is treated as a bare name with an empty namespace.
func splitNamespacedName(id string) (name, namespace string) {
	if i := strings.IndexByte(id, '/'); i >= 0 {
		return id[i+1:], id[:i]
	}
	return id, ""
}

// requestInflightLabels carries the gauge label values captured at admission so
// the decrement, which may run on a background reaper goroutine, uses the same
// series the increment did.
type requestInflightLabels struct {
	modelName       string
	targetModelName string
	fairnessID      string
	priority        string
}

func newRequestInflightLabels(request *fwksched.InferenceRequest) requestInflightLabels {
	fairnessID := request.FairnessID
	if fairnessID == "" {
		fairnessID = metadata.DefaultFairnessID
	}
	return requestInflightLabels{
		modelName:       request.IncomingModel,
		targetModelName: request.TargetModel,
		fairnessID:      fairnessID,
		priority:        strconv.Itoa(request.Objectives.Priority),
	}
}

func incRequestInflight(l requestInflightLabels) {
	if l.modelName == "" {
		return
	}
	requestInflight.WithLabelValues(l.modelName, l.targetModelName, l.fairnessID, l.priority).Inc()
}

func decRequestInflight(l requestInflightLabels) {
	if l.modelName == "" {
		return
	}
	requestInflight.WithLabelValues(l.modelName, l.targetModelName, l.fairnessID, l.priority).Dec()
}
