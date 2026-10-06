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

package tokenizer

import (
	"context"
	"errors"
	"net/url"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Values of the result label on renderDurationSeconds.
const (
	renderResultSuccess  = "success"
	renderResultTimeout  = "timeout"
	renderResultCanceled = "canceled"
	renderResultError    = "error"
)

// Values of the reason label on renderFailuresTotal.
const (
	renderReasonTimeout     = "timeout"
	renderReasonStatus      = "status"
	renderReasonConnection  = "connection"
	renderReasonDecode      = "decode"
	renderReasonNoEndpoints = "no_endpoints"
	renderReasonOther       = "other"
)

// renderFailureLogInterval bounds the render failure log to one line per
// interval per plugin instance. A saturated render endpoint fails every
// request, and the metrics carry the per-request signal.
const renderFailureLogInterval = 10 * time.Second

var (
	// renderDurationSeconds observes one render call per request, including a
	// retry on an alternate endpoint. The buckets extend past the default 5s
	// completions and 30s chat render timeouts so a timed-out call lands in a
	// finite bucket.
	renderDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "token_producer_render_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration of token-producer render calls, by backend and result.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60},
		},
		[]string{"plugin_type", "plugin_name", "backend", "result"},
	)

	// renderFailuresTotal counts render calls that returned an error other than
	// a caller cancellation.
	renderFailuresTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "token_producer_render_failures_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of failed token-producer render calls, by backend and reason.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_type", "plugin_name", "backend", "reason"},
	)

	registerOnce sync.Once
)

func registerRenderMetrics() {
	registerOnce.Do(func() {
		metrics.Registry.MustRegister(renderDurationSeconds)
		metrics.Registry.MustRegister(renderFailuresTotal)
	})
}

// unobservedRenderCtxKey marks render calls that are not made for a request.
type unobservedRenderCtxKey struct{}

// withUnobservedRender returns ctx whose render calls are excluded from the
// render metrics and the failure log. The warmup probe uses it: its failures
// while the render endpoint starts are expected.
func withUnobservedRender(ctx context.Context) context.Context {
	return context.WithValue(ctx, unobservedRenderCtxKey{}, true)
}

// classifyRenderError maps a render call's error to the result and reason label
// values. The reason is empty when the call is not counted as a failure.
func classifyRenderError(err error) (result, reason string) {
	var statusErr *renderStatusError
	var urlErr *url.Error
	switch {
	case err == nil:
		return renderResultSuccess, ""
	case errors.Is(err, context.DeadlineExceeded):
		return renderResultTimeout, renderReasonTimeout
	case errors.Is(err, context.Canceled):
		return renderResultCanceled, ""
	case errors.As(err, &statusErr):
		return renderResultError, renderReasonStatus
	case errors.Is(err, errNoRenderEndpoints):
		return renderResultError, renderReasonNoEndpoints
	case errors.Is(err, errRenderDecode):
		return renderResultError, renderReasonDecode
	case errors.As(err, &urlErr):
		return renderResultError, renderReasonConnection
	default:
		return renderResultError, renderReasonOther
	}
}

// observeRender records the outcome of one render call. The failure log is
// emitted here because the Director's producer deadline can fire before the
// render call returns, in which case the returned error is discarded.
func (r *vllmHTTPRenderer) observeRender(ctx context.Context, path string, timeout, elapsed time.Duration, err error) {
	if ctx.Value(unobservedRenderCtxKey{}) != nil {
		return
	}
	result, reason := classifyRenderError(err)
	renderDurationSeconds.WithLabelValues(PluginType, r.pluginName, backendVLLM, result).Observe(elapsed.Seconds())
	if reason == "" {
		return
	}
	renderFailuresTotal.WithLabelValues(PluginType, r.pluginName, backendVLLM, reason).Inc()
	r.failureLog.Do(func() {
		log.FromContext(ctx).Error(err, "token-producer render call failed",
			"backend", backendVLLM, "path", path, "reason", reason,
			"elapsed", elapsed.String(), "timeout", timeout.String())
	})
}
