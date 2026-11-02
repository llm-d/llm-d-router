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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/codes"
	otelsemconv "go.opentelemetry.io/otel/semconv/v1.41.0"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/time/rate"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

func renderToolFieldCounts(t *testing.T) map[string]float64 {
	t.Helper()
	eppmetrics.Register()
	families, err := metrics.Registry.Gather()
	require.NoError(t, err)
	counts := map[string]float64{}
	for _, family := range families {
		if family.GetName() != "llm_d_epp_tool_calling_field_status_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels[toolcalling.MetricLabelComponent] == "tokenizer_http" {
				require.Len(t, labels, 4)
				require.Equal(t, toolcalling.DirectionRequest, labels[toolcalling.MetricLabelDirection])
				counts[labels[toolcalling.MetricLabelField]+"/"+labels[toolcalling.MetricLabelStatus]] = metric.GetCounter().GetValue()
			}
		}
	}
	return counts
}

func TestRenderToolFieldMetrics(t *testing.T) {
	for _, tc := range []struct {
		name, path, body string
		fields           []toolcalling.Field
		status           int
		warmup           bool
		transform        bool
		disabled         bool
	}{
		{name: "disabled", path: chatRenderPath, body: `{"tools":[]}`, disabled: true},
		{name: "chat", path: chatRenderPath, body: `{"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"response_format":{"type":"json_object"}}`, fields: []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice, toolcalling.FieldParallelToolCalls, toolcalling.FieldResponseFormat}, transform: true},
		{name: "messages", path: messagesRenderPath, body: `{"tools":[],"tool_choice":{"type":"any"}}`, fields: []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice}},
		{name: "responses", path: responsesRenderPath, body: `{"tools":[],"tool_choice":"auto"}`, fields: []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice}},
		{name: "plain", path: chatRenderPath, body: `{"messages":[]}`},
		{name: "completions", path: completionsRenderPath, body: `{"tools":[]}`},
		{name: "invalid capture", path: chatRenderPath, body: `{"tools":[`},
		{name: "warmup", path: chatRenderPath, body: `{"tools":[]}`, warmup: true},
		{name: "HTTP error", path: chatRenderPath, body: `{"tools":[]}`, fields: []toolcalling.Field{toolcalling.FieldTools}, status: http.StatusUnprocessableEntity},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := renderToolFieldCounts(t)
			renderer := newRenderServer(t, vllmConfig{PrefillOnly: tc.transform, OmitMMKwargs: tc.transform}, func(w http.ResponseWriter, _ *http.Request) {
				if tc.status != 0 {
					w.WriteHeader(tc.status)
				}
				_, _ = io.WriteString(w, `{"token_ids":[1]}`)
			})
			ctx := t.Context()
			if !tc.disabled {
				ctx = toolcalling.WithTelemetryEnabled(ctx)
			}
			if tc.warmup {
				ctx = withUnobservedRender(ctx)
			}
			var out renderResponse
			err := renderer.postJSON(ctx, tc.path, fwkrh.RawPayload(tc.body), time.Second, &out)
			require.Equal(t, tc.status != 0, err != nil)
			after := renderToolFieldCounts(t)
			for key, value := range before {
				after[key] -= value
			}
			for key, value := range after {
				if value == 0 {
					delete(after, key)
				}
			}
			want := map[string]float64{}
			for _, field := range tc.fields {
				want[string(field)+"/"+string(toolcalling.FieldStatusPreserved)] = 1
			}
			assert.Equal(t, want, after)
		})
	}
}

func TestRenderToolFieldMetrics_Retry(t *testing.T) {
	recorder := setupSpanRecorder(t)
	before := renderToolFieldCounts(t)
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	for _, name := range []string{"a", "b"} {
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, name, "8000").GetMetadata()))
	}
	calls := 0
	renderer := &vllmHTTPRenderer{
		endpointPicker: picker,
		timeout:        time.Second,
		pluginName:     fmt.Sprintf("%s-%d", t.Name(), renderTestRuns.Add(1)),
		client: &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			calls++
			status := http.StatusOK
			if calls == 1 {
				status = http.StatusServiceUnavailable
			}
			return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(`{"token_ids":[1]}`))}, nil
		})},
	}
	request := chatRequest()
	request.Body.Payload = fwkrh.RawPayload(`{"tools":[]}`)
	err = newTestPlugin(renderer).Produce(toolcalling.WithTelemetryEnabled(t.Context()), request, nil)
	require.NoError(t, err)
	require.Equal(t, 2, calls)
	after := renderToolFieldCounts(t)
	assert.Equal(t, float64(1), after["tools/preserved"]-before["tools/preserved"])
	assert.Equal(t, before["tools/rejected"], after["tools/rejected"])
	count, _ := renderHistogram(t, renderer.pluginName, renderResultSuccess)
	assert.Equal(t, uint64(1), count)
	assert.Zero(t, renderSeriesCount(t, renderFailuresTotal, renderer.pluginName))
	assert.Equal(t, renderResultSuccess, spanAttrs(tokenizeSpan(t, recorder))[semconv.LLMDEPPTokenProducerResultKey].AsString())
}

func TestProduce_ToolRenderOutcomes(t *testing.T) {
	for _, tc := range []struct {
		name, body, result, reason, spanResult string
		status                                 int
	}{
		{"success", `{"token_ids":[1]}`, renderResultSuccess, "", renderResultSuccess, http.StatusOK},
		{"HTTP rejection", `{"detail":"private_tool"}`, renderResultError, renderReasonStatus, renderResultError, http.StatusUnprocessableEntity},
		{"server failure", `{"detail":"private_tool"}`, renderResultError, renderReasonStatus, renderResultError, http.StatusServiceUnavailable},
		{"decode failure", `{"token_ids":"private_tool"}`, renderResultError, renderReasonDecode, renderResultError, http.StatusOK},
		{"no tokens", `{"token_ids":[]}`, renderResultSuccess, "", resultSkippedNoTokens, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := setupSpanRecorder(t)
			renderer := newRenderServer(t, vllmConfig{}, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			})
			request := chatRequest()
			request.Body.Payload = fwkrh.RawPayload(`{"tools":[{"type":"function","function":{"name":"private_tool"}}]}`)
			err := newTestPlugin(renderer).Produce(toolcalling.WithTelemetryEnabled(t.Context()), request, nil)
			require.Equal(t, tc.reason != "", err != nil)
			assert.Equal(t, tc.reason == "" && tc.spanResult != resultSkippedNoTokens, request.Body.TokenizedRequest != nil)
			count, _ := renderHistogram(t, renderer.pluginName, tc.result)
			assert.Equal(t, uint64(1), count)
			if tc.reason == "" {
				assert.Zero(t, renderSeriesCount(t, renderFailuresTotal, renderer.pluginName))
			} else {
				assert.Equal(t, float64(1), renderFailures(renderer.pluginName, tc.reason))
				assert.Equal(t, codes.Error, tokenizeSpan(t, recorder).Status().Code)
			}
			attrs := spanAttrs(tokenizeSpan(t, recorder))
			assert.Equal(t, tc.spanResult, attrs[semconv.LLMDEPPTokenProducerResultKey].AsString())
			assert.True(t, attrs[semconv.LLMDToolCallingPresentKey].AsBool())
			assert.Equal(t, string(toolcalling.FieldStatusPreserved), attrs[semconv.LLMDToolCallingFieldToolsStatusKey].AsString())
			require.Len(t, recorder.Ended(), 2)
			for _, span := range recorder.Ended() {
				if span.SpanKind() == trace.SpanKindClient {
					assert.Equal(t, int64(tc.status), spanAttrs(span)[otelsemconv.HTTPResponseStatusCodeKey].AsInt64())
				}
			}
		})
	}
}

func TestRenderToolFieldMetrics_NoEndpoints(t *testing.T) {
	before := renderToolFieldCounts(t)
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	renderer := &vllmHTTPRenderer{endpointPicker: picker, timeout: time.Second}
	_, _, err = renderer.RenderChat(t.Context(), fwkrh.RawPayload(`{"tools":[]}`))
	require.ErrorIs(t, err, errNoRenderEndpoints)
	assert.Equal(t, before, renderToolFieldCounts(t))
}

// renderHistogram returns the sample count and sum of renderDurationSeconds for
// one plugin name and result.
func renderHistogram(t *testing.T, pluginName, result string) (uint64, float64) {
	t.Helper()
	m := &dto.Metric{}
	observer := renderDurationSeconds.WithLabelValues(PluginType, pluginName, BackendVLLM, result)
	require.NoError(t, observer.(prometheus.Metric).Write(m))
	return m.GetHistogram().GetSampleCount(), m.GetHistogram().GetSampleSum()
}

func renderFailures(pluginName, reason string) float64 {
	return testutil.ToFloat64(renderFailuresTotal.WithLabelValues(PluginType, pluginName, BackendVLLM, reason))
}

// renderSeriesCount returns the number of series c holds for one plugin name.
func renderSeriesCount(t *testing.T, c prometheus.Collector, pluginName string) int {
	t.Helper()
	registry := prometheus.NewRegistry()
	require.NoError(t, registry.Register(c))
	families, err := registry.Gather()
	require.NoError(t, err)
	count := 0
	for _, family := range families {
		for _, m := range family.GetMetric() {
			for _, label := range m.GetLabel() {
				if label.GetName() == "plugin_name" && label.GetValue() == pluginName {
					count++
				}
			}
		}
	}
	return count
}

// logCapture collects the lines written through the logger it places on a context.
type logCapture struct {
	mu    sync.Mutex
	lines []string
}

func (c *logCapture) context(ctx context.Context) context.Context {
	return log.IntoContext(ctx, funcr.New(func(_, args string) {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.lines = append(c.lines, args)
	}, funcr.Options{}))
}

func (c *logCapture) snapshot() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.lines...)
}

// renderTestRuns makes plugin names unique across repeated runs of one test.
var renderTestRuns atomic.Int64

// newRenderServer serves every render path with handler and gives the returned
// renderer a plugin name of its own, so each test run reads its own metric series.
func newRenderServer(t *testing.T, cfg vllmConfig, handler http.HandlerFunc) *vllmHTTPRenderer {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	cfg.URL = srv.URL
	r, err := newVLLMHTTPRenderer(&cfg)
	require.NoError(t, err)
	r.pluginName = fmt.Sprintf("%s-%d", t.Name(), renderTestRuns.Add(1))
	return r
}

// newSaturatedRenderServer holds every request until the test ends, as a render
// endpoint whose queue exceeds the render timeout does. onRequest, when set,
// runs as each request arrives.
func newSaturatedRenderServer(t *testing.T, cfg vllmConfig, onRequest func()) *vllmHTTPRenderer {
	t.Helper()
	release := make(chan struct{})
	r := newRenderServer(t, cfg, func(http.ResponseWriter, *http.Request) {
		if onRequest != nil {
			onRequest()
		}
		<-release
	})
	// Registered after the server so held requests return before it closes.
	t.Cleanup(func() { close(release) })
	return r
}

func renderPath(ctx context.Context, r *vllmHTTPRenderer, path string) error {
	payload := fwkrh.RawPayload(`{"model":"m"}`)
	var err error
	switch path {
	case completionsRenderPath:
		_, _, err = r.Render(ctx, payload)
	case chatRenderPath:
		_, _, err = r.RenderChat(ctx, payload)
	case messagesRenderPath:
		_, _, err = r.RenderMessages(ctx, payload)
	case responsesRenderPath:
		_, _, err = r.RenderResponses(ctx, payload)
	}
	return err
}

func TestRenderMetrics_SaturatedEndpointTimesOut(t *testing.T) {
	const (
		timeout   = 50 * time.Millisecond
		mmTimeout = 200 * time.Millisecond
	)
	for _, tc := range []struct {
		name   string
		path   string
		budget time.Duration
		// parent is the caller's deadline, as set by the Director for a producer.
		parent time.Duration
	}{
		{"completions", completionsRenderPath, timeout, 0},
		{"chat", chatRenderPath, mmTimeout, 0},
		{"messages", messagesRenderPath, mmTimeout, 0},
		{"responses", responsesRenderPath, mmTimeout, 0},
		// The Director's producer deadline equals the chat render budget and is
		// armed first, so it is the deadline that expires.
		{"chat under producer deadline", chatRenderPath, mmTimeout, mmTimeout},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newSaturatedRenderServer(t, vllmConfig{Timeout: timeout.String(), MMTimeout: mmTimeout.String()}, nil)
			logs := &logCapture{}
			ctx := logs.context(context.Background())
			if tc.parent > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, tc.parent)
				defer cancel()
			}

			err := renderPath(ctx, r, tc.path)
			require.ErrorIs(t, err, context.DeadlineExceeded)

			count, sum := renderHistogram(t, r.pluginName, renderResultTimeout)
			assert.Equal(t, uint64(1), count)
			// The parent deadline is armed before the render call starts.
			if tc.parent == 0 {
				assert.GreaterOrEqual(t, sum, tc.budget.Seconds())
			}
			assert.Less(t, sum, 10*tc.budget.Seconds())
			assert.Equal(t, 1.0, renderFailures(r.pluginName, renderReasonTimeout))

			lines := logs.snapshot()
			require.Len(t, lines, 1)
			assert.Contains(t, lines[0], `"msg"="token-producer render call failed"`)
			assert.Contains(t, lines[0], `"backend"="vllm"`)
			assert.Contains(t, lines[0], `"path"="`+tc.path+`"`)
			assert.Contains(t, lines[0], `"reason"="timeout"`)
			assert.Contains(t, lines[0], `"timeout"="`+tc.budget.String()+`"`)
			assert.Contains(t, lines[0], `"elapsed"=`)
		})
	}
}

func TestRenderMetrics_FailureReasons(t *testing.T) {
	for _, tc := range []struct {
		name    string
		handler http.HandlerFunc
		// setup adjusts the renderer after the server is up.
		setup  func(t *testing.T, r *vllmHTTPRenderer)
		reason string
	}{
		{
			name: "non-2xx status",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "overloaded", http.StatusServiceUnavailable)
			},
			reason: renderReasonStatus,
		},
		{
			name: "undecodable body",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte("not json"))
			},
			reason: renderReasonDecode,
		},
		{
			name:    "connection refused",
			handler: func(http.ResponseWriter, *http.Request) {},
			setup: func(t *testing.T, r *vllmHTTPRenderer) {
				t.Helper()
				srv := httptest.NewServer(http.NotFoundHandler())
				srv.Close()
				r.endpointPicker = fixedEndpointPicker(srv.URL)
			},
			reason: renderReasonConnection,
		},
		{
			name:    "no discovered endpoints",
			handler: func(http.ResponseWriter, *http.Request) {},
			setup: func(t *testing.T, r *vllmHTTPRenderer) {
				t.Helper()
				picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
				require.NoError(t, err)
				r.endpointPicker = picker
			},
			reason: renderReasonNoEndpoints,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newRenderServer(t, vllmConfig{}, tc.handler)
			if tc.setup != nil {
				tc.setup(t, r)
			}
			logs := &logCapture{}

			require.Error(t, renderPath(logs.context(context.Background()), r, chatRenderPath))

			count, _ := renderHistogram(t, r.pluginName, renderResultError)
			assert.Equal(t, uint64(1), count)
			assert.Equal(t, 1.0, renderFailures(r.pluginName, tc.reason))
			lines := logs.snapshot()
			require.Len(t, lines, 1)
			assert.Contains(t, lines[0], `"reason"="`+tc.reason+`"`)
		})
	}
}

func TestRenderMetrics_Success(t *testing.T) {
	r := newRenderServer(t, vllmConfig{}, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"token_ids":[1,2]}`))
	})
	logs := &logCapture{}

	require.NoError(t, renderPath(logs.context(context.Background()), r, chatRenderPath))

	count, _ := renderHistogram(t, r.pluginName, renderResultSuccess)
	assert.Equal(t, uint64(1), count)
	assert.Zero(t, renderSeriesCount(t, renderFailuresTotal, r.pluginName))
	assert.Empty(t, logs.snapshot())
}

// A caller that goes away is not a render failure: it is observed in the
// histogram only.
func TestRenderMetrics_CallerCancellation(t *testing.T) {
	started := make(chan struct{})
	r := newSaturatedRenderServer(t, vllmConfig{}, func() { close(started) })
	logs := &logCapture{}
	ctx, cancel := context.WithCancel(logs.context(context.Background()))
	go func() {
		<-started
		cancel()
	}()

	require.ErrorIs(t, renderPath(ctx, r, chatRenderPath), context.Canceled)

	count, _ := renderHistogram(t, r.pluginName, renderResultCanceled)
	assert.Equal(t, uint64(1), count)
	assert.Zero(t, renderSeriesCount(t, renderFailuresTotal, r.pluginName))
	assert.Empty(t, logs.snapshot())
}

func TestRenderMetrics_FailureLogIsRateLimited(t *testing.T) {
	r := newRenderServer(t, vllmConfig{}, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	})
	logs := &logCapture{}
	ctx := logs.context(context.Background())

	const calls = 5
	for range calls {
		require.Error(t, renderPath(ctx, r, chatRenderPath))
	}

	assert.Equal(t, float64(calls), renderFailures(r.pluginName, renderReasonStatus))
	lines := logs.snapshot()
	require.Len(t, lines, 1)
	assert.Contains(t, lines[0], `"suppressed"=0`)

	// A fresh limiter stands in for the interval elapsing.
	r.failureLog = rate.Sometimes{Interval: renderFailureLogInterval}
	require.Error(t, renderPath(ctx, r, chatRenderPath))

	lines = logs.snapshot()
	require.Len(t, lines, 2)
	assert.Contains(t, lines[1], fmt.Sprintf(`"suppressed"=%d`, calls-1))
}

func TestRenderMetrics_WarmupIsNotObserved(t *testing.T) {
	r := newRenderServer(t, vllmConfig{}, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	})
	logs := &logCapture{}

	renderBackend{tk: r}.warmup(logs.context(context.Background()))

	assert.Zero(t, renderSeriesCount(t, renderDurationSeconds, r.pluginName))
	assert.Zero(t, renderSeriesCount(t, renderFailuresTotal, r.pluginName))
	for _, line := range logs.snapshot() {
		assert.NotContains(t, line, "token-producer render call failed")
	}
}
