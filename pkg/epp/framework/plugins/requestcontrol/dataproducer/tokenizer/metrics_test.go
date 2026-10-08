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
	"net/http"
	"net/http/httptest"
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
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// renderHistogram returns the sample count and sum of renderDurationSeconds for
// one plugin name and result.
func renderHistogram(t *testing.T, pluginName, result string) (uint64, float64) {
	t.Helper()
	m := &dto.Metric{}
	observer := renderDurationSeconds.WithLabelValues(PluginType, pluginName, backendVLLM, result)
	require.NoError(t, observer.(prometheus.Metric).Write(m))
	return m.GetHistogram().GetSampleCount(), m.GetHistogram().GetSampleSum()
}

func renderFailures(pluginName, reason string) float64 {
	return testutil.ToFloat64(renderFailuresTotal.WithLabelValues(PluginType, pluginName, backendVLLM, reason))
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
	assert.Len(t, logs.snapshot(), 1)
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
