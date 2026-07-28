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

package datalayer

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer/mocks"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	datasourcemocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

func defaultEndpoint() fwkdl.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod-name",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}
	return fwkdl.NewEndpoint(meta, nil)
}

var (
	endpoint = defaultEndpoint()
	sources  = []fwkdl.PollingDispatcher{&datasourcemocks.MetricsDataSource{}}
)

// Mock PollingDispatchers for collector tests. Each tracks its own
// invocation count and (for dataSource) runs bound extractors with metric
// instrumentation. Same behavior the framework collector did before the
// dispatcher pattern moved extractor handling into each dispatcher.

type errSource struct {
	kind      string
	err       error
	CallCount int64
}

func (e *errSource) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: e.kind, Name: e.kind}
}
func (e *errSource) Dispatch(_ context.Context, _ fwkdl.Endpoint) error {
	atomic.AddInt64(&e.CallCount, 1)
	return e.err
}
func (e *errSource) AppendExtractor(_ fwkplugin.Plugin) error { return nil }

type dataSource struct {
	kind      string
	CallCount int64
	mu        sync.Mutex
	exts      []*stubExtractor
}

func (d *dataSource) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: d.kind, Name: d.kind}
}
func (d *dataSource) Dispatch(ctx context.Context, ep fwkdl.Endpoint) error {
	atomic.AddInt64(&d.CallCount, 1)
	d.mu.Lock()
	exts := append([]*stubExtractor(nil), d.exts...)
	d.mu.Unlock()
	for _, ext := range exts {
		if err := ext.Extract(ctx, fwkdl.PollInput[any]{Payload: struct{}{}, Endpoint: ep}); err != nil {
			//nolint:staticcheck // SA1019: mock mirrors production's dual-record during deprecation window.
			metrics.DataLayerExtractErrorsTotal.WithLabelValues(d.kind, ext.kind).Inc()
			metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(d.kind, ext.kind).Inc()
		}
	}
	return nil
}
func (d *dataSource) AppendExtractor(p fwkplugin.Plugin) error {
	s, ok := p.(*stubExtractor)
	if !ok {
		return nil
	}
	d.mu.Lock()
	d.exts = append(d.exts, s)
	d.mu.Unlock()
	return nil
}

type stubExtractor struct {
	kind string
	err  error
}

func (s *stubExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: s.kind, Name: s.kind}
}
func (s *stubExtractor) Extract(_ context.Context, _ fwkdl.PollInput[any]) error { return s.err }

func TestCollectorStartInputs(t *testing.T) {
	tests := []struct {
		name        string
		ctxCanceled bool
		sources     []fwkdl.PollingDispatcher
		streamers   []fwkdl.StreamingDispatcher
		wantErr     bool
		wantErrIs   error
	}{
		{name: "valid sources, live ctx", sources: sources},
		{name: "valid streamers, live ctx", streamers: []fwkdl.StreamingDispatcher{&mockStreamingDispatcher{kind: "stream"}}},
		{name: "empty sources and streamers", sources: []fwkdl.PollingDispatcher{}, wantErr: true},
		{name: "nil poller", sources: []fwkdl.PollingDispatcher{nil}, wantErr: true},
		{name: "nil streamer", streamers: []fwkdl.StreamingDispatcher{nil}, wantErr: true},
		{name: "cancelled parent ctx", ctxCanceled: true, sources: sources, wantErr: true, wantErrIs: context.Canceled},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tt.ctxCanceled {
				cancel()
			}

			c := NewCollector()
			ticker := mocks.NewTicker()
			err := c.Start(ctx, ticker, endpoint, tt.sources, tt.streamers)
			if tt.wantErr {
				require.Error(t, err)
				if tt.wantErrIs != nil {
					assert.ErrorIs(t, err, tt.wantErrIs)
				}
				require.NoError(t, c.Start(context.Background(), ticker, endpoint, sources, nil),
					"retry after failed Start should succeed")
			} else {
				require.NoError(t, err)
			}
			c.Stop()
		})
	}
}

func TestCollectorCanStartOnlyOnce(t *testing.T) {
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, sources, nil))
	assert.Error(t, c.Start(ctx, ticker, endpoint, sources, nil),
		"second Start after success should error")
	c.Stop()
}

func TestCollectorStop(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T) *Collector
	}{
		{
			name:  "before any Start",
			setup: func(t *testing.T) *Collector { return NewCollector() },
		},
		{
			name: "after failed Start",
			setup: func(t *testing.T) *Collector {
				c := NewCollector()
				ticker := mocks.NewTicker()
				_ = c.Start(context.Background(), ticker, endpoint, []fwkdl.PollingDispatcher{}, nil)
				return c
			},
		},
		{
			name: "after successful Start",
			setup: func(t *testing.T) *Collector {
				c := NewCollector()
				ticker := mocks.NewTicker()
				require.NoError(t, c.Start(context.Background(), ticker, endpoint, sources, nil))
				return c
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.setup(t)
			c.Stop()
			c.Stop()
			c.Stop()
		})
	}
}

// TestCollectorCollectsOnTicks confirms ticks drive Poll calls.
func TestCollectorCollectsOnTicks(t *testing.T) {
	source := &datasourcemocks.MetricsDataSource{}
	c := NewCollector()
	ticker := mocks.NewTicker()

	require.NoError(t, c.Start(context.Background(), ticker, endpoint, []fwkdl.PollingDispatcher{source}, nil))
	defer c.Stop()

	ticker.Tick()
	ticker.Tick()

	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&source.CallCount) == 2
	}, 1*time.Second, 2*time.Millisecond, "expected 2 collections")
}

// TestCollectorStreamingDispatcher verifies the streaming dispatcher lifecycle in Collector.
func TestCollectorStreamingDispatcher(t *testing.T) {
	started := make(chan struct{})
	streamer := &mockStreamingDispatcher{kind: "zmq-test", started: started}

	c := NewCollector()
	ticker := mocks.NewTicker()
	require.NoError(t, c.Start(context.Background(), ticker, endpoint, nil, []fwkdl.StreamingDispatcher{streamer}))

	select {
	case <-started:
	case <-time.After(1 * time.Second):
		t.Fatal("expected streaming dispatcher to start")
	}

	c.Stop()
}

type mockStreamingDispatcher struct {
	kind     string
	started  chan struct{}
	stopWait chan struct{}
	err      error
}

func (m *mockStreamingDispatcher) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: m.kind, Name: m.kind}
}
func (m *mockStreamingDispatcher) Start(ctx context.Context, _ fwkdl.Endpoint) error {
	if m.started != nil {
		close(m.started)
	}
	if m.stopWait != nil {
		<-m.stopWait
	} else {
		<-ctx.Done()
	}
	return m.err
}
func (m *mockStreamingDispatcher) AppendExtractor(_ fwkplugin.Plugin) error { return nil }

// TestCollectorErrorMetrics confirms Poll/Extract errors increment per-event
// counters (no transition dedup) and successes do not.
func TestCollectorErrorMetrics(t *testing.T) {
	pollErr := errors.New("poll boom")
	extErr := errors.New("extract boom")

	tests := []struct {
		name          string
		srcType       string
		srcErr        error // if non-nil, source.Poll returns it
		extType       string
		extErr        error // if extType set and this non-nil, extractor.Extract returns it
		ticks         int
		wantPollDelta float64
		wantExtDelta  float64
	}{
		{
			name:          "poll errors increment per tick",
			srcType:       "table-poll-err",
			srcErr:        pollErr,
			ticks:         3,
			wantPollDelta: 3,
		},
		{
			name:         "extract errors increment per tick",
			srcType:      "table-ext-src",
			extType:      "table-ext-err",
			extErr:       extErr,
			ticks:        2,
			wantExtDelta: 2,
		},
		{
			name:    "success records nothing",
			srcType: "table-success-src",
			extType: "table-success-ext",
			ticks:   2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var src fwkdl.PollingDispatcher
			if tt.srcErr != nil {
				src = &errSource{kind: tt.srcType, err: tt.srcErr}
			} else {
				src = &dataSource{kind: tt.srcType}
			}

			if tt.extType != "" {
				ext := &stubExtractor{kind: tt.extType, err: tt.extErr}
				require.NoError(t, src.AppendExtractor(ext))
			}

			pollBefore := testutil.ToFloat64(metrics.LlmdDataLayerPollErrorsTotal.WithLabelValues(tt.srcType))
			var extBefore float64
			if tt.extType != "" {
				extBefore = testutil.ToFloat64(metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(tt.srcType, tt.extType))
			}

			c := NewCollector()
			ticker := mocks.NewTicker()
			require.NoError(t, c.Start(context.Background(), ticker, endpoint, []fwkdl.PollingDispatcher{src}, nil))
			defer c.Stop()

			for i := 0; i < tt.ticks; i++ {
				ticker.Tick()
			}

			require.Eventually(t, func() bool {
				gotPoll := testutil.ToFloat64(metrics.LlmdDataLayerPollErrorsTotal.WithLabelValues(tt.srcType)) - pollBefore
				if gotPoll != tt.wantPollDelta {
					return false
				}
				if tt.extType != "" {
					gotExt := testutil.ToFloat64(metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(tt.srcType, tt.extType)) - extBefore
					if gotExt != tt.wantExtDelta {
						return false
					}
				}
				// For success case, also verify polls actually happened (otherwise
				// the deltas being zero is trivially satisfied).
				if d, ok := src.(*dataSource); ok && atomic.LoadInt64(&d.CallCount) < int64(tt.ticks) {
					return false
				}
				return true
			}, 1*time.Second, 5*time.Millisecond, "expected counter deltas to settle")
		})
	}
}

// TestCollectorRapidStartStopRaceFree simulates the Runtime's pattern of
// ReleaseEndpoint immediately followed by NewEndpoint for the same key.
// Stop returns asynchronously, so a fresh Collector's goroutine may briefly
// coexist with a prior Collector's goroutine still exiting. -race surfaces
// any data race in the lifecycle.
func TestCollectorRapidStartStopRaceFree(t *testing.T) {
	src := &datasourcemocks.MetricsDataSource{}
	for i := 0; i < 100; i++ {
		c := NewCollector()
		ticker := mocks.NewTicker()
		require.NoError(t, c.Start(context.Background(), ticker, endpoint, []fwkdl.PollingDispatcher{src}, nil))
		ticker.Tick()
		c.Stop()
	}
}

// TestCollectorConcurrentStopRaceFree drives multiple Stop calls in parallel
// to verify the cancel field's mutex serializes them.
func TestCollectorConcurrentStopRaceFree(t *testing.T) {
	c := NewCollector()
	ticker := mocks.NewTicker()
	require.NoError(t, c.Start(context.Background(), ticker, endpoint, sources, nil))

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Stop()
		}()
	}
	wg.Wait()
}
