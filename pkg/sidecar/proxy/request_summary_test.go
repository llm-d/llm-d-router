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

package proxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

func TestObserveStageRecordsMetricAndSummary(t *testing.T) {
	before := snapshotStageMetrics(t)
	ctx, summary := withRequestSummary(context.Background())

	observeStageDuration(ctx, metrics.StagePrefill, 10*time.Millisecond)
	observeStageDuration(ctx, metrics.StagePrefill, 5*time.Millisecond)
	observeStageError(ctx, metrics.StageDecode)
	observeStageError(ctx, metrics.StagePrefill)

	assert.Equal(t, stageMetrics{prefillCount: 2, prefillErrors: 1, decodeErrors: 1}, snapshotStageMetrics(t).delta(before))
	assert.Equal(t, 15*time.Millisecond, summary.durations[metrics.StagePrefill], "time in a stage adds up across attempts")
	assert.Equal(t, metrics.StageDecode, summary.failedStage, "the first failed stage is kept")
}

func TestObserveStageWithoutSummaryStillRecordsMetrics(t *testing.T) {
	before := snapshotStageMetrics(t)

	observeStageDuration(context.Background(), metrics.StageEncode, time.Millisecond)
	observeStageError(context.Background(), metrics.StageEncode)

	assert.Equal(t, stageMetrics{encodeCount: 1, encodeErrors: 1}, snapshotStageMetrics(t).delta(before))
}

// The sidecar runs at --zap-log-level=1 in the guides, so the request line must
// be written at V(0) to be seen at all. The logger here drops everything above
// V(0), which is what makes the test fail if the line is demoted to debug.
func TestRequestFinishedLineAtDefaultLevel(t *testing.T) {
	provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	prevProvider, prevPropagator := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() {
		otel.SetTracerProvider(prevProvider)
		otel.SetTextMapPropagator(prevPropagator)
		_ = provider.Shutdown(context.Background())
	})

	const upstreamTraceID = "0af7651916cd43dd8448eb211c80319c"

	for _, tt := range []struct {
		name          string
		prefill       bool
		decodeStatus  int
		wantStatus    string
		wantPath      string
		wantStages    []string
		missingStages []string
		wantFailed    string
	}{
		{
			name:          "decode only",
			decodeStatus:  http.StatusOK,
			wantStatus:    "200",
			wantPath:      "decode-only",
			missingStages: []string{"prefillDuration", "decodeDuration", "failedStage"},
		},
		{
			name:         "prefill and decode",
			prefill:      true,
			decodeStatus: http.StatusOK,
			wantStatus:   "200",
			wantPath:     "prefill-decode",
			wantStages:   []string{"prefillDuration", "decodeDuration"},
		},
		{
			name:         "decode fails",
			prefill:      true,
			decodeStatus: http.StatusBadRequest,
			wantStatus:   "400",
			wantPath:     "prefill-decode",
			wantStages:   []string{"prefillDuration", "decodeDuration"},
			wantFailed:   "decode",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var mu sync.Mutex
			var logged []string
			logger := funcr.New(func(prefix, args string) {
				mu.Lock()
				defer mu.Unlock()
				logged = append(logged, prefix+" "+args)
			}, funcr.Options{Verbosity: 0})

			prefill := httptest.NewServer(statusHandler(http.StatusOK, `{"kv_transfer_params":{}}`))
			defer prefill.Close()
			decoder := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				statusHandler(tt.decodeStatus, `{"content":"hi"}`).ServeHTTP(w, r)
			}))
			defer decoder.Close()
			decoderURL, err := url.Parse(decoder.URL)
			require.NoError(t, err)
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "8000", DecoderURL: decoderURL, KVConnector: constants.KVConnectorNIXLV2})
			s.logger = logger
			s.allowlistValidator = &AllowlistValidator{}
			sidecar := httptest.NewServer(otelhttp.NewHandler(s.createRoutes(), "sidecar-http"))
			defer sidecar.Close()

			r, err := http.NewRequest(http.MethodPost, sidecar.URL+reqcommon.PathChatCompletions,
				strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
			require.NoError(t, err)
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("traceparent", "00-"+upstreamTraceID+"-b7ad6b7169203331-01")
			if tt.prefill {
				r.Header.Set(routing.PrefillEndpointHeader, prefillURL.Host)
			}
			resp, err := sidecar.Client().Do(r)
			require.NoError(t, err)
			_, _ = io.Copy(io.Discard, resp.Body)
			require.NoError(t, resp.Body.Close())

			// The line is written by a deferred call once the handler returns,
			// which can land just after the client has read the response.
			var lines []string
			require.Eventually(t, func() bool {
				mu.Lock()
				defer mu.Unlock()
				lines = lines[:0]
				for _, line := range logged {
					if strings.Contains(line, `"msg"="request finished"`) {
						lines = append(lines, line)
					}
				}
				return len(lines) > 0
			}, 2*time.Second, 10*time.Millisecond)
			require.Len(t, lines, 1, "exactly one summary line per request")
			line := lines[0]

			assert.Contains(t, line, `"trace_id"="`+upstreamTraceID+`"`, "the line must be findable by the request's trace_id")
			assert.Contains(t, line, `"path"="`+tt.wantPath+`"`)
			assert.Contains(t, line, `"status"=`+tt.wantStatus)
			assert.Contains(t, line, `"decoder"="`+decoderURL.Host+`"`)
			if tt.prefill {
				assert.Contains(t, line, `"prefiller"="`+prefillURL.Host+`"`)
			} else {
				assert.NotContains(t, line, `"prefiller"=`)
			}
			for _, key := range tt.wantStages {
				assert.Contains(t, line, `"`+key+`"=`)
			}
			for _, key := range tt.missingStages {
				assert.NotContains(t, line, `"`+key+`"=`)
			}
			if tt.wantFailed != "" {
				assert.Contains(t, line, `"failedStage"="`+tt.wantFailed+`"`)
			}
		})
	}
}
