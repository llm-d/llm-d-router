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
	"sync"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// requestSummary collects what the stages of one request took, whichever
// connector served it, so the handler can log the request as a single line when
// it finishes. Connectors may run prefill and decode concurrently, so access is
// guarded.
type requestSummary struct {
	mu          sync.Mutex
	durations   map[string]time.Duration
	failedStage string
}

type requestSummaryKey struct{}

// withRequestSummary returns ctx carrying a new, empty summary.
func withRequestSummary(ctx context.Context) (context.Context, *requestSummary) {
	summary := &requestSummary{durations: map[string]time.Duration{}}
	return context.WithValue(ctx, requestSummaryKey{}, summary), summary
}

// requestSummaryFrom returns the summary carried by ctx, or nil.
func requestSummaryFrom(ctx context.Context) *requestSummary {
	summary, _ := ctx.Value(requestSummaryKey{}).(*requestSummary)
	return summary
}

// observeStageDuration records how long a stage took in that stage's duration
// histogram, with ctx's trace as the exemplar, and in the request's summary.
func observeStageDuration(ctx context.Context, stage string, d time.Duration) {
	switch stage {
	case routing.StageEncode:
		metrics.RecordEncodeDuration(ctx, d)
	case routing.StagePrefill:
		metrics.RecordPrefillDuration(ctx, d)
	case routing.StageDecode:
		metrics.RecordDecodeDuration(ctx, d)
	}
	if summary := requestSummaryFrom(ctx); summary != nil {
		summary.mu.Lock()
		summary.durations[stage] += d
		summary.mu.Unlock()
	}
}

// observeStageError counts a failed stage and marks it on the request's
// summary. The first failure is kept, since later stages usually fail because
// of it.
func observeStageError(ctx context.Context, stage string) {
	metrics.RecordError(stage)
	if summary := requestSummaryFrom(ctx); summary != nil {
		summary.mu.Lock()
		if summary.failedStage == "" {
			summary.failedStage = stage
		}
		summary.mu.Unlock()
	}
}

// failedStageMarker returns the routing.FailedStageHeader value for a response
// written now: the stage that failed, else whether prefill completed. Every
// connector records the prefill stage's duration once the prefill response is
// in, and records a prefill failure before it writes a response, so a recorded
// prefill with no failed stage is a prefill that completed.
func (s *requestSummary) failedStageMarker() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failedStage != "" {
		return s.failedStage
	}
	if _, ok := s.durations[routing.StagePrefill]; ok {
		return routing.StageNonePrefilled
	}
	return routing.StageNone
}

// requestRoute is where the handler sent a request, for its summary line.
type requestRoute struct {
	path      string
	prefiller string
	decoder   string
	encoders  int
}

// log writes the request's summary line at the default level, so a request can
// be found by its trace_id without raising the sidecar's verbosity. Stages that
// did not run are left out.
// grpcStatus is the response grpc-status, or "" for a non-gRPC response.
func (s *requestSummary) log(logger logr.Logger, route requestRoute, status *responseStatus, grpcStatus string, elapsed time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()

	keysAndValues := []any{"path", route.path, "status", status.statusCode, "duration", elapsed, "decoder", route.decoder}
	if grpcStatus != "" {
		keysAndValues = append(keysAndValues, "grpcStatus", grpcStatus)
	}
	if route.prefiller != "" {
		keysAndValues = append(keysAndValues, "prefiller", route.prefiller)
	}
	if route.encoders > 0 {
		keysAndValues = append(keysAndValues, "encoders", route.encoders)
	}
	for _, stage := range []string{routing.StageEncode, routing.StagePrefill, routing.StageDecode} {
		if d, ok := s.durations[stage]; ok {
			keysAndValues = append(keysAndValues, stage+"Duration", d)
		}
	}
	if s.failedStage != "" {
		keysAndValues = append(keysAndValues, "failedStage", s.failedStage)
	}
	if status.writeFailed {
		keysAndValues = append(keysAndValues, "clientWriteFailed", true)
	}
	logger.Info("request finished", keysAndValues...)
}
