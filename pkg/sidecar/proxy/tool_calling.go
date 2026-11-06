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
	"errors"
	"fmt"
	"net/http"

	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

const toolCallingBaselineKey contextKey = "toolCallingBaseline"
const toolCallingRequestSpanKey contextKey = "toolCallingRequestSpan"

type toolCallingTelemetryEnabledKey struct{}

func toolCallingTelemetryEnabled(ctx context.Context) bool {
	enabled, _ := ctx.Value(toolCallingTelemetryEnabledKey{}).(bool)
	return enabled
}

// toolCallingRequest is owned by the dispatching goroutine. Comparisons for
// asynchronous prefill must be aggregated before starting its goroutine, so
// reporting does not require waiting for prefill to finish.
type toolCallingRequest struct {
	surface  reqcommon.APIType
	original toolcalling.RequestSnapshot
	statuses []toolcalling.FieldStatus
}

// Capture errors disable observation; callers must preserve routing behavior.
func captureToolCallingRequest(apiType reqcommon.APIType, body []byte) (*toolCallingRequest, error) {
	return captureToolCallingRequestBody(apiType, body, false)
}

func captureToolCallingRequestBody(apiType reqcommon.APIType, body []byte, validated bool) (*toolCallingRequest, error) {
	switch apiType {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages:
	default:
		return nil, fmt.Errorf("unsupported tool-calling API type %s", apiType)
	}
	capture := toolcalling.CaptureRequestJSON
	if validated {
		capture = toolcalling.CaptureValidatedRequestJSON
	}
	original, err := capture(apiType, body)
	if err != nil {
		return nil, err
	}
	return &toolCallingRequest{surface: apiType, original: original}, nil
}

// Inference handoffs reuse the client snapshot, with independent dispatch outcomes.
func captureInferenceToolCallingRequest(r *http.Request, apiType reqcommon.APIType, body []byte) (*toolCallingRequest, error) {
	if !toolCallingTelemetryEnabled(r.Context()) {
		return nil, nil //nolint:nilnil // Disabled observation has no capture or error.
	}
	baseline, ok := r.Context().Value(toolCallingBaselineKey).(*toolCallingRequest)
	var captured *toolCallingRequest
	if !ok {
		var err error
		// Inference callers supply unchanged bytes from readJSONBody or json.Marshal.
		captured, err = captureToolCallingRequestBody(apiType, body, true)
		if err != nil {
			return nil, err
		}
	} else {
		if baseline == nil {
			return nil, errors.New("tool-calling client baseline unavailable")
		}
		if baseline.surface != apiType {
			return nil, fmt.Errorf("tool-calling baseline API %s does not match %s", baseline.surface, apiType)
		}
		captured = &toolCallingRequest{surface: apiType, original: baseline.original}
	}
	captured.setRequestSpanAttributes(r.Context())
	return captured, nil
}

func (r *toolCallingRequest) setRequestSpanAttributes(ctx context.Context) {
	if r == nil {
		return
	}
	span, _ := ctx.Value(toolCallingRequestSpanKey).(trace.Span)
	if span != nil && span.IsRecording() {
		span.SetAttributes(r.original.SpanAttributes(r.statuses)...)
	}
}

// The dispatch owner finalizes the request before its span ends, even on abort.
func (r *toolCallingRequest) report(ctx context.Context) {
	if r == nil {
		return
	}
	metrics.RecordToolCallingFieldStatuses(r.fieldStatuses())
	r.setRequestSpanAttributes(ctx)
}

// compareDispatched must be called at dispatch, after local routing checks.
// Prepared bodies for requests that will not be sent provide no evidence.
// Returned statuses describe this dispatch; the aggregate deduplicates retries
// and decode chunks by keeping the worst outcome for each observed field.
func (r *toolCallingRequest) compareDispatched(body []byte) ([]toolcalling.FieldStatus, error) {
	return r.compareDispatchedBody(body, false)
}

// compareValidatedDispatched requires an unchanged body from readJSONBody or json.Marshal.
func (r *toolCallingRequest) compareValidatedDispatched(body []byte) ([]toolcalling.FieldStatus, error) {
	return r.compareDispatchedBody(body, true)
}

func (r *toolCallingRequest) compareDispatchedBody(body []byte, validated bool) ([]toolcalling.FieldStatus, error) {
	if r == nil {
		return nil, nil
	}
	capture := toolcalling.CaptureRequestJSON
	if validated {
		capture = toolcalling.CaptureValidatedRequestJSON
	}
	after, err := capture(r.surface, body)
	if err != nil {
		return nil, err
	}
	// Empty captures would only allocate unobserved outcomes.
	if r.original.CanSkipCapture(nil) && after.CanSkipCapture(nil) {
		return nil, nil
	}
	statuses, err := toolcalling.CompareRequests(r.original, after)
	if err != nil {
		return nil, err
	}
	r.mergeStatuses(statuses)
	return statuses, nil
}

// Stage spans describe their dispatch; request metrics keep the worst outcome.
func (r *toolCallingRequest) compareDispatchedForSpan(body []byte, span trace.Span) {
	statuses, err := r.compareDispatched(body)
	if r != nil && err == nil && span.IsRecording() {
		span.SetAttributes(r.original.SpanAttributes(statuses)...)
	}
}

// compareValidatedDispatchedForSpan requires an unchanged body from readJSONBody or json.Marshal.
func (r *toolCallingRequest) compareValidatedDispatchedForSpan(body []byte, span trace.Span) {
	statuses, err := r.compareValidatedDispatched(body)
	if r != nil && err == nil && span.IsRecording() {
		span.SetAttributes(r.original.SpanAttributes(statuses)...)
	}
}

func (r *toolCallingRequest) mergeStatuses(statuses []toolcalling.FieldStatus) {
	for _, result := range statuses {
		if !result.Observed {
			continue
		}
		found := false
		for i := range r.statuses {
			if r.statuses[i].Field == result.Field {
				r.statuses[i].Status = worstToolCallingFieldStatus(r.statuses[i].Status, result.Status)
				found = true
				break
			}
		}
		if !found {
			r.statuses = append(r.statuses, result)
		}
	}
}

func (r *toolCallingRequest) fieldStatuses() []toolcalling.FieldStatus {
	if r == nil {
		return nil
	}
	return append([]toolcalling.FieldStatus(nil), r.statuses...)
}

func worstToolCallingFieldStatus(first, second toolcalling.FieldStatusValue) toolcalling.FieldStatusValue {
	worst := toolcalling.FieldStatusPreserved
	for _, status := range []toolcalling.FieldStatusValue{toolcalling.FieldStatusChanged, toolcalling.FieldStatusDropped, toolcalling.FieldStatusRejected} {
		if first == status || second == status {
			worst = status
		}
	}
	return worst
}
