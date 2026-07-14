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

package server

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

type stubStep struct {
	name string
	err  error
}

func (s stubStep) Name() string { return s.name }

func (s stubStep) Execute(context.Context, *pipeline.RequestContext) error { return s.err }

func newTestServer(stepErr error) *Server {
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub", err: stepErr}})
	srv, err := New(config.ServerConfig{}, p)
	if err != nil {
		panic(err) // default config is always valid
	}
	return srv
}

func postInference(t *testing.T, srv *Server) *httptest.ResponseRecorder {
	t.Helper()
	return postInferenceWithRequestID(t, srv, "")
}

func postInferenceWithRequestID(t *testing.T, srv *Server, requestID string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	if requestID != "" {
		req.Header.Set(reqcommon.RequestIDHeaderKey, requestID)
	}
	rec := httptest.NewRecorder()
	srv.handleInference(rec, req)
	return rec
}

func TestHandleInference_ClientErrorMapsTo400(t *testing.T) {
	// A step that wraps ErrBadRequest signals invalid client input; the handler
	// must surface 400, not 502.
	stepErr := fmt.Errorf("render: prompt must be a string: %w", pipeline.ErrBadRequest)
	rec := postInference(t, newTestServer(stepErr))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for client error, got %d", rec.Code)
	}
}

func TestHandleInference_UnclassifiedErrorMapsTo502(t *testing.T) {
	// A plain step error (no ErrBadRequest, no UpstreamError) stays 502.
	stepErr := errors.New("prefill: connect: connection refused")
	rec := postInference(t, newTestServer(stepErr))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for unclassified error, got %d", rec.Code)
	}
}

func TestHandleInference_Upstream4xxIsForwarded(t *testing.T) {
	// An upstream 4xx means the request was the root cause; forward the status.
	stepErr := fmt.Errorf("wrapped: %w", &pipeline.UpstreamError{Step: "render", StatusCode: http.StatusUnprocessableEntity, Body: "bad tokens"})
	rec := postInference(t, newTestServer(stepErr))
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("expected upstream 422 to be forwarded, got %d", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "bad tokens") {
		t.Fatalf("upstream body must not leak to the client: %q", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "render") {
		t.Fatalf("client message should name the step: %q", rec.Body.String())
	}
}

func TestHandleInference_Upstream5xxMapsTo502(t *testing.T) {
	// An upstream 5xx is a gateway fault, not the client's; report 502.
	stepErr := fmt.Errorf("wrapped: %w", &pipeline.UpstreamError{Step: "prefill", StatusCode: http.StatusServiceUnavailable, Body: "down"})
	rec := postInference(t, newTestServer(stepErr))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for upstream 5xx, got %d", rec.Code)
	}
}

func TestHandleInference_SuccessMapsTo200(t *testing.T) {
	rec := postInference(t, newTestServer(nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 on success, got %d", rec.Code)
	}
}

func TestHandleInference_NullBodyMapsTo400(t *testing.T) {
	// JSON `null` unmarshals to a nil map without error; reject it before steps
	// write to the body and panic.
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("null"))
	rec := httptest.NewRecorder()
	newTestServer(nil).handleInference(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for null body, got %d", rec.Code)
	}
}

func TestHandleInference_BodyOverConfiguredCapMapsTo413(t *testing.T) {
	// A body larger than server.max_request_body_size (in MB) is rejected before parsing.
	// Use a 1 MB cap and send 1 MB + 1 byte to trigger the limit.
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	srv, err := New(config.ServerConfig{MaxRequestBodySize: 1}, p)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	oversize := strings.Repeat("x", config.BytesPerMB+1)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(oversize))
	rec := httptest.NewRecorder()
	srv.handleInference(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413 for oversize body, got %d", rec.Code)
	}
}

func TestNew_RejectsNegativeMaxRequestBodySize(t *testing.T) {
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	if _, err := New(config.ServerConfig{MaxRequestBodySize: -1}, p); err == nil {
		t.Fatal("expected error for negative MaxRequestBodySize")
	}
}

func TestNew_RejectsOverflowMaxRequestBodySize(t *testing.T) {
	// MaxInt64 would cause maxRequestBodySize+1 to overflow to a negative
	// io.LimitReader limit, making it return immediate EOF.
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	if _, err := New(config.ServerConfig{MaxRequestBodySize: math.MaxInt64}, p); err == nil {
		t.Fatal("expected error for MaxRequestBodySize > MaxInt64-1")
	}
}

// deadlineRecorder wraps httptest.ResponseRecorder with a SetWriteDeadline
// method so http.NewResponseController can reach it; it records the deadline
// the handler sets.
type deadlineRecorder struct {
	*httptest.ResponseRecorder
	deadlineSet   bool
	deadlineValue time.Time
}

func (d *deadlineRecorder) SetWriteDeadline(t time.Time) error {
	d.deadlineSet = true
	d.deadlineValue = t
	return nil
}

func TestHandleInference_StreamingClearsWriteDeadline(t *testing.T) {
	// A streaming request must clear the connection write deadline (zero value)
	// before executing the pipeline, so a long stream is not cut by WriteTimeout.
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m","stream":true}`))
	rec := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	newTestServer(nil).handleInference(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for streaming request, got %d", rec.Code)
	}
	if !rec.deadlineSet {
		t.Fatal("streaming request did not clear the write deadline")
	}
	if !rec.deadlineValue.IsZero() {
		t.Fatalf("expected zero deadline to disable the timeout, got %v", rec.deadlineValue)
	}
}

func TestHandleInference_NonStreamingDoesNotTouchWriteDeadline(t *testing.T) {
	// A non-streaming request must leave the write deadline alone, keeping
	// WriteTimeout as a slow-client guard.
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	rec := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	newTestServer(nil).handleInference(rec, req)
	if rec.deadlineSet {
		t.Fatal("non-streaming request must not change the write deadline")
	}
}

func TestHandleInference_ValidRequestIDIsReflected(t *testing.T) {
	// A well-formed client request ID is echoed in the error response.
	stepErr := fmt.Errorf("render: %w", pipeline.ErrBadRequest)
	rec := postInferenceWithRequestID(t, newTestServer(stepErr), "req-abc-123")
	if !strings.Contains(rec.Body.String(), "req-abc-123") {
		t.Fatalf("expected valid request_id in response, got %q", rec.Body.String())
	}
}

func TestHandleInference_MaliciousRequestIDIsRejected(t *testing.T) {
	// A request ID with disallowed characters must not be reflected into the
	// error response; the handler substitutes a generated one.
	stepErr := fmt.Errorf("render: %w", pipeline.ErrBadRequest)
	malicious := "evil\r\nInjected: header value with spaces"
	rec := postInferenceWithRequestID(t, newTestServer(stepErr), malicious)
	if strings.Contains(rec.Body.String(), malicious) || strings.Contains(rec.Body.String(), "Injected") {
		t.Fatalf("malicious request_id must not leak to the client: %q", rec.Body.String())
	}
}

func TestHandleInference_OverlongRequestIDIsRejected(t *testing.T) {
	// A request ID over the length cap is replaced with a generated one.
	stepErr := fmt.Errorf("render: %w", pipeline.ErrBadRequest)
	overlong := strings.Repeat("a", 129)
	rec := postInferenceWithRequestID(t, newTestServer(stepErr), overlong)
	if strings.Contains(rec.Body.String(), overlong) {
		t.Fatalf("overlong request_id must not leak to the client: %q", rec.Body.String())
	}
}
