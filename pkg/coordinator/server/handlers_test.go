package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

type stubStep struct {
	name string
	err  error
}

func (s stubStep) Name() string { return s.name }

func (s stubStep) Execute(context.Context, *pipeline.RequestContext) error { return s.err }

func newTestServer(stepErr error) *Server {
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub", err: stepErr}})
	return New(config.ServerConfig{}, p)
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
