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
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

type stubStep struct {
	name string
	err  error
	// fn, when non-nil, takes precedence over err. Tests use it to observe
	// state (e.g. gauge labels) at pipeline-execution time.
	fn func(context.Context, *pipeline.RequestContext) error
}

func (s stubStep) Name() string { return s.name }

func (s stubStep) Execute(ctx context.Context, rc *pipeline.RequestContext) error {
	if s.fn != nil {
		return s.fn(ctx, rc)
	}
	return s.err
}

// stubGatewayURL is a placeholder used by tests that never actually issue a
// passthrough request. A real value only matters in passthrough_test.go.
const stubGatewayURL = "http://gateway-stub.invalid"

func newTestServer(stepErr error) *Server {
	return newTestServerWithGateway(stepErr, stubGatewayURL)
}

func newTestServerWithGateway(stepErr error, gatewayURL string) *Server {
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub", err: stepErr}})
	// A default *http.Transport{} is safe here: the passthrough handler needs a
	// non-typed-nil RoundTripper so httputil.ReverseProxy dials the test server.
	gw := gateway.NewWithTransport(&http.Transport{}, gatewayURL)
	srv, err := New(config.ServerConfig{}, p, gw)
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
	srv, err := New(config.ServerConfig{MaxRequestBodySize: 1}, p, gateway.NewWithTransport(nil, stubGatewayURL))
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
	if _, err := New(config.ServerConfig{MaxRequestBodySize: -1}, p, gateway.NewWithTransport(nil, stubGatewayURL)); err == nil {
		t.Fatal("expected error for negative MaxRequestBodySize")
	}
}

func TestNew_RejectsOverflowMaxRequestBodySize(t *testing.T) {
	// MaxInt64 would cause maxRequestBodySize+1 to overflow to a negative
	// io.LimitReader limit, making it return immediate EOF.
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	if _, err := New(config.ServerConfig{MaxRequestBodySize: math.MaxInt64}, p, gateway.NewWithTransport(nil, stubGatewayURL)); err == nil {
		t.Fatal("expected error for MaxRequestBodySize > MaxInt64-1")
	}
}

func TestNew_RejectsNilGatewayClient(t *testing.T) {
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	if _, err := New(config.ServerConfig{}, p, nil); err == nil {
		t.Fatal("expected error for nil gateway client")
	}
}

func TestNew_RejectsGatewayURLWithoutHost(t *testing.T) {
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	if _, err := New(config.ServerConfig{}, p, gateway.NewWithTransport(nil, "not-a-url")); err == nil {
		t.Fatal("expected error for gateway URL without a host")
	}
}

func TestNew_RejectsUnparsableGatewayURL(t *testing.T) {
	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	if _, err := New(config.ServerConfig{}, p, gateway.NewWithTransport(nil, "http://gw:notaport")); err == nil {
		t.Fatal("expected error for unparsable gateway URL")
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

func TestRoutesRegistered(t *testing.T) {
	// With the NotFound passthrough in place, an unregistered path no longer
	// returns 404 — it reverse-proxies to the stub gateway and returns 502.
	// Assert 200 to prove the path reaches its handler (stub step + handleHealth
	// both leave the recorder at its default 200).
	const inferenceBody = `{"model":"m","token_ids":[1,2,3]}`
	tests := []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{"chat completions", http.MethodPost, gateway.PathChatCompletions, inferenceBody},
		{"completions", http.MethodPost, gateway.PathCompletions, inferenceBody},
		{"generate", http.MethodPost, gateway.DefaultGeneratePath, inferenceBody},
		{"healthz", http.MethodGet, "/healthz", ""},
		{"readyz", http.MethodGet, "/readyz", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newTestServer(nil)
			req := httptest.NewRequest(tt.method, tt.path, strings.NewReader(tt.body))
			rec := httptest.NewRecorder()
			srv.httpServer.Handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("route %s %s expected 200, got HTTP %d", tt.method, tt.path, rec.Code)
			}
		})
	}
}

// newMetricsRegistry wires the coordinator's metric vectors onto a fresh
// registry so a test can inspect their state end-to-end via the same package-
// level vectors the handler mutates. Reset clears the vectors' state so
// concurrent tests do not see each other's increments.
func newMetricsRegistry(t *testing.T) *prometheus.Registry {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	return reg
}

func TestHandleInference_SuccessRecordsRequestFamily(t *testing.T) {
	reg := newMetricsRegistry(t)

	rec := postInference(t, newTestServer(nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(mustCounter(t, reg, "llm_d_coordinator_request_total", map[string]string{"model_name": "m"})),
		1e-9,
	)
	require.Zero(t, seriesCount(t, reg, "llm_d_coordinator_request_error_total"), "no errors expected on the success path")
	if hc := histogramCount(t, reg, "llm_d_coordinator_request_duration_seconds", map[string]string{"model_name": "m"}); hc != 1 {
		t.Fatalf("expected 1 request_duration_seconds observation, got %d", hc)
	}
	if hc := histogramCount(t, reg, "llm_d_coordinator_request_size_bytes", map[string]string{"model_name": "m"}); hc != 1 {
		t.Fatalf("expected 1 request_size_bytes observation, got %d", hc)
	}
}

func TestHandleInference_UpstreamErrorRecordsErrorCode(t *testing.T) {
	reg := newMetricsRegistry(t)

	stepErr := fmt.Errorf("wrapped: %w", &pipeline.UpstreamError{Step: "prefill", StatusCode: http.StatusServiceUnavailable, Body: "down"})
	rec := postInference(t, newTestServer(stepErr))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", rec.Code)
	}

	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(mustCounter(t, reg, "llm_d_coordinator_request_error_total", map[string]string{"model_name": "m", "error_code": coordmetrics.ErrorCodeUpstream5xx})),
		1e-9,
	)
}

func TestHandleInference_StreamedUpstreamErrorClassifiedAndNotOverwritten(t *testing.T) {
	reg := newMetricsRegistry(t)

	// stubStep writes a streamed response body first, then returns an
	// UpstreamStreamedError. The handler must classify it for
	// request_error_total but must not call http.Error on top of the bytes
	// already on the wire.
	const streamedBody = "streamed 500 body from decode worker"
	step := stubStep{name: "decode", fn: func(_ context.Context, rc *pipeline.RequestContext) error {
		rc.ResponseWriter.WriteHeader(http.StatusInternalServerError)
		_, _ = rc.ResponseWriter.Write([]byte(streamedBody))
		return &pipeline.UpstreamStreamedError{Step: "decode", StatusCode: http.StatusInternalServerError}
	}}
	p := pipeline.New([]pipeline.Step{step})
	srv, err := New(config.ServerConfig{}, p, gateway.NewWithTransport(nil, stubGatewayURL))
	require.NoError(t, err)

	rec := postInference(t, srv)
	require.Equal(t, http.StatusInternalServerError, rec.Code, "streamed status must survive: no http.Error overwrite")
	require.Equal(t, streamedBody, rec.Body.String(), "streamed body must survive: no http.Error overwrite")

	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(mustCounter(t, reg, "llm_d_coordinator_request_error_total", map[string]string{"model_name": "m", "error_code": coordmetrics.ErrorCodeUpstream5xx})),
		1e-9, "streamed upstream 5xx must classify as upstream_5xx in request_error_total",
	)
}

func TestHandleInference_PipelinePanicRecordsErrorAndPropagates(t *testing.T) {
	reg := newMetricsRegistry(t)

	panicky := stubStep{name: "prefill", fn: func(_ context.Context, _ *pipeline.RequestContext) error {
		panic("boom")
	}}
	p := pipeline.New([]pipeline.Step{panicky})
	srv, err := New(config.ServerConfig{}, p, gateway.NewWithTransport(nil, stubGatewayURL))
	require.NoError(t, err)

	defer func() {
		r := recover()
		require.NotNil(t, r, "pipeline panic must propagate out of handleInference so chi Recoverer at the server edge can answer 500")

		require.InDelta(t, 1.0,
			promtestutil.ToFloat64(mustCounter(t, reg, "llm_d_coordinator_request_error_total", map[string]string{"model_name": "m", "error_code": coordmetrics.ErrorCodeInternal})),
			1e-9, "request_error_total must record the panic under error_code=internal",
		)
		require.InDelta(t, 1.0,
			promtestutil.ToFloat64(mustCounter(t, reg, "llm_d_coordinator_request_total", map[string]string{"model_name": "m"})),
			1e-9, "request_total must still fire on panic so error-rate ratios stay meaningful",
		)
		require.InDelta(t, 0.0,
			promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": "m"})),
			1e-9, "request_running must be balanced back to 0 after a panic",
		)
	}()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	rec := httptest.NewRecorder()
	srv.handleInference(rec, req)
}

func TestHandleInference_MalformedBodyRecordsBadRequestWithUnknownModel(t *testing.T) {
	reg := newMetricsRegistry(t)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("not-json"))
	rec := httptest.NewRecorder()
	newTestServer(nil).handleInference(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for malformed body, got %d", rec.Code)
	}

	// Malformed pre-parse: model attributes to "unknown".
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(mustCounter(t, reg, "llm_d_coordinator_request_total", map[string]string{"model_name": coordmetrics.ModelUnknown})),
		1e-9,
	)
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(mustCounter(t, reg, "llm_d_coordinator_request_error_total", map[string]string{"model_name": coordmetrics.ModelUnknown, "error_code": coordmetrics.ErrorCodeBadRequest})),
		1e-9,
	)
	// request_size_bytes covers the malformed branch under the unknown label.
	if hc := histogramCount(t, reg, "llm_d_coordinator_request_size_bytes", map[string]string{"model_name": coordmetrics.ModelUnknown}); hc != 1 {
		t.Fatalf("expected 1 request_size_bytes observation on invalid-JSON path, got %d", hc)
	}
}

func TestHandleInference_NullBodyRecordsRequestSize(t *testing.T) {
	// JSON null returns nil parsed; the size histogram must still be observed
	// under the unknown label so the family is consistent across early returns.
	reg := newMetricsRegistry(t)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("null"))
	rec := httptest.NewRecorder()
	newTestServer(nil).handleInference(rec, req)
	require.Equal(t, http.StatusBadRequest, rec.Code)

	if hc := histogramCount(t, reg, "llm_d_coordinator_request_size_bytes", map[string]string{"model_name": coordmetrics.ModelUnknown}); hc != 1 {
		t.Fatalf("expected 1 request_size_bytes observation on null-body path, got %d", hc)
	}
}

// erroringReader returns some bytes and then fails, so a test can drive the
// io.ReadAll error branch without racing on connection resets.
type erroringReader struct {
	prefix []byte
	pos    int
}

func (e *erroringReader) Read(p []byte) (int, error) {
	if e.pos < len(e.prefix) {
		n := copy(p, e.prefix[e.pos:])
		e.pos += n
		return n, nil
	}
	return 0, io.ErrUnexpectedEOF
}

func TestHandleInference_ReadErrorRecordsPartialRequestSize(t *testing.T) {
	// A partial-read failure must still observe the size histogram; the value
	// recorded is what was read before the error, not the promised size.
	reg := newMetricsRegistry(t)

	partial := []byte(`{"mod`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", &erroringReader{prefix: partial})
	rec := httptest.NewRecorder()
	newTestServer(nil).handleInference(rec, req)
	require.Equal(t, http.StatusBadRequest, rec.Code)

	if hc := histogramCount(t, reg, "llm_d_coordinator_request_size_bytes", map[string]string{"model_name": coordmetrics.ModelUnknown}); hc != 1 {
		t.Fatalf("expected 1 request_size_bytes observation on read-error path, got %d", hc)
	}
	if hs := histogramSum(t, reg, "llm_d_coordinator_request_size_bytes", map[string]string{"model_name": coordmetrics.ModelUnknown}); hs != float64(len(partial)) {
		t.Fatalf("expected partial-read length %d recorded on read-error path, got %v", len(partial), hs)
	}
}

// blockingReader gates the first Read call on unblock, so a test can catch
// the handler mid-io.ReadAll and inspect the in-flight gauge before the JSON
// parse has run.
type blockingReader struct {
	once    sync.Once
	started chan struct{}
	unblock chan struct{}
	body    io.Reader
}

func (b *blockingReader) Read(p []byte) (int, error) {
	b.once.Do(func() {
		close(b.started)
		<-b.unblock
	})
	return b.body.Read(p)
}

func TestHandleInference_PreparseWorkIsCountedAsInflight(t *testing.T) {
	reg := newMetricsRegistry(t)

	br := &blockingReader{
		started: make(chan struct{}),
		unblock: make(chan struct{}),
		body:    strings.NewReader(`{"model":"m"}`),
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", br)
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		newTestServer(nil).handleInference(rec, req)
		close(done)
	}()

	// Wait until the handler is inside io.ReadAll before observing. The gauge
	// must already reflect the request under the "unknown" label; the JSON
	// has not been parsed yet, so no other label is possible.
	<-br.started
	require.InDelta(t, 1.0,
		promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": coordmetrics.ModelUnknown})),
		1e-9, "gauge must be 1 during pre-parse under the unknown label",
	)

	close(br.unblock)
	<-done

	require.InDelta(t, 0.0,
		promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": coordmetrics.ModelUnknown})),
		1e-9, "unknown label must balance to 0 after the label was swapped and the handler returned",
	)
	require.InDelta(t, 0.0,
		promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": "m"})),
		1e-9, "parsed model label must balance to 0 after the handler returned",
	)
}

func TestHandleInference_OversizeBodyBalancesGauge(t *testing.T) {
	reg := newMetricsRegistry(t)

	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	srv, err := New(config.ServerConfig{MaxRequestBodySize: 1}, p, gateway.NewWithTransport(nil, stubGatewayURL))
	require.NoError(t, err)

	oversize := strings.Repeat("x", config.BytesPerMB+1)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(oversize))
	rec := httptest.NewRecorder()
	srv.handleInference(rec, req)
	require.Equal(t, http.StatusRequestEntityTooLarge, rec.Code)

	// The 413 path must touch request_running: mustRequestRunningGauge fails
	// if the series is absent, so an omitted Inc on the pre-parse path is
	// caught as a hard failure and cannot be read as a legitimate 0.
	require.InDelta(t, 0.0,
		promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": coordmetrics.ModelUnknown})),
		1e-9, "gauge must be back at 0 for the unknown label after 413",
	)
	// request_size_bytes covers the 413 branch under the unknown label.
	if hc := histogramCount(t, reg, "llm_d_coordinator_request_size_bytes", map[string]string{"model_name": coordmetrics.ModelUnknown}); hc != 1 {
		t.Fatalf("expected 1 request_size_bytes observation on 413 path, got %d", hc)
	}
}

func TestHandleInference_ParsedModelSwapsLabel(t *testing.T) {
	reg := newMetricsRegistry(t)

	var (
		unknownDuring float64
		parsedDuring  float64
	)
	observer := stubStep{
		name: "observe",
		fn: func(_ context.Context, _ *pipeline.RequestContext) error {
			unknownDuring = promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": coordmetrics.ModelUnknown}))
			parsedDuring = promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": "m"}))
			return nil
		},
	}
	p := pipeline.New([]pipeline.Step{observer})
	srv, err := New(config.ServerConfig{}, p, gateway.NewWithTransport(nil, stubGatewayURL))
	require.NoError(t, err)

	rec := postInference(t, srv)
	require.Equal(t, http.StatusOK, rec.Code)

	require.InDelta(t, 0.0, unknownDuring, 1e-9, "unknown must be 0 during pipeline: the entry Inc was swapped out on parse")
	require.InDelta(t, 1.0, parsedDuring, 1e-9, "parsed model must be 1 during pipeline: the entry Inc was swapped to this label on parse")

	require.InDelta(t, 0.0,
		promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": coordmetrics.ModelUnknown})),
		1e-9,
	)
	require.InDelta(t, 0.0,
		promtestutil.ToFloat64(mustRequestRunningGauge(t, reg, map[string]string{"model_name": "m"})),
		1e-9,
	)
}

// mustRequestRunningGauge finds the request_running gauge series matching all
// of labels in reg. A missing series fails the test, so an untouched series
// cannot be mistaken for a legitimate 0 by promtestutil.ToFloat64.
func mustRequestRunningGauge(t *testing.T, reg *prometheus.Registry, labels map[string]string) prometheus.Gauge {
	t.Helper()
	const name = "llm_d_coordinator_request_running"
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if labelsMatch(m.GetLabel(), labels) {
				g := prometheus.NewGauge(prometheus.GaugeOpts{Name: "shadow"})
				g.Set(m.GetGauge().GetValue())
				return g
			}
		}
	}
	t.Fatalf("gauge %s%v not present", name, labels)
	return nil
}

// mustCounter looks up the counter series matching all of labels from reg. The
// series must exist; a missing series is a test failure so promtestutil.ToFloat64
// cannot silently see zero from an absent series.
func mustCounter(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) prometheus.Counter {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if !labelsMatch(m.GetLabel(), labels) {
				continue
			}
			c := prometheus.NewCounter(prometheus.CounterOpts{Name: "shadow"})
			c.Add(m.GetCounter().GetValue())
			return c
		}
	}
	t.Fatalf("counter %s%v not present", name, labels)
	return nil
}

func histogramCount(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) uint64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if labelsMatch(m.GetLabel(), labels) {
				return m.GetHistogram().GetSampleCount()
			}
		}
	}
	t.Fatalf("histogram %s%v not present", name, labels)
	return 0
}

func histogramSum(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if labelsMatch(m.GetLabel(), labels) {
				return m.GetHistogram().GetSampleSum()
			}
		}
	}
	t.Fatalf("histogram %s%v not present", name, labels)
	return 0
}

// seriesCount returns the number of series (label-set combinations) that the
// metric family with the given name currently carries. Zero means the vector
// has never been observed under any label combination.
func seriesCount(t *testing.T, reg *prometheus.Registry, name string) int {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() == name {
			return len(mf.GetMetric())
		}
	}
	return 0
}

func labelsMatch(actual []*dto.LabelPair, want map[string]string) bool {
	if want == nil {
		return true
	}
	got := map[string]string{}
	for _, l := range actual {
		got[l.GetName()] = l.GetValue()
	}
	for k, v := range want {
		if got[k] != v {
			return false
		}
	}
	return true
}

func TestRoutesRegistered_MethodMismatchReturns405(t *testing.T) {
	// A registered route with the wrong method must return 405, not fall
	// through to the passthrough. Chi's default MethodNotAllowed handler
	// produces this; the coordinator does not override it.
	srv := newTestServer(nil)
	req := httptest.NewRequest(http.MethodGet, gateway.PathChatCompletions, nil)
	rec := httptest.NewRecorder()
	srv.httpServer.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 for GET on POST-only %s, got %d", gateway.PathChatCompletions, rec.Code)
	}
}
