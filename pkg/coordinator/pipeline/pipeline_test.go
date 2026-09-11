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

package pipeline

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
)

type mockStep struct {
	name string
	fn   func(ctx context.Context, rc *RequestContext) error
}

func (m *mockStep) Name() string { return m.name }
func (m *mockStep) Execute(ctx context.Context, rc *RequestContext) error {
	return m.fn(ctx, rc)
}

func TestPipeline_ExecutesStepsInOrder(t *testing.T) {
	order := []string{}
	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			order = append(order, "a")
			return nil
		}},
		&mockStep{name: "b", fn: func(_ context.Context, _ *RequestContext) error {
			order = append(order, "b")
			return nil
		}},
		&mockStep{name: "c", fn: func(_ context.Context, _ *RequestContext) error {
			order = append(order, "c")
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(context.Background(), &RequestContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(order) != 3 || order[0] != "a" || order[1] != "b" || order[2] != "c" {
		t.Fatalf("unexpected execution order: %v", order)
	}
}

func TestPipeline_GeneratesRevisionDecisionID(t *testing.T) {
	var decisionID string
	p := New([]Step{&mockStep{name: "capture", fn: func(_ context.Context, rc *RequestContext) error {
		decisionID = rc.RevisionDecisionID
		return nil
	}}})

	if err := p.Execute(context.Background(), &RequestContext{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if decisionID == "" {
		t.Fatal("pipeline did not generate a revision decision ID")
	}
}

func TestPipeline_AbortsOnError(t *testing.T) {
	executed := map[string]bool{}
	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			executed["a"] = true
			return errors.New("step a failed")
		}},
		&mockStep{name: "b", fn: func(_ context.Context, _ *RequestContext) error {
			executed["b"] = true
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(context.Background(), &RequestContext{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !executed["a"] {
		t.Fatal("step a should have executed")
	}
	if executed["b"] {
		t.Fatal("step b should NOT have executed")
	}
}

func TestPipeline_StopsOnErrPipelineDone(t *testing.T) {
	executed := map[string]bool{}
	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			executed["a"] = true
			return ErrPipelineDone
		}},
		&mockStep{name: "b", fn: func(_ context.Context, _ *RequestContext) error {
			executed["b"] = true
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(context.Background(), &RequestContext{})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if !executed["a"] {
		t.Fatal("step a should have executed")
	}
	if executed["b"] {
		t.Fatal("step b should NOT have executed after ErrPipelineDone")
	}
}

func TestPipeline_RespectsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			t.Fatal("should not execute")
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(ctx, &RequestContext{})
	if err == nil {
		t.Fatal("expected cancellation error")
	}
}

// newMetricsRegistry wires the coordinator's package-level metric vectors
// onto a fresh registry so the pipeline's step recording is observable in
// isolation, and clears their state so concurrent tests do not see each
// other's increments.
func newMetricsRegistry(t *testing.T) *prometheus.Registry {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	return reg
}

// stepErrorCount reads the error counter for the given step and error_code
// via the passed gatherer. Absent series is 0.
func stepErrorCount(t *testing.T, reg *prometheus.Registry, step, errorCode string) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != "llm_d_coordinator_step_errors_total" {
			continue
		}
		for _, m := range mf.GetMetric() {
			labels := map[string]string{}
			for _, l := range m.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			if labels["step"] == step && labels["error_code"] == errorCode {
				return m.GetCounter().GetValue()
			}
		}
	}
	return 0
}

func TestExecute_SuccessRecordsStepDuration(t *testing.T) {
	reg := newMetricsRegistry(t)
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
		&mockStep{name: "decode", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
	}

	if err := New(steps).Execute(context.Background(), &RequestContext{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	require.InDelta(t, 0.0,
		mustGauge(t, reg, "llm_d_coordinator_step_running", map[string]string{"step": "render"}),
		1e-9, "step_running must be balanced back to 0 after success",
	)
	require.InDelta(t, 0.0, stepErrorCount(t, reg, "render", coordmetrics.ErrorCodeInternal), 1e-9)
}

func TestExecute_StepPanicRecordsErrorAndBalancesGauge(t *testing.T) {
	reg := newMetricsRegistry(t)
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, _ *RequestContext) error {
			panic("boom")
		}},
	}
	defer func() {
		r := recover()
		require.NotNil(t, r, "step panic must propagate out of Execute so chi Recoverer handles it")
		require.InDelta(t, 0.0,
			mustGauge(t, reg, "llm_d_coordinator_step_running", map[string]string{"step": "render"}),
			1e-9, "step_running must be balanced back to 0 after a step panic",
		)
		require.InDelta(t, 1.0,
			stepErrorCount(t, reg, "render", coordmetrics.ErrorCodeInternal),
			1e-9, "step_errors_total must record the panic under error_code=internal",
		)
	}()
	_ = New(steps).Execute(context.Background(), &RequestContext{})
}

func TestExecute_BadRequestErrorClassified(t *testing.T) {
	reg := newMetricsRegistry(t)
	stepErr := fmt.Errorf("render: prompt must be a string: %w", ErrBadRequest)
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, _ *RequestContext) error { return stepErr }},
	}
	if err := New(steps).Execute(context.Background(), &RequestContext{}); err == nil {
		t.Fatal("expected error")
	}
	require.InDelta(t, 1.0, stepErrorCount(t, reg, "render", coordmetrics.ErrorCodeBadRequest), 1e-9)
}

func TestExecute_Upstream4xxAnd5xxClassified(t *testing.T) {
	cases := []struct {
		name       string
		status     int
		wantCode   string
		otherCodes []string
	}{
		{"4xx", http.StatusUnprocessableEntity, coordmetrics.ErrorCodeUpstream4xx, []string{coordmetrics.ErrorCodeUpstream5xx, coordmetrics.ErrorCodeInternal}},
		{"5xx", http.StatusServiceUnavailable, coordmetrics.ErrorCodeUpstream5xx, []string{coordmetrics.ErrorCodeUpstream4xx, coordmetrics.ErrorCodeInternal}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := newMetricsRegistry(t)
			stepErr := fmt.Errorf("prefill: %w", &UpstreamError{Step: "prefill", StatusCode: tc.status})
			steps := []Step{
				&mockStep{name: "prefill", fn: func(_ context.Context, _ *RequestContext) error { return stepErr }},
			}
			if err := New(steps).Execute(context.Background(), &RequestContext{}); err == nil {
				t.Fatal("expected error")
			}
			require.InDelta(t, 1.0, stepErrorCount(t, reg, "prefill", tc.wantCode), 1e-9)
			for _, other := range tc.otherCodes {
				require.Zero(t, stepErrorCount(t, reg, "prefill", other), "series for %q must be absent", other)
			}
		})
	}
}

func TestExecute_ErrPipelineDoneIsNotAnError(t *testing.T) {
	reg := newMetricsRegistry(t)
	steps := []Step{
		&mockStep{name: "conditional-decode", fn: func(_ context.Context, _ *RequestContext) error { return ErrPipelineDone }},
	}
	if err := New(steps).Execute(context.Background(), &RequestContext{}); err != nil {
		t.Fatalf("expected nil (clean early exit), got %v", err)
	}
	// No error code should have any observations; the metric family may still
	// carry other series from prior tests, so scan for our step name.
	for _, code := range []string{coordmetrics.ErrorCodeBadRequest, coordmetrics.ErrorCodeUpstream4xx, coordmetrics.ErrorCodeUpstream5xx, coordmetrics.ErrorCodeInternal} {
		require.Zero(t, stepErrorCount(t, reg, "conditional-decode", code))
	}
}

// pathCount reads the execution_path_total counter for the given path under
// the test's canonical model_name "m".
func pathCount(t *testing.T, reg *prometheus.Registry, path string) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != "llm_d_coordinator_execution_path_total" {
			continue
		}
		for _, m := range mf.GetMetric() {
			labels := map[string]string{}
			for _, l := range m.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			if labels["model_name"] == "m" && labels["path"] == path {
				return m.GetCounter().GetValue()
			}
		}
	}
	return 0
}

// inputTokensCount reads the request_input_tokens histogram observation count
// for the given model. Zero means the histogram never observed a value for
// that label combination.
func inputTokensCount(t *testing.T, reg *prometheus.Registry, model string) uint64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != "llm_d_coordinator_request_input_tokens" {
			continue
		}
		for _, m := range mf.GetMetric() {
			labels := map[string]string{}
			for _, l := range m.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			if labels["model_name"] == model {
				return m.GetHistogram().GetSampleCount()
			}
		}
	}
	return 0
}

func TestExecute_ExecutionPathTable(t *testing.T) {
	cases := []struct {
		name      string
		stepNames []string
		wantPath  string
	}{
		{"decode-only via cache hit", []string{"conditional-decode"}, coordmetrics.PathDecodeOnly},
		{"prefill-decode without encode", []string{"prefill", "decode"}, coordmetrics.PathPrefillDecode},
		{"encode-prefill-decode full path", []string{"encode", "prefill", "decode"}, coordmetrics.PathEncodePrefillDecode},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := newMetricsRegistry(t)
			steps := make([]Step, 0, len(tc.stepNames))
			for _, n := range tc.stepNames {
				fn := func(_ context.Context, _ *RequestContext) error { return nil }
				if n == "conditional-decode" {
					// The cache-hit case returns ErrPipelineDone; the executor
					// must still count the step toward the path.
					fn = func(_ context.Context, _ *RequestContext) error { return ErrPipelineDone }
				}
				steps = append(steps, &mockStep{name: n, fn: fn})
			}
			if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			require.InDelta(t, 1.0, pathCount(t, reg, tc.wantPath), 1e-9)
		})
	}
}

func TestExecute_ExecutionPathRecordedOnDecodeFailure(t *testing.T) {
	// Regular decode fails after running: path totals must still record the
	// request as prefill-decode.
	reg := newMetricsRegistry(t)
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
		&mockStep{name: "prefill", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
		&mockStep{name: "decode", fn: func(_ context.Context, _ *RequestContext) error {
			return &UpstreamStreamedError{Step: "decode", StatusCode: http.StatusServiceUnavailable}
		}},
	}
	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err == nil {
		t.Fatal("expected error")
	}
	require.InDelta(t, 1.0, pathCount(t, reg, coordmetrics.PathPrefillDecode), 1e-9)
}

func TestExecute_ExecutionPathRecordedOnConditionalDecodeFailure(t *testing.T) {
	// Conditional-decode probe fails (transport error / worker 4xx/5xx):
	// the request was routed via decode-only and must record that.
	reg := newMetricsRegistry(t)
	steps := []Step{
		&mockStep{name: "conditional-decode", fn: func(_ context.Context, _ *RequestContext) error {
			return &UpstreamStreamedError{Step: "conditional-decode", Cause: errors.New("connection reset")}
		}},
	}
	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err == nil {
		t.Fatal("expected error")
	}
	require.InDelta(t, 1.0, pathCount(t, reg, coordmetrics.PathDecodeOnly), 1e-9)
}

func TestExecute_ExecutionPathNotRecordedWhenDecodeAbsent(t *testing.T) {
	reg := newMetricsRegistry(t)
	// A pipeline that aborts before decode-ish must not record any path.
	stepErr := fmt.Errorf("prefill: %w", &UpstreamError{Step: "prefill", StatusCode: http.StatusServiceUnavailable})
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
		&mockStep{name: "prefill", fn: func(_ context.Context, _ *RequestContext) error { return stepErr }},
	}
	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err == nil {
		t.Fatal("expected error")
	}
	for _, p := range []string{coordmetrics.PathDecodeOnly, coordmetrics.PathPrefillDecode, coordmetrics.PathEncodePrefillDecode} {
		require.Zero(t, pathCount(t, reg, p))
	}
}

func TestExecute_InputTokensRecordedOnlyWhenPopulated(t *testing.T) {
	// TokenIDs is populated by render; the executor records after any success
	// or clean early-exit path.
	reg := newMetricsRegistry(t)
	steps := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, rc *RequestContext) error {
			rc.TokenIDs = []int{1, 2, 3, 4}
			return nil
		}},
		&mockStep{name: "decode", fn: func(_ context.Context, _ *RequestContext) error { return nil }},
	}
	if err := New(steps).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	require.Equal(t, uint64(1), inputTokensCount(t, reg, "m"))

	// A pipeline that never runs render leaves TokenIDs empty, so the
	// histogram must not observe zero.
	reg2 := newMetricsRegistry(t)
	steps2 := []Step{
		&mockStep{name: "prefill", fn: func(_ context.Context, _ *RequestContext) error {
			return fmt.Errorf("prefill: %w", &UpstreamError{Step: "prefill", StatusCode: http.StatusServiceUnavailable})
		}},
	}
	_ = New(steps2).Execute(context.Background(), &RequestContext{Model: "m"})
	require.Equal(t, uint64(0), inputTokensCount(t, reg2, "m"))

	// A render that succeeds with a zero-length prompt (e.g. completions
	// empty array) is a valid observation and must be recorded.
	reg3 := newMetricsRegistry(t)
	steps3 := []Step{
		&mockStep{name: "render", fn: func(_ context.Context, rc *RequestContext) error {
			rc.TokenIDs = []int{}
			return nil
		}},
	}
	if err := New(steps3).Execute(context.Background(), &RequestContext{Model: "m"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	require.Equal(t, uint64(1), inputTokensCount(t, reg3, "m"))
}

// mustGauge finds a gauge series matching all of labels in reg and returns
// its value. A missing series is a test failure, so callers cannot silently
// see 0 when the series was never emitted.
func mustGauge(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			match := true
			got := map[string]string{}
			for _, l := range m.GetLabel() {
				got[l.GetName()] = l.GetValue()
			}
			for k, v := range labels {
				if got[k] != v {
					match = false
					break
				}
			}
			if match {
				return m.GetGauge().GetValue()
			}
		}
	}
	t.Fatalf("gauge %s%v not present", name, labels)
	return 0
}
