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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
)

// ErrPipelineDone is returned by a step to signal successful early exit.
// The pipeline treats this as success and stops executing further steps.
var ErrPipelineDone = errors.New("pipeline done")

// ErrBadRequest marks a step failure as caused by invalid client input rather
// than an internal or upstream fault. Steps wrap it (with %w) when rejecting a
// malformed request so the server can answer 400 instead of 502.
var ErrBadRequest = errors.New("bad request")

// UpstreamError carries the HTTP status a step received from an upstream
// service (render, gateway). The server forwards a 4xx status to the client
// (the request was the root cause) and treats 5xx as a 502 gateway fault.
// Body holds the upstream response for programmatic inspection only; it is
// kept out of Error() (which may be logged) and off the client response, since
// it can carry prompt or user data.
type UpstreamError struct {
	Step       string
	StatusCode int
	Body       string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("%s: upstream returned HTTP %d", e.Step, e.StatusCode)
}

// UpstreamStreamedError signals that a streaming step (decode,
// conditional-decode) saw an upstream failure after the response was already
// committed to the client. The reverse proxy either streamed a 4xx/5xx body
// from the worker (StatusCode carries that status) or wrote 502 via its
// ErrorHandler on a transport failure (StatusCode is 0, Cause carries the
// transport error). Callers must classify this for request_error_total /
// step_errors_total but must not write another response body: the client is
// already committed.
type UpstreamStreamedError struct {
	Step       string
	StatusCode int
	Cause      error
}

func (e *UpstreamStreamedError) Error() string {
	if e.StatusCode == 0 {
		return fmt.Sprintf("%s: upstream transport error: %v", e.Step, e.Cause)
	}
	return fmt.Sprintf("%s: upstream returned HTTP %d (response already streamed to client)", e.Step, e.StatusCode)
}

func (e *UpstreamStreamedError) Unwrap() error { return e.Cause }

// Pipeline orchestrates the sequential execution of steps.
type Pipeline struct {
	steps []Step
}

// New creates a pipeline from an ordered list of steps.
func New(steps []Step) *Pipeline {
	return &Pipeline{steps: steps}
}

// Execute runs all steps in order. Any error aborts immediately.
func (p *Pipeline) Execute(ctx context.Context, reqCtx *RequestContext) error {
	logger := log.FromContext(ctx)

	type stepTiming struct {
		name     string
		duration time.Duration
	}
	timings := make([]stepTiming, len(p.steps))
	executed := map[string]bool{}
	defer func() {
		stats := make([]any, 0, (len(timings)+1)*2)
		if reqCtx.ParseDuration > 0 {
			stats = append(stats, "parse", reqCtx.ParseDuration.String())
		}
		for _, t := range timings {
			stats = append(stats, t.name, t.duration.String())
		}
		logger.V(logutil.DEFAULT).Info("pipeline step timings", stats...)

		if path, ok := classifyExecutionPath(executed); ok {
			coordmetrics.IncExecutionPath(reqCtx.Model, path)
		}
		// Render populates TokenIDs on success (chat, completions, and generate
		// paths); recording only when non-empty keeps failed-before-render
		// requests out of the histogram.
		if n := len(reqCtx.TokenIDs); n > 0 {
			coordmetrics.RecordRequestInputTokens(reqCtx.Model, n)
		}
	}()

	for idx, step := range p.steps {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("pipeline cancelled: %w", err)
		}
		name := step.Name()
		logger.V(logutil.TRACE).Info("step starting", "step", name)
		coordmetrics.IncStepRunning(name)
		start := time.Now()
		var err error
		func() {
			// defer covers all three step-observability signals on every exit
			// path (normal return, error return, panic). On panic recovery,
			// step_errors_total records the failure under error_code=internal
			// and the panic re-propagates into the chi Recoverer at the
			// server edge.
			defer func() {
				d := time.Since(start)
				coordmetrics.RecordStepDuration(name, d)
				coordmetrics.DecStepRunning(name)
				timings[idx] = stepTiming{name: name, duration: d}
				if r := recover(); r != nil {
					coordmetrics.IncStepErrorTotal(name, coordmetrics.ErrorCodeInternal)
					panic(r)
				}
			}()
			err = step.Execute(ctx, reqCtx)
		}()
		if err != nil {
			if errors.Is(err, ErrPipelineDone) {
				// Clean early exit (e.g. conditional-decode cache hit); not an
				// error, and the step's work counts toward the execution path.
				executed[name] = true
				return nil
			}
			coordmetrics.IncStepErrorTotal(name, coordmetrics.ClassifyErrorCode(err, classifyOpts))
			return fmt.Errorf("step %q failed: %w", name, err)
		}
		executed[name] = true
		logger.V(logutil.TRACE).Info("step complete", "step", name)
	}
	return nil
}

// classifyExecutionPath maps the set of successfully-executed steps to the
// execution_path_total label. Returns false when no decode-ish step ran, so
// pipelines aborted before decode do not spuriously record a path. Step-name
// strings are hardcoded here because the pipeline package cannot import the
// steps package without introducing a dependency cycle. They match each step
// file's own StepName constant by contract; keep the two in sync.
func classifyExecutionPath(executed map[string]bool) (string, bool) {
	decodeIsh := executed["decode"] || executed["conditional-decode"]
	if !decodeIsh {
		return "", false
	}
	switch {
	case executed["encode"] && executed["prefill"]:
		return coordmetrics.PathEncodePrefillDecode, true
	case executed["prefill"]:
		return coordmetrics.PathPrefillDecode, true
	default:
		return coordmetrics.PathDecodeOnly, true
	}
}

// classifyOpts injects the pipeline's error sentinels into the shared
// coordmetrics.ClassifyErrorCode so step_errors_total and request_error_total
// share one mapping and cannot drift.
var classifyOpts = coordmetrics.ClassifyOptions{
	BadRequest: ErrBadRequest,
	IsUpstream: func(err error) (int, bool) {
		var u *UpstreamError
		if errors.As(err, &u) {
			return u.StatusCode, true
		}
		var s *UpstreamStreamedError
		if errors.As(err, &s) {
			return s.StatusCode, true
		}
		return 0, false
	},
}
