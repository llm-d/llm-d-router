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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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
// Body holds the upstream response for server-side logging only; it is not
// sent to the client.
type UpstreamError struct {
	Step       string
	StatusCode int
	Body       string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("%s: upstream returned HTTP %d: %s", e.Step, e.StatusCode, e.Body)
}

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

	for _, step := range p.steps {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("pipeline cancelled: %w", err)
		}
		logger.V(logutil.TRACE).Info("step starting", "step", step.Name())
		if err := step.Execute(ctx, reqCtx); err != nil {
			if errors.Is(err, ErrPipelineDone) {
				return nil
			}
			return fmt.Errorf("step %q failed: %w", step.Name(), err)
		}
		logger.V(logutil.TRACE).Info("step complete", "step", step.Name())
	}
	return nil
}
