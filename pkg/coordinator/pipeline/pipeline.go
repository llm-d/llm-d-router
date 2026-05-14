package pipeline

import (
	"context"
	"fmt"

	"github.com/llm-d/coordinator/pkg/logging"
)

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
	logger := logging.FromContext(ctx)

	for _, step := range p.steps {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("pipeline cancelled: %w", err)
		}
		logger.V(logging.TRACE).Info("step starting", "step", step.Name())
		if err := step.Execute(ctx, reqCtx); err != nil {
			return fmt.Errorf("step %q failed: %w", step.Name(), err)
		}
		logger.V(logging.TRACE).Info("step complete", "step", step.Name())
	}
	return nil
}
