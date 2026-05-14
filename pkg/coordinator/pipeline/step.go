package pipeline

import "context"

// Step is the fundamental unit of work in the coordinator pipeline.
type Step interface {
	Name() string
	Execute(ctx context.Context, reqCtx *RequestContext) error
}

// StepFactory creates a Step from YAML configuration parameters.
type StepFactory func(params map[string]any) (Step, error)
