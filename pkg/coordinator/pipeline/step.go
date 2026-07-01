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

	"github.com/llm-d/coordinator/pkg/gateway"
)

// Step is the fundamental unit of work in the coordinator pipeline.
type Step interface {
	Name() string
	Execute(ctx context.Context, reqCtx *RequestContext) error
}

// StepFactory creates a Step from the gateway client and YAML configuration
// parameters. Steps that issue upstream requests store the client at
// construction; steps that do not need it ignore the argument.
type StepFactory func(gwClient *gateway.Client, params map[string]any) (Step, error)
