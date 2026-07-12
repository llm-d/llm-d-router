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

// Package builder assembles a coordinator pipeline from configuration. It lives
// apart from pipeline and steps because it depends on both: steps imports
// pipeline, so this glue cannot live in pipeline without an import cycle.
package builder

import (
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

// validatePipeline rejects configurations that cannot work before any step runs.
// The tokens-in format (use_openai_format=false) sends token IDs that only the
// render step produces, so it requires a render step in the pipeline.
func validatePipeline(p config.PipelineConfig) error {
	if p.UseOpenAIFormat {
		return nil
	}
	for _, s := range p.Steps {
		if s.Type == steps.RenderStepName {
			return nil
		}
	}
	return fmt.Errorf("pipeline.use_openai_format=false requires a %q step (the tokens-in format sends token IDs that render produces)", steps.RenderStepName)
}

func mergeConnectorDefaults(params map[string]any, kvConnector, ecConnector string) map[string]any {
	out := make(map[string]any, len(params))
	for k, v := range params {
		out[k] = v
	}
	if _, ok := out[steps.ParamKVConnector]; !ok && kvConnector != "" {
		out[steps.ParamKVConnector] = kvConnector
	}
	if _, ok := out[steps.ParamECConnector]; !ok && ecConnector != "" {
		out[steps.ParamECConnector] = ecConnector
	}
	return out
}

// Build validates cfg.Pipeline and constructs its steps in order.
func Build(cfg *config.Config, gwClient *gateway.Client) ([]pipeline.Step, error) {
	if err := validatePipeline(cfg.Pipeline); err != nil {
		return nil, err
	}

	var steps []pipeline.Step
	for _, stepCfg := range cfg.Pipeline.Steps {
		params := mergeConnectorDefaults(stepCfg.Params, cfg.Pipeline.KVConnector, cfg.Pipeline.ECConnector)
		if _, ok := params["use_openai_format"]; !ok {
			params["use_openai_format"] = cfg.Pipeline.UseOpenAIFormat
		}
		step, err := pipeline.Build(stepCfg.Type, gwClient, params)
		if err != nil {
			return nil, err
		}

		steps = append(steps, step)
	}
	return steps, nil
}
