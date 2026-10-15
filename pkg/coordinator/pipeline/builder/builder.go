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

	"github.com/go-logr/logr"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps/asyncbroker"
)

// validatePipeline rejects configurations that cannot work before any step runs.
// The async-broker step must intercept requests before any processing step
// touches them (queued bodies are stored verbatim, and passthrough stamping
// must precede routing), so it is only valid in first position. The tokens-in
// format (use_openai_format=false) sends token IDs that only the render step
// produces, so it requires a render step in the pipeline.
func validatePipeline(p config.PipelineConfig) error {
	for i, s := range p.Steps {
		if s.Type == asyncbroker.StepName && i > 0 {
			return fmt.Errorf("the %q step must be the first pipeline step, found it at position %d", asyncbroker.StepName, i+1)
		}
	}
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

// usesOpenAIFormatParam reports whether stepType reads the use_openai_format
// parameter. Only encode and prefill resolve their body format from it; decode
// and conditional-decode derive it directly from the request's original path
// and reject the key instead of silently ignoring it.
func usesOpenAIFormatParam(stepType string) bool {
	return stepType == steps.EncodeStepName || stepType == steps.PrefillStepName
}

func mergePipelineDefaults(params map[string]any, cfg config.PipelineConfig) map[string]any {
	out := make(map[string]any, len(params))
	for k, v := range params {
		out[k] = v
	}
	if _, ok := out[steps.ParamKVConnector]; !ok && cfg.KVConnector != "" {
		out[steps.ParamKVConnector] = cfg.KVConnector
	}
	if _, ok := out[steps.ParamECConnector]; !ok && cfg.ECConnector != "" {
		out[steps.ParamECConnector] = cfg.ECConnector
	}
	return out
}

// kvConnectorName returns the KV connector a step's merged params select, with
// an unset name resolved to the default.
func kvConnectorName(params map[string]any) string {
	name, _ := params[steps.ParamKVConnector].(string)
	return kv.ResolveName(name)
}

// warnKVConnectorMismatch logs when prefill and decode resolve to different KV
// connectors. Each step builds its own connector, so the mismatch is valid
// config, but decode then does not ask for the KV prefill stored and
// recomputes the prompt, and no request fails to show it.
func warnKVConnectorMismatch(logger logr.Logger, prefill, decode string) {
	if prefill != "" && decode != "" && prefill != decode {
		logger.Info("warning: prefill and decode use different kv_connector values; decode does not read the KV prefill stores",
			"prefill", prefill, "decode", decode)
	}
}

// Build validates cfg.Pipeline and constructs its pipeline.
func Build(cfg *config.Config, gwClient *gateway.Client) (*pipeline.Pipeline, error) {
	if err := validatePipeline(cfg.Pipeline); err != nil {
		return nil, err
	}

	pipelineSteps := make([]pipeline.Step, 0, len(cfg.Pipeline.Steps))
	var prefillKV, decodeKV string
	for _, stepCfg := range cfg.Pipeline.Steps {
		params := mergePipelineDefaults(stepCfg.Params, cfg.Pipeline)
		if usesOpenAIFormatParam(stepCfg.Type) {
			if _, ok := params["use_openai_format"]; !ok {
				params["use_openai_format"] = cfg.Pipeline.UseOpenAIFormat
			}
		}
		step, err := pipeline.Build(stepCfg.Type, gwClient, params)
		if err != nil {
			return nil, err
		}

		switch stepCfg.Type {
		case steps.PrefillStepName:
			prefillKV = kvConnectorName(params)
		case steps.DecodeStepName:
			decodeKV = kvConnectorName(params)
		}
		pipelineSteps = append(pipelineSteps, step)
	}
	warnKVConnectorMismatch(ctrl.Log.WithName("builder"), prefillKV, decodeKV)
	return pipeline.NewWithForwardResponseHeaders(pipelineSteps, cfg.Pipeline.ForwardResponseHeaders)
}
