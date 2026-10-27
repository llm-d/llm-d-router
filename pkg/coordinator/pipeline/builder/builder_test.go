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

package builder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps/asyncbroker"
)

func TestValidatePipeline(t *testing.T) {
	render := config.StepConfig{Type: steps.RenderStepName}
	decode := config.StepConfig{Type: steps.DecodeStepName}
	broker := config.StepConfig{Type: asyncbroker.StepName}

	tests := []struct {
		name    string
		cfg     config.PipelineConfig
		wantErr bool
	}{
		{
			name:    "openai format needs no render",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{decode}},
			wantErr: false,
		},
		{
			name:    "tokens-in with render",
			cfg:     config.PipelineConfig{UseOpenAIFormat: false, Steps: []config.StepConfig{render, decode}},
			wantErr: false,
		},
		{
			name:    "tokens-in without render is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: false, Steps: []config.StepConfig{decode}},
			wantErr: true,
		},
		{
			name:    "async-broker first is accepted",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{broker, decode}},
			wantErr: false,
		},
		{
			name:    "async-broker after another step is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{decode, broker}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePipeline(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validatePipeline() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestMergePipelineDefaultsDoesNotInjectResponseHeadersIntoSteps(t *testing.T) {
	params := mergePipelineDefaults(nil, config.PipelineConfig{
		ForwardResponseHeaders: []string{"x-llm-d-disagg-revision"},
	})
	if _, found := params["forward_response_headers"]; found {
		t.Fatalf("pipeline response headers leaked into step parameters: %v", params)
	}
}

func TestBuildDoesNotInjectUseOpenAIFormatIntoDecodeOrConditionalDecode(t *testing.T) {
	cfg := &config.Config{Pipeline: config.PipelineConfig{
		UseOpenAIFormat: true,
		Steps: []config.StepConfig{
			{Type: steps.ConditionalDecodeStepName},
			{Type: steps.DecodeStepName},
		},
	}}
	if _, err := Build(cfg, gateway.New(config.GatewayConfig{})); err != nil {
		t.Fatalf("Build() unexpected error: %v", err)
	}
}

// TestBuildRejectsUseOpenAIFormatOverrideInCoordinatorYAML reproduces a
// coordinator.yaml carrying the deprecated per-step override this PR removed:
// decode or conditional-decode setting use_openai_format under their own
// params. Build must fail config loading rather than build a pipeline that
// silently ignores the setting.
func TestBuildRejectsUseOpenAIFormatOverrideInCoordinatorYAML(t *testing.T) {
	for _, stepType := range []string{steps.DecodeStepName, steps.ConditionalDecodeStepName} {
		t.Run(stepType, func(t *testing.T) {
			body := "pipeline:\n" +
				"  use_openai_format: true\n" +
				"  steps:\n" +
				"    - type: " + stepType + "\n" +
				"      params:\n" +
				"        use_openai_format: false\n"
			path := filepath.Join(t.TempDir(), "coordinator.yaml")
			if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
				t.Fatalf("writing fixture: %v", err)
			}

			cfg, err := config.Load(path)
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			if _, err := Build(cfg, gateway.New(config.GatewayConfig{})); err == nil {
				t.Fatalf("Build() expected an error for %q overriding use_openai_format", stepType)
			}
		})
	}
}

func TestBuildRejectsInvalidForwardResponseHeaders(t *testing.T) {
	cfg := &config.Config{Pipeline: config.PipelineConfig{
		UseOpenAIFormat:        true,
		ForwardResponseHeaders: []string{"content-type"},
	}}
	if _, err := Build(cfg, nil); err == nil {
		t.Fatal("Build() expected an error for a non-forwardable response header")
	}
}
