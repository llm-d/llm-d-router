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

package main

import (
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

func TestValidatePipeline(t *testing.T) {
	render := config.StepConfig{Type: steps.RenderStepName}
	decode := config.StepConfig{Type: steps.DecodeStepName}

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
