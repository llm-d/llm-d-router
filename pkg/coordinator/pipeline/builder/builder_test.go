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
	"reflect"
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
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, DPSize: 1, Steps: []config.StepConfig{decode}},
			wantErr: false,
		},
		{
			name:    "tokens-in with render",
			cfg:     config.PipelineConfig{UseOpenAIFormat: false, DPSize: 1, Steps: []config.StepConfig{render, decode}},
			wantErr: false,
		},
		{
			name:    "tokens-in without render is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: false, DPSize: 1, Steps: []config.StepConfig{decode}},
			wantErr: true,
		},
		{
			name:    "non-positive dp_size is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, DPSize: 0, Steps: []config.StepConfig{decode}},
			wantErr: true,
		},
		{
			name:    "negative dp_size is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, DPSize: -1, Steps: []config.StepConfig{decode}},
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

func TestMergeConnectorDefaults_DPSize(t *testing.T) {
	tests := []struct {
		name   string
		params map[string]any
		dpSize int
		want   map[string]any
	}{
		{
			name:   "injected when absent",
			params: map[string]any{},
			dpSize: 4,
			want:   map[string]any{steps.ParamDPSize: 4},
		},
		{
			name:   "step override wins",
			params: map[string]any{steps.ParamDPSize: 2},
			dpSize: 4,
			want:   map[string]any{steps.ParamDPSize: 2},
		},
		{
			name:   "zero pipeline default is not injected",
			params: map[string]any{},
			dpSize: 0,
			want:   map[string]any{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeConnectorDefaults(tt.params, "", "", tt.dpSize)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("mergeConnectorDefaults() = %v, want %v", got, tt.want)
			}
		})
	}
}
