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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps/asyncbroker"
)

func TestValidatePipeline(t *testing.T) {
	render := config.StepConfig{Type: steps.RenderStepName}
	decode := config.StepConfig{Type: steps.DecodeStepName}
	broker := config.StepConfig{Type: asyncbroker.StepName}
	forward := map[string]any{steps.ParamForwardPromptTokenIDs: true}
	forwardDecode := config.StepConfig{Type: steps.DecodeStepName, Params: forward}
	forwardCondDecode := config.StepConfig{Type: steps.ConditionalDecodeStepName, Params: forward}
	forwardPrefill := config.StepConfig{Type: steps.PrefillStepName, Params: forward}

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
			name:    "forward_prompt_token_ids with render",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{render, forwardCondDecode, forwardPrefill, forwardDecode}},
			wantErr: false,
		},
		{
			name:    "forward_prompt_token_ids on decode without render is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{forwardDecode}},
			wantErr: true,
		},
		{
			name:    "forward_prompt_token_ids on conditional-decode without render is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{forwardCondDecode, decode}},
			wantErr: true,
		},
		{
			name:    "forward_prompt_token_ids_on_prefill_without_render_is_rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{forwardPrefill, decode}},
			wantErr: true,
		},
		{
			name: "forward_prompt_token_ids false without render",
			cfg: config.PipelineConfig{UseOpenAIFormat: true, Steps: []config.StepConfig{
				{Type: steps.DecodeStepName, Params: map[string]any{steps.ParamForwardPromptTokenIDs: false}},
			}},
			wantErr: false,
		},
		{
			name:    "grpc worker protocol still requires render when use_openai_format is false",
			cfg:     config.PipelineConfig{WorkerProtocol: config.WorkerProtocolGRPC, UseOpenAIFormat: false, Steps: []config.StepConfig{decode}},
			wantErr: true,
		},
		{
			name:    "grpc worker protocol with use_openai_format false and render is accepted",
			cfg:     config.PipelineConfig{WorkerProtocol: config.WorkerProtocolGRPC, UseOpenAIFormat: false, Steps: []config.StepConfig{render, decode}},
			wantErr: false,
		},
		{
			name:    "invalid worker_protocol is rejected",
			cfg:     config.PipelineConfig{WorkerProtocol: "ws", UseOpenAIFormat: true, Steps: []config.StepConfig{decode}},
			wantErr: true,
		},
		{
			name:    "max_step_reruns is accepted",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, MaxStepReruns: 2, Steps: []config.StepConfig{decode}},
			wantErr: false,
		},
		{
			name:    "negative max_step_reruns is rejected",
			cfg:     config.PipelineConfig{UseOpenAIFormat: true, MaxStepReruns: -1, Steps: []config.StepConfig{decode}},
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

// loadYAML writes body to a coordinator.yaml in a temp dir and loads it.
func loadYAML(t *testing.T, body string) *config.Config {
	t.Helper()
	path := filepath.Join(t.TempDir(), "coordinator.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing fixture: %v", err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	return cfg
}

// TestBuildForceStreamFromPipelineConfig loads the pipeline-level force-stream
// settings and builds a pipeline whose steps share them; a size that fails
// validation fails Build, with force_stream on or off.
func TestBuildForceStreamFromPipelineConfig(t *testing.T) {
	const pipelineYAML = "pipeline:\n" +
		"  force_stream: %t\n" +
		"  force_stream_buffer_size: %s\n" +
		"  steps:\n" +
		"    - type: " + steps.ConditionalDecodeStepName + "\n" +
		"    - type: " + steps.DecodeStepName + "\n"

	cfg := loadYAML(t, fmt.Sprintf(pipelineYAML, true, "512MiB"))
	if !cfg.Pipeline.ForceStream || cfg.Pipeline.ForceStreamBufferSize != "512MiB" {
		t.Fatalf("pipeline force-stream settings not loaded: %+v", cfg.Pipeline)
	}
	if _, err := Build(cfg, gateway.New(config.GatewayConfig{})); err != nil {
		t.Fatalf("Build() unexpected error: %v", err)
	}

	for _, forceStream := range []bool{true, false} {
		cfg = loadYAML(t, fmt.Sprintf(pipelineYAML, forceStream, "1KiB"))
		if _, err := Build(cfg, gateway.New(config.GatewayConfig{})); err == nil {
			t.Fatalf("Build() expected an error for a buffer below the reply-envelope overhead (force_stream: %t)", forceStream)
		}
	}
}

// TestBuildForceStreamsRequest runs one non-streaming request through the
// pipeline Build returns with pipeline.force_stream set: the step that serves it
// sends the request upstream as a stream, and the client gets one JSON reply.
func TestBuildForceStreamsRequest(t *testing.T) {
	for _, stepTypes := range [][]string{
		{steps.DecodeStepName},
		{steps.ConditionalDecodeStepName, steps.DecodeStepName},
	} {
		t.Run(strings.Join(stepTypes, ","), func(t *testing.T) {
			var (
				mu      sync.Mutex
				streams []any
			)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				_ = json.NewDecoder(r.Body).Decode(&body)
				mu.Lock()
				streams = append(streams, body["stream"])
				mu.Unlock()
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, `data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`+"\n\n"+
					"data: [DONE]\n\n")
			}))
			defer server.Close()

			body := "pipeline:\n  force_stream: true\n  steps:\n    - type: " + strings.Join(stepTypes, "\n    - type: ") + "\n"
			p, err := Build(loadYAML(t, body), gateway.New(config.GatewayConfig{Address: server.URL}))
			if err != nil {
				t.Fatalf("Build() error = %v", err)
			}

			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-built",
				OriginalPath:     reqcommon.PathChatCompletions,
				Model:            "llama-3",
				KVTransferParams: map[string]any{},
				Body: map[string]any{
					"model": "llama-3", "stream": false, "max_tokens": 16,
					"messages": []any{map[string]any{"role": "user", "content": "hi"}},
				},
				ResponseWriter: recorder,
			}
			if err := p.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			mu.Lock()
			defer mu.Unlock()
			if len(streams) != 1 || streams[0] != true {
				t.Fatalf("upstream stream fields = %v, want one request with stream: true", streams)
			}
			if got := recorder.Header().Get("Content-Type"); got != reqcommon.ContentTypeJSON {
				t.Fatalf("Content-Type = %q, want %q", got, reqcommon.ContentTypeJSON)
			}
			if !strings.Contains(recorder.Body.String(), `"content":"ok"`) {
				t.Fatalf("reply %q does not carry the reassembled content", recorder.Body.String())
			}
		})
	}
}

// TestBuildRejectsStepLevelForceStream covers a coordinator.yaml that sets
// force_stream under a step's params. Build must fail rather than build a
// pipeline that does not force-stream.
func TestBuildRejectsStepLevelForceStream(t *testing.T) {
	for _, stepType := range []string{steps.DecodeStepName, steps.ConditionalDecodeStepName} {
		t.Run(stepType, func(t *testing.T) {
			cfg := loadYAML(t, "pipeline:\n"+
				"  steps:\n"+
				"    - type: "+stepType+"\n"+
				"      params:\n"+
				"        force_stream: true\n")
			if _, err := Build(cfg, gateway.New(config.GatewayConfig{})); err == nil {
				t.Fatalf("Build() expected an error for %q setting force_stream", stepType)
			}
		})
	}
}

func TestBuildValidatesWorkerProtocolAndRenderRequirement(t *testing.T) {
	noRender := &config.Config{Pipeline: config.PipelineConfig{
		WorkerProtocol:  config.WorkerProtocolGRPC,
		UseOpenAIFormat: false,
		Steps:           []config.StepConfig{{Type: steps.DecodeStepName}},
	}}
	if _, err := Build(noRender, gateway.New(config.GatewayConfig{})); err == nil {
		t.Fatal("Build() expected an error when worker_protocol=grpc, use_openai_format=false, and render step is absent")
	}

	withRender := &config.Config{Pipeline: config.PipelineConfig{
		WorkerProtocol:  config.WorkerProtocolGRPC,
		UseOpenAIFormat: false,
		Steps: []config.StepConfig{
			{Type: steps.RenderStepName, Params: map[string]any{"address": "http://127.0.0.1:8000"}},
			{Type: steps.DecodeStepName},
		},
	}}
	if _, err := Build(withRender, gateway.New(config.GatewayConfig{})); err != nil {
		t.Fatalf("Build() unexpected error with render step: %v", err)
	}
}

// rerunOnceStep asks for a re-run on its first attempt, then succeeds.
type rerunOnceStep struct{ attempts int }

func (*rerunOnceStep) Name() string { return "rerun-once" }

func (s *rerunOnceStep) Execute(context.Context, *pipeline.RequestContext) error {
	s.attempts++
	if s.attempts > 1 {
		return nil
	}
	return &pipeline.StepRerunError{Step: s.Name(), Cause: errors.New("unavailable")}
}

func TestBuildAppliesMaxStepReruns(t *testing.T) {
	const stepType = "test-rerun-once"
	pipeline.Register(stepType, func(*gateway.Client, map[string]any) (pipeline.Step, error) {
		return &rerunOnceStep{}, nil
	})

	tests := []struct {
		name      string
		maxReruns int
		wantErr   bool
	}{
		{name: "re-runs are off by default", maxReruns: 0, wantErr: true},
		{name: "max_step_reruns reaches the pipeline", maxReruns: 1, wantErr: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{Pipeline: config.PipelineConfig{
				UseOpenAIFormat: true,
				MaxStepReruns:   tt.maxReruns,
				Steps:           []config.StepConfig{{Type: stepType}},
			}}
			p, err := Build(cfg, nil)
			if err != nil {
				t.Fatalf("Build() unexpected error: %v", err)
			}
			err = p.Execute(context.Background(), &pipeline.RequestContext{})
			if (err != nil) != tt.wantErr {
				t.Fatalf("Execute() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
