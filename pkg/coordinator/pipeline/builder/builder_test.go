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
	"encoding/json"
	"io"
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

// TestBuild_KVConnectorParams loads a coordinator YAML, builds the pipeline and
// runs a request through it, checking the bootstrap_port the prefill pod receives.
func TestBuild_KVConnectorParams(t *testing.T) {
	const prefillDecode = `
  steps:
    - type: prefill
    - type: decode
`
	tests := []struct {
		name     string
		yaml     string
		env      string
		wantPort int // 0: no bootstrap_port on the prefill request
		wantErr  string
	}{
		{
			name:     "pipeline params",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_port: 9100\n" + prefillDecode,
			wantPort: 9100,
		},
		{
			name: "prefill step params",
			yaml: `pipeline:
  kv_connector: kv-sglang
  steps:
    - type: prefill
      params:
        kv_connector_params:
          bootstrap_port: 9200
    - type: decode
`,
			wantPort: 9200,
		},
		{
			name: "step params replace pipeline params",
			yaml: `pipeline:
  kv_connector: kv-sglang
  kv_connector_params:
    bootstrap_port: 9100
  steps:
    - type: prefill
      params:
        kv_connector_params:
          bootstrap_port: 9200
    - type: decode
`,
			wantPort: 9200,
		},
		{
			name:     "param takes precedence over env",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_port: 9100\n" + prefillDecode,
			env:      "9300",
			wantPort: 9100,
		},
		{
			name:     "env fallback without params",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n" + prefillDecode,
			env:      "9300",
			wantPort: 9300,
		},
		{
			name:     "default without params or env",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n" + prefillDecode,
			wantPort: 8998,
		},
		{
			name:     "invalid env falls back to default",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n" + prefillDecode,
			env:      "abc",
			wantPort: 8998,
		},
		{
			name: "step repeating the pipeline connector inherits pipeline params",
			yaml: `pipeline:
  kv_connector: kv-sglang
  kv_connector_params:
    bootstrap_port: 9100
  steps:
    - type: prefill
      params:
        kv_connector: kv-sglang
    - type: decode
      params:
        kv_connector: kv-sglang
`,
			wantPort: 9100,
		},
		{
			name: "step with a different connector does not inherit pipeline params",
			yaml: `pipeline:
  kv_connector: kv-sglang
  kv_connector_params:
    bootstrap_port: 9100
  steps:
    - type: prefill
      params:
        kv_connector: kv-nixl
    - type: decode
      params:
        kv_connector: kv-nixl
`,
			wantPort: 0,
		},
		{
			name:    "port zero rejected",
			yaml:    "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_port: 0\n" + prefillDecode,
			wantErr: "kv_connector_params",
		},
		{
			name:     "max port accepted",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_port: 65535\n" + prefillDecode,
			wantPort: 65535,
		},
		{
			name:     "null kv_connector_params is the same as absent",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n" + prefillDecode,
			env:      "9300",
			wantPort: 9300,
		},
		{
			name:     "empty map kv_connector_params is the same as absent",
			yaml:     "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params: {}\n" + prefillDecode,
			env:      "9300",
			wantPort: 9300,
		},
		{
			name:    "negative port rejected",
			yaml:    "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_port: -1\n" + prefillDecode,
			wantErr: "kv_connector_params",
		},
		{
			name:    "port above range rejected",
			yaml:    "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_port: 65536\n" + prefillDecode,
			wantErr: "kv_connector_params",
		},
		{
			name:    "non-numeric port rejected",
			yaml:    "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_port: abc\n" + prefillDecode,
			wantErr: "kv_connector_params",
		},
		{
			name:    "unknown key rejected",
			yaml:    "pipeline:\n  kv_connector: kv-sglang\n  kv_connector_params:\n    bootstrap_prot: 9100\n" + prefillDecode,
			wantErr: "bootstrap_prot",
		},
		{
			name:    "params on a connector that takes none rejected",
			yaml:    "pipeline:\n  kv_connector: kv-nixl\n  kv_connector_params:\n    bootstrap_port: 9100\n" + prefillDecode,
			wantErr: "kv_connector_params",
		},
		{
			name: "params that are not a map rejected",
			yaml: `pipeline:
  kv_connector: kv-sglang
  steps:
    - type: prefill
      params:
        kv_connector_params: "9100"
    - type: decode
`,
			wantErr: "kv_connector_params",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("SGLANG_BOOTSTRAP_PORT", tt.env)

			var mu sync.Mutex
			var prefillKV map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
				case gateway.PhasePrefill:
					body, _ := io.ReadAll(r.Body)
					var parsed map[string]any
					_ = json.Unmarshal(body, &parsed)
					mu.Lock()
					prefillKV, _ = parsed[reqcommon.FieldKVTransferParams].(map[string]any)
					mu.Unlock()
					_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": map[string]any{}})
				default:
					_ = json.NewEncoder(w).Encode(map[string]any{
						"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
					})
				}
			}))
			defer srv.Close()

			path := filepath.Join(t.TempDir(), "coordinator.yaml")
			if err := os.WriteFile(path, []byte(tt.yaml), 0o600); err != nil {
				t.Fatal(err)
			}
			cfg, err := config.Load(path)
			if err != nil {
				t.Fatalf("config.Load: %v", err)
			}

			p, err := Build(cfg, gateway.New(config.GatewayConfig{Address: srv.URL}))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("Build() error = %v, want it to contain %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}

			reqCtx := &pipeline.RequestContext{
				RequestID:        "req",
				OriginalPath:     reqcommon.PathChatCompletions,
				Model:            "m",
				Body:             map[string]any{"model": "m", "messages": []any{map[string]any{"role": "user", "content": "hi"}}},
				KVTransferParams: map[string]any{},
				ResponseWriter:   httptest.NewRecorder(),
			}
			if err := p.Execute(t.Context(), reqCtx); err != nil {
				t.Fatalf("Execute: %v", err)
			}

			mu.Lock()
			defer mu.Unlock()
			if prefillKV == nil {
				t.Fatal("prefill request carried no kv_transfer_params")
			}
			got, present := prefillKV["bootstrap_port"]
			switch {
			case tt.wantPort == 0 && present:
				t.Errorf("bootstrap_port = %v, want it absent", got)
			case tt.wantPort != 0 && got != float64(tt.wantPort):
				t.Errorf("bootstrap_port = %v, want %d", got, tt.wantPort)
			}
		})
	}
}
