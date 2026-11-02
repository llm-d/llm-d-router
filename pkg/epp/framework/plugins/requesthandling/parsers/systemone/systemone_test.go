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

package systemone

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwkplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins"
)

var systemOneHeaders = map[string]string{":path": "/v1/systemone"}

// A request as the SDKs send it: a JSON state, questions in request order,
// and numbers the Go float64 round trip would alter.
const sampleRequest = `{"model":"Qwen/Qwen3-0.6B",` +
	`"state":{"ticket":"Charged twice","z":9007199254740993,"a":1e0},` +
	`"questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":null,"shipping":null}},` +
	`"urgent":{"type":"noul","instructions":"Is it urgent?"}},` +
	`"seed":7,"priority":3,"cache_salt":"tenant-a","chat_template_kwargs":{"enable_thinking":false}}`

func TestNewSystemOneParser(t *testing.T) {
	parser := NewSystemOneParser()
	want := fwkplugin.TypedName{Type: SystemOneParserType, Name: SystemOneParserType}
	if diff := cmp.Diff(want, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}
}

func TestSystemOneParserPluginFactory(t *testing.T) {
	plugin, err := SystemOneParserPluginFactory("named", nil, nil)
	if err != nil {
		t.Fatalf("SystemOneParserPluginFactory() error = %v", err)
	}
	if _, ok := plugin.(*SystemOneParser); !ok {
		t.Fatalf("SystemOneParserPluginFactory() = %T, want *SystemOneParser", plugin)
	}
	want := fwkplugin.TypedName{Type: SystemOneParserType, Name: "named"}
	if diff := cmp.Diff(want, plugin.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}
}

func TestSystemOneParser_Claims(t *testing.T) {
	got := NewSystemOneParser().Claims()
	want := fwkrh.Claims{
		Paths:     []string{"systemone"},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Claims() mismatch (-want +got):\n%s", diff)
	}
}

func TestSystemOneParser_ParseRequest(t *testing.T) {
	result, err := NewSystemOneParser().ParseRequest(context.Background(), []byte(sampleRequest), systemOneHeaders)
	if err != nil {
		t.Fatalf("ParseRequest() error = %v", err)
	}
	if result.SkipResponseProcessing || !result.JSONValidated {
		t.Errorf("SkipResponseProcessing = %v, JSONValidated = %v, want false, true", result.SkipResponseProcessing, result.JSONValidated)
	}
	body := result.Body
	if body.Model != "Qwen/Qwen3-0.6B" || !body.RenderRequest {
		t.Errorf("body = {Model:%q RenderRequest:%v}, want model set and render request", body.Model, body.RenderRequest)
	}
	if !bytes.Equal(body.RawBody, []byte(sampleRequest)) {
		t.Errorf("RawBody = %s, want the request bytes", body.RawBody)
	}
}

func TestSystemOneParser_ParseRequest_StateShapes(t *testing.T) {
	parser := NewSystemOneParser()
	for name, state := range map[string]string{
		"string": `"Charged twice"`,
		"array":  `[1,2,3]`,
	} {
		t.Run(name, func(t *testing.T) {
			raw := `{"model":"m","state":` + state + `,"questions":{"q":{"type":"noul"}}}`
			result, err := parser.ParseRequest(context.Background(), []byte(raw), systemOneHeaders)
			if err != nil {
				t.Fatalf("ParseRequest() error = %v", err)
			}
			got := result.Body.Payload.(fwkrh.PayloadMap)["state"]
			if diff := cmp.Diff(json.RawMessage(state), got); diff != "" {
				t.Errorf("state mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSystemOneParser_ParseRequest_NoModel(t *testing.T) {
	result, err := NewSystemOneParser().ParseRequest(context.Background(),
		[]byte(`{"state":"s","questions":{"q":{"type":"noul"}}}`), systemOneHeaders)
	if err != nil {
		t.Fatalf("ParseRequest() error = %v", err)
	}
	if result.Body.Model != "" {
		t.Errorf("Model = %q, want empty", result.Body.Model)
	}
}

func TestSystemOneParser_ParseRequest_Errors(t *testing.T) {
	parser := NewSystemOneParser()
	tests := []struct {
		name    string
		headers map[string]string
		body    string
		wantErr string
	}{
		{
			name:    "unsupported path",
			headers: map[string]string{":path": "/v1/decisions"},
			body:    `{"model":"m","state":"s","questions":{"q":{"type":"noul"}}}`,
			wantErr: "unsupported API endpoint: /v1/decisions",
		},
		{
			name:    "invalid json",
			headers: systemOneHeaders,
			body:    `{"model":`,
			wantErr: "error unmarshaling request body",
		},
		{
			name:    "missing questions",
			headers: systemOneHeaders,
			body:    `{"model":"m","state":"s"}`,
			wantErr: "questions must be a JSON object",
		},
		{
			name:    "questions as array",
			headers: systemOneHeaders,
			body:    `{"model":"m","state":"s","questions":[{"type":"noul"}]}`,
			wantErr: "questions must be a JSON object",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parser.ParseRequest(context.Background(), []byte(tt.body), tt.headers)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ParseRequest() error = %v, want it to contain %q", err, tt.wantErr)
			}
			if result != nil {
				t.Errorf("ParseRequest() result = %v, want nil", result)
			}
		})
	}
}

// The forwarded body keeps state, questions and extension fields byte for byte
// when the router rewrites the model.
func TestSystemOneParser_RewriteModelName_RoundTrip(t *testing.T) {
	parser := NewSystemOneParser()
	result, err := parser.ParseRequest(context.Background(), []byte(sampleRequest), systemOneHeaders)
	if err != nil {
		t.Fatalf("ParseRequest() error = %v", err)
	}
	rewritten, err := parser.RewriteModelName(result.Body.Payload.(fwkrh.MarshalablePayload), "served-model")
	if err != nil {
		t.Fatalf("RewriteModelName() error = %v", err)
	}
	out, err := rewritten.(fwkrh.Marshaler).Marshal()
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	var got, want map[string]json.RawMessage
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(sampleRequest), &want); err != nil {
		t.Fatal(err)
	}
	if string(got["model"]) != `"served-model"` {
		t.Errorf("model = %s, want \"served-model\"", got["model"])
	}
	delete(got, "model")
	delete(want, "model")
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("forwarded fields mismatch (-want +got):\n%s", diff)
	}
}

func TestSystemOneParser_RewritePriority(t *testing.T) {
	parser := NewSystemOneParser()
	vllm := fwkrh.PriorityRewriteContext{TargetEndpoint: &fwkdl.EndpointMetadata{
		Labels: map[string]string{fwkplugins.EngineTypeLabelKey: "vllm"}}}
	sglang := fwkrh.PriorityRewriteContext{TargetEndpoint: &fwkdl.EndpointMetadata{
		Labels: map[string]string{fwkplugins.EngineTypeLabelKey: "sglang"}}}

	tests := []struct {
		name string
		ctx  fwkrh.PriorityRewriteContext
		want int
	}{
		{name: "vllm negates the EPP priority", ctx: vllm, want: -5},
		{name: "sglang keeps the EPP priority", ctx: sglang, want: 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parser.ParseRequest(context.Background(), []byte(sampleRequest), systemOneHeaders)
			if err != nil {
				t.Fatalf("ParseRequest() error = %v", err)
			}
			payload, mutated, err := parser.RewritePriority(tt.ctx, result.Body.Payload.(fwkrh.MarshalablePayload), 5)
			if err != nil {
				t.Fatalf("RewritePriority() error = %v", err)
			}
			if !mutated {
				t.Errorf("RewritePriority() mutated = false, want true")
			}
			// The client-supplied priority is replaced.
			if got := payload.(fwkrh.PayloadMap)["priority"]; got != tt.want {
				t.Errorf("priority = %v, want %d", got, tt.want)
			}
		})
	}
}

func TestSystemOneParser_ParseResponse(t *testing.T) {
	parser := NewSystemOneParser()
	tests := []struct {
		name      string
		body      string
		wantUsage *fwkrh.Usage
		wantErr   bool
	}{
		{
			name: "decision response",
			body: `{"id":"decision-1","object":"structured_decision","model":"m",` +
				`"answers":{"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"shipping":0.1},"confidence":0.9}},` +
				`"usage":{"input_tokens":254,"output_tokens":2},` +
				`"diagnostics":{"team":{"label_mass":0.99,"argmax_is_label":true}}}`,
			wantUsage: &fwkrh.Usage{PromptTokens: 254, CompletionTokens: 2, TotalTokens: 256},
		},
		{
			name: "error response without usage",
			body: `{"error":{"message":"model does not support structured decisions","type":"NotImplementedError","code":501}}`,
		},
		{
			name: "empty body",
			body: "",
		},
		{
			name:    "invalid json",
			body:    `{"usage":`,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseResponse(context.Background(), []byte(tt.body), nil, true)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseResponse() error = %v, wantErr %v", err, tt.wantErr)
			}
			var want *fwkrh.ParsedResponse
			if tt.wantUsage != nil {
				want = &fwkrh.ParsedResponse{Usage: tt.wantUsage}
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("ParseResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
