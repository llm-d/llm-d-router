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

package vertexai

import (
	"bytes"
	"context"
	"strings"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	restBase = "/v1/projects/my-project/locations/us-central1/endpoints/1234567890"

	chatBody       = `{"model":"m","messages":[{"role":"user","content":"Hello"}],"stream":true}`
	completionBody = `{"model":"m","prompt":"Hello","max_tokens":8}`
	tfBody         = `{"instances":[[1,2,3]]}`
	opaqueBody     = `{"image_bytes":"aGVsbG8="}`
)

// requestKind names the OpenAI request a parse produced, or "raw" when the
// request was forwarded without being interpreted.
func requestKind(body *fwkrh.InferenceRequestBody) string {
	switch {
	case body.ChatCompletions != nil:
		return "chat"
	case body.Completions != nil:
		return "completions"
	case body.Responses != nil:
		return "responses"
	default:
		return "raw"
	}
}

// TestParseRequestREST covers the REST APIs a Vertex AI dedicated endpoint
// forwards: predict, rawPredict, streamRawPredict, chat completions and invoke.
func TestParseRequestREST(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		body     string
		wantKind string
	}{
		{"chat completions", restBase + "/chat/completions", chatBody, "chat"},
		{"chat completions with query", restBase + "/chat/completions?alt=sse", chatBody, "chat"},
		{"chat completions v1beta1", "/v1beta1/projects/p/locations/l/endpoints/e/chat/completions", chatBody, "chat"},
		{"chat completions bad payload", restBase + "/chat/completions", `{"messages":"not a list"}`, "raw"},
		{"rawPredict chat payload", restBase + ":rawPredict", chatBody, "chat"},
		{"rawPredict tensorflow payload", restBase + ":rawPredict", tfBody, "raw"},
		{"rawPredict non-json payload", restBase + ":rawPredict", "\x89PNG", "raw"},
		{"streamRawPredict completions payload", restBase + ":streamRawPredict", completionBody, "completions"},
		{"streamRawPredict with alt=sse", restBase + ":streamRawPredict?alt=sse", chatBody, "chat"},
		{"predict tensorflow payload", restBase + ":predict", tfBody, "raw"},
		{"predict project-number path", "/v1/projects/123456/locations/l/endpoints/e:predict", tfBody, "raw"},
		{"ui predict", "/ui/projects/p/locations/l/endpoints/e:predict", tfBody, "raw"},
		{"invoke chat payload", restBase + "/invoke/v1/chat/completions", chatBody, "chat"},
		{"invoke opaque payload", restBase + "/invoke/predict", opaqueBody, "raw"},
		{"invoke deployed model", restBase + "/deployedModels/42/invoke/generate", completionBody, "completions"},
	}

	parser := NewVertexAIParser()
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if clean, _, _ := strings.Cut(tc.path, "?"); !claimed(clean) {
				t.Fatalf("path %q is not claimed by the parser", tc.path)
			}
			res, err := parser.ParseRequest(context.Background(), []byte(tc.body), map[string]string{":path": tc.path})
			if err != nil {
				t.Fatalf("ParseRequest: %v", err)
			}
			if res == nil || res.Body == nil {
				t.Fatal("ParseRequest returned no body")
			}
			if got := requestKind(res.Body); got != tc.wantKind {
				t.Errorf("request kind = %q, want %q", got, tc.wantKind)
			}
			raw, isRaw := res.Body.Payload.(fwkrh.RawPayload)
			if tc.wantKind == "raw" {
				// An unparsed request is forwarded byte for byte, and its response is not processed.
				if !isRaw || !bytes.Equal(raw, []byte(tc.body)) || !res.SkipResponseProcessing {
					t.Errorf("ParseRequest = %+v, want the body passed through unchanged", res)
				}
			} else if res.SkipResponseProcessing {
				t.Error("SkipResponseProcessing = true, want response processing for a parsed request")
			}
		})
	}
}

// TestParseResponseREST covers responses to dedicated-endpoint REST traffic, which
// are the model server's JSON or SSE as is, with no gRPC framing.
func TestParseResponseREST(t *testing.T) {
	const usageJSON = `{"object":"chat.completion","usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`
	wantUsage := fwkrh.Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30}

	tests := []struct {
		name    string
		body    string
		headers map[string]string
	}{
		{"json", usageJSON, map[string]string{"content-type": "application/json"}},
		{"json without content type", usageJSON, nil},
		{"sse", "data: " + usageJSON + "\n\ndata: [DONE]\n\n", map[string]string{"Content-Type": "text/event-stream; charset=utf-8"}},
	}

	parser := NewVertexAIParser()
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := parser.ParseResponse(context.Background(), []byte(tc.body), tc.headers, true)
			if err != nil {
				t.Fatalf("ParseResponse: %v", err)
			}
			if resp == nil || resp.Usage == nil {
				t.Fatalf("ParseResponse = %+v, want usage", resp)
			}
			if *resp.Usage != wantUsage {
				t.Errorf("usage = %+v, want %+v", *resp.Usage, wantUsage)
			}
		})
	}
}

func TestIsGRPCResponse(t *testing.T) {
	frame := []byte{0, 0, 0, 0, 2, '{', '}'}
	tests := []struct {
		name    string
		headers map[string]string
		body    []byte
		want    bool
	}{
		{"grpc content type", map[string]string{"content-type": "application/grpc"}, frame, true},
		{"grpc+proto content type", map[string]string{"Content-Type": "application/grpc+proto"}, frame, true},
		{"json content type", map[string]string{"content-type": "application/json"}, []byte(`{}`), false},
		{"no content type, frame", nil, frame, true},
		{"no content type, json", nil, []byte(` {"a":1}`), false},
		{"no content type, sse", nil, []byte("data:{}\n\n"), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isGRPCResponse(tc.headers, tc.body); got != tc.want {
				t.Errorf("isGRPCResponse() = %v, want %v", got, tc.want)
			}
		})
	}
}
