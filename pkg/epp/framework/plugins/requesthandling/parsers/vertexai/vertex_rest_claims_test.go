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
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

// matchClaim mirrors pkg/epp/handlers.matchClaim so the parser's own claims can be
// checked against realistic Vertex paths without importing the handlers package.
func matchClaim(path, claim string) bool {
	claim = strings.Trim(strings.TrimSpace(claim), "/")
	if segment, ok := strings.CutSuffix(claim, "/*"); ok {
		return segment != "" && strings.Contains(path, "/"+segment+"/")
	}
	return request.MatchPathSuffix(path, claim)
}

func claimed(path string) bool {
	for _, c := range (&VertexAIParser{}).Claims().Paths {
		if matchClaim(path, c) {
			return true
		}
	}
	return false
}

func TestVertexAIParserClaimsRealPaths(t *testing.T) {
	const base = "/v1/projects/tianjiaoliu-test/locations/us-central1/endpoints/1343561702578323456"

	tests := []struct {
		name string
		path string
		want bool
	}{
		// REST forms a dedicated CloudESF endpoint sends.
		{"rest predict", base + ":predict", true},
		{"rest rawPredict", base + ":rawPredict", true},
		{"rest streamRawPredict", base + ":streamRawPredict", true},
		{"rest chat completions", base + "/chat/completions", true},
		{"rest invoke with container path", base + "/invoke/generate", true},
		{"rest invoke nested container path", base + "/invoke/v2/models/x:infer", true},

		// The existing gRPC service methods must keep working.
		{"grpc RawPredict", "/google.cloud.aiplatform.v1.PredictionService/RawPredict", true},
		{"grpc StreamRawPredict", "/google.cloud.aiplatform.v1.PredictionService/StreamRawPredict", true},
		{"grpc ChatCompletions", "/google.cloud.aiplatform.v1.PredictionService/ChatCompletions", true},

		// Must not over-claim.
		{"unrelated path", "/v1/models/foo:somethingElse", false},
		{"bare invoke without container path", base + ":invoke", false},
		{"health", "/healthz", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := claimed(tc.path); got != tc.want {
				t.Errorf("claimed(%q) = %v, want %v", tc.path, got, tc.want)
			}
		})
	}
}

// TestSuffixesDoNotCrossMatch guards the subtle part: ":rawPredict" must not be
// satisfied by ":streamRawPredict", or streaming traffic would be parsed as unary.
func TestSuffixesDoNotCrossMatch(t *testing.T) {
	const p = "/v1/projects/p/locations/l/endpoints/e:streamRawPredict"

	if request.MatchPathSuffix(p, rawPredictPathSuffix) {
		t.Error(":streamRawPredict matched the :rawPredict claim, want distinct claims")
	}
	if request.MatchPathSuffix(p, predictPathSuffix) {
		t.Error(":streamRawPredict matched the :predict claim, want distinct claims")
	}
	if !request.MatchPathSuffix(p, streamRawPredictPathSuffix) {
		t.Error(":streamRawPredict did not match its own claim")
	}
}

func TestResolveOpenAIPath(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{"messages routes to chat completions", `{"messages":[{"role":"user","content":"hi"}]}`, openAIChatCompletionsPath},
		{"prompt routes to completions", `{"prompt":"hello"}`, openAICompletionsPath},
		{"input routes to responses", `{"input":"hello"}`, openAIResponsesPath},
		// A classic Vertex predict body is not OpenAI-shaped; it must fall through
		// to passthrough rather than being forced onto /responses.
		{"tensorflow instances payload", `{"instances":[[1,2,3]]}`, ""},
		{"not json", `not json at all`, ""},
		{"empty", ``, ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveOpenAIPath([]byte(tc.payload)); got != tc.want {
				t.Errorf("resolveOpenAIPath(%q) = %q, want %q", tc.payload, got, tc.want)
			}
		})
	}
}
