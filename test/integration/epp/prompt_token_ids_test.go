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

package epp

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/test/integration"
)

// renderRecorder captures the bodies of the render requests a token-producer
// sends, so a test can assert which requests reached the renderer. The
// token-producer's warmup probes hit the same server; their bodies carry
// "warmup", so counts are taken by request marker rather than in total.
type renderRecorder struct {
	mutex  sync.Mutex
	bodies []string
}

func (r *renderRecorder) handle(w http.ResponseWriter, req *http.Request) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	r.mutex.Lock()
	r.bodies = append(r.bodies, string(body))
	r.mutex.Unlock()
	// The chat and messages render response shape; completions is not exercised here.
	_, _ = w.Write([]byte(`{"token_ids":[5,6,7]}`))
}

func (r *renderRecorder) callsContaining(marker string) int {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	n := 0
	for _, body := range r.bodies {
		if strings.Contains(body, marker) {
			n++
		}
	}
	return n
}

// TestClientPromptTokenIDsSkipRender drives a chat-completions request through
// the ext-proc stream with a token-producer backed by a recording render
// server. A request carrying kv_transfer_params.prompt_token_ids must route
// without any render call; a request without it must render once.
func TestClientPromptTokenIDsSkipRender(t *testing.T) {
	recorder := &renderRecorder{}
	renderSrv := httptest.NewServer(http.HandlerFunc(recorder.handle))
	t.Cleanup(renderSrv.Close)

	configText := fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: token-producer
    parameters:
      modelName: %s
      vllm:
        url: %s
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: openai-parser
  - type: mock-metrics-source
  - type: core-metrics-extractor
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: kv-cache-utilization-scorer
requestHandler:
  propagatePriority: true
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
`, modelMyModelTarget, renderSrv.URL)

	cases := []struct {
		name        string
		marker      string
		body        string
		wantBody    map[string]any
		wantRenders int
	}{
		{
			name:   "prompt_token_ids skips rendering",
			marker: "ids-marker",
			body: `{"model":"` + modelMyModel + `","messages":[{"role":"user","content":"ids-marker"}],` +
				`"max_tokens":100,"kv_transfer_params":{"prompt_token_ids":[7,8,9]}}`,
			wantBody: map[string]any{
				"kv_transfer_params": json.RawMessage(`{"prompt_token_ids":[7,8,9]}`),
				"max_tokens":         json.Number("100"),
				"messages":           json.RawMessage(`[{"role":"user","content":"ids-marker"}]`),
				"model":              modelMyModelTarget,
				"priority":           2,
			},
			wantRenders: 0,
		},
		{
			name:   "no prompt_token_ids renders once",
			marker: "no-ids-marker",
			body:   `{"model":"` + modelMyModel + `","messages":[{"role":"user","content":"no-ids-marker"}],"max_tokens":100}`,
			wantBody: map[string]any{
				"max_tokens": json.Number("100"),
				"messages":   json.RawMessage(`[{"role":"user","content":"no-ids-marker"}]`),
				"model":      modelMyModelTarget,
				"priority":   2,
			},
			wantRenders: 1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := NewTestHarness(t.Context(), t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
			pods := []PodState{
				P(0, 0, 0.1, modelMyModelTarget), // Winner: lowest queue and KV utilization
				P(1, 5, 0.5, modelMyModelTarget),
			}
			h.WithPods(pods).WaitForSync(len(pods), modelMyModel)

			requests := integration.ReqRaw(map[string]string{
				":path":                      "/v1/chat/completions",
				"hi":                         "mom",
				reqcommon.RequestIDHeaderKey: "test-request-id",
				metadata.ObjectiveKey:        modelMyModel,
				metadata.ModelNameRewriteKey: modelMyModelTarget,
			}, tc.body)

			responses, err := integration.StreamedRequest(t, h.Client, requests, 2)
			require.NoError(t, err)

			// The repackage rewrites the model to the target, injects the
			// resolved priority, and re-marshals the payload map with sorted
			// keys; nested objects keep their original bytes.
			expectedBody, err := json.Marshal(tc.wantBody)
			require.NoError(t, err)
			wantResponses := integration.NewRequestBufferedResponse("192.168.1.1:8000", expectedBody,
				&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: ":path", RawValue: []byte("/v1/chat/completions")}},
				&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
				&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
					Key: reqcommon.RequestIDHeaderKey, RawValue: []byte("test-request-id"),
				}},
			)
			if diff := cmp.Diff(wantResponses, responses,
				protocmp.Transform(),
				protocmp.SortRepeated(func(a, b *envoyCorev3.HeaderValueOption) bool {
					return a.GetHeader().GetKey() < b.GetHeader().GetKey()
				}),
				protocmp.SortRepeated(func(a, b string) bool { return a < b }),
			); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}

			require.Equal(t, tc.wantRenders, recorder.callsContaining(tc.marker),
				"render calls carrying the request body")
		})
	}
}
