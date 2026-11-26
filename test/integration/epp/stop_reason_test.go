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
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

const stopReasonRecorderType = "test-stop-reason-recorder"

var observedStopReasons chan fwkrh.StopReason

func init() {
	fwkplugin.Register(stopReasonRecorderType, fwkplugin.StabilityAlpha, func(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &stopReasonRecorder{
			typedName: fwkplugin.TypedName{Type: stopReasonRecorderType, Name: name},
		}, nil
	})
}

type stopReasonRecorder struct {
	typedName fwkplugin.TypedName
}

func (p *stopReasonRecorder) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *stopReasonRecorder) ResponseBody(_ context.Context, _ *fwksched.InferenceRequest, response *fwkrh.Response, _ *fwkdl.EndpointMetadata) {
	if response == nil || !response.EndOfStream || observedStopReasons == nil {
		return
	}
	select {
	case observedStopReasons <- response.StopReason:
	default:
	}
}

func TestStopReasonReachesResponsePlugin(t *testing.T) {
	config := fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: openai-parser
- type: anthropic-parser
- type: mock-metrics-source
- type: %s
schedulingProfiles:
- name: default
requestHandler:
  parsers:
  - pluginRef: openai-parser
  - pluginRef: anthropic-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
`, stopReasonRecorderType)

	observedStopReasons = make(chan fwkrh.StopReason, 4)
	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithConfigText(config)).WithBaseResources()
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)

	tests := []struct {
		name       string
		path       string
		body       string
		response   map[string]string
		respChunks []string
		want       fwkrh.StopReason
	}{
		{
			name:     "openai_chat_turn_ended",
			path:     reqcommon.APITypeChatCompletions.Path(),
			body:     `{"model":"` + modelMyModelTarget + `","stream":true,"messages":[{"role":"user","content":"hello"}]}`,
			response: map[string]string{reqcommon.HeaderContentType: reqcommon.MediaTypeEventStream, ":status": "200"},
			respChunks: []string{
				"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
				"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
				"data: [DONE]\n\n",
			},
			want: fwkrh.StopReasonTurnEnded,
		},
		{
			name:     "openai_responses_split_function_call",
			path:     reqcommon.APITypeResponses.Path(),
			body:     `{"model":"` + modelMyModelTarget + `","stream":true,"input":"hello"}`,
			response: map[string]string{reqcommon.HeaderContentType: reqcommon.MediaTypeEventStream, ":status": "200"},
			respChunks: []string{
				"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\"}}\n\n",
				"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\"}}\n\n",
			},
			want: fwkrh.StopReasonToolCall,
		},
		{
			name:     "anthropic_tool_use",
			path:     reqcommon.APITypeMessages.Path(),
			body:     `{"model":"` + modelMyModelTarget + `","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hello"}]}`,
			response: map[string]string{reqcommon.HeaderContentType: reqcommon.MediaTypeEventStream, ":status": "200"},
			respChunks: []string{
				"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":2}}\n\n",
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
			},
			want: fwkrh.StopReasonToolCall,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			headers := map[string]string{
				":method":                    "POST",
				":path":                      tt.path,
				reqcommon.HeaderContentType:  reqcommon.ContentTypeJSON,
				reqcommon.RequestIDHeaderKey: tt.name,
				metadata.ObjectiveKey:        modelMyModel,
				metadata.ModelNameRewriteKey: modelMyModelTarget,
			}
			requests := fwkepp.ReqRaw(headers, tt.body)
			requests = append(requests, ReqResponseOnly(tt.response, tt.respChunks...)...)

			client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(t.Context())
			require.NoError(t, err)
			responses, err := fwkepp.StreamedRequest(t, client, requests, len(requests))
			require.NoError(t, err)
			require.Len(t, responses, len(requests))

			select {
			case got := <-observedStopReasons:
				require.Equal(t, tt.want, got)
			case <-time.After(5 * time.Second):
				t.Fatal("timed out waiting for response plugin stop reason")
			}
		})
	}
}
