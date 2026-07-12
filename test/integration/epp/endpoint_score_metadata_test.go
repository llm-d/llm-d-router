/*
Copyright 2026 The Kubernetes Authors.

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
	"fmt"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	integration "github.com/llm-d/llm-d-router/test/integration"
)

const endpointScoreMetadataConfig = `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: endpoint-attribute-scorer
    parameters:
      source: metadata
      metadataNamespace: llm-d.routing
      metadataField: endpoint_scores
      algorithm:
        type: linear_higher_is_better
        normalization:
          adaptiveRange: {}
  - type: max-score-picker
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: max-score-picker
      - pluginRef: endpoint-attribute-scorer
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
`

func TestEndpointScoreFromMetadata(t *testing.T) {
	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(endpointScoreMetadataConfig)).WithBaseResources()

	pods := []PodState{
		P(0, 0, 0.1, modelMyModelTarget),
		P(1, 0, 0.1, modelMyModelTarget),
		P(2, 0, 0.1, modelMyModelTarget),
	}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	id := func(idx int) string { return fmt.Sprintf("%s/pod-%d-rank-0", h.Namespace, idx) }
	ep := func(idx int) string { return fmt.Sprintf("192.168.1.%d:8000", idx+1) }

	cases := []struct {
		name   string
		scores map[string]float64
		want   string
	}{
		{"highest upstream-scored endpoint wins", map[string]float64{id(0): 0.9, id(1): 0.1, id(2): 0.1}, ep(0)},
		{"flipping the scores flips the pick", map[string]float64{id(0): 0.1, id(1): 0.1, id(2): 0.9}, ep(2)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, sendScoreRequest(t, h, tc.scores))
		})
	}
}

// sendScoreRequest drives one request whose llm-d.routing dynamic metadata carries a per-endpoint score
// map (endpoint_scores), and returns the destination endpoint the EPP picked.
func sendScoreRequest(t *testing.T, h *TestHarness, scores map[string]float64) string {
	t.Helper()

	scoreMap := make(map[string]any, len(scores))
	for k, v := range scores {
		scoreMap[k] = v
	}
	routing, err := structpb.NewStruct(map[string]any{"endpoint_scores": scoreMap})
	require.NoError(t, err)
	md := &envoyCorev3.Metadata{FilterMetadata: map[string]*structpb.Struct{"llm-d.routing": routing}}

	headers := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: []*envoyCorev3.HeaderValue{
					{Key: metadata.ObjectiveKey, Value: modelMyModel},
					{Key: metadata.ModelNameRewriteKey, Value: modelMyModelTarget},
					{Key: reqcommon.RequestIDHeaderKey, Value: "score-req"},
				}},
			},
		},
		MetadataContext: md,
	}
	body := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{
				Body:        []byte(`{"model":"` + modelMyModel + `","prompt":"hello","max_tokens":10,"temperature":0}`),
				EndOfStream: true,
			},
		},
		MetadataContext: md,
	}

	client, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(t.Context())
	require.NoError(t, err)
	responses, err := integration.StreamedRequest(t, client, []*extProcPb.ProcessingRequest{headers, body}, 2)
	require.NoError(t, err)

	for _, r := range responses {
		if e := headerValue(r.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey); e != "" {
			return e
		}
		if e := headerValue(r.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey); e != "" {
			return e
		}
	}
	t.Fatal("no destination endpoint set in the EPP responses")
	return ""
}
