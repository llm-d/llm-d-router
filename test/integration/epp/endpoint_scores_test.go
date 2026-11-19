/*
Copyright 2025 The llm-d Authors.

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
	"strings"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/approximateprefix"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

// TestEndpointScoresMetadata verifies the opt-in request-path dynamic metadata contract: with
// --emit-endpoint-scores, the envoy.lb namespace carries an x-gateway-destination-endpoint-scores
// struct mapping every endpoint the scheduler scored to its score, including candidates the picker
// did not select, while the endpoint key itself is unchanged.
func TestEndpointScoresMetadata(t *testing.T) {
	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithEmitEndpointScores()).WithBaseResources()

	// Both pods stay strictly below the utilization-detector limits (queue < 5, KV < 0.8)
	// so the default filter drops neither and the scheduler scores both.
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget), eppharness.P(1, 4, 0.5, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel).WaitForMetricsDelivery()

	envoyLb := requestHeaderEnvoyLbMetadata(t, h)

	endpointValue, ok := envoyLb.Fields[metadata.DestinationEndpointKey]
	require.True(t, ok, "expected destination endpoint in envoy.lb namespace")
	endpoints := strings.Split(endpointValue.GetStringValue(), ",")
	require.NotEmpty(t, endpoints, "expected at least one destination endpoint")

	scoresValue, ok := envoyLb.Fields[metadata.DestinationEndpointScoresKey]
	require.True(t, ok, "expected destination endpoint scores in envoy.lb namespace")
	scoreFields := scoresValue.GetStructValue().Fields
	require.Len(t, scoreFields, len(pods), "expected a score for every scored candidate endpoint")
	for _, endpoint := range endpoints {
		score, ok := scoreFields[endpoint]
		require.True(t, ok, "expected a score for destination endpoint %s", endpoint)
		require.GreaterOrEqual(t, score.GetNumberValue(), 0.0, "expected a non-negative score for endpoint %s", endpoint)
	}
	// The picker selects a subset of the candidates it scores, so the runner-up's score
	// is present even though it is not a destination.
	require.Greater(t, len(scoreFields), len(endpoints), "expected scores for candidates the picker did not select")
}

// TestEndpointScoresMetadataOffByDefault verifies that without --emit-endpoint-scores the
// request-path dynamic metadata carries only the destination endpoint key.
func TestEndpointScoresMetadataOffByDefault(t *testing.T) {
	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode()).WithBaseResources()

	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget), eppharness.P(1, 5, 0.5, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)

	envoyLb := requestHeaderEnvoyLbMetadata(t, h)

	_, ok := envoyLb.Fields[metadata.DestinationEndpointKey]
	require.True(t, ok, "expected destination endpoint in envoy.lb namespace")
	_, ok = envoyLb.Fields[metadata.DestinationEndpointScoresKey]
	require.False(t, ok, "expected destination endpoint scores to be absent from envoy.lb namespace by default")
}

func TestTextToSpeechCacheSaltEndpointScores(t *testing.T) {
	configText := `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: token-producer
    parameters:
      backend: estimate
  - type: approx-prefix-cache-producer
    parameters:
      autoTune: false
      blockSizeTokens: 64
  - type: prefix-cache-scorer
  - type: max-score-picker
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: prefix-cache-scorer
      - pluginRef: max-score-picker
requestHandler:
  parsers:
    - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
    - pluginRef: mock-metrics-source
`
	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithConfigText(configText), eppharness.WithEmitEndpointScores()).WithBaseResources()
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget), eppharness.P(1, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)

	sendSpeech := func(t *testing.T, requestID, salt string) (string, map[string]*structpb.Value) {
		t.Helper()
		body := map[string]any{
			"model": modelMyModel,
			"input": strings.Repeat("Welcome to customer support. ", 16),
		}
		if salt != "" {
			body["cache_salt"] = salt
		}
		raw, err := json.Marshal(body)
		require.NoError(t, err)
		requests := fwkepp.ReqRaw(map[string]string{
			":path":                      "/v1/audio/speech",
			metadata.ObjectiveKey:        modelMyModel,
			metadata.ModelNameRewriteKey: modelMyModelTarget,
			reqcommon.RequestIDHeaderKey: requestID,
		}, string(raw))
		requests = append(requests, ReqResponseOnly(
			map[string]string{"content-type": "audio/wav", ":status": "200"}, "RIFF",
		)...)

		client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(t.Context())
		require.NoError(t, err)
		defer func() { _ = client.CloseSend() }()
		responses, err := fwkepp.StreamedRequest(t, client, requests, 4)
		require.NoError(t, err)
		require.Len(t, responses, 4)

		envoyLb := responses[0].DynamicMetadata.GetFields()[metadata.DestinationEndpointNamespace].GetStructValue()
		require.NotNil(t, envoyLb)
		endpoint := envoyLb.Fields[metadata.DestinationEndpointKey].GetStringValue()
		require.NotEmpty(t, endpoint)
		scores := envoyLb.Fields[metadata.DestinationEndpointScoresKey].GetStructValue()
		require.NotNil(t, scores)
		require.Len(t, scores.Fields, len(pods))
		return endpoint, scores.Fields
	}

	cachedEndpoint, coldScores := sendSpeech(t, "warmup", "tenant-a")
	for _, score := range coldScores {
		require.Zero(t, score.GetNumberValue())
	}

	// PreRequest updates the prefix index asynchronously.
	dumper, ok := h.Runner.PluginHandle.Plugin(approximateprefix.ApproxPrefixCachePluginType).(plugin.StateDumper)
	require.True(t, ok)
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		raw, err := dumper.DumpState()
		require.NoError(c, err)
		var state struct {
			TotalPods int `json:"totalPods"`
			Pods      []struct {
				Blocks int `json:"blocks"`
			} `json:"pods"`
		}
		require.NoError(c, json.Unmarshal(raw, &state))
		require.Equal(c, 1, state.TotalPods)
		require.Len(c, state.Pods, 1)
		require.Greater(c, state.Pods[0].Blocks, 0)
	}, 5*time.Second, 10*time.Millisecond)

	for _, tc := range []struct {
		name      string
		salt      string
		wantMatch bool
	}{
		{name: "same-salt", salt: "tenant-a", wantMatch: true},
		{name: "different-salt", salt: "tenant-b"},
		{name: "missing-salt"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, scores := sendSpeech(t, tc.name, tc.salt)
			require.Contains(t, scores, cachedEndpoint)
			if tc.wantMatch {
				require.Greater(t, scores[cachedEndpoint].GetNumberValue(), 0.0)
			} else {
				for _, score := range scores {
					require.Zero(t, score.GetNumberValue(), "isolated requests must not receive a cache-hit bonus")
				}
			}
		})
	}
}

// TestMultiModalGenerateEndpointScores sends pre-rendered prompts that differ
// only in the image. vLLM renders every same-size image as the same run of
// placeholder token IDs, so the image's identity is only in mm_hashes; an
// endpoint that served one image must not score as a full prefix hit for
// another.
func TestMultiModalGenerateEndpointScores(t *testing.T) {
	configText := `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: vllmhttp-parser
  - type: token-producer
    parameters:
      backend: estimate
  - type: approx-prefix-cache-producer
    parameters:
      autoTune: false
      blockSizeTokens: 64
  - type: prefix-cache-scorer
  - type: max-score-picker
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: prefix-cache-scorer
      - pluginRef: max-score-picker
requestHandler:
  parsers:
    - pluginRef: vllmhttp-parser
dataLayer:
  injectDefaults: false
  sources:
    - pluginRef: mock-metrics-source
`
	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithConfigText(configText), eppharness.WithEmitEndpointScores()).WithBaseResources()
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget), eppharness.P(1, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)

	// 100 text tokens, a 256-token image, 20 text tokens: six 64-token blocks,
	// the image starting in block 1.
	const placeholder = 7
	tokenIDs := make([]int, 0, 100+256+20)
	for i := range 100 {
		tokenIDs = append(tokenIDs, 1000+i)
	}
	for range 256 {
		tokenIDs = append(tokenIDs, placeholder)
	}
	for i := range 20 {
		tokenIDs = append(tokenIDs, 2000+i)
	}

	sendGenerate := func(t *testing.T, requestID, imageHash string) (string, map[string]*structpb.Value) {
		t.Helper()
		raw, err := json.Marshal(map[string]any{
			"model":     modelMyModel,
			"token_ids": tokenIDs,
			"features": map[string]any{
				"mm_hashes":       map[string][]string{"image": {imageHash}},
				"mm_placeholders": map[string][]map[string]int{"image": {{"offset": 100, "length": 256}}},
			},
		})
		require.NoError(t, err)
		requests := fwkepp.ReqRaw(map[string]string{
			":path":                      reqcommon.PathVLLMGenerate,
			metadata.ObjectiveKey:        modelMyModel,
			metadata.ModelNameRewriteKey: modelMyModelTarget,
			reqcommon.RequestIDHeaderKey: requestID,
		}, string(raw))
		requests = append(requests, ReqResponseOnly(
			map[string]string{"content-type": "application/json", ":status": "200"}, `{}`,
		)...)

		client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(t.Context())
		require.NoError(t, err)
		defer func() { _ = client.CloseSend() }()
		responses, err := fwkepp.StreamedRequest(t, client, requests, 4)
		require.NoError(t, err)
		require.Len(t, responses, 4)

		envoyLb := responses[0].DynamicMetadata.GetFields()[metadata.DestinationEndpointNamespace].GetStructValue()
		require.NotNil(t, envoyLb)
		endpoint := envoyLb.Fields[metadata.DestinationEndpointKey].GetStringValue()
		require.NotEmpty(t, endpoint)
		scores := envoyLb.Fields[metadata.DestinationEndpointScoresKey].GetStructValue()
		require.NotNil(t, scores)
		require.Len(t, scores.Fields, len(pods))
		return endpoint, scores.Fields
	}

	cachedEndpoint, coldScores := sendGenerate(t, "warmup", "image-a")
	for _, score := range coldScores {
		require.Zero(t, score.GetNumberValue())
	}

	// PreRequest updates the prefix index asynchronously.
	dumper, ok := h.Runner.PluginHandle.Plugin(approximateprefix.ApproxPrefixCachePluginType).(plugin.StateDumper)
	require.True(t, ok)
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		raw, err := dumper.DumpState()
		require.NoError(c, err)
		var state struct {
			TotalPods int `json:"totalPods"`
		}
		require.NoError(c, json.Unmarshal(raw, &state))
		require.Equal(c, 1, state.TotalPods)
	}, 5*time.Second, 10*time.Millisecond)

	for _, tc := range []struct {
		name      string
		imageHash string
		wantScore float64
	}{
		{name: "same-image", imageHash: "image-a", wantScore: 1},
		{name: "different-image", imageHash: "image-b", wantScore: 1.0 / 6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, scores := sendGenerate(t, tc.name, tc.imageHash)
			require.Contains(t, scores, cachedEndpoint)
			assert.InDelta(t, tc.wantScore, scores[cachedEndpoint].GetNumberValue(), 1e-9)
		})
	}
}

// requestHeaderEnvoyLbMetadata drives one LLM request through the EPP and returns the envoy.lb
// dynamic metadata struct attached to the request-headers ext_proc response.
func requestHeaderEnvoyLbMetadata(t *testing.T, h *eppharness.TestHarness) *structpb.Struct {
	t.Helper()

	requests := fwkepp.ReqLLM(eppharness.Logger(), "hello", modelMyModel, modelMyModelTarget)

	// RequestHeaders + RequestBody -> request header response and request body response.
	responses, err := fwkepp.StreamedRequest(t, h.Client, requests, 2)
	require.NoError(t, err)
	require.Len(t, responses, 2)

	res := responses[0]
	require.NotNil(t, res.DynamicMetadata, "expected DynamicMetadata in the request headers ext_proc response")
	envoyLb, ok := res.DynamicMetadata.Fields[metadata.DestinationEndpointNamespace]
	require.True(t, ok, "expected envoy.lb namespace in DynamicMetadata")
	return envoyLb.GetStructValue()
}
