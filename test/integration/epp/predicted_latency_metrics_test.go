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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
)

func TestPredictedLatencyModelLabels(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/status", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"is_ready":true,"model_type":"bayesian_ridge"}`))
	})
	mux.HandleFunc("/model/download/info", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model_type":"bayesian_ridge"}`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ttft_intercept 10\ntpot_intercept 1\n"))
	})
	mux.HandleFunc("/predict/bulk/strict", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"predictions":[{"ttft_ms":10,"tpot_ms":1}],"total_requests":1,"successful_predictions":1}`))
	})
	predictor := httptest.NewServer(mux)
	t.Cleanup(predictor.Close)
	t.Setenv("PREDICTION_SERVER_URL", predictor.URL)
	t.Setenv("TRAINING_SERVER_URL", predictor.URL)

	pluginName := strings.ToLower(t.Name())
	configText := fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: predicted-latency-producer
    name: %s
    parameters:
      predictInProduce: true
  - type: queue-scorer
  - type: openai-parser
  - type: mock-metrics-source
  - type: core-metrics-extractor
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
requestHandler:
  parsers:
    - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
    - pluginRef: mock-metrics-source
`, pluginName)
	h := NewTestHarness(t.Context(), t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
	h.WithPods([]PodState{P(0, 0, 0.1)}).WaitForSync(1, modelMyModel).WaitForMetricsDelivery()

	predictionCounts := func(t *testing.T, targetLabel string) map[string]uint64 {
		t.Helper()
		families, err := crmetrics.Registry.Gather()
		require.NoError(t, err)
		counts := make(map[string]uint64)
		for _, family := range families {
			switch family.GetName() {
			case "llm_d_epp_request_ttft_prediction_duration_seconds", "llm_d_epp_request_tpot_prediction_duration_seconds":
				for _, metric := range family.GetMetric() {
					labels := make(map[string]string)
					for _, label := range metric.GetLabel() {
						labels[label.GetName()] = label.GetValue()
					}
					if labels["plugin_name"] == pluginName && labels["target_model_name"] == targetLabel {
						counts[family.GetName()] = metric.GetHistogram().GetSampleCount()
					}
				}
			}
		}
		return counts
	}

	tests := []struct {
		name      string
		target    string
		wantLabel string
	}{
		{name: "invalid UTF-8", target: "model\xff", wantLabel: metricsutil.OverflowValue},
		{
			name:      "oversized",
			target:    strings.Repeat("m", metricsutil.MaxLabelValueLength+1),
			wantLabel: metricsutil.OverflowValue,
		},
		{
			name:      "at length limit",
			target:    strings.Repeat("m", metricsutil.MaxLabelValueLength),
			wantLabel: strings.Repeat("m", metricsutil.MaxLabelValueLength),
		},
		{name: "valid request after invalid requests", target: modelDirect, wantLabel: modelDirect},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := predictionCounts(t, tt.wantLabel)
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(ctx)
			require.NoError(t, err)
			requests := fwkepp.ReqRaw(map[string]string{
				":path":        "/v1/completions",
				"content-type": "application/json",
			}, `{"model":"`+modelDirect+`","prompt":[1,2,3],"max_tokens":1}`)
			// Envoy carries arbitrary header bytes in RawValue, not the UTF-8 Value field.
			requests[0].GetRequestHeaders().Headers.Headers = append(requests[0].GetRequestHeaders().Headers.Headers,
				&envoyCorev3.HeaderValue{Key: metadata.ModelNameRewriteKey, RawValue: []byte(tt.target)})

			responses, err := fwkepp.StreamedRequest(t, client, requests, 2)
			require.NoError(t, err)
			require.Len(t, responses, 2)
			require.NotNil(t, responses[0].GetRequestHeaders())
			require.NotNil(t, responses[1].GetRequestBody())
			var body map[string]any
			require.NoError(t, json.Unmarshal(responses[1].GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetBody(), &body))
			require.Equal(t, strings.ToValidUTF8(tt.target, "\ufffd"), body["model"])
			require.NoError(t, client.CloseSend())

			after := predictionCounts(t, tt.wantLabel)
			require.Len(t, after, 2, "both prediction duration metrics must use the bounded target label")
			for name, count := range after {
				require.Equal(t, before[name]+1, count, name)
			}
		})
	}
}
