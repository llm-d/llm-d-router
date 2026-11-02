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

package predictedlatency

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
)

// TestRenamedPrefillProfile_TTFTTrainingAttribution drives Produce, PreRequest
// and the first streamed ResponseBody chunk for a P/D request and checks that
// the TTFT sample is attributed to the prefill endpoint whatever the prefill
// profile is named.
func TestRenamedPrefillProfile_TTFTTrainingAttribution(t *testing.T) {
	const roleLabel = "llm-d.ai/role"
	for _, prefillProfile := range []string{"prefill", "prefill-h200"} {
		t.Run(prefillProfile, func(t *testing.T) {
			cfg := DefaultConfig
			cfg.StreamingMode = true
			cfg.EndpointRoleLabel = roleLabel
			// The mock keys predictions on KV usage: 0.4 is the prefill endpoint, 0.5 the decode endpoint.
			mp := &mockPredictor{predictions: map[string]*latencypredictor.PredictionResponse{
				"0.4": {TTFT: 95, TPOT: 0},
				"0.5": {TTFT: 40, TPOT: 12},
			}}
			pl := NewPredictedLatency(LatencyDataProviderPluginType, cfg, mp)

			prefill := createTestEndpointWithLabels("prefill-pod", 0.4, 3, 2, map[string]string{roleLabel: "prefill"})
			decode := createTestEndpointWithLabels("decode-pod", 0.5, 1, 1, map[string]string{roleLabel: "decode"})
			request := createTestInferenceRequest("pd-req", 1000, 50)
			result := &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				PrefillProfileName: prefillProfile,
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode":       {TargetEndpoints: []fwksched.Endpoint{decode}},
					prefillProfile: {TargetEndpoints: []fwksched.Endpoint{prefill}},
				},
			}

			ctx := context.Background()
			require.NoError(t, pl.Produce(ctx, request, []fwksched.Endpoint{prefill, decode}))
			require.Len(t, mp.capturedBulkStrictRequests, 2, "Produce predicts every candidate endpoint")
			require.NoError(t, pl.PreRequest(ctx, request, result))
			pl.ResponseBody(ctx, request, &fwkrh.Response{}, decode.GetMetadata())

			require.Len(t, mp.capturedTrainingEntries, 1, "first token records one TTFT sample")
			entry := mp.capturedTrainingEntries[0]
			assert.Equal(t, "prefill", entry.PodType, "TTFT sample attributed to the wrong endpoint role")
			assert.Equal(t, 0.4, entry.KVCachePercentage, "TTFT sample uses the wrong profile's metrics")
			assert.Equal(t, 3, entry.NumRequestRunning, "TTFT sample uses the wrong endpoint's dispatch snapshot")
			if assert.NotNil(t, entry.PredictedTTFT) {
				assert.Equal(t, 95.0, *entry.PredictedTTFT, "TTFT sample paired with the wrong endpoint's prediction")
			}
		})
	}
}
