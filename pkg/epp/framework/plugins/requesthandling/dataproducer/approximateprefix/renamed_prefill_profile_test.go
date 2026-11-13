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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixmetrics"
)

// A disaggregated request's prefill profile may have any name. Its target
// should be indexed and predicted for as it is under the name "prefill".
func TestRenamedPrefillProfile(t *testing.T) {
	disableMinBlockSizeClamp(t)
	tokens := []uint32{1, 2, 3, 4}

	for _, prefillProfile := range []string{"prefill", "prefill-h200"} {
		t.Run(prefillProfile, func(t *testing.T) {
			name := "approx-renamed-" + prefillProfile
			p := producerForPrediction(t, name, 2)
			decode, prefill := namedEndpoint("decode"), namedEndpoint("prefill")
			endpoints := []fwksched.Endpoint{decode, prefill}
			disaggregated := &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				PrefillProfileName: prefillProfile,
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode":       {TargetEndpoints: []fwksched.Endpoint{decode}},
					prefillProfile: {TargetEndpoints: []fwksched.Endpoint{prefill}},
				},
			}

			runPrediction(t, p, "first", tokens, endpoints, disaggregated)
			assert.Equal(t, float64(len(tokens)), metricSum(t, promptTokensMetric, name, prefixmetrics.RolePrefill),
				"prompt tokens recorded under endpoint_role=prefill")

			probe := &fwksched.InferenceRequest{RequestID: "probe", TargetModel: "m", Body: tokenizedBody(tokens)}
			require.NoError(t, p.Produce(context.Background(), probe, endpoints))
			key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name)
			for _, endpoint := range endpoints {
				info, ok := endpoint.Get(key)
				require.True(t, ok)
				assert.Equal(t, 2, info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks(),
					"blocks indexed on %s", endpoint.GetMetadata().ID.Name)
			}

			runPrediction(t, p, "second", tokens, endpoints, disaggregated)
			assert.Equal(t, float64(len(tokens)), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RolePrefill),
				"repeat predicted as a hit on the prefill endpoint")
		})
	}
}
