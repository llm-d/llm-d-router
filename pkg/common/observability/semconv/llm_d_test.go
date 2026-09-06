/*
Copyright 2025 The Kubernetes Authors.

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

package semconv

import (
	"testing"

	"go.opentelemetry.io/otel/attribute"
)

func TestLLMDSemanticConventions(t *testing.T) {
	tests := []struct {
		name     string
		got      attribute.KeyValue
		wantKey  string
		wantType attribute.Type
	}{
		{
			name:     "LLMDEPPScorerType",
			got:      LLMDEPPScorerType("precise_prefix_cache"),
			wantKey:  "llm_d.epp.scorer.type",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPScorerScoreMax",
			got:      LLMDEPPScorerScoreMax(95.5),
			wantKey:  "llm_d.epp.scorer.score.max",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDEPPProfileHandlerDecision",
			got:      LLMDEPPProfileHandlerDecision("run_decode"),
			wantKey:  "llm_d.epp.profile_handler.decision",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPPDDisaggregationUsed",
			got:      LLMDEPPPDDisaggregationUsed(true),
			wantKey:  "llm_d.epp.pd.disaggregation_used",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDKVCacheBlockKeysCount",
			got:      LLMDKVCacheBlockKeysCount(16),
			wantKey:  "llm_d.kv_cache.block_keys.count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPDisaggReason",
			got:      LLMDEPPDisaggReason("prefix_cache"),
			wantKey:  "llm_d.epp.disagg.reason",
			wantType: attribute.STRING,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.got.Key) != tt.wantKey {
				t.Errorf("got key %q, want %q", tt.got.Key, tt.wantKey)
			}
			if tt.got.Value.Type() != tt.wantType {
				t.Errorf("got type %v, want %v", tt.got.Value.Type(), tt.wantType)
			}
		})
	}
}
