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

package nohitlru_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/nohitlru"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestRenamedPrefillProfileUpdatesLRU(t *testing.T) {
	for _, prefillProfile := range []string{"prefill", "prefill-h200"} {
		t.Run(prefillProfile, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)
			prefillA := newColdNS("prefill-a")
			prefillB := newColdNS("prefill-b")
			decodeA := newColdNS("decode-a")

			req := &scheduling.InferenceRequest{RequestID: "cold-pd"}
			scorer.Score(ctx, req, []scheduling.Endpoint{decodeA})
			scorer.Score(ctx, req, []scheduling.Endpoint{prefillA, prefillB})
			_ = scorer.PreRequest(ctx, req, &scheduling.SchedulingResult{
				PrimaryProfileName: "decode",
				PrefillProfileName: prefillProfile,
				ProfileResults: map[string]*scheduling.ProfileRunResult{
					"decode":       {TargetEndpoints: []scheduling.Endpoint{decodeA}},
					prefillProfile: {TargetEndpoints: []scheduling.Endpoint{prefillA}},
				},
			})

			want := []string{"default/decode-a", "default/prefill-a"}
			if diff := cmp.Diff(want, dumpLRU(t, scorer).Endpoints); diff != "" {
				t.Errorf("LRU after cold P/D request (-want +got):\n%s", diff)
			}

			got := scorer.Score(ctx, &scheduling.InferenceRequest{RequestID: "next-cold"}, []scheduling.Endpoint{prefillA, prefillB})
			if got[prefillB] <= got[prefillA] {
				t.Errorf("next cold prefill request: prefill-b scored %v, want above prefill-a's %v", got[prefillB], got[prefillA])
			}
		})
	}
}
