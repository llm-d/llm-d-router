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

package preciseprefixcache

import (
	"context"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

// The prefill profile name is set by the disagg profile handler's
// configuration, so speculative seeding reaches the prefill endpoint under any
// name the deployment gives that profile.
func TestRenamedPrefillProfile_SeedsSpeculativeForPrefill(t *testing.T) {
	for _, prefillProfile := range []string{"prefill", "prefill-h200"} {
		t.Run(prefillProfile, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			index, err := kvblock.NewInMemoryIndex(nil)
			require.NoError(t, err)
			keys := []kvblock.BlockHash{0xA1, 0xA2}
			idx := &fakeKVCacheIndexer{
				computeFromTokens: func(context.Context, []uint32, string, []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
					return keys, nil
				},
				index: index,
			}
			p := newProducerForProduceAndPreRequest(ctx, "renamed-prefill-"+prefillProfile, idx)
			p.speculativeEnabled = true
			p.speculativeCache = ttlcache.New[string, *speculativeEntries]()
			p.speculativeTTL = time.Minute

			endpoints := freshEndpoints()
			req := tokenizedRequest("req-renamed-prefill", len(keys)*testBlockSize)
			req.TargetModel = "test-model"
			require.NoError(t, p.Produce(ctx, req, endpoints))
			require.NoError(t, p.PreRequest(ctx, req, &scheduling.SchedulingResult{
				PrimaryProfileName: "decode",
				PrefillProfileName: prefillProfile,
				ProfileResults: map[string]*scheduling.ProfileRunResult{
					"decode":       {TargetEndpoints: []scheduling.Endpoint{endpoints[0]}},
					prefillProfile: {TargetEndpoints: []scheduling.Endpoint{endpoints[1]}},
				},
			}))

			for _, target := range []struct{ role, podID string }{
				{"decode", "10.0.0.1:8080"},
				{"prefill", "10.0.0.2:8080"},
			} {
				hits, err := index.Lookup(ctx, keys, sets.New(target.podID))
				require.NoError(t, err)
				for _, key := range keys {
					assert.Equal(t, []kvblock.PodEntry{{PodIdentifier: target.podID, Speculative: true}}, hits[key],
						"%s endpoint %s must hold a speculative entry for key %v", target.role, target.podID, key)
				}
			}
		})
	}
}
