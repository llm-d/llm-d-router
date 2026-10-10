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

package internal

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	detectorconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
)

// TestProjectedCostAdmissionRegression connects the real processor and detector.
// Queue bookkeeping uses the package's existing harness. Endpoint token estimates
// are supplied directly; tokenization, prefix lookup, HTTP and GPUs are not run.
func TestProjectedCostAdmissionRegression(t *testing.T) {
	for _, tc := range []struct {
		name       string
		loads      []int64
		incoming   int64
		admissible int
	}{
		{"projected_cost_exceeds_all_endpoints", []int64{60, 100}, 50, 0},
		{"request_fits_one_endpoint", []int64{60, 100}, 20, 1},
		{"all_endpoints_full", []int64{100, 100}, 20, 0},
		{"oversized_request_can_use_empty_endpoint", []int64{0, 100}, 150, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newTestHarness(t, testCleanupTick)
			t.Cleanup(h.Stop)
			params := json.NewDecoder(strings.NewReader(`{"concurrencyMode":"tokens","maxTokenConcurrency":100,"headroom":0,"failOpen":false}`))
			instance, err := detectorconcurrency.ConcurrencyDetectorFactory("cost-repro", params,
				plugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil }))
			require.NoError(t, err)
			detector := instance.(flowcontrol.SaturationDetector)
			filter := instance.(scheduling.Filter)
			h.processor.saturationDetector = detector
			pool := make([]datalayer.Endpoint, 0, len(tc.loads))
			candidates := make([]scheduling.Endpoint, 0, len(tc.loads))
			for i, load := range tc.loads {
				ep := datalayer.NewEndpoint(&datalayer.EndpointMetadata{
					ID: types.NamespacedName{Namespace: "test", Name: fmt.Sprintf("endpoint-%d", i)},
				}, nil)
				ep.GetAttributes().Put(attrconcurrency.InFlightLoadDataKey,
					&attrconcurrency.InFlightLoad{Requests: 1, Tokens: load})
				pool = append(pool, ep)
				attrs := ep.GetAttributes().Clone()
				attrs.Put(attrconcurrency.UncachedRequestTokensDataKey,
					&attrconcurrency.UncachedRequestTokens{Tokens: tc.incoming})
				candidates = append(candidates, scheduling.NewEndpoint(ep.GetMetadata(), ep.GetMetrics(), attrs))
			}
			h.endpointCandidates.Candidates = pool
			q := h.addQueue(testFlow)
			item := h.newTestItem("cost-repro-request", testFlow, testTTL)
			metadata := make(map[types.NamespacedName]*datalayer.EndpointMetadata, len(pool))
			for _, ep := range pool {
				metadata[ep.GetMetadata().ID] = ep.GetMetadata().Clone()
			}
			item.SetPreparation(t.Context(), func(context.Context, func(*contracts.PreparedRequest) bool, func(*contracts.PreparedRequest)) *contracts.PreparedRequest {
				return &contracts.PreparedRequest{Request: item.OriginalRequest().InferenceRequest(), Endpoints: candidates, Candidates: metadata}
			})
			require.NoError(t, q.Add(item))
			filtered := filter.Filter(t.Context(), nil, candidates)
			require.Len(t, filtered, tc.admissible)
			saturation := detector.Saturation(t.Context(), pool)
			dispatched := h.processor.dispatchCycle(t.Context())
			if saturation < 1 {
				require.False(t, dispatched, "preparation must leave the item queued")
				require.Eventually(t, func() bool { return len(item.preparationDone) > 0 }, time.Second, time.Millisecond)
				dispatched = h.processor.dispatchCycle(t.Context())
			}
			t.Logf("saturation=%.2f incoming_tokens=%d admissible_endpoints=%d dispatched=%t queue_remaining=%d",
				saturation, tc.incoming, len(filtered), dispatched, q.Len())
			require.Equal(t, tc.admissible > 0, dispatched,
				"a request without an admissible endpoint should stay queued")
		})
	}
}

func TestProjectedAdmissionHonorsRankedBandOpportunities(t *testing.T) {
	for _, preparing := range []bool{true, false} {
		t.Run(fmt.Sprintf("preparation_pending=%v", preparing), func(t *testing.T) {
			h := newTestHarness(t, testCleanupTick)
			t.Cleanup(h.Stop)
			instance, err := detectorconcurrency.ConcurrencyDetectorFactory("ranked-cost", plugin.StrictDecoder([]byte(
				`{"concurrencyMode":"tokens","maxTokenConcurrency":100,"headroom":0,"failOpen":false}`)),
				plugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil }))
			require.NoError(t, err)
			h.processor.saturationDetector = instance.(flowcontrol.SaturationDetector)
			endpoint := datalayer.NewEndpoint(&datalayer.EndpointMetadata{
				ID: types.NamespacedName{Namespace: "test", Name: "ranked-endpoint"},
			}, nil)
			endpoint.GetAttributes().Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: 1, Tokens: 60})
			h.endpointCandidates.Candidates = []datalayer.Endpoint{endpoint}

			started, release := make(chan struct{}, 1), make(chan struct{})
			defer close(release)
			newPreparedItem := func(key flowcontrol.FlowKey, incoming int64, pending bool) *FlowItem {
				item := h.newTestItem(key.ID, key, testTTL)
				attrs := endpoint.GetAttributes().Clone()
				attrs.Put(attrconcurrency.UncachedRequestTokensDataKey, &attrconcurrency.UncachedRequestTokens{Tokens: incoming})
				prepared := &contracts.PreparedRequest{
					Request:    item.OriginalRequest().InferenceRequest(),
					Endpoints:  []scheduling.Endpoint{scheduling.NewEndpoint(endpoint.GetMetadata(), endpoint.GetMetrics(), attrs)},
					Candidates: map[datalayer.ID]*datalayer.EndpointMetadata{endpoint.GetMetadata().ID: endpoint.GetMetadata().Clone()},
				}
				item.SetPreparation(h.ctx, func(ctx context.Context, _ func(*contracts.PreparedRequest) bool, _ func(*contracts.PreparedRequest)) *contracts.PreparedRequest {
					if pending {
						started <- struct{}{}
						select {
						case <-release:
						case <-ctx.Done():
						}
					}
					return prepared
				})
				if !pending {
					item.prepared = prepared
					item.refreshAfter = h.clock.Now().Add(time.Minute)
				}
				require.NoError(t, h.addQueue(key).Add(item))
				return item
			}

			high := newPreparedItem(flowcontrol.FlowKey{ID: "high", Priority: 30}, 10, false)
			ready := newPreparedItem(flowcontrol.FlowKey{ID: "ready", Priority: 20}, 20, false)
			blocked := newPreparedItem(flowcontrol.FlowKey{ID: "blocked", Priority: 10}, 50, preparing)
			policy := &fakeBandSelectionPolicy{order: []int{2, 1, 0}}
			h.processor.bandSelectionPolicy = policy

			require.True(t, h.processor.dispatchCycle(h.ctx))
			require.NotNil(t, ready.FinalState())
			require.NoError(t, ready.FinalState().Err)
			require.Nil(t, high.FinalState(), "the ranked order must take precedence over numeric priority")
			require.Nil(t, blocked.FinalState())
			require.Equal(t, []int{20}, policy.dispatched)
			if preparing {
				require.Eventually(t, func() bool { return len(started) == 1 }, time.Second, time.Millisecond)
			}

			require.True(t, h.processor.dispatchCycle(h.ctx))
			require.NotNil(t, high.FinalState())
			require.NoError(t, high.FinalState().Err)
			require.Equal(t, []int{20, 30}, policy.dispatched)
			require.False(t, h.processor.dispatchCycle(h.ctx))
			require.Nil(t, blocked.FinalState(), "the inadmissible request must remain queued")
			require.Equal(t, []int{20, 30}, policy.dispatched, "a cycle with no dispatch must not charge the band policy")
		})
	}
}

func TestProjectedAdmissionWaitPreservesTailReclamation(t *testing.T) {
	h := newTestHarness(t, testCleanupTick)
	t.Cleanup(h.Stop)
	h.saturationDetector.SaturationFunc = func(context.Context, []datalayer.Endpoint) float64 { return 0.6 }
	h.processor.usageLimitPolicy = usagelimits.NewPolicyFunc("admission-ceilings",
		func(_ context.Context, _ float64, priorities []int, ceilings []float64) {
			for i, priority := range priorities {
				if priority == 10 {
					ceilings[i] = 0.5
				}
			}
		})
	policy := &fakeBandSelectionPolicy{}
	h.processor.bandSelectionPolicy = policy
	evictor := &fakeInFlightEvictor{inFlight: 6, evictable: 3, victimPriority: 0, hasVictim: true}
	withReclamation(h, evictor, testReclamationConfig)

	key := flowcontrol.FlowKey{ID: "preparing", Priority: 20}
	preparing := h.newTestItem(key.ID, key, testTTL)
	preparing.SetPreparation(h.ctx, func(context.Context, func(*contracts.PreparedRequest) bool, func(*contracts.PreparedRequest)) *contracts.PreparedRequest {
		t.Fatal("the pending preparation must not be restarted")
		return nil
	})
	preparing.preparationDone = make(chan preparationResult)
	require.NoError(t, h.addQueue(key).Add(preparing))
	gatedKey := flowcontrol.FlowKey{ID: "gated", Priority: 10}
	gated := h.newTestItem(gatedKey.ID, gatedKey, testTTL)
	require.NoError(t, h.addQueue(gatedKey).Add(gated))

	require.False(t, h.processor.dispatchCycle(h.ctx))
	require.Nil(t, preparing.FinalState())
	require.Nil(t, gated.FinalState())
	require.Empty(t, policy.dispatched)
	require.Equal(t, []int{1}, evictor.totalEvictCalls())
	require.Equal(t, []int{10}, evictor.boundCalls, "reclamation must target the usage-gated band's demand")
}
