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

package prefixcacheaffinity

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// With PeakPrefillThroughput 1000, tokens equal the TTFT estimate in ms. In
// spillPool the sticky endpoint holds 90 of 100 blocks of 16 tokens and has
// 5000 ms of backlog; six cold endpoints are idle.
func spillConfig(maxSpill int) Config {
	return Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 1000, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, TTFTEstimate: TTFTEstimateBackLog, MaxSpillEndpoints: maxSpill}
}

func spillPool() []fwksched.Endpoint {
	eps := make([]fwksched.Endpoint, 0, 7)
	eps = append(eps, makeEndpoint("sticky", 90, -1, 5000, -1))
	for i := range 6 {
		eps = append(eps, makeEndpoint(fmt.Sprintf("cold-%d", i), 0, -1, 0, -1))
	}
	return eps
}

// prefixRequest has a 1440-token prefix (the sticky endpoint's 90 blocks)
// identified by seed, then a suffix that differs per request.
func prefixRequest(seed, suffix uint32) *fwksched.InferenceRequest {
	tokens := make([]uint32, 0, 1500)
	for i := range uint32(1440) {
		tokens = append(tokens, seed*100000+i)
	}
	for i := range uint32(60) {
		tokens = append(tokens, suffix*1000+i)
	}
	return &fwksched.InferenceRequest{
		TargetModel: "m",
		Body:        &fwkrequest.InferenceRequestBody{TokenizedRequest: fwkrequest.NewTokenizedRequest([][]uint32{tokens})},
	}
}

func names(eps []fwksched.Endpoint) []string {
	out := make([]string, 0, len(eps))
	for _, ep := range eps {
		out = append(out, ep.GetMetadata().ID.Name)
	}
	return out
}

func replace(pool []fwksched.Endpoint, ep fwksched.Endpoint) []fwksched.Endpoint {
	out := slices.Clone(pool)
	for i := range out {
		if out[i].GetMetadata().ID == ep.GetMetadata().ID {
			out[i] = ep
		}
	}
	return out
}

// spillTarget returns the one cold endpoint a k=1 override keeps.
func spillTarget(t *testing.T, p *Plugin, request *fwksched.InferenceRequest, pool []fwksched.Endpoint) string {
	t.Helper()
	cold := slices.DeleteFunc(names(p.Filter(context.Background(), request, pool)), func(name string) bool {
		return !strings.HasPrefix(name, "cold-")
	})
	require.Len(t, cold, 1)
	return cold[0]
}

func TestSpill_DisabledKeepsAllEndpoints(t *testing.T) {
	p := newTestPlugin(spillConfig(0))
	pool := spillPool()
	assert.Equal(t, pool, p.Filter(context.Background(), prefixRequest(1, 1), pool))
}

func TestSpill_KeepsStickyPlusAtMostK(t *testing.T) {
	for _, k := range []int{1, 2, 6, 10} {
		p := newTestPlugin(spillConfig(k))
		got := names(p.Filter(context.Background(), prefixRequest(1, 1), spillPool()))
		assert.Len(t, got, 1+min(k, 6), "k=%d", k)
		assert.Equal(t, "sticky", got[0], "k=%d", k)
	}
}

func TestSpill_SamePrefixSameTarget(t *testing.T) {
	pool := spillPool()
	want := spillTarget(t, newTestPlugin(spillConfig(1)), prefixRequest(7, 1), pool)
	for suffix := range uint32(20) {
		// A second instance stands in for another EPP replica.
		other := newTestPlugin(spillConfig(1))
		assert.Equal(t, want, spillTarget(t, other, prefixRequest(7, suffix), pool), "suffix %d", suffix)
	}
}

func TestSpill_PrefixesSpread(t *testing.T) {
	p := newTestPlugin(spillConfig(1))
	pool := spillPool()
	counts := map[string]int{}
	for seed := range uint32(600) {
		counts[spillTarget(t, p, prefixRequest(seed, 0), pool)]++
	}
	require.Len(t, counts, 6)
	for name, n := range counts {
		assert.Greater(t, n, 50, "%s got %d of 600 prefixes", name, n)
	}
}

func TestSpill_RemovingAnEndpointMovesOnlyItsPrefixes(t *testing.T) {
	p := newTestPlugin(spillConfig(1))
	pool := spillPool()
	removed := "cold-3"
	without := slices.DeleteFunc(slices.Clone(pool), func(ep fwksched.Endpoint) bool {
		return ep.GetMetadata().ID.Name == removed
	})
	moved := 0
	for seed := range uint32(300) {
		before := spillTarget(t, p, prefixRequest(seed, 0), pool)
		after := spillTarget(t, p, prefixRequest(seed, 0), without)
		if before == removed {
			moved++
			continue
		}
		assert.Equal(t, before, after, "prefix %d moved off %s", seed, before)
	}
	assert.Positive(t, moved)
}

// A cold endpoint whose own estimate does not beat the sticky one by more than
// maxTTFTPenaltyMs is skipped, and the next endpoint in rendezvous order takes
// the spill.
func TestSpill_SkipsEndpointThatDoesNotBeatSticky(t *testing.T) {
	p := newTestPlugin(spillConfig(1))
	request := prefixRequest(3, 0)
	pool := spillPool()
	first := spillTarget(t, p, request, pool)

	busy := replace(pool, makeEndpoint(first, 0, -1, 4000, -1)) // 5000 - 4000 does not exceed 1000
	second := spillTarget(t, p, request, busy)
	assert.NotEqual(t, first, second)

	key := p.spillKey(request, pool[:1])
	ranked := slices.Clone(pool[1:])
	slices.SortFunc(ranked, func(a, b fwksched.Endpoint) int {
		return cmp.Compare(rendezvousScore(key, b.GetMetadata().ID), rendezvousScore(key, a.GetMetadata().ID))
	})
	assert.Equal(t, []string{first, second}, names(ranked[:2]))
}

// With ttftEstimate full, a cold endpoint is charged the request's uncached
// prefill, so one that beats the sticky endpoint only on backlog is skipped.
func TestSpill_FullEstimateChargesThePrefill(t *testing.T) {
	full := spillConfig(1)
	full.TTFTEstimate = TTFTEstimateFull
	request := prefixRequest(3, 0)
	pool := make([]fwksched.Endpoint, 0, 7)
	pool = append(pool, makeEndpoint("sticky", 90, -1, 5000, 60))
	for i := range 6 {
		pool = append(pool, makeEndpoint(fmt.Sprintf("cold-%d", i), 0, -1, 0, 1500))
	}
	first := spillTarget(t, newTestPlugin(full), request, pool)

	// Backlog 3000 ms beats the sticky 5000 ms by 2000; adding the 1500-token
	// prefill leaves 5060 - 4500 = 560, under maxTTFTPenaltyMs.
	busy := replace(pool, makeEndpoint(first, 0, -1, 3000, 1500))
	assert.Equal(t, first, spillTarget(t, newTestPlugin(spillConfig(1)), request, busy))
	assert.NotEqual(t, first, spillTarget(t, newTestPlugin(full), request, busy))
}

func TestSpill_WithoutTokensKeysOnTheStickyEndpoint(t *testing.T) {
	p := newTestPlugin(spillConfig(1))
	pool := spillPool()
	want := spillTarget(t, p, nil, pool)
	assert.Equal(t, want, spillTarget(t, p, &fwksched.InferenceRequest{TargetModel: "m"}, pool))

	// Two sticky endpoints at the same depth: the key does not depend on their order.
	a, b := makeEndpoint("sticky-a", 90, -1, 5000, -1), makeEndpoint("sticky-b", 90, -1, 5000, -1)
	cold := spillPool()[1:]
	ab := spillTarget(t, p, nil, append([]fwksched.Endpoint{a, b}, cold...))
	assert.Equal(t, ab, spillTarget(t, p, nil, append([]fwksched.Endpoint{b, a}, cold...)))
}

func TestSpill_KeyCoversTheDeepestCachedPrefix(t *testing.T) {
	p := newTestPlugin(spillConfig(1))
	deep := []fwksched.Endpoint{makeEndpoint("deep", 90, -1, 0, -1)}
	key := p.spillKey(prefixRequest(5, 1), deep)
	assert.Equal(t, key, p.spillKey(prefixRequest(5, 2), deep), "suffix past the cached prefix")
	assert.NotEqual(t, key, p.spillKey(prefixRequest(6, 1), deep), "different prefix")

	salted := prefixRequest(5, 1)
	salted.Body.TokenizedRequest.CacheSalt = "tenant-a"
	assert.NotEqual(t, key, p.spillKey(salted, deep), "cache salt")
	model := prefixRequest(5, 1)
	model.TargetModel = "other"
	assert.NotEqual(t, key, p.spillKey(model, deep), "target model")

	shallow := makeEndpoint("shallow", 85, -1, 0, -1)
	assert.Equal(t, key, p.spillKey(prefixRequest(5, 1), append([]fwksched.Endpoint{shallow}, deep...)))
	assert.NotEqual(t, key, p.spillKey(prefixRequest(5, 1), []fwksched.Endpoint{shallow}))
}

func TestFactory_MaxSpillEndpoints(t *testing.T) {
	assert.Zero(t, DefaultConfig.MaxSpillEndpoints)

	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"maxSpillEndpoints": 2}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, 2, plugin.(*Plugin).config.MaxSpillEndpoints)

	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"maxSpillEndpoints": -1}`)), nil)
	assert.ErrorContains(t, err, "maxSpillEndpoints must be >= 0")
}
