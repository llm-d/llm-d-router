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
	"context"
	"math"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
)

// testSteps is the profile of the engine measuredCases come from.
var testSteps = &PrefillSteps{
	TokensPerStep:            8192,
	RequestsPerStep:          4,
	StepMs:                   345,
	RequestMs:                200,
	ReadMicrosecondsPerToken: 5,
	HandoffMs:                500,
}

// measuredCases are median times from dispatch to the first streamed token
// through a router, on a GLM 5.3 prefill engine (8x H100,
// --max-num-batched-tokens 8192, --max-num-seqs 4) of a P/D deployment. The
// cases with nothing cached are lone prompts; the others are session prompts
// with most of their prefix cached, under open-loop load, grouped by the
// requests pending on the engine when the router scheduled them. A prefill
// request ends at its first token, so every request in flight was pending.
var measuredCases = []struct {
	requests, promptTokens, uncachedTokens int64
	measuredMs                             float64
}{
	{requests: 0, promptTokens: 11240, uncachedTokens: 11240, measuredMs: 1004},
	{requests: 0, promptTokens: 14980, uncachedTokens: 14980, measuredMs: 1038},
	{requests: 0, promptTokens: 29940, uncachedTokens: 29940, measuredMs: 2015},
	{requests: 0, promptTokens: 59861, uncachedTokens: 59861, measuredMs: 3752},
	{requests: 0, promptTokens: 112220, uncachedTokens: 112220, measuredMs: 6492},
	{requests: 0, promptTokens: 38621, uncachedTokens: 3088, measuredMs: 975},
	{requests: 0, promptTokens: 95848, uncachedTokens: 2664, measuredMs: 1382},
	{requests: 0, promptTokens: 40160, uncachedTokens: 2982, measuredMs: 1086},
	{requests: 1, promptTokens: 42696, uncachedTokens: 3122, measuredMs: 1403},
	{requests: 2, promptTokens: 66145, uncachedTokens: 3073, measuredMs: 1622},
	{requests: 3, promptTokens: 65789, uncachedTokens: 3022, measuredMs: 1799},
	{requests: 4, promptTokens: 41837, uncachedTokens: 2991, measuredMs: 1907},
	{requests: 5, promptTokens: 40825, uncachedTokens: 3077, measuredMs: 2067},
	{requests: 6, promptTokens: 39529, uncachedTokens: 2919, measuredMs: 2140},
}

func TestPrefillSteps_MatchesMeasuredEngine(t *testing.T) {
	errs := make([]float64, 0, len(measuredCases))
	for _, c := range measuredCases {
		got := testSteps.ttftMs(c.requests, c.uncachedTokens, c.promptTokens-c.uncachedTokens, 0)
		err := math.Abs(got-c.measuredMs) / c.measuredMs
		assert.LessOrEqualf(t, err, 0.30, "requests=%d prompt=%d uncached=%d: estimate %.0f ms, measured %.0f ms",
			c.requests, c.promptTokens, c.uncachedTokens, got, c.measuredMs)
		errs = append(errs, err)
	}
	slices.Sort(errs)
	assert.LessOrEqual(t, errs[len(errs)/2], 0.15, "median error")
}

func TestPrefillSteps_TTFT(t *testing.T) {
	tests := []struct {
		name                           string
		requests, tokens, cachedTokens int64
		readUs, loadMs, requestMs      float64
		want                           float64
	}{
		{name: "a short request takes one step and the handoff", tokens: 10, want: 150},
		{name: "tokens fill whole steps", tokens: 2001, want: 3*100 + 50},
		{name: "no tokens still take a step", tokens: 0, want: 150},
		{name: "every request in flight costs a step", requests: 3, tokens: 10, want: 3*100 + 150},
		{name: "or requestMs when set", requests: 3, tokens: 10, requestMs: 40, want: 3*40 + 150},
		{name: "a prompt longer than one step shares its steps", requests: 1, tokens: 1001, want: 100 + 3*100 + 50},
		{name: "with at most RequestsPerStep requests", requests: 3, tokens: 1001, want: 3*100 + 3*100 + 50},
		{name: "the whole prompt is read", tokens: 1000, cachedTokens: 4000, readUs: 2, want: 10 + 150},
		{name: "the wait for the cached prefix comes first", tokens: 10, loadMs: 20, want: 20 + 150},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &PrefillSteps{TokensPerStep: 1000, RequestsPerStep: 2, StepMs: 100, RequestMs: tt.requestMs, HandoffMs: 50, ReadMicrosecondsPerToken: tt.readUs}
			assert.InDelta(t, tt.want, s.ttftMs(tt.requests, tt.tokens, tt.cachedTokens, tt.loadMs), 1e-9)
		})
	}
}

// makeStepEndpoint creates an endpoint that holds cachedBlocks of the
// request's 100 blocks of 100 tokens, with the given pending requests and
// tokens.
func makeStepEndpoint(name string, cachedBlocks int, requests, tokens int64) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name, Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, 100, 100))
	ep.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: requests, PendingRequests: requests, Tokens: tokens})
	ep.Put(attrconcurrency.UncachedRequestTokensDataKey, &attrconcurrency.UncachedRequestTokens{Tokens: int64(100-cachedBlocks) * 100})
	return ep
}

func stepsConfig() Config {
	return Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 1000, TTFTSource: TTFTSourcePrefillSteps, PrefillSteps: testSteps}
}

// Twelve short pending requests cost the request 2.4 s on the warm endpoint,
// about 3.3 s in all, against two steps (about 1.2 s) to compute the 10K-token
// prompt on the idle one. Their 2400 tokens read as
// 0.2 s at 12500 tokens/s.
func TestFilter_PrefillStepsBreaksOnQueuedRequests(t *testing.T) {
	p := newTestPlugin(stepsConfig())
	endpoints := []fwksched.Endpoint{
		makeStepEndpoint("warm", 100, 12, 2400),
		makeStepEndpoint("idle", 0, 0, 0),
	}
	assert.Len(t, p.Filter(context.Background(), nil, endpoints), 2)

	p = newTestPlugin(Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 1000, TTFTSource: TTFTSourcePrefillThroughput, PeakPrefillThroughput: 12500, TTFTEstimate: TTFTEstimateFull})
	assert.Len(t, p.Filter(context.Background(), nil, endpoints), 1, "the throughput estimate does not count requests")
}

// One pending request costs the request 200 ms on the warm endpoint;
// computing the 10K-token prompt on the idle endpoint takes two steps.
func TestFilter_PrefillStepsKeepsWarmEndpoint(t *testing.T) {
	p := newTestPlugin(stepsConfig())
	endpoints := []fwksched.Endpoint{
		makeStepEndpoint("warm", 100, 1, 16384),
		makeStepEndpoint("idle", 0, 0, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Len(t, result, 1)
	assert.Equal(t, "warm", result[0].GetMetadata().ID.Name)
}

// The warm endpoint holds the whole 10K-token prompt, 9K of it outside GPU
// memory. The 1.8 s restore takes longer than computing the prompt on the idle
// endpoint.
func TestFilter_PrefillStepsCountsRestore(t *testing.T) {
	cfg := stepsConfig()
	steps := *testSteps
	steps.RestoreMs = 1800
	cfg.PrefillSteps = &steps
	p := newTestPlugin(cfg)

	withTiers := func(gpuBlocks int) []fwksched.Endpoint {
		warm := makeStepEndpoint("warm", 100, 0, 0)
		warm.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(100, 100, 100).
			WithCachedBlocksByTier(map[string]int{kvcache.GPUTier: gpuBlocks, "cpu": 100}))
		return []fwksched.Endpoint{warm, makeStepEndpoint("idle", 0, 0, 0)}
	}
	assert.Len(t, p.Filter(context.Background(), nil, withTiers(10)), 2)
	assert.Len(t, p.Filter(context.Background(), nil, withTiers(100)), 1, "a prefix in GPU memory needs no restore")
}

// The endpoint holds the first 90 of the 100K-token prompt's 100 blocks, 40 of
// them in GPU memory and the other 50 only in CPU memory. The in-flight load
// producer counts cached blocks unweighted, so it reports the 10K tokens past
// the cached prefix as uncached. The endpoint restores the 50 blocks and
// computes the 10K tokens in two steps: read, restore, two steps and the
// handoff.
func TestEndpointTTFT_PrefillStepsMixedTierPrefix(t *testing.T) {
	cfg := stepsConfig()
	steps := *testSteps
	steps.RestoreMs = 1000
	cfg.PrefillSteps = &steps
	p := newTestPlugin(cfg)

	ep := makeStepEndpoint("warm", 75, 0, 0)
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(75, 100, 1000).
		WithCachedBlockCount(90).
		WithCachedBlocksByTier(map[string]int{kvcache.GPUTier: 40, "cpu": 90}))
	ep.Put(attrconcurrency.UncachedRequestTokensDataKey, &attrconcurrency.UncachedRequestTokens{Tokens: 10000})

	ttft, ok := p.endpointTTFT(ep)
	assert.True(t, ok)
	assert.InDelta(t, 500+1000+2*345+500, ttft, 1e-9)
}

// Speculative index entries count toward the cached prefix but not as
// offloaded blocks, so an endpoint whose confirmed blocks are all in GPU
// memory pays no restore.
func TestEndpointTTFT_PrefillStepsSpeculativePrefix(t *testing.T) {
	cfg := stepsConfig()
	steps := *testSteps
	steps.RestoreMs = 1000
	cfg.PrefillSteps = &steps
	p := newTestPlugin(cfg)

	ep := makeStepEndpoint("warm", 75, 0, 0)
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(75, 100, 1000).
		WithCachedBlockCount(90).
		WithConfirmedCachedBlockCount(40).
		WithCachedBlocksByTier(map[string]int{kvcache.GPUTier: 40, attrprefix.SpeculativeTierKey: 50}))
	ep.Put(attrconcurrency.UncachedRequestTokensDataKey, &attrconcurrency.UncachedRequestTokens{Tokens: 10000})

	ttft, ok := p.endpointTTFT(ep)
	assert.True(t, ok)
	assert.InDelta(t, 500+2*345+500, ttft, 1e-9)
}

// Without tier data the prefix producer reports no blocks outside GPU memory,
// so the endpoint pays no restore: read, one step and the handoff.
func TestEndpointTTFT_PrefillStepsRestoresNothingWithoutTiers(t *testing.T) {
	cfg := stepsConfig()
	steps := *testSteps
	steps.RestoreMs = 1000
	cfg.PrefillSteps = &steps
	p := newTestPlugin(cfg)

	ttft, ok := p.endpointTTFT(makeStepEndpoint("warm", 100, 0, 0))
	assert.True(t, ok)
	assert.InDelta(t, 50+345+500, ttft, 1e-9)
}

// Requests decoding on an aggregated engine are in flight but not pending, so
// only the 2 pending requests cost requestMs: read, 2 x 200 ms, one step and
// the handoff.
func TestEndpointTTFT_PrefillStepsChargesPendingRequests(t *testing.T) {
	p := newTestPlugin(stepsConfig())
	ep := makeStepEndpoint("warm", 100, 0, 0)
	ep.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: 20, PendingRequests: 2})

	ttft, ok := p.endpointTTFT(ep)
	assert.True(t, ok)
	assert.InDelta(t, 50+2*200+345+500, ttft, 1e-9)
}

// 10 requests are pending on an engine that runs 4 at a time and completes 2
// per second. With ttftEstimate full, the 6 queued requests wait 3 s in place
// of requestMs, and the 4 others cost 200 ms each. With backlog all 10 cost
// requestMs.
func TestEndpointTTFT_PrefillStepsAddsQueueWait(t *testing.T) {
	ep := makeStepEndpoint("warm", 100, 0, 0)
	ep.Put(attrconcurrency.InFlightLoadDataKey,
		&attrconcurrency.InFlightLoad{Requests: 10, PendingRequests: 10, CompletionsPerSecond: 2})
	base := 50.0 + 345 + 500

	cfg := stepsConfig()
	cfg.TTFTEstimate = TTFTEstimateFull
	cfg.MaxNumSeqs = 4
	ttft, ok := newTestPlugin(cfg).endpointTTFT(ep)
	assert.True(t, ok)
	assert.InDelta(t, base+4*200+3000, ttft, 1e-9)

	cfg.TTFTEstimate = TTFTEstimateBackLog
	ttft, ok = newTestPlugin(cfg).endpointTTFT(ep)
	assert.True(t, ok)
	assert.InDelta(t, base+10*200, ttft, 1e-9)
}

func TestFilter_PrefillStepsMissingUncachedTokens(t *testing.T) {
	p := newTestPlugin(stepsConfig())
	warm := makeStepEndpoint("warm", 100, 12, 2400)
	idle := makeEndpoint("idle", 0, -1, 0, -1)
	assert.Len(t, p.Filter(context.Background(), nil, []fwksched.Endpoint{warm, idle}), 2, "missing signal keeps all endpoints")
}

func TestConsumes_PrefillSteps(t *testing.T) {
	p := newTestPlugin(stepsConfig())
	consumed := p.Consumes()
	_, ok := consumed.Required[p.inFlightLoadDataKey]
	assert.True(t, ok)
	_, ok = consumed.Required[p.uncachedRequestDataKey]
	assert.True(t, ok)
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.False(t, ok)
}

func TestFactory_PrefillSteps(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftSource": "prefillSteps", "prefillSteps": {"tokensPerStep": 8192, "requestsPerStep": 4, "stepMs": 345, "requestMs": 200, "readMicrosecondsPerToken": 5, "handoffMs": 500}}`)), nil)
	assert.NoError(t, err)
	assert.Equal(t, testSteps, plugin.(*Plugin).config.PrefillSteps)
}

func TestFactory_PrefillStepsNeedsNoPeakThroughput(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftSource": "prefillSteps", "peakPrefillThroughput": 0, "prefillSteps": {"tokensPerStep": 8192, "requestsPerStep": 4, "stepMs": 345}}`)), nil)
	assert.NoError(t, err)
}

func TestFactory_InvalidPrefillSteps(t *testing.T) {
	tests := []struct{ name, params, want string }{
		{"profile missing", `{"ttftSource": "prefillSteps"}`, "prefillSteps must be set"},
		{"no tokens per step", `{"ttftSource": "prefillSteps", "prefillSteps": {"requestsPerStep": 4, "stepMs": 345}}`, "tokensPerStep must be > 0"},
		{"no requests per step", `{"ttftSource": "prefillSteps", "prefillSteps": {"tokensPerStep": 8192, "stepMs": 345}}`, "requestsPerStep must be > 0"},
		{"no step time", `{"ttftSource": "prefillSteps", "prefillSteps": {"tokensPerStep": 8192, "requestsPerStep": 4}}`, "stepMs must be > 0"},
		{"negative request time", `{"ttftSource": "prefillSteps", "prefillSteps": {"tokensPerStep": 8192, "requestsPerStep": 4, "stepMs": 345, "requestMs": -1}}`, "requestMs must be >= 0"},
		{"negative read cost", `{"ttftSource": "prefillSteps", "prefillSteps": {"tokensPerStep": 8192, "requestsPerStep": 4, "stepMs": 345, "readMicrosecondsPerToken": -1}}`, "readMicrosecondsPerToken must be >= 0"},
		{"negative restore cost", `{"ttftSource": "prefillSteps", "prefillSteps": {"tokensPerStep": 8192, "requestsPerStep": 4, "stepMs": 345, "restoreMs": -1}}`, "restoreMs must be >= 0"},
		{"negative handoff", `{"ttftSource": "prefillSteps", "prefillSteps": {"tokensPerStep": 8192, "requestsPerStep": 4, "stepMs": 345, "handoffMs": -1}}`, "handoffMs must be >= 0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Factory("test", fwkplugin.StrictDecoder([]byte(tt.params)), nil)
			assert.ErrorContains(t, err, tt.want)
		})
	}
}
