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

package disagg

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	testclock "k8s.io/utils/clock/testing"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/test/utils"
)

const testNixlHealthDeciderName = "nixl-health"

// countingPDDecider returns allow and counts its calls.
type countingPDDecider struct {
	allow bool
	calls int
}

func (m *countingPDDecider) TypedName() plugin.TypedName { return plugin.TypedName{} }

func (m *countingPDDecider) disaggregate(_ context.Context, _ *scheduling.InferenceRequest, _ scheduling.Endpoint) bool {
	m.calls++
	return m.allow
}

func makeDecodeEndpoint(name string) scheduling.Endpoint {
	return makeEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: name}, "10.0.0.1", testPodPort, nil)
}

func setFailedTransfers(ep scheduling.Endpoint, value float64) {
	ep.Put(attrmetrics.NixlFailedTransfersDataKey, attrmetrics.ScalarMetricValue(value))
}

func newTestNixlHealthDecider(t *testing.T, inner deciderPlugin, threshold int) (*NixlHealthPDDecider, *testclock.FakeClock) {
	t.Helper()
	d, err := NewNixlHealthPDDecider(inner, threshold, time.Minute)
	require.NoError(t, err)
	clock := testclock.NewFakeClock(time.Unix(1_000_000, 0))
	d.clock = clock
	return d, clock
}

func TestNixlHealthPDDeciderFactory(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := handleWithDeciders(ctx)

	tests := []struct {
		name          string
		params        string
		wantErr       bool
		wantThreshold int
		wantWindow    time.Duration
	}{
		{
			name:          "defaults",
			params:        `{"decider": "prefix-based-pd-decider"}`,
			wantThreshold: defaultNixlFailedTransfersThreshold,
			wantWindow:    defaultNixlFailureWindow,
		},
		{
			name:          "explicit threshold and window",
			params:        `{"decider": "always-disagg-pd-decider", "failedTransfersThreshold": 2, "window": "30s"}`,
			wantThreshold: 2,
			wantWindow:    30 * time.Second,
		},
		{name: "missing decider", params: `{}`, wantErr: true},
		{name: "unknown decider", params: `{"decider": "no-such-plugin"}`, wantErr: true},
		{name: "decider is not a PD decider", params: `{"decider": "` + PrefixBasedPDDeciderPluginType + `-handler"}`, wantErr: true},
		{name: "zero threshold", params: `{"decider": "prefix-based-pd-decider", "failedTransfersThreshold": 0}`, wantErr: true},
		{name: "negative threshold", params: `{"decider": "prefix-based-pd-decider", "failedTransfersThreshold": -1}`, wantErr: true},
		{name: "zero window", params: `{"decider": "prefix-based-pd-decider", "window": "0s"}`, wantErr: true},
		{name: "unknown field", params: `{"decider": "prefix-based-pd-decider", "threshold": 3}`, wantErr: true},
		{name: "invalid JSON", params: `{"decider": `, wantErr: true},
	}

	// A plugin that exists but is not a PD decider.
	notDecider, err := HandlerFactory("h", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(PrefixBasedPDDeciderPluginType+"-handler", notDecider)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := NixlHealthPDDeciderPluginFactory(testNixlHealthDeciderName,
				plugin.StrictDecoder(json.RawMessage(tt.params)), handle)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, p)
				return
			}
			require.NoError(t, err)
			d, ok := p.(*NixlHealthPDDecider)
			require.True(t, ok)
			assert.Equal(t, plugin.TypedName{Type: NixlHealthPDDeciderPluginType, Name: testNixlHealthDeciderName}, d.TypedName())
			assert.InDelta(t, float64(tt.wantThreshold), d.threshold, 0)
			assert.Equal(t, tt.wantWindow, d.window)
		})
	}
}

func TestNixlHealthPDDeciderConfigParser_DeclaresDeciderDependency(t *testing.T) {
	cfg, err := NixlHealthPDDeciderConfigParser(
		plugin.StrictDecoder(json.RawMessage(`{"decider": "prefix-based-pd-decider"}`)), nil)
	require.NoError(t, err)
	parsed, ok := cfg.(NixlHealthPDDeciderConfig)
	require.True(t, ok)
	assert.Equal(t, "prefix-based-pd-decider", parsed.Decider)
}

func TestNixlHealthPDDecider_DelegatesWhenCounterAbsent(t *testing.T) {
	ctx := utils.NewTestContext(t)
	for _, allow := range []bool{true, false} {
		inner := &countingPDDecider{allow: allow}
		d, _ := newTestNixlHealthDecider(t, inner, 1)
		ep := makeDecodeEndpoint("pod1")

		assert.Equal(t, allow, d.disaggregate(ctx, completionsRequest(testLongPrompt), ep))
		assert.Equal(t, 1, inner.calls)
	}
}

func TestNixlHealthPDDecider_DelegatesForNilEndpoint(t *testing.T) {
	ctx := utils.NewTestContext(t)
	inner := &countingPDDecider{allow: true}
	d, _ := newTestNixlHealthDecider(t, inner, 1)

	assert.True(t, d.disaggregate(ctx, completionsRequest(testLongPrompt), nil))
	assert.Equal(t, 1, inner.calls)
}

// Failures counted before the decider first evaluated the endpoint are history,
// not a current failure rate.
func TestNixlHealthPDDecider_FirstObservationIsBaseline(t *testing.T) {
	ctx := utils.NewTestContext(t)
	inner := &countingPDDecider{allow: true}
	d, _ := newTestNixlHealthDecider(t, inner, 1)
	ep := makeDecodeEndpoint("pod1")
	setFailedTransfers(ep, 44)

	assert.True(t, d.disaggregate(ctx, completionsRequest(testLongPrompt), ep))
	assert.True(t, d.disaggregate(ctx, completionsRequest(testLongPrompt), ep))
}

func TestNixlHealthPDDecider_ThresholdWithinWindow(t *testing.T) {
	ctx := utils.NewTestContext(t)
	inner := &countingPDDecider{allow: true}
	d, clock := newTestNixlHealthDecider(t, inner, 3)
	ep := makeDecodeEndpoint("pod1")
	req := completionsRequest(testLongPrompt)

	setFailedTransfers(ep, 10)
	require.True(t, d.disaggregate(ctx, req, ep), "baseline")

	clock.Step(10 * time.Second)
	setFailedTransfers(ep, 12)
	assert.True(t, d.disaggregate(ctx, req, ep), "2 failures in the window are below the threshold of 3")

	clock.Step(10 * time.Second)
	setFailedTransfers(ep, 13)
	callsBefore := inner.calls
	assert.False(t, d.disaggregate(ctx, req, ep), "3 failures in the window reach the threshold")
	assert.Equal(t, callsBefore, inner.calls, "the wrapped decider is not consulted for a failing endpoint")

	clock.Step(10 * time.Second)
	assert.False(t, d.disaggregate(ctx, req, ep), "failures stay in the window without new increases")
}

func TestNixlHealthPDDecider_FailuresAgeOutOfWindow(t *testing.T) {
	ctx := utils.NewTestContext(t)
	inner := &countingPDDecider{allow: true}
	d, clock := newTestNixlHealthDecider(t, inner, 2)
	ep := makeDecodeEndpoint("pod1")
	req := completionsRequest(testLongPrompt)

	setFailedTransfers(ep, 0)
	require.True(t, d.disaggregate(ctx, req, ep))

	clock.Step(time.Second)
	setFailedTransfers(ep, 1)
	require.True(t, d.disaggregate(ctx, req, ep))

	clock.Step(50 * time.Second)
	setFailedTransfers(ep, 2)
	require.False(t, d.disaggregate(ctx, req, ep), "two failures 50s apart are inside a 60s window")

	clock.Step(15 * time.Second)
	assert.True(t, d.disaggregate(ctx, req, ep), "the first failure aged out, leaving one")

	clock.Step(time.Minute)
	assert.True(t, d.disaggregate(ctx, req, ep), "no failures left in the window")
}

// A counter that goes down means the engine restarted. The failures counted
// before the restart are dropped and the post-restart count is new failures.
func TestNixlHealthPDDecider_CounterReset(t *testing.T) {
	ctx := utils.NewTestContext(t)
	inner := &countingPDDecider{allow: true}
	d, clock := newTestNixlHealthDecider(t, inner, 3)
	ep := makeDecodeEndpoint("pod1")
	req := completionsRequest(testLongPrompt)

	setFailedTransfers(ep, 40)
	require.True(t, d.disaggregate(ctx, req, ep))
	clock.Step(time.Second)
	setFailedTransfers(ep, 43)
	require.False(t, d.disaggregate(ctx, req, ep))

	clock.Step(time.Second)
	setFailedTransfers(ep, 0)
	assert.True(t, d.disaggregate(ctx, req, ep), "a restart clears the failures counted before it")

	clock.Step(time.Second)
	setFailedTransfers(ep, 2)
	assert.True(t, d.disaggregate(ctx, req, ep), "2 failures after the restart")

	clock.Step(time.Second)
	setFailedTransfers(ep, 1)
	assert.True(t, d.disaggregate(ctx, req, ep), "a second restart that already failed once")
	clock.Step(time.Second)
	setFailedTransfers(ep, 3)
	assert.False(t, d.disaggregate(ctx, req, ep), "1 failure at the restart plus 2 after it reach the threshold")
}

func TestNixlHealthPDDecider_EndpointsAreIndependent(t *testing.T) {
	ctx := utils.NewTestContext(t)
	inner := &countingPDDecider{allow: true}
	d, clock := newTestNixlHealthDecider(t, inner, 1)
	failing := makeDecodeEndpoint("pod1")
	healthy := makeDecodeEndpoint("pod2")
	req := completionsRequest(testLongPrompt)

	setFailedTransfers(failing, 0)
	setFailedTransfers(healthy, 0)
	require.True(t, d.disaggregate(ctx, req, failing))
	require.True(t, d.disaggregate(ctx, req, healthy))

	clock.Step(time.Second)
	setFailedTransfers(failing, 5)
	assert.False(t, d.disaggregate(ctx, req, failing))
	assert.True(t, d.disaggregate(ctx, req, healthy))
}

func TestNixlHealthPDDecider_ForgetsIdleEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	inner := &countingPDDecider{allow: true}
	d, clock := newTestNixlHealthDecider(t, inner, 1)
	req := completionsRequest(testLongPrompt)

	gone := makeDecodeEndpoint("replaced-pod")
	setFailedTransfers(gone, 7)
	require.True(t, d.disaggregate(ctx, req, gone))

	active := makeDecodeEndpoint("pod1")
	setFailedTransfers(active, 0)
	for range 3 {
		clock.Step(time.Minute)
		require.True(t, d.disaggregate(ctx, req, active))
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	assert.NotContains(t, d.endpoints, gone.GetMetadata().ID)
	assert.Contains(t, d.endpoints, active.GetMetadata().ID)
}

// The scheduler evaluates requests concurrently; run under -race.
func TestNixlHealthPDDecider_ConcurrentRequests(t *testing.T) {
	ctx := utils.NewTestContext(t)
	d, err := NewNixlHealthPDDecider(newAlwaysDisaggPDDecider(), 1, time.Minute)
	require.NoError(t, err)
	endpoints := []scheduling.Endpoint{makeDecodeEndpoint("pod1"), makeDecodeEndpoint("pod2")}

	var wg sync.WaitGroup
	for worker := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 100 {
				ep := endpoints[(worker+i)%len(endpoints)]
				setFailedTransfers(ep, float64(i))
				d.disaggregate(ctx, completionsRequest(testLongPrompt), ep)
			}
		}()
	}
	wg.Wait()
}

func TestNixlHealthPDDecider_DataDeclarations(t *testing.T) {
	prefixDecider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{
		NonCachedTokens:             4,
		PrefixMatchInfoProducerName: "precise-prefix-cache-producer",
	})
	require.NoError(t, err)

	t.Run("wraps a decider with declarations", func(t *testing.T) {
		d, err := NewNixlHealthPDDecider(prefixDecider, 1, time.Minute)
		require.NoError(t, err)

		consumed := d.deciderConsumes()
		assert.Contains(t, consumed.Optional, attrmetrics.NixlFailedTransfersDataKey)
		assert.Equal(t, attrmetrics.ScalarMetricValue(0), consumed.Optional[attrmetrics.NixlFailedTransfersDataKey])
		for key := range prefixDecider.Consumes().Required {
			assert.Contains(t, consumed.Required, key)
		}
		assert.Equal(t, prefixDecider.Produces(), d.Produces())
		assert.Equal(t, prefixDecider.prefixMatchInfoDataKey(), d.prefixMatchInfoDataKey())
	})

	t.Run("wraps a decider without declarations", func(t *testing.T) {
		d, err := NewNixlHealthPDDecider(newAlwaysDisaggPDDecider(), 1, time.Minute)
		require.NoError(t, err)

		consumed := d.deciderConsumes()
		assert.Empty(t, consumed.Required)
		assert.Contains(t, consumed.Optional, attrmetrics.NixlFailedTransfersDataKey)
		assert.Empty(t, d.Produces())
		assert.Equal(t, attrprefix.PrefixCacheMatchInfoDataKey, d.prefixMatchInfoDataKey())
	})
}

// The handler reads the prefix producer and the optional NIXL key through the
// wrapper, so the data graph sees what the wrapped decider needs.
func TestHandler_Consumes_NixlHealthDecider(t *testing.T) {
	prefixDecider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{
		NonCachedTokens:             4,
		PrefixMatchInfoProducerName: "precise-prefix-cache-producer",
	})
	require.NoError(t, err)
	d, err := NewNixlHealthPDDecider(prefixDecider, 1, time.Minute)
	require.NoError(t, err)

	h := NewDisaggProfileHandler(defaultDecodeProfile, DefaultPrefillProfile, defaultEncodeProfile, d, nil)
	consumed := h.Consumes()
	assert.Contains(t, consumed.Required, prefixDecider.prefixMatchInfoDataKey())
	assert.NotContains(t, consumed.Required, attrprefix.PrefixCacheMatchInfoDataKey)
	assert.Contains(t, consumed.Required, tokenproducer.TokenizedPromptDataKey)
	assert.Contains(t, consumed.Optional, attrmetrics.NixlFailedTransfersDataKey)
}

type tokenizedPromptTestProducer struct{}

func (p *tokenizedPromptTestProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "token-producer", Name: "token-producer"}
}

func (p *tokenizedPromptTestProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: fwkrequest.TokenizedRequest{}}
}

func (p *tokenizedPromptTestProducer) Produce(_ context.Context, _ *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	return nil
}

type prefixMatchTestProducer struct{}

func (p *prefixMatchTestProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "approx-prefix-cache-producer", Name: "approx-prefix-cache-producer"}
}

func (p *prefixMatchTestProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{attrprefix.PrefixCacheMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{}}
}

func (p *prefixMatchTestProducer) Produce(_ context.Context, _ *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	return nil
}

// The decider runs only inside the handler, so the startup data graph must
// accept it alongside the request-control producers of what it reads.
func TestNixlHealthPDDecider_DataGraphAcceptsConfiguration(t *testing.T) {
	prefixDecider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 4})
	require.NoError(t, err)
	d, err := NewNixlHealthPDDecider(prefixDecider, 1, time.Minute)
	require.NoError(t, err)
	h := NewDisaggProfileHandler(defaultDecodeProfile, DefaultPrefillProfile, defaultEncodeProfile, d, nil)

	_, err = datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{
		&tokenizedPromptTestProducer{}, &prefixMatchTestProducer{}, prefixDecider, d.WithName(testNixlHealthDeciderName), h,
	})
	require.NoError(t, err)
}

// A failing decode endpoint completes decode-only. The skip is a decline, so
// the request does not fail the way a required prefill with no endpoint does.
func TestHandler_PD_NixlFailingDecodeEndpointServesDecodeOnly(t *testing.T) {
	ctx := utils.NewTestContext(t)
	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		DefaultPrefillProfile: &mockProfile{},
	}
	d, clock := newTestNixlHealthDecider(t, newAlwaysDisaggPDDecider(), 1)
	h := NewDisaggProfileHandler(defaultDecodeProfile, DefaultPrefillProfile, defaultEncodeProfile, d, nil)

	decodeResults := map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: makeProfileRunResult("pod1")}
	decodeEndpoint := decodeResults[defaultDecodeProfile].TargetEndpoints[0]

	setFailedTransfers(decodeEndpoint, 0)
	got := h.Pick(ctx, completionsRequest(testLongPrompt), profiles, decodeResults)
	require.ElementsMatch(t, []string{DefaultPrefillProfile}, profileNames(got), "healthy endpoint runs prefill")

	clock.Step(time.Second)
	setFailedTransfers(decodeEndpoint, 1)
	req := completionsRequest(testLongPrompt)
	decodeResults = map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: decodeResults[defaultDecodeProfile]}
	got = h.Pick(ctx, req, profiles, decodeResults)
	assert.ElementsMatch(t, []string{}, profileNames(got), "failing endpoint skips prefill")

	res, err := h.ProcessResults(ctx, req, decodeResults)
	require.NoError(t, err)
	assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
	assert.NotContains(t, res.ProfileResults, DefaultPrefillProfile)
}

// The handler factory resolves the wrapper like any other PD decider.
func TestHandlerFactory_NixlHealthDecider(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := handleWithDeciders(ctx)
	d, err := NixlHealthPDDeciderPluginFactory(testNixlHealthDeciderName,
		plugin.StrictDecoder(json.RawMessage(`{"decider": "prefix-based-pd-decider"}`)), handle)
	require.NoError(t, err)
	handle.AddPlugin(testNixlHealthDeciderName, d)

	p, err := HandlerFactory("h", plugin.StrictDecoder(json.RawMessage(
		`{"deciders": {"prefill": "`+testNixlHealthDeciderName+`"}}`)), handle)
	require.NoError(t, err)
	h, ok := p.(*Handler)
	require.True(t, ok)
	assert.Same(t, d, h.pdDecider)
}
