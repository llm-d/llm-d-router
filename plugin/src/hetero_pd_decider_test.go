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
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/hetero"
)

const (
	heteroClassA30   = "nvidia-a30"
	heteroClassGaudi = "intel-gaudi3"
	heteroPrompt     = 2000
)

// heteroLabConfig models one A30 prefill pod feeding Gaudi 3 decode pods over 25 GbE.
// Throughputs are placeholders, not measurements.
func heteroLabConfig() HeteroPDDeciderConfig {
	c := defaultHeteroPDDeciderConfig()
	c.Strategy = hetero.StrategyModel // these tests exercise the cost model directly
	// KV compatibility has its own tests below; the cost-model tests leave it out.
	c.Compatibility = hetero.CompatibilityOff
	c.TTFTSLOMs = 1000
	c.BandwidthGbps = 25
	c.Profiles = map[string]hetero.Profile{
		heteroClassA30:   {PrefillTokensPerSec: 5000},
		heteroClassGaudi: {PrefillTokensPerSec: 20000},
	}
	return c
}

func heteroEndpoint(name, role, class string, running, waiting int, kvUsage float64, extra map[string]string) scheduling.Endpoint {
	labels := map[string]string{bylabel.RoleLabel: role, defaultAcceleratorLabel: class}
	for k, v := range extra {
		labels[k] = v
	}
	m := fwkdl.NewMetrics()
	m.RunningRequestsSize = running
	m.WaitingQueueSize = waiting
	m.KVCacheUsagePercent = kvUsage
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Namespace: "default", Name: name},
			Address: testEndpointAddr,
			Port:    testEndpointPort,
			Labels:  labels,
		},
		m,
		fwkdl.NewAttributes(),
	)
}

func newTestHeteroDecider(t *testing.T, c HeteroPDDeciderConfig) *HeteroPDDecider {
	t.Helper()
	d, err := NewHeteroPDDecider(c)
	require.NoError(t, err)
	return d.WithName("hetero")
}

func TestHeteroPDDeciderLabScenarios(t *testing.T) {
	a30Idle := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, nil)
	a30Queued := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 3, 0, nil)

	tests := []struct {
		name    string
		tokens  int
		decode  scheduling.Endpoint
		pool    []scheduling.Endpoint
		produce bool
		want    bool
	}{
		{
			name:    "busy decode pod offloads a long prompt",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil),
			pool:    []scheduling.Endpoint{a30Idle},
			produce: true,
			want:    true,
		},
		{
			name:    "idle decode pod prefills locally",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 0, 0, 0.1, nil),
			pool:    []scheduling.Endpoint{a30Idle},
			produce: true,
			want:    false,
		},
		{
			name:    "short prompt stays local",
			tokens:  100,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil),
			pool:    []scheduling.Endpoint{a30Idle},
			produce: true,
			want:    false,
		},
		{
			name:    "queued prefill pod would miss the SLO",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil),
			pool:    []scheduling.Endpoint{a30Queued},
			produce: true,
			want:    false,
		},
		{
			name:    "decode KV nearly full",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.95, nil),
			pool:    []scheduling.Endpoint{a30Idle},
			produce: true,
			want:    false,
		},
		{
			name:    "decode pods are not prefill candidates",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil),
			pool:    []scheduling.Endpoint{heteroEndpoint("gaudi-1", bylabel.RoleDecode, heteroClassGaudi, 0, 0, 0, nil)},
			produce: true,
			want:    false,
		},
		{
			name:    "prefill-decode peer is a candidate",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RolePrefillDecode, heteroClassGaudi, 40, 0, 0.3, nil),
			pool:    []scheduling.Endpoint{heteroEndpoint("gaudi-1", bylabel.RolePrefillDecode, heteroClassGaudi, 0, 0, 0, nil)},
			produce: true,
			want:    true,
		},
		{
			name:    "disabled prefill pod is skipped",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil),
			pool:    []scheduling.Endpoint{heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, map[string]string{defaultDisableLabel: "true"})},
			produce: true,
			want:    false,
		},
		{
			name:    "no prefill view stays local",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil),
			produce: false,
			want:    false,
		},
		{
			name:    "unprofiled decode class uses the fallback threshold",
			tokens:  heteroPrompt,
			decode:  heteroEndpoint("x-0", bylabel.RoleDecode, "unprofiled", 40, 0, 0.3, nil),
			pool:    []scheduling.Endpoint{a30Idle},
			produce: true,
			want:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := newTestHeteroDecider(t, heteroLabConfig())
			ctx := context.Background()
			if tt.produce {
				require.NoError(t, d.Produce(ctx, nil, append([]scheduling.Endpoint{tt.decode}, tt.pool...)))
			}
			assert.Equal(t, tt.want, d.disaggregate(ctx, makeRequestWithTokens(tt.tokens), tt.decode))
		})
	}
}

func TestHeteroPDDeciderSubtractsCachedPrefix(t *testing.T) {
	d := newTestHeteroDecider(t, heteroLabConfig())
	ctx := context.Background()
	decode := heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil)
	pool := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, nil)
	require.NoError(t, d.Produce(ctx, nil, []scheduling.Endpoint{decode, pool}))
	require.True(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode))

	// 1900 of 2000 tokens cached leaves 100 uncached, under minUncachedTokens.
	decode.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(1900, heteroPrompt, 1))
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode))
}

func TestHeteroPDDeciderFailsSoft(t *testing.T) {
	d := newTestHeteroDecider(t, heteroLabConfig())
	ctx := context.Background()
	decode := heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil)
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), nil))
	assert.False(t, d.disaggregate(ctx, &scheduling.InferenceRequest{}, decode))
	decode.Put(attrprefix.PrefixCacheMatchInfoDataKey, &notPrefixCacheMatchInfo{})
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode))
}

func TestHeteroPDDeciderRecordsReason(t *testing.T) {
	d := newTestHeteroDecider(t, heteroLabConfig())
	ctx := context.Background()
	decode := heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, nil)
	pool := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, nil)
	require.NoError(t, d.Produce(ctx, nil, []scheduling.Endpoint{decode, pool}))

	counter := LlmdHeteroPDDecisionCount.WithLabelValues("hetero", hetero.ReasonRemoteBetter, "true")
	before := testutil.ToFloat64(counter)
	require.True(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode))
	assert.InDelta(t, before+1, testutil.ToFloat64(counter), 1e-9)
}

func TestHeteroPDDeciderFactory(t *testing.T) {
	raw := `{
		"ttftSLOMs": 1000,
		"bandwidthGbps": 25,
		"stallWeight": 0.25,
		"acceleratorLabel": "example.com/accel",
		"profiles": {"nvidia-a30": {"prefillTokensPerSec": 5000}}
	}`
	p, err := HeteroPDDeciderPluginFactory("h", json.NewDecoder(strings.NewReader(raw)), nil)
	require.NoError(t, err)
	d, ok := p.(*HeteroPDDecider)
	require.True(t, ok)
	assert.Equal(t, "h", d.TypedName().Name)
	assert.Equal(t, HeteroPDDeciderPluginType, d.TypedName().Type)
	assert.Equal(t, "example.com/accel", d.config.AcceleratorLabel)
	assert.InDelta(t, 1000.0, d.config.TTFTSLOMs, 1e-9)
	assert.Equal(t, 131072, int(d.config.KVBytesPerToken))
	assert.InDelta(t, 5000.0, d.config.Profiles["nvidia-a30"].PrefillTokensPerSec, 1e-9)

	p, err = HeteroPDDeciderPluginFactory("h", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, defaultAcceleratorLabel, p.(*HeteroPDDecider).config.AcceleratorLabel)

	for _, bad := range []string{
		`{"acceleratorLabel": ""}`,
		`{"bandwidthGbps": 0}`,
		`{"safetyMargin": 1}`,
		`{"fallbackNonCachedTokens": -1}`,
		`{"profiles": {"x": {"prefillTokensPerSec": 0}}}`,
		`{"bandwidthGbps": "fast"}`,
	} {
		_, err := HeteroPDDeciderPluginFactory("h", json.NewDecoder(strings.NewReader(bad)), nil)
		assert.Error(t, err, bad)
	}
}

func TestHeteroPDDeciderConsumes(t *testing.T) {
	d := newTestHeteroDecider(t, heteroLabConfig())
	required := d.Consumes().Required
	assert.Contains(t, required, d.prefixMatchInfoDataKey())
	assert.Len(t, required, 5)
	assert.Empty(t, d.Produces())
}

func TestHeteroPDDeciderDefaultsToAuto(t *testing.T) {
	assert.Equal(t, hetero.StrategyAuto, defaultHeteroPDDeciderConfig().Strategy)
}

// compatLabConfig is the lab config with the compatibility rule on.
func compatLabConfig() HeteroPDDeciderConfig {
	c := heteroLabConfig()
	c.Compatibility = hetero.CompatibilityAuto
	return c
}

func TestHeteroPDDeciderCompatibility(t *testing.T) {
	key := func(k string) map[string]string { return map[string]string{defaultCompatibilityLabel: k} }
	busyGaudi := func(extra map[string]string) scheduling.Endpoint {
		return heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, extra)
	}
	a30With := func(extra map[string]string) scheduling.Endpoint {
		return heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, extra)
	}
	tests := []struct {
		name     string
		approved []string
		decode   scheduling.Endpoint
		prefill  scheduling.Endpoint
		want     bool
	}{
		{"mixed pool without keys keeps local", nil, busyGaudi(nil), a30With(nil), false},
		{"equal keys split", nil, busyGaudi(key("k1")), a30With(key("k1")), true},
		{"different keys keep local", nil, busyGaudi(key("gaudi")), a30With(key("a30")), false},
		{"approved pair splits", []string{"a30->gaudi"}, busyGaudi(key("gaudi")), a30With(key("a30")), true},
		{"approval is directional", []string{"gaudi->a30"}, busyGaudi(key("gaudi")), a30With(key("a30")), false},
		{"live approval label splits", nil,
			busyGaudi(map[string]string{defaultCompatibilityLabel: "gaudi", defaultAcceptLabelPrefix + "a30": "true"}), a30With(key("a30")), true},
		{"live approval for another key does not", nil,
			busyGaudi(map[string]string{defaultCompatibilityLabel: "gaudi", defaultAcceptLabelPrefix + "t4": "true"}), a30With(key("a30")), false},
		{"live approval label must be true", nil,
			busyGaudi(map[string]string{defaultCompatibilityLabel: "gaudi", defaultAcceptLabelPrefix + "a30": "false"}), a30With(key("a30")), false},
		{"legacy fingerprint label is read", nil,
			busyGaudi(map[string]string{defaultFingerprintLabel: "f"}), a30With(map[string]string{defaultFingerprintLabel: "f"}), true},
		{"one key missing in a mixed pool keeps local", nil, busyGaudi(key("k1")), a30With(nil), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := compatLabConfig()
			c.ApprovedPairs = tt.approved
			d := newTestHeteroDecider(t, c)
			ctx := context.Background()
			require.NoError(t, d.Produce(ctx, nil, []scheduling.Endpoint{tt.decode, tt.prefill}))
			assert.Equal(t, tt.want, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), tt.decode))
			got := d.CompatiblePrefill(tt.decode, []scheduling.Endpoint{tt.prefill})
			assert.Equal(t, tt.want, len(got) == 1, "filter and decider must agree")
		})
	}
}

func TestHeteroPDDeciderFallbackNeedsCompatiblePod(t *testing.T) {
	d := newTestHeteroDecider(t, compatLabConfig())
	ctx := context.Background()
	decode := heteroEndpoint("x-0", bylabel.RoleDecode, "unprofiled", 40, 0, 0.3, nil)
	pool := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, nil)
	require.NoError(t, d.Produce(ctx, nil, []scheduling.Endpoint{decode, pool}))
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode),
		"the fallback threshold must not split toward an incompatible pod")
}

func TestHeteroKVCompatFilter(t *testing.T) {
	d := newTestHeteroDecider(t, compatLabConfig())
	f := &HeteroKVCompatFilter{typedName: d.TypedName(), decider: d}
	ctx := context.Background()
	decode := heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, map[string]string{defaultCompatibilityLabel: "k1"})
	same := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, map[string]string{defaultCompatibilityLabel: "k1"})
	other := heteroEndpoint("a30-1", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, map[string]string{defaultCompatibilityLabel: "k2"})
	pool := []scheduling.Endpoint{same, other}

	req := makeRequestWithTokens(heteroPrompt)
	assert.Equal(t, pool, f.Filter(ctx, req, pool), "without a peer endpoint candidates pass")
	assert.Equal(t, pool, f.Filter(ctx, nil, pool))

	req.PutAttribute(PeerEndpointAttributeKey, decode)
	assert.Equal(t, []scheduling.Endpoint{same}, f.Filter(ctx, req, pool))
	assert.Empty(t, f.Filter(ctx, req, []scheduling.Endpoint{other}), "fails closed")

	unresolved := &HeteroKVCompatFilter{}
	assert.Equal(t, pool, unresolved.Filter(ctx, req, pool), "no decider: candidates pass unfiltered")
	assert.Contains(t, f.Consumes().Optional, PeerEndpointAttributeKey)
}

func TestHeteroKVCompatFilterFactory(t *testing.T) {
	p, err := HeteroKVCompatFilterFactory("", nil, nil)
	require.NoError(t, err)
	f := p.(*HeteroKVCompatFilter)
	assert.Equal(t, HeteroKVCompatFilterType, f.TypedName().Name)
	assert.Equal(t, HeteroPDDeciderPluginType, f.deciderName)
	p, err = HeteroKVCompatFilterFactory("c", json.NewDecoder(strings.NewReader(`{"decider":"h"}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, "h", p.(*HeteroKVCompatFilter).deciderName)
	_, err = HeteroKVCompatFilterFactory("c", json.NewDecoder(strings.NewReader(`{"decider":1}`)), nil)
	assert.Error(t, err)
}

func pdResult(decode, prefill scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{decode}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{prefill}},
		},
	}
}

func TestHeteroPDDeciderExcludesFailingPair(t *testing.T) {
	c := compatLabConfig()
	c.FailureThreshold, c.ExclusionSeconds = 2, 60
	d := newTestHeteroDecider(t, c)
	t0 := time.Unix(1000, 0)
	d.now = func() time.Time { return t0 }
	ctx := context.Background()
	k := map[string]string{defaultCompatibilityLabel: "k1"}
	decode := heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, k)
	pool := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, k)
	require.NoError(t, d.Produce(ctx, nil, []scheduling.Endpoint{decode, pool}))
	require.True(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode))

	exclusions := LlmdHeteroPDExclusionCount.WithLabelValues("hetero")
	before := testutil.ToFloat64(exclusions)
	fail := func(id string) {
		req := makeRequestWithTokens(heteroPrompt)
		req.RequestID = id
		require.NoError(t, d.PreRequest(ctx, req, pdResult(decode, pool)))
		d.ResponseHeader(ctx, req, &fwkrc.Response{Headers: map[string]string{":status": "502"}}, nil)
		// The end of the same response must not count it twice.
		d.ResponseBody(ctx, req, &fwkrc.Response{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseError}, nil)
	}
	fail("r1")
	assert.True(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode), "one failure is below the threshold")
	fail("r2")
	assert.InDelta(t, before+1, testutil.ToFloat64(exclusions), 1e-9)
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode), "excluded pair keeps requests local")
	assert.Empty(t, d.CompatiblePrefill(decode, []scheduling.Endpoint{pool}), "the filter applies the exclusion too")

	t0 = t0.Add(61 * time.Second)
	assert.True(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode), "exclusion ends after the TTL")
}

func TestHeteroPDDeciderResponseOutcomes(t *testing.T) {
	c := compatLabConfig()
	c.FailureThreshold, c.ExclusionSeconds = 1, 60
	d := newTestHeteroDecider(t, c)
	ctx := context.Background()
	k := map[string]string{defaultCompatibilityLabel: "k1"}
	decode := heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, k)
	pool := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, k)
	pair := hetero.PairKey(d.toPod(pool), d.toPod(decode))
	start := func(id string) *scheduling.InferenceRequest {
		req := makeRequestWithTokens(heteroPrompt)
		req.RequestID = id
		require.NoError(t, d.PreRequest(ctx, req, pdResult(decode, pool)))
		return req
	}

	// Success: a 200 header, then a natural end.
	req := start("ok")
	d.ResponseHeader(ctx, req, &fwkrc.Response{Headers: map[string]string{":status": "200"}}, nil)
	d.ResponseBody(ctx, req, &fwkrc.Response{EndOfStream: false}, nil)
	d.ResponseBody(ctx, req, &fwkrc.Response{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseNatural}, nil)
	_, pending := d.inflight.Load("ok")
	assert.False(t, pending)
	assert.False(t, d.exclusions.Excluded(pair, time.Now()))

	// A client disconnect says nothing about the pair.
	req = start("gone")
	d.ResponseBody(ctx, req, &fwkrc.Response{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseClientDisconnect}, nil)
	_, pending = d.inflight.Load("gone")
	assert.False(t, pending)
	assert.False(t, d.exclusions.Excluded(pair, time.Now()))

	// An error end counts, read from the response's ID when the request is absent.
	start("err")
	d.ResponseBody(ctx, nil, &fwkrc.Response{RequestID: "err", EndOfStream: true, TerminationCause: fwkrc.TerminationCauseError}, nil)
	assert.True(t, d.exclusions.Excluded(pair, time.Now()))

	// Decode-only results and nil inputs are ignored.
	decodeOnly := &scheduling.SchedulingResult{PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{"decode": {TargetEndpoints: []scheduling.Endpoint{decode}}}}
	req = makeRequestWithTokens(heteroPrompt)
	req.RequestID = "local"
	require.NoError(t, d.PreRequest(ctx, req, decodeOnly))
	require.NoError(t, d.PreRequest(ctx, nil, decodeOnly))
	require.NoError(t, d.PreRequest(ctx, req, nil))
	_, pending = d.inflight.Load("local")
	assert.False(t, pending)
	d.ResponseHeader(ctx, req, nil, nil)
	d.ResponseBody(ctx, req, nil, nil)
	d.ResponseHeader(ctx, req, &fwkrc.Response{Headers: map[string]string{"status": "503"}}, nil) // unknown ID
}

func TestHeteroPDDeciderExclusionDisabled(t *testing.T) {
	c := compatLabConfig()
	c.FailureThreshold = 0
	d := newTestHeteroDecider(t, c)
	ctx := context.Background()
	k := map[string]string{defaultCompatibilityLabel: "k1"}
	decode := heteroEndpoint("gaudi-0", bylabel.RoleDecode, heteroClassGaudi, 40, 0, 0.3, k)
	pool := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, k)
	req := makeRequestWithTokens(heteroPrompt)
	req.RequestID = "x"
	require.NoError(t, d.PreRequest(ctx, req, pdResult(decode, pool)))
	_, pending := d.inflight.Load("x")
	assert.False(t, pending, "nothing is tracked when exclusion is off")
}

func TestHeteroPDDeciderReadsMeasuredSpeedLabel(t *testing.T) {
	d := newTestHeteroDecider(t, heteroLabConfig())
	speed := func(v string) map[string]string { return map[string]string{defaultSpeedLabel: v} }
	cases := []struct {
		label string
		want  float64
	}{
		{"7200", 7200}, {"7200.5", 7200.5}, {"", 0}, {"fast", 0}, {"0", 0}, {"-5", 0}, {"+Inf", 0},
	}
	for _, c := range cases {
		ep := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, speed(c.label))
		assert.Equal(t, c.want, d.toPod(ep).PrefillTokensPerSec, "label %q", c.label)
	}
	cfg := heteroLabConfig()
	cfg.SpeedLabel = ""
	off := newTestHeteroDecider(t, cfg)
	ep := heteroEndpoint("a30-0", bylabel.RolePrefill, heteroClassA30, 0, 0, 0, speed("7200"))
	assert.Zero(t, off.toPod(ep).PrefillTokensPerSec, "an empty speedLabel turns the label off")
}
