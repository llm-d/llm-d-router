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
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/hetero"
	"github.com/llm-d/llm-d-router/test/utils"
)

const (
	heteroClassA30   = "nvidia-a30"
	heteroClassGaudi = "intel-gaudi3"
	heteroPrompt     = 2000
)

// heteroLabConfig models A30 prefill endpoints feeding Gaudi 3 decode endpoints over 25 GbE.
// The throughputs are illustrative values.
func heteroLabConfig() HeteroPDDeciderConfig {
	c := defaultHeteroPDDeciderConfig()
	c.Strategy = hetero.StrategyModel
	c.TTFTSLOMs = 1000
	c.BandwidthGbps = 25
	c.Profiles = map[string]hetero.Profile{
		heteroClassA30:   {PrefillTokensPerSec: 5000},
		heteroClassGaudi: {PrefillTokensPerSec: 20000},
	}
	return c
}

// heteroEndpoint returns the scheduling and datalayer views of one endpoint.
func heteroEndpoint(name, role, class string, running, waiting int, extra map[string]string) (scheduling.Endpoint, fwkdl.Endpoint) {
	labels := map[string]string{bylabel.RoleLabel: role, defaultAcceleratorLabel: class}
	for k, v := range extra {
		labels[k] = v
	}
	md := &fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: name}, Address: name, Labels: labels}
	m := fwkdl.NewMetrics()
	m.RunningRequestsSize = running
	m.WaitingQueueSize = waiting
	return scheduling.NewEndpoint(md, m, fwkdl.NewAttributes()), fwkdl.NewEndpoint(md, m)
}

func newHeteroDecider(t testing.TB, c HeteroPDDeciderConfig, eps ...fwkdl.Endpoint) *HeteroPDDecider {
	t.Helper()
	d, err := NewHeteroPDDecider(c)
	require.NoError(t, err)
	for _, ep := range eps {
		require.NoError(t, d.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	}
	d.refresh(context.Background())
	return d
}

func candidateNames(d *HeteroPDDecider, decodeID string) []string {
	ns, name, _ := strings.Cut(decodeID, "/")
	candidates := d.snapshot.Load().byDecode[k8stypes.NamespacedName{Namespace: ns, Name: name}].candidates
	names := make([]string, 0, len(candidates))
	for _, p := range candidates {
		names = append(names, p.Name)
	}
	return names
}

func TestHeteroPDDeciderLabScenarios(t *testing.T) {
	tests := []struct {
		name       string
		running    int
		waiting    int
		tokens     int
		wantRemote bool
	}{
		{"long prompt, busy decode endpoint offloads", 40, 0, heteroPrompt, true},
		{"long prompt, idle decode endpoint prefills locally", 0, 0, heteroPrompt, false},
		{"short prompt stays local", 40, 0, 100, false},
		{"prefill queue breaks the SLO", 40, 3, heteroPrompt, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, prefill := heteroEndpoint("a30", bylabel.RolePrefill, heteroClassA30, 0, tt.waiting, nil)
			decode, decodeDL := heteroEndpoint("gaudi", bylabel.RoleDecode, heteroClassGaudi, tt.running, 0, nil)
			d := newHeteroDecider(t, heteroLabConfig(), prefill, decodeDL)
			assert.Equal(t, tt.wantRemote, d.disaggregate(context.Background(), makeRequestWithTokens(tt.tokens), decode))
		})
	}
}

func TestHeteroPDDeciderRanksCandidates(t *testing.T) {
	c := heteroLabConfig()
	c.MaxCandidates = 2
	_, slow := heteroEndpoint("slow", bylabel.RolePrefill, heteroClassA30, 0, 0, nil)
	_, fast := heteroEndpoint("fast", bylabel.RolePrefill, heteroClassA30, 0, 0, map[string]string{defaultSpeedLabel: "40000"})
	_, queued := heteroEndpoint("queued", bylabel.RolePrefill, heteroClassA30, 0, 9, map[string]string{defaultSpeedLabel: "40000"})
	_, unprofiled := heteroEndpoint("unprofiled", bylabel.RolePrefill, "other", 0, 0, nil)
	_, decodeDL := heteroEndpoint("gaudi", bylabel.RoleDecode, heteroClassGaudi, 40, 0, nil)
	d := newHeteroDecider(t, c, slow, fast, queued, unprofiled, decodeDL)
	assert.Equal(t, []string{"default/fast", "default/slow"}, candidateNames(d, "default/gaudi"),
		"cheapest first, a long queue counts against an endpoint, and only MaxCandidates are kept")

	require.NoError(t, d.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: fast}))
	d.refresh(context.Background())
	assert.Equal(t, []string{"default/slow", "default/queued"}, candidateNames(d, "default/gaudi"))
	require.NoError(t, d.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate}))
}

func TestHeteroPDDeciderUsesKVCompat(t *testing.T) {
	_, compatible := heteroEndpoint("p1", bylabel.RolePrefill, heteroClassA30, 0, 0, map[string]string{defaultKVCompatKeyLabel: "k1"})
	_, other := heteroEndpoint("p2", bylabel.RolePrefill, heteroClassA30, 0, 0,
		map[string]string{defaultKVCompatKeyLabel: "k2", defaultSpeedLabel: "90000"})
	decode, decodeDL := heteroEndpoint("d", bylabel.RoleDecode, heteroClassGaudi, 40, 0, map[string]string{defaultKVCompatKeyLabel: "k1"})
	d := newHeteroDecider(t, heteroLabConfig(), compatible, other, decodeDL)
	assert.Equal(t, []string{"default/p2", "default/p1"}, candidateNames(d, "default/d"), "without kv-compat every endpoint is a candidate")

	handle := utils.NewTestHandle(context.Background())
	handle.AddPlugin("hetero", d)
	p, err := KVCompatPDDeciderPluginFactory("compat", json.NewDecoder(strings.NewReader(`{"decider":"hetero","failureThreshold":1}`)), handle)
	require.NoError(t, err)
	compat := p.(*KVCompatPDDecider)
	for _, ep := range []fwkdl.Endpoint{compatible, other, decodeDL} {
		require.NoError(t, compat.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	}
	d.refresh(context.Background())
	assert.Equal(t, []string{"default/p1"}, candidateNames(d, "default/d"), "the kv-compat rule limits the candidates")
	assert.True(t, compat.disaggregate(context.Background(), makeRequestWithTokens(heteroPrompt), decode))

	req := makeRequestWithTokens(heteroPrompt)
	pSched, _ := heteroEndpoint("p1", bylabel.RolePrefill, heteroClassA30, 0, 0, map[string]string{defaultKVCompatKeyLabel: "k1"})
	require.NoError(t, compat.PreRequest(context.Background(), req, &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{decode}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{pSched}},
		},
	}))
	compat.finish(context.Background(), req.RequestID, false)
	d.refresh(context.Background())
	assert.Empty(t, candidateNames(d, "default/d"), "an excluded pair is not a candidate")
	assert.False(t, d.disaggregate(context.Background(), makeRequestWithTokens(heteroPrompt), decode))
}

func TestHeteroPDDeciderFallbackAndErrors(t *testing.T) {
	ctx := context.Background()
	_, prefill := heteroEndpoint("a30", bylabel.RolePrefill, heteroClassA30, 0, 0, nil)
	decode, decodeDL := heteroEndpoint("x", bylabel.RoleDecode, "unknown", 40, 0, nil)
	d := newHeteroDecider(t, heteroLabConfig(), prefill, decodeDL)
	assert.True(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode), "an unprofiled decode class falls back to the threshold")
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(100), decode))

	c := heteroLabConfig()
	c.FallbackNonCachedTokens = 0
	off := newHeteroDecider(t, c, prefill, decodeDL)
	assert.False(t, off.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), decode))

	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), nil))
	assert.False(t, d.disaggregate(ctx, completionsRequest(""), decode), "a request without a tokenized prompt is served decode-only")
	bad, _ := heteroEndpoint("x", bylabel.RoleDecode, "unknown", 0, 0, nil)
	bad.Put(d.prefixMatchInfoDK, &KVCompatExclusions{})
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), bad))

	disabled, disabledDL := heteroEndpoint("g", bylabel.RoleDecode, heteroClassGaudi, 40, 0, map[string]string{defaultHeteroDisableLabel: "true"})
	dd := newHeteroDecider(t, heteroLabConfig(), prefill, disabledDL)
	assert.False(t, dd.disaggregate(ctx, makeRequestWithTokens(heteroPrompt), disabled))
}

func TestHeteroPDDeciderSubtractsCachedPrefix(t *testing.T) {
	_, prefill := heteroEndpoint("a30", bylabel.RolePrefill, heteroClassA30, 0, 0, nil)
	decode, decodeDL := heteroEndpoint("gaudi", bylabel.RoleDecode, heteroClassGaudi, 40, 0, nil)
	d := newHeteroDecider(t, heteroLabConfig(), prefill, decodeDL)
	decode.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(heteroPrompt-100, heteroPrompt, 1))
	assert.False(t, d.disaggregate(context.Background(), makeRequestWithTokens(heteroPrompt), decode), "100 uncached tokens stay local")
}

func TestHeteroPDDeciderRecordsReason(t *testing.T) {
	_, prefill := heteroEndpoint("a30", bylabel.RolePrefill, heteroClassA30, 0, 0, nil)
	decode, decodeDL := heteroEndpoint("gaudi", bylabel.RoleDecode, heteroClassGaudi, 40, 0, nil)
	d := newHeteroDecider(t, heteroLabConfig(), prefill, decodeDL)
	d.typedName.Name = t.Name()
	counter := LlmdHeteroPDDecisionCount.WithLabelValues(t.Name(), hetero.ReasonRemoteBetter, "true")
	before := testutil.ToFloat64(counter)
	require.True(t, d.disaggregate(context.Background(), makeRequestWithTokens(heteroPrompt), decode))
	assert.Equal(t, before+1, testutil.ToFloat64(counter))
}

func TestHeteroPDDeciderFactory(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	handle := utils.NewTestHandle(ctx)
	decode := func(raw string) *json.Decoder { return json.NewDecoder(strings.NewReader(raw)) }
	p, err := HeteroPDDeciderPluginFactory("h", decode(`{"refreshInterval":"5ms","profiles":{"a":{"prefillTokensPerSec":100}}}`), handle)
	require.NoError(t, err)
	d := p.(*HeteroPDDecider)
	assert.Equal(t, "h", d.TypedName().Name)
	assert.Equal(t, attrprefix.PrefixCacheMatchInfoDataKey, d.prefixMatchInfoDataKey())
	_, prefill := heteroEndpoint("a30", bylabel.RolePrefill, "a", 0, 0, nil)
	_, decodeDL := heteroEndpoint("d", bylabel.RoleDecode, "a", 0, 0, nil)
	for _, ep := range []fwkdl.Endpoint{prefill, decodeDL} {
		require.NoError(t, d.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	}
	assert.Eventually(t, func() bool { return len(candidateNames(d, "default/d")) == 1 }, time.Second, 5*time.Millisecond,
		"the refresh loop ranks the candidates in the background")
	cancel()

	for name, raw := range map[string]string{
		"invalid json":          `{"maxCandidates":`,
		"zero candidates":       `{"maxCandidates":0}`,
		"zero refresh":          `{"refreshInterval":"0s"}`,
		"empty class label":     `{"acceleratorLabel":""}`,
		"negative fallback":     `{"fallbackNonCachedTokens":-1}`,
		"invalid decision core": `{"kvBytesPerToken":0}`,
	} {
		_, err := HeteroPDDeciderPluginFactory("h", decode(raw), handle)
		assert.Error(t, err, name)
	}
	_, err = HeteroPDDeciderPluginFactory("h", nil, nil)
	assert.Error(t, err)
	_, err = HeteroPDDeciderPluginFactory("h", nil, handle)
	assert.NoError(t, err)

	r := &recordingRegistrar{}
	require.NoError(t, d.RegisterDependencies(r))
	assert.Len(t, r.got, 1)
}

// benchPool builds a pool of n prefill endpoints and one busy decode endpoint, refreshed once.
func benchPool(b *testing.B, n int, withCompat bool) (*HeteroPDDecider, deciderPlugin, scheduling.Endpoint) {
	b.Helper()
	c := heteroLabConfig()
	c.Strategy = hetero.StrategyAuto
	extra := func(i int) map[string]string {
		return map[string]string{defaultKVCompatKeyLabel: "k" + strconv.Itoa(i%2), defaultSpeedLabel: strconv.Itoa(4000 + i*10)}
	}
	eps := make([]fwkdl.Endpoint, 0, n+1)
	for i := range n {
		_, ep := heteroEndpoint("p"+strconv.Itoa(i), bylabel.RolePrefill, heteroClassA30, 0, i%4, extra(i))
		eps = append(eps, ep)
	}
	decode, decodeDL := heteroEndpoint("d", bylabel.RoleDecode, heteroClassGaudi, 32, 0, map[string]string{defaultKVCompatKeyLabel: "k0"})
	eps = append(eps, decodeDL)
	d := newHeteroDecider(b, c, eps...)
	if !withCompat {
		return d, d, decode
	}
	compat, err := NewKVCompatPDDecider(d, KVCompatPDDeciderParameters{FailureThreshold: ptr.To(3), ExclusionDuration: &metav1.Duration{Duration: time.Minute}})
	require.NoError(b, err)
	d.useKVCompat(compat)
	for _, ep := range eps {
		require.NoError(b, compat.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	}
	d.refresh(context.Background())
	return d, compat, decode
}

// BenchmarkHeteroPDDeciderDecision measures one decision on the request path. Its cost does not
// depend on the pool size, since the candidates are ranked in the background.
func BenchmarkHeteroPDDeciderDecision(b *testing.B) {
	for _, n := range []int{8, 64, 512, 2048} {
		for _, withCompat := range []bool{false, true} {
			_, decider, decode := benchPool(b, n, withCompat)
			req := makeRequestWithTokens(heteroPrompt)
			ctx := context.Background()
			b.Run(fmt.Sprintf("prefillEndpoints=%d/kvCompat=%v", n, withCompat), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					decider.disaggregate(ctx, req, decode)
				}
			})
		}
	}
}

// BenchmarkHeteroPDDeciderRefresh measures one background ranking pass.
func BenchmarkHeteroPDDeciderRefresh(b *testing.B) {
	for _, n := range []int{8, 64, 512, 2048} {
		d, _, _ := benchPool(b, n, true)
		ctx := context.Background()
		b.Run(fmt.Sprintf("prefillEndpoints=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				d.refresh(ctx)
			}
		})
	}
}
