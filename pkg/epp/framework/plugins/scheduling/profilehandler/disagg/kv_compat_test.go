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

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/test/utils"
)

type fakeDecider struct {
	result bool
	calls  int
}

func (f *fakeDecider) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "fake-decider", Name: "fake"}
}

func (f *fakeDecider) disaggregate(context.Context, *scheduling.InferenceRequest, scheduling.Endpoint) bool {
	f.calls++
	return f.result
}

func kvEndpoint(name, role, key string, extra map[string]string) scheduling.Endpoint {
	labels := map[string]string{bylabel.RoleLabel: role}
	if key != "" {
		labels[defaultKVCompatKeyLabel] = key
	}
	for k, v := range extra {
		labels[k] = v
	}
	return makeEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: name}, "10.0.0.1", "8000", labels)
}

// dlEndpoint returns the datalayer endpoint an endpoint event carries for ep.
func dlEndpoint(ep scheduling.Endpoint) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(ep.GetMetadata(), nil)
}

func newKVCompatDecider(t *testing.T, inner deciderPlugin, params KVCompatPDDeciderParameters, eps ...scheduling.Endpoint) *KVCompatPDDecider {
	t.Helper()
	d, err := NewKVCompatPDDecider(inner, params)
	require.NoError(t, err)
	for _, ep := range eps {
		require.NoError(t, d.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: dlEndpoint(ep)}))
	}
	return d
}

func TestKVCompatPDDeciderDecision(t *testing.T) {
	accept := func(key string) map[string]string {
		return map[string]string{defaultKVCompatAcceptLabelPrefix + key: "true"}
	}
	tests := []struct {
		name      string
		params    KVCompatPDDeciderParameters
		decode    scheduling.Endpoint
		prefill   []scheduling.Endpoint
		wantInner bool
	}{
		{"no endpoint has a key", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "", nil), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "", nil)}, true},
		{"equal keys", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "k1", nil), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k1", nil)}, true},
		{"different keys", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "k1", nil), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k2", nil)}, false},
		{"one compatible endpoint among several", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "k1", nil),
			[]scheduling.Endpoint{kvEndpoint("p1", bylabel.RolePrefill, "k2", nil), kvEndpoint("p2", bylabel.RolePrefill, "k1", nil)}, true},
		{"decode endpoint without a key while keys are in use", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "", nil), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k1", nil)}, false},
		{"accept label on the decode endpoint", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "k1", accept("k2")), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k2", nil)}, true},
		{"accept label for another key", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "k1", accept("k3")), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k2", nil)}, false},
		{"accept labels disabled", KVCompatPDDeciderParameters{AcceptLabelPrefix: "-"},
			kvEndpoint("d", bylabel.RoleDecode, "k1", accept("k2")), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k2", nil)}, false},
		{"approved pair", KVCompatPDDeciderParameters{ApprovedPairs: []string{"k2->k1"}},
			kvEndpoint("d", bylabel.RoleDecode, "k1", nil), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k2", nil)}, true},
		{"approval applies in one direction", KVCompatPDDeciderParameters{ApprovedPairs: []string{"k1->k2"}},
			kvEndpoint("d", bylabel.RoleDecode, "k1", nil), []scheduling.Endpoint{kvEndpoint("p", bylabel.RolePrefill, "k2", nil)}, false},
		{"decode endpoints do not count as prefill candidates", KVCompatPDDeciderParameters{},
			kvEndpoint("d", bylabel.RoleDecode, "k1", nil), []scheduling.Endpoint{kvEndpoint("d2", bylabel.RoleDecode, "k1", nil)}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := &fakeDecider{result: true}
			d := newKVCompatDecider(t, inner, tt.params, append(tt.prefill, tt.decode)...)
			got := d.disaggregate(context.Background(), makeRequestWithTokens(2000), tt.decode)
			assert.Equal(t, tt.wantInner, got)
			assert.Equal(t, tt.wantInner, inner.calls == 1, "the wrapped decider runs only when a compatible prefill endpoint exists")
		})
	}
}

func TestKVCompatPDDeciderReturnsWrappedAnswer(t *testing.T) {
	inner := &fakeDecider{result: false}
	decode := kvEndpoint("d", bylabel.RoleDecode, "k1", nil)
	d := newKVCompatDecider(t, inner, KVCompatPDDeciderParameters{}, decode, kvEndpoint("p", bylabel.RolePrefill, "k1", nil))
	assert.False(t, d.disaggregate(context.Background(), makeRequestWithTokens(2000), decode))
	assert.Equal(t, 1, inner.calls)
	assert.False(t, d.disaggregate(context.Background(), makeRequestWithTokens(2000), nil))
}

func TestKVCompatPDDeciderIndexFollowsEvents(t *testing.T) {
	decode := kvEndpoint("d", bylabel.RoleDecode, "k1", nil)
	prefill := kvEndpoint("p", bylabel.RolePrefill, "k1", nil)
	d := newKVCompatDecider(t, &fakeDecider{result: true}, KVCompatPDDeciderParameters{}, decode, prefill)
	ctx := context.Background()
	require.True(t, d.disaggregate(ctx, makeRequestWithTokens(2000), decode))

	relabelled := kvEndpoint("p", bylabel.RolePrefill, "k2", nil)
	require.NoError(t, d.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: dlEndpoint(relabelled)}))
	assert.False(t, d.disaggregate(ctx, makeRequestWithTokens(2000), decode), "a key change replaces the old key")

	require.NoError(t, d.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: dlEndpoint(relabelled)}))
	require.NoError(t, d.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: dlEndpoint(decode)}))
	assert.False(t, d.keysInUse(), "deleting every keyed endpoint turns the rule off")
	assert.Empty(t, d.prefillKeys)
	require.NoError(t, d.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate}))
}

func TestKVCompatPDDeciderCountsDeclines(t *testing.T) {
	decode := kvEndpoint("d", bylabel.RoleDecode, "k1", nil)
	d := newKVCompatDecider(t, &fakeDecider{result: true}, KVCompatPDDeciderParameters{},
		decode, kvEndpoint("p", bylabel.RolePrefill, "k2", nil))
	d.typedName.Name = "kv-compat-declines-test"
	before := testutil.ToFloat64(LlmdKVCompatDeclinedCount.WithLabelValues(d.typedName.Name))
	d.disaggregate(context.Background(), makeRequestWithTokens(2000), decode)
	assert.Equal(t, before+1, testutil.ToFloat64(LlmdKVCompatDeclinedCount.WithLabelValues(d.typedName.Name)))
}

func TestKVCompatFilter(t *testing.T) {
	decode := kvEndpoint("d", bylabel.RoleDecode, "k1", map[string]string{defaultKVCompatAcceptLabelPrefix + "k3": "true"})
	same := kvEndpoint("p1", bylabel.RolePrefill, "k1", nil)
	other := kvEndpoint("p2", bylabel.RolePrefill, "k2", nil)
	accepted := kvEndpoint("p3", bylabel.RolePrefill, "k3", nil)
	unkeyed := kvEndpoint("p4", bylabel.RolePrefill, "", nil)
	candidates := []scheduling.Endpoint{same, other, accepted, unkeyed}
	d := newKVCompatDecider(t, &fakeDecider{}, KVCompatPDDeciderParameters{}, decode, same, other, accepted, unkeyed)
	f := &KVCompatFilter{typedName: plugin.TypedName{Type: KVCompatFilterPluginType, Name: "f"}, decider: d}
	ctx := context.Background()

	request := makeRequestWithTokens(2000)
	assert.Equal(t, candidates, f.Filter(ctx, request, candidates), "no decode pick: every candidate passes")
	assert.Equal(t, candidates, f.Filter(ctx, nil, candidates))

	request.PutAttribute(PeerEndpointAttributeKey, decode)
	assert.Equal(t, []scheduling.Endpoint{same, accepted}, f.Filter(ctx, request, candidates))

	request.PutAttribute(PeerEndpointAttributeKey, kvEndpoint("d2", bylabel.RoleDecode, "k9", nil))
	assert.Empty(t, f.Filter(ctx, request, candidates), "an incompatible candidate is never kept")

	unkeyedPool := newKVCompatDecider(t, &fakeDecider{}, KVCompatPDDeciderParameters{}, unkeyed)
	f.decider = unkeyedPool
	request.PutAttribute(PeerEndpointAttributeKey, kvEndpoint("d3", bylabel.RoleDecode, "", nil))
	assert.Equal(t, candidates, f.Filter(ctx, request, candidates), "without keys the filter keeps every candidate")
}

func TestKVCompatFactories(t *testing.T) {
	handle := utils.NewTestHandle(context.Background())
	stock, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 16})
	require.NoError(t, err)
	handle.AddPlugin("stock", stock)
	handle.AddPlugin("fake", &fakeDecider{})

	decode := func(raw string) *json.Decoder { return json.NewDecoder(strings.NewReader(raw)) }
	p, err := KVCompatPDDeciderPluginFactory("compat", decode(`{"decider":"stock","approvedPairs":["a->b"]}`), handle)
	require.NoError(t, err)
	d := p.(*KVCompatPDDecider)
	assert.Equal(t, plugin.TypedName{Type: KVCompatPDDeciderPluginType, Name: "compat"}, d.TypedName())
	assert.Equal(t, stock.prefixMatchInfoDataKey(), d.prefixMatchInfoDataKey(), "the wrapped decider's prefix key is forwarded")
	handle.AddPlugin("compat", d)

	fake, err := NewKVCompatPDDecider(&fakeDecider{}, KVCompatPDDeciderParameters{})
	require.NoError(t, err)
	assert.Equal(t, attrprefix.PrefixCacheMatchInfoDataKey, fake.prefixMatchInfoDataKey())

	for name, raw := range map[string]string{
		"missing decider":    `{}`,
		"unknown decider":    `{"decider":"nope"}`,
		"not a decider":      `{"decider":"compat-filter-target"}`,
		"bad approved pair":  `{"decider":"stock","approvedPairs":["a-b"]}`,
		"empty approved key": `{"decider":"stock","approvedPairs":["->b"]}`,
		"invalid json":       `{"decider":`,
	} {
		t.Run(name, func(t *testing.T) {
			handle.AddPlugin("compat-filter-target", &KVCompatFilter{})
			_, err := KVCompatPDDeciderPluginFactory("x", decode(raw), handle)
			assert.Error(t, err)
		})
	}
	_, err = KVCompatPDDeciderPluginFactory("x", nil, handle)
	assert.Error(t, err)
	_, err = KVCompatPDDeciderPluginFactory("x", decode(`{"decider":"stock"}`), nil)
	assert.Error(t, err)
	_, err = NewKVCompatPDDecider(nil, KVCompatPDDeciderParameters{})
	assert.Error(t, err)

	fp, err := KVCompatFilterPluginFactory("", decode(`{"decider":"compat"}`), handle)
	require.NoError(t, err)
	assert.Equal(t, plugin.TypedName{Type: KVCompatFilterPluginType, Name: KVCompatFilterPluginType}, fp.TypedName())
	assert.Contains(t, fp.(*KVCompatFilter).Consumes().Optional, PeerEndpointAttributeKey)
	for _, raw := range []string{`{}`, `{"decider":"stock"}`, `{"decider":`} {
		_, err := KVCompatFilterPluginFactory("f", decode(raw), handle)
		assert.Error(t, err, raw)
	}
	_, err = KVCompatFilterPluginFactory("f", decode(`{"decider":"compat"}`), nil)
	assert.Error(t, err)
}

func TestKVCompatPDDeciderRegistersEndpointSource(t *testing.T) {
	d, err := NewKVCompatPDDecider(&fakeDecider{}, KVCompatPDDeciderParameters{})
	require.NoError(t, err)
	r := &recordingRegistrar{}
	require.NoError(t, d.RegisterDependencies(r))
	require.Len(t, r.got, 1)
	assert.Equal(t, d, r.got[0].Extractor)
	assert.NotNil(t, r.got[0].DefaultSource)
}

type recordingRegistrar struct {
	got []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(p fwkdl.PendingRegistration) error {
	r.got = append(r.got, p)
	return nil
}
