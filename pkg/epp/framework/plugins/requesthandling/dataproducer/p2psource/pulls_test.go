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

package p2psource

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

// newTrackingProducer builds a producer through the factory with
// trackInFlightPulls set and a fresh gauge.
func newTrackingProducer(t *testing.T, name string) *trackingProducer {
	t.Helper()
	inFlightPulls.Reset()
	t.Cleanup(inFlightPulls.Reset)
	ctx, cancel := context.WithCancel(utils.NewTestContext(t))
	t.Cleanup(cancel)
	dec := json.NewDecoder(strings.NewReader(`{"trackInFlightPulls": true}`))
	p, err := PluginFactory(name, dec, utils.NewTestHandle(ctx))
	require.NoError(t, err)
	return p.(*trackingProducer)
}

// pull runs Produce and PreRequest for a request whose best source, pod-b,
// out-caches the computing pod, pod-a, so the source is written.
func pull(t *testing.T, p *Producer, requestID string) *scheduling.InferenceRequest {
	t.Helper()
	ctx := utils.NewTestContext(t)
	computing := endpoint(p, "pod-a", "10.0.0.1", 1)
	req := &scheduling.InferenceRequest{RequestID: requestID, Headers: map[string]string{}, Body: bodyWith(map[string]any{})}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{computing, endpoint(p, "pod-b", "10.0.0.2", 3)}))
	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(computing)))
	assertSource(t, req, "10.0.0.2", defaultP2PConnectorPort)
	return req
}

func chunk(start, end bool) *fwkrh.Response {
	return &fwkrh.Response{StartOfStream: start, EndOfStream: end}
}

// requirePulls checks the gauge: one pod-b series with the given value, or
// no series at all when want is 0.
func requirePulls(t *testing.T, producerName string, want int) {
	t.Helper()
	if want == 0 {
		require.Equal(t, 0, promtestutil.CollectAndCount(inFlightPulls))
		return
	}
	expected := fmt.Sprintf(`
# HELP llm_d_epp_p2p_inflight_pulls [ALPHA] Current number of P2P KV cache pulls directed at each source endpoint, as tracked by the p2p-source-producer.
# TYPE llm_d_epp_p2p_inflight_pulls gauge
llm_d_epp_p2p_inflight_pulls{endpoint_name="pod-b",namespace="",producer_name=%q} %d
`, producerName, want)
	require.NoError(t, promtestutil.CollectAndCompare(inFlightPulls, strings.NewReader(expected), "llm_d_epp_p2p_inflight_pulls"))
}

// Only a tracking producer takes the response and abandonment hooks.
func TestPluginFactory_TrackInFlightPulls(t *testing.T) {
	var on plugin.Plugin = newTrackingProducer(t, "test")
	assert.Implements(t, (*fwkrh.ResponseBodyProcessor)(nil), on)
	assert.Implements(t, (*fwkrh.RequestAbandoned)(nil), on)

	off, err := PluginFactory("test", nil, nil)
	require.NoError(t, err)
	assert.Nil(t, off.(*Producer).pulls)
	_, ok := off.(fwkrh.ResponseBodyProcessor)
	assert.False(t, ok)
	_, ok = off.(fwkrh.RequestAbandoned)
	assert.False(t, ok)

	_, err = PluginFactory("test", json.NewDecoder(strings.NewReader(`{"trackInFlightPulls": true}`)), nil)
	require.Error(t, err)
}

func TestProduce_StashesSourceID(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{})
	req := &scheduling.InferenceRequest{RequestID: "req-id"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		endpoint(p, "pod-a", "10.0.0.1", 1),
		endpoint(p, "pod-b", "10.0.0.2", 3),
	}))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, "pod-b", best.id.Name)
}

// A streamed request's pull is released at its first response chunk; later
// chunks, the end of stream and an abandonment do not release it again.
func TestInFlightPulls_ReleasedAtFirstChunk(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newTrackingProducer(t, "p2p")

	first := pull(t, p.Producer, "req-1")
	second := pull(t, p.Producer, "req-2")
	requirePulls(t, "p2p", 2)

	p.ResponseBody(ctx, first, chunk(true, false), nil)
	requirePulls(t, "p2p", 1)

	p.ResponseBody(ctx, first, chunk(false, false), nil)
	p.ResponseBody(ctx, first, chunk(false, true), nil)
	p.RequestAbandoned(ctx, first)
	requirePulls(t, "p2p", 1)

	p.ResponseBody(ctx, second, chunk(true, true), nil)
	requirePulls(t, "p2p", 0)
}

// A request that ends without a first chunk is released at end of stream.
func TestInFlightPulls_ReleasedAtEndOfStream(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newTrackingProducer(t, "test")

	req := pull(t, p.Producer, "req-eos")
	requirePulls(t, "test", 1)

	p.ResponseBody(ctx, req, chunk(false, true), nil)
	requirePulls(t, "test", 0)
}

func TestInFlightPulls_ReleasedOnAbandon(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newTrackingProducer(t, "test")

	req := pull(t, p.Producer, "req-abandoned")
	requirePulls(t, "test", 1)

	p.RequestAbandoned(ctx, req)
	requirePulls(t, "test", 0)
}

// A second PreRequest for the same request does not count the pull twice.
func TestInFlightPulls_OnePullPerRequest(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newTrackingProducer(t, "test")

	req := pull(t, p.Producer, "req-twice")
	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(endpoint(p.Producer, "pod-a", "10.0.0.1", 1))))
	requirePulls(t, "test", 1)

	p.ResponseBody(ctx, req, chunk(true, false), nil)
	requirePulls(t, "test", 0)
}

// Without a source header there is no pull to count.
func TestInFlightPulls_NoHeaderNoCount(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newTrackingProducer(t, "test")

	computing := endpoint(p.Producer, "pod-a", "10.0.0.1", 3)
	req := &scheduling.InferenceRequest{RequestID: "req-local", Headers: map[string]string{}, Body: bodyWith(map[string]any{})}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{computing, endpoint(p.Producer, "pod-b", "10.0.0.2", 3)}))
	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(computing)))
	require.NotContains(t, req.Headers, routing.KVCacheSourceHeader)

	requirePulls(t, "test", 0)
}

// With tracking off, a pull is not counted.
func TestInFlightPulls_Off(t *testing.T) {
	inFlightPulls.Reset()
	p, err := PluginFactory("test", nil, nil)
	require.NoError(t, err)

	pull(t, p.(*Producer), "req-off")
	requirePulls(t, "test", 0)
}
