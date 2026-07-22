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
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

const testBlockSize = 16

// endpoint builds a candidate carrying the producer's PrefixCacheMatchInfo
// with the given unweighted cached-block count.
func endpoint(p *Producer, name, address string, cachedBlocks int) scheduling.Endpoint {
	e := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: name},
		Address:        address,
		Port:           "8080",
	}, nil, nil)
	e.Put(p.prefixMatchDataKey.String(),
		attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, 4, testBlockSize).WithCachedBlockCount(cachedBlocks))
	return e
}

func decodeOnly(ep scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode": {TargetEndpoints: []scheduling.Endpoint{ep}},
		},
	}
}

// Factory defaults: token delta 1, default PrefixCacheMatchInfo producer.
func TestPluginFactory_Defaults(t *testing.T) {
	p, err := PluginFactory("test", nil, nil)
	require.NoError(t, err)
	producer := p.(*Producer)
	assert.Equal(t, 1, producer.minCachedTokenDelta)
	assert.Equal(t, attrprefix.PrefixCacheMatchInfoDataKey.String(), producer.prefixMatchDataKey.String())
}

// Factory wires minCachedTokenDelta and binds the data key to the configured
// producer name.
func TestPluginFactory_WiresConfig(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(
		`{"prefixMatchInfoProducerName": "precise", "minCachedTokenDelta": 33}`))
	p, err := PluginFactory("test", dec, nil)
	require.NoError(t, err)
	producer := p.(*Producer)
	assert.Equal(t, 33, producer.minCachedTokenDelta)
	assert.Equal(t,
		attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("precise").String(),
		producer.prefixMatchDataKey.String())
}

// Factory rejects an explicit delta below 1.
func TestPluginFactory_RejectsZeroDelta(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(`{"minCachedTokenDelta": 0}`))
	_, err := PluginFactory("test", dec, nil)
	require.Error(t, err)
}

// Produce stashes the endpoint holding the most cached prompt tokens.
func TestProduce_StashesBestMatchPeer(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-stash"}
	eps := []scheduling.Endpoint{
		endpoint(p, "pod-a", "10.0.0.1", 1),
		endpoint(p, "pod-b", "10.0.0.2", 3),
	}
	require.NoError(t, p.Produce(ctx, req, eps))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok, "expected best-match attribute to be stashed")
	assert.Equal(t, "10.0.0.2:8080", best.hostPort)
	assert.Equal(t, 48, best.cachedTokens)
}

// No candidate holds any cached block: nothing to pull, no attribute.
func TestProduce_NoCachedBlocks_NoStash(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-nocache"}
	eps := []scheduling.Endpoint{
		endpoint(p, "pod-a", "10.0.0.1", 0),
		endpoint(p, "pod-b", "10.0.0.2", 0),
	}
	require.NoError(t, p.Produce(ctx, req, eps))

	_, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	assert.False(t, ok)
}

// Endpoints without PrefixCacheMatchInfo are treated as holding 0 blocks.
func TestProduce_MissingMatchInfo_TreatedAsZero(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	bare := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: "pod-bare"},
		Address:        "10.0.0.9",
		Port:           "8080",
	}, nil, nil)

	req := &scheduling.InferenceRequest{RequestID: "req-bare"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{bare, endpoint(p, "pod-b", "10.0.0.2", 2)}))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, "10.0.0.2:8080", best.hostPort)
}

// Best peer exceeds the decode pod's cached tokens by >= delta: header set.
func TestPreRequest_SetsKVCacheSourceHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-hdr", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedTokens: 48})

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1)))

	assert.Equal(t, "10.0.0.2:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Delta below threshold: header not set.
func TestPreRequest_DeltaBelowThreshold_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 17})

	req := &scheduling.InferenceRequest{RequestID: "req-low", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedTokens: 32})

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// The chosen decode pod is itself the best match: header not set.
func TestPreRequest_BestIsChosen_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-self", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.1:8080", cachedTokens: 32})

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 2)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// P/D: the prefill pod computes the prefix; when it is the best match the
// header is not set even if the decode pod holds fewer blocks.
func TestPreRequest_PrefillProfile_BestIsPrefill_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-pd-self", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedTokens: 48})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-a", "10.0.0.1", 0)}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-b", "10.0.0.2", 3)}},
		},
	}
	p.PreRequest(ctx, req, result)

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// P/D: a third pod out-caches the chosen prefill pod by >= delta: header set.
func TestPreRequest_PrefillProfile_HeaderFromThirdPod(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-pd-third", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.3:8080", cachedTokens: 64})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-a", "10.0.0.1", 0)}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-b", "10.0.0.2", 1)}},
		},
	}
	p.PreRequest(ctx, req, result)

	assert.Equal(t, "10.0.0.3:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Inbound (spoofed) header is removed even when no best-match attribute was
// stashed.
func TestPreRequest_DeletesInboundHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{
		RequestID: "req-spoof",
		Headers:   map[string]string{routing.KVCacheSourceHeader: "evil:1234"},
	}

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// Consumes declares the PrefixCacheMatchInfo dependency name-bound to the
// configured producer.
func TestConsumes_DeclaresPrefixCacheMatchInfo(t *testing.T) {
	p := New("test", Config{PrefixMatchInfoProducerName: "precise", MinCachedTokenDelta: 1})
	deps := p.Consumes()
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("precise")
	_, ok := deps.Required[key]
	assert.True(t, ok)
}

// IPv6 endpoint addresses are emitted bracketed via net.JoinHostPort so the
// sidecar's host:port validation accepts them.
func TestPreRequest_IPv6HeaderBracketed(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	best := net.JoinHostPort("fd00::2", "8080")
	req := &scheduling.InferenceRequest{RequestID: "req-ipv6", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: best, cachedTokens: 48})

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "fd00::1", 1)))

	assert.Equal(t, best, req.Headers[routing.KVCacheSourceHeader])
	// Round-trips through the same validation the sidecar applies.
	_, _, err := net.SplitHostPort(req.Headers[routing.KVCacheSourceHeader])
	assert.NoError(t, err)
}

// Produce emits a bracketed host:port for an IPv6 candidate.
func TestProduce_IPv6BestMatchBracketed(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-ipv6-produce"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{endpoint(p, "pod-a", "fd00::9", 2)}))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, net.JoinHostPort("fd00::9", "8080"), best.hostPort)
}

// A renamed prefill profile is honored: the comparison is against the prefill
// pod under the configured name, not the primary decode pod.
func TestPreRequest_ConfiguredPrefillProfileName(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, PrefillProfileName: "P"})

	req := &scheduling.InferenceRequest{RequestID: "req-custom-profile", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedTokens: 48})

	// Best match IS the renamed prefill pod -> pulling from self, no header.
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode": {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-a", "10.0.0.1", 0)}},
			"P":      {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-b", "10.0.0.2", 3)}},
		},
	}
	p.PreRequest(ctx, req, result)
	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// Factory wires a custom prefillProfileName; default is "prefill".
func TestPluginFactory_PrefillProfileName(t *testing.T) {
	def, err := PluginFactory("d", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, "prefill", def.(*Producer).prefillProfile)

	dec := json.NewDecoder(strings.NewReader(`{"prefillProfileName": "P"}`))
	custom, err := PluginFactory("c", dec, nil)
	require.NoError(t, err)
	assert.Equal(t, "P", custom.(*Producer).prefillProfile)
}

// endpointWithLoad builds a candidate carrying both PrefixCacheMatchInfo and
// a waiting-queue depth.
func endpointWithLoad(p *Producer, name, address string, cachedBlocks, waiting int) scheduling.Endpoint {
	e := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: name},
		Address:        address,
		Port:           "8080",
	}, &fwkdl.Metrics{WaitingQueueSize: waiting}, nil)
	e.Put(p.prefixMatchDataKey.String(),
		attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, 4, testBlockSize).WithCachedBlockCount(cachedBlocks))
	return e
}

// Equally-cached peers share pull traffic proportionally to 1/(1+queue):
// the shortest queue receives the most requests, deeper queues fewer, and a
// small queue difference shifts share without starving anyone.
func TestProduce_SharesByInverseQueueWeight(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	// Weights: pod-a 1/13, pod-b 1/3, pod-c 1/8.
	eps := []scheduling.Endpoint{
		endpointWithLoad(p, "pod-a", "10.0.0.1", 3, 12),
		endpointWithLoad(p, "pod-b", "10.0.0.2", 3, 2),
		endpointWithLoad(p, "pod-c", "10.0.0.3", 3, 7),
	}
	picks := map[string]int{}
	for i := 0; i < 400; i++ {
		req := &scheduling.InferenceRequest{RequestID: fmt.Sprintf("req-%d", i)}
		require.NoError(t, p.Produce(ctx, req, eps))
		best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
		require.True(t, ok)
		picks[best.hostPort]++
	}
	assert.Greater(t, picks["10.0.0.2:8080"], picks["10.0.0.3:8080"], "shortest queue must lead: %v", picks)
	assert.Greater(t, picks["10.0.0.3:8080"], picks["10.0.0.1:8080"], "middle queue must beat deepest: %v", picks)
	assert.Greater(t, picks["10.0.0.1:8080"], 0, "deepest queue must still receive some share: %v", picks)
}

// A one-request queue difference (noise at scrape granularity) shifts share
// roughly 2:1 instead of starving the deeper peer.
func TestProduce_NearTie_NoHerding(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	eps := []scheduling.Endpoint{
		endpointWithLoad(p, "pod-idle", "10.0.0.1", 3, 0),
		endpointWithLoad(p, "pod-busy", "10.0.0.2", 3, 1),
	}
	picks := map[string]int{}
	for i := 0; i < 600; i++ {
		req := &scheduling.InferenceRequest{RequestID: fmt.Sprintf("near-%d", i)}
		require.NoError(t, p.Produce(ctx, req, eps))
		best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
		require.True(t, ok)
		picks[best.hostPort]++
	}
	idle, busy := picks["10.0.0.1:8080"], picks["10.0.0.2:8080"]
	assert.Greater(t, idle, busy, "idle peer must lead: %v", picks)
	// Expected ratio 2:1 (weights 1 and 0.5); allow generous slack around it.
	assert.Greater(t, busy, 600/6, "busy peer must not be starved: %v", picks)
}

// A peer more than one block ahead wins regardless of load: the queue
// weighting applies within the one-block band only.
func TestProduce_BeyondBandWinsOverIdle(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-strict"}
	eps := []scheduling.Endpoint{
		endpointWithLoad(p, "pod-a", "10.0.0.1", 5, 20),
		endpointWithLoad(p, "pod-b", "10.0.0.2", 3, 0),
	}
	require.NoError(t, p.Produce(ctx, req, eps))
	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, "10.0.0.1:8080", best.hostPort)
	assert.Equal(t, 5*testBlockSize, best.cachedTokens)
}

// A peer one block short of the maximum competes on queue weight: an idle
// one-block-short peer takes most of the traffic from a deeply-queued
// maximum, and the stashed count is the chosen peer's own.
func TestProduce_OneBlockShort_SharesByQueue(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	// Weights: pod-ahead 1/21, pod-short 1.
	eps := []scheduling.Endpoint{
		endpointWithLoad(p, "pod-ahead", "10.0.0.1", 4, 20),
		endpointWithLoad(p, "pod-short", "10.0.0.2", 3, 0),
	}
	picks := map[string]int{}
	for i := 0; i < 400; i++ {
		req := &scheduling.InferenceRequest{RequestID: fmt.Sprintf("band-%d", i)}
		require.NoError(t, p.Produce(ctx, req, eps))
		best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
		require.True(t, ok)
		picks[best.hostPort]++
		want := 4 * testBlockSize
		if best.hostPort == "10.0.0.2:8080" {
			want = 3 * testBlockSize
		}
		assert.Equal(t, want, best.cachedTokens)
	}
	assert.Greater(t, picks["10.0.0.2:8080"], picks["10.0.0.1:8080"], "idle one-block-short peer must lead: %v", picks)
	assert.Greater(t, picks["10.0.0.1:8080"], 0, "max-cached peer must keep some share: %v", picks)
}

// Peers tied on cached count and queue depth: requests spread across them by
// request-ID hash instead of converging on iteration order, and the same
// request always maps to the same peer.
func TestProduce_EqualQueues_SpreadByRequestID(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	eps := []scheduling.Endpoint{
		endpointWithLoad(p, "pod-a", "10.0.0.1", 3, 0),
		endpointWithLoad(p, "pod-b", "10.0.0.2", 3, 0),
		endpointWithLoad(p, "pod-c", "10.0.0.3", 3, 0),
	}

	picks := map[string]int{}
	for i := 0; i < 64; i++ {
		req := &scheduling.InferenceRequest{RequestID: fmt.Sprintf("req-%d", i)}
		require.NoError(t, p.Produce(ctx, req, eps))
		best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
		require.True(t, ok)
		picks[best.hostPort]++
	}
	assert.Len(t, picks, 3, "ties must spread across all tied peers, got %v", picks)

	// Determinism: the same request ID picks the same peer.
	reqA := &scheduling.InferenceRequest{RequestID: "req-7"}
	reqB := &scheduling.InferenceRequest{RequestID: "req-7"}
	require.NoError(t, p.Produce(ctx, reqA, eps))
	require.NoError(t, p.Produce(ctx, reqB, eps))
	bestA, _ := scheduling.ReadRequestAttribute[*bestMatchPeer](reqA, p.attrKey())
	bestB, _ := scheduling.ReadRequestAttribute[*bestMatchPeer](reqB, p.attrKey())
	assert.Equal(t, bestA.hostPort, bestB.hostPort)
}

// Endpoints without metrics are treated as load 0: selection stays purely
// delta-driven, matching the previous behavior.
func TestProduce_NilMetrics_NeutralLoad(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-nilmetrics"}
	eps := []scheduling.Endpoint{
		endpoint(p, "pod-a", "10.0.0.1", 1),
		endpoint(p, "pod-b", "10.0.0.2", 3),
	}
	require.NoError(t, p.Produce(ctx, req, eps))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, "10.0.0.2:8080", best.hostPort)
}
