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
	"net"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

const testBlockSize = 16

// endpoint builds a candidate carrying the producer's PrefixCacheMatchInfo
// with the given unweighted cached-block count.
func endpoint(p *Producer, name, address string, cachedBlocks int) scheduling.Endpoint {
	e := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: name},
		Address: address,
		Port:    "8080",
	}, nil, nil)
	e.Put(p.prefixMatchDataKey,
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

func readReusablePrefixTokens(request *scheduling.InferenceRequest, producerName string) (attrprefix.ReusablePrefixTokens, bool) {
	key := attrprefix.ReusablePrefixTokensDataKey.WithNonEmptyProducerName(producerName)
	return scheduling.ReadRequestAttribute[attrprefix.ReusablePrefixTokens](request, key)
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

func TestNew_DefaultsMinCachedTokenDelta(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{})
	req := &scheduling.InferenceRequest{RequestID: "req-new-default"}

	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		tierEndpoint(p, "pod-a", "10.0.0.1", 4, map[string]int{cpuDeviceTier: 4}),
	}))

	assert.Equal(t, defaultMinCachedTokenDelta, p.minCachedTokenDelta)
	got, ok := readReusablePrefixTokens(req, "test")
	require.True(t, ok)
	assert.Equal(t, attrprefix.ReusablePrefixTokens(4*testBlockSize), got)
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

func TestProduce_PublishesReusablePrefixTokens(t *testing.T) {
	tests := []struct {
		name        string
		cached      int
		delta       int
		wantTokens  attrprefix.ReusablePrefixTokens
		wantPresent bool
	}{
		{name: "delta one", cached: 4, delta: 1, wantTokens: 4 * testBlockSize, wantPresent: true},
		{name: "delta equals source", cached: 4, delta: 4 * testBlockSize, wantTokens: 1, wantPresent: true},
		{name: "delta exceeds source", cached: 4, delta: 4*testBlockSize + 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			p := New("test", Config{MinCachedTokenDelta: tt.delta})
			req := &scheduling.InferenceRequest{RequestID: "req-floor-" + tt.name}

			require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
				tierEndpoint(p, "pod-a", "10.0.0.1", tt.cached, map[string]int{cpuDeviceTier: tt.cached}),
			}))

			got, ok := readReusablePrefixTokens(req, "test")
			assert.Equal(t, tt.wantPresent, ok)
			if tt.wantPresent {
				assert.Equal(t, tt.wantTokens, got)
			}
			_, hasBest := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
			assert.True(t, hasBest)
		})
	}
}

func TestProduce_NoSource_NoReusablePrefixTokens(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})
	req := &scheduling.InferenceRequest{RequestID: "req-no-floor"}

	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		endpoint(p, "pod-a", "10.0.0.1", 0),
	}))

	_, ok := readReusablePrefixTokens(req, "test")
	assert.False(t, ok)
}

func TestProduce_CancelledContextDoesNotStoreAttributes(t *testing.T) {
	ctx, cancel := context.WithCancel(utils.NewTestContext(t))
	cancel()
	p := New("test", Config{MinCachedTokenDelta: 1})
	req := &scheduling.InferenceRequest{RequestID: "req-cancelled"}

	err := p.Produce(ctx, req, []scheduling.Endpoint{
		tierEndpoint(p, "pod-a", "10.0.0.1", 4, map[string]int{cpuDeviceTier: 4}),
	})

	require.ErrorIs(t, err, context.Canceled)
	_, hasBest := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	assert.False(t, hasBest)
	_, hasReusable := readReusablePrefixTokens(req, "test")
	assert.False(t, hasReusable)
}

func TestProduce_ReusablePrefixTokensUsesSampledSource(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})
	eps := []scheduling.Endpoint{
		tierEndpointWithLoad(p, "pod-ahead", "10.0.0.1", 4, 20),
		tierEndpointWithLoad(p, "pod-short", "10.0.0.2", 3, 0),
	}

	foundShort := false
	for i := 0; i < 400; i++ {
		req := &scheduling.InferenceRequest{RequestID: fmt.Sprintf("floor-band-%d", i)}
		require.NoError(t, p.Produce(ctx, req, eps))
		best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
		require.True(t, ok)
		if best.hostPort != "10.0.0.2:8080" {
			continue
		}

		got, ok := readReusablePrefixTokens(req, "test")
		require.True(t, ok)
		assert.Equal(t, attrprefix.ReusablePrefixTokens(3*testBlockSize), got)
		foundShort = true
		break
	}
	assert.True(t, foundShort, "expected the one-block-short source to be sampled")
}

func TestProduce_ReusablePrefixTokensKeyUsesInstanceName(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("custom", Config{MinCachedTokenDelta: 1})
	req := &scheduling.InferenceRequest{RequestID: "req-custom-key"}

	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		tierEndpoint(p, "pod-a", "10.0.0.1", 2, map[string]int{cpuDeviceTier: 2}),
	}))

	got, ok := readReusablePrefixTokens(req, "custom")
	require.True(t, ok)
	assert.Equal(t, attrprefix.ReusablePrefixTokens(2*testBlockSize), got)
	_, ok = scheduling.ReadRequestAttribute[attrprefix.ReusablePrefixTokens](req, attrprefix.ReusablePrefixTokensDataKey)
	assert.False(t, ok)
}

func TestProduces_DeclaresReusablePrefixTokens(t *testing.T) {
	p := New("custom", Config{MinCachedTokenDelta: 1})
	key := attrprefix.ReusablePrefixTokensDataKey.WithNonEmptyProducerName("custom")

	produced := p.Produces()
	require.Len(t, produced, 1)
	assert.Equal(t, attrprefix.ReusablePrefixTokens(0), produced[key])
	assert.IsType(t, attrprefix.ReusablePrefixTokens(0), produced[key])
	assert.Equal(t, plugin.NewDataKey("ReusablePrefixTokensDataKey", "custom").String(), key.String())
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
		ID:      k8stypes.NamespacedName{Name: "pod-bare"},
		Address: "10.0.0.9",
		Port:    "8080",
	}, nil, nil)

	req := &scheduling.InferenceRequest{RequestID: "req-bare"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{bare, endpoint(p, "pod-b", "10.0.0.2", 2)}))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, "10.0.0.2:8080", best.hostPort)
}

// A metadata-less endpoint must not pin the pool maximum: it cannot serve as
// a source itself, and an inflated maximum would exclude every real endpoint
// and silently suppress the stash.
func TestProduce_MetadataNilMax_DoesNotSuppressStash(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	noMD := scheduling.NewEndpoint(nil, nil, nil)
	noMD.Put(p.prefixMatchDataKey,
		attrprefix.NewPrefixCacheMatchInfo(10, 4, testBlockSize).WithCachedBlockCount(10))

	req := &scheduling.InferenceRequest{RequestID: "req-nil-md"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{noMD, endpoint(p, "pod-b", "10.0.0.2", 2)}))

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

	_ = p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1)))

	assert.Equal(t, "10.0.0.2:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Delta below threshold: header not set.
func TestPreRequest_DeltaBelowThreshold_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 17})

	req := &scheduling.InferenceRequest{RequestID: "req-low", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedTokens: 32})

	_ = p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// The chosen decode pod is itself the best match: header not set.
func TestPreRequest_BestIsChosen_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-self", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.1:8080", cachedTokens: 32})

	_ = p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 2)))

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
	_ = p.PreRequest(ctx, req, result)

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
	_ = p.PreRequest(ctx, req, result)

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

	_ = p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 0)))

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

	_ = p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "fd00::1", 1)))

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
	_ = p.PreRequest(ctx, req, result)
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
		ID:      k8stypes.NamespacedName{Name: name},
		Address: address,
		Port:    "8080",
	}, &fwkdl.Metrics{WaitingQueueSize: waiting}, nil)
	e.Put(p.prefixMatchDataKey,
		attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, 4, testBlockSize).WithCachedBlockCount(cachedBlocks))
	return e
}

func tierEndpointWithLoad(p *Producer, name, address string, cachedBlocks, waiting int) scheduling.Endpoint {
	e := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: name},
		Address: address,
		Port:    "8080",
	}, &fwkdl.Metrics{WaitingQueueSize: waiting}, nil)
	e.Put(p.prefixMatchDataKey,
		attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, 4, testBlockSize).
			WithCachedBlockCount(cachedBlocks).
			WithConfirmedCachedBlockCount(cachedBlocks).
			WithCachedBlocksByTier(map[string]int{cpuDeviceTier: cachedBlocks}))
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

// tierEndpoint builds a candidate whose PrefixCacheMatchInfo carries a
// per-tier cached-block map alongside the unweighted total.
func tierEndpoint(p *Producer, name, address string, cachedBlocks int, byTier map[string]int) scheduling.Endpoint {
	return tierEndpointWithConfirmed(p, name, address, cachedBlocks, cachedBlocks, byTier)
}

func tierEndpointWithConfirmed(
	p *Producer, name, address string, cachedBlocks, confirmedCachedBlocks int, byTier map[string]int,
) scheduling.Endpoint {
	e := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: name},
		Address: address,
		Port:    "8080",
	}, nil, nil)
	e.Put(p.prefixMatchDataKey,
		attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, 4, testBlockSize).
			WithCachedBlockCount(cachedBlocks).
			WithConfirmedCachedBlockCount(confirmedCachedBlocks).
			WithCachedBlocksByTier(byTier))
	return e
}

// A GPU-only holder cannot serve a pull and must lose to a CPU-tier holder.
func TestProduce_GPUOnlySource_NotChosen(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-gpu-only"}
	eps := []scheduling.Endpoint{
		tierEndpoint(p, "pod-gpu", "10.0.0.1", 10, map[string]int{"gpu": 10}),
		tierEndpoint(p, "pod-cpu", "10.0.0.2", 3, map[string]int{"gpu": 3, cpuDeviceTier: 3}),
	}
	require.NoError(t, p.Produce(ctx, req, eps))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, "10.0.0.2:8080", best.hostPort)
	assert.Equal(t, 3*testBlockSize, best.cachedTokens)
}

func TestProduce_GPUOnlySource_NoReusablePrefixTokens(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})
	req := &scheduling.InferenceRequest{RequestID: "req-gpu-only-floor"}

	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		tierEndpoint(p, "pod-gpu", "10.0.0.1", 10, map[string]int{"gpu": 10}),
	}))

	_, ok := readReusablePrefixTokens(req, "test")
	assert.False(t, ok)
}

// Speculative entries must not make an endpoint a pull source.
func TestProduce_SpeculativeOnlySource_NoStash(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-spec"}
	eps := []scheduling.Endpoint{
		tierEndpoint(p, "pod-spec", "10.0.0.1", 10,
			map[string]int{attrprefix.SpeculativeTierKey: 10}),
	}
	require.NoError(t, p.Produce(ctx, req, eps))

	_, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	assert.False(t, ok)
}

// Producers without tier data keep the unweighted count.
func TestProduce_NoTierData_FallsBackToUnweighted(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-no-tier"}
	eps := []scheduling.Endpoint{
		endpoint(p, "pod-a", "10.0.0.1", 4),
	}
	require.NoError(t, p.Produce(ctx, req, eps))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, 4*testBlockSize, best.cachedTokens)
	_, hasReusable := readReusablePrefixTokens(req, "test")
	assert.False(t, hasReusable)
}

// The computing side counts confirmed local blocks across every real tier.
func TestPreRequest_ComputingSideCountsAllTiers(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 2 * testBlockSize})

	src := tierEndpoint(p, "pod-src", "10.0.0.1", 6, map[string]int{cpuDeviceTier: 6})
	computing := tierEndpoint(p, "pod-comp", "10.0.0.2", 5, map[string]int{"gpu": 5})

	req := &scheduling.InferenceRequest{RequestID: "req-comp-tiers"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{src, computing}))
	_ = p.PreRequest(ctx, req, decodeOnly(computing))

	// source 6 cpu blocks minus computing 5 (gpu, but local) = 1 block < delta.
	assert.Empty(t, req.Headers[routing.KVCacheSourceHeader])
}

func TestPreRequest_TierDataWithoutConfirmedCountUsesCachedCount(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 2 * testBlockSize})

	src := tierEndpoint(p, "pod-src", "10.0.0.1", 6, map[string]int{cpuDeviceTier: 6})
	computing := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: "pod-comp"},
		Address: "10.0.0.2",
		Port:    "8080",
	}, nil, nil)
	computing.Put(p.prefixMatchDataKey,
		attrprefix.NewPrefixCacheMatchInfo(5, 6, testBlockSize).
			WithCachedBlockCount(5).
			WithCachedBlocksByTier(map[string]int{"gpu": 5}))

	req := &scheduling.InferenceRequest{RequestID: "req-tier-default", Headers: map[string]string{}}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{src, computing}))
	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(computing)))

	// The source leads by one block, below the two-block delta. A producer that
	// publishes tier data without setting a separate confirmed count preserves
	// its cached count, so the pull remains suppressed.
	assert.Empty(t, req.Headers[routing.KVCacheSourceHeader])
}

func TestPreRequest_SpeculativeComputingCacheDoesNotSuppressHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	src := tierEndpoint(p, "pod-src", "10.0.0.1", 6, map[string]int{cpuDeviceTier: 6})
	computing := tierEndpointWithConfirmed(p, "pod-comp", "10.0.0.2", 6, 0,
		map[string]int{attrprefix.SpeculativeTierKey: 6})
	req := &scheduling.InferenceRequest{RequestID: "req-speculative-computing", Headers: map[string]string{}}

	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{src, computing}))
	require.NoError(t, p.PreRequest(ctx, req, decodeOnly(computing)))

	assert.Equal(t, "10.0.0.1:8080", req.Headers[routing.KVCacheSourceHeader])
}

// ---- cost model ----

// rigCostModel carries the constants measured on the B200 RDMA testbed.
func rigCostModel() *CostModelConfig {
	return &CostModelConfig{
		PrefillMicrosecondsPerToken:  19,
		TransferMicrosecondsPerToken: 3.9,
		TransferFixedMs:              15,
		SourceWaitMs:                 350,
		RequeueMs:                    500,
		FleetWeight:                  0,
		BusyQueueThreshold:           1,
	}
}

// queuedEndpoint builds a candidate with the given waiting-queue depth.
func queuedEndpoint(p *Producer, name, address string, cachedBlocks, waiting int) scheduling.Endpoint {
	e := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: name},
		Address: address,
		Port:    "8080",
	}, &fwkdl.Metrics{WaitingQueueSize: waiting}, nil)
	e.Put(p.prefixMatchDataKey,
		attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, 4, testBlockSize).WithCachedBlockCount(cachedBlocks))
	return e
}

func costRequest(p *Producer, id string, sourceCachedTokens, sourceWaiting int) *scheduling.InferenceRequest {
	req := &scheduling.InferenceRequest{RequestID: id, Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedTokens: sourceCachedTokens, waitingQueue: sourceWaiting})
	return req
}

// Factory parses the optional costModel block.
func TestPluginFactory_CostModel_Parsed(t *testing.T) {
	raw := `{"minCachedTokenDelta": 256, "costModel": {"prefillMicrosecondsPerToken": 19, "transferMicrosecondsPerToken": 3.9,
		"transferFixedMs": 15, "sourceWaitMs": 350, "requeueMs": 500, "fleetWeight": 0.5, "busyQueueThreshold": 2}}`
	pl, err := PluginFactory("test", json.NewDecoder(strings.NewReader(raw)), nil)
	require.NoError(t, err)
	p := pl.(*Producer)
	require.NotNil(t, p.costModel)
	assert.Equal(t, 19.0, p.costModel.PrefillMicrosecondsPerToken)
	assert.Equal(t, 3.9, p.costModel.TransferMicrosecondsPerToken)
	assert.Equal(t, 15.0, p.costModel.TransferFixedMs)
	assert.Equal(t, 350.0, p.costModel.SourceWaitMs)
	assert.Equal(t, 500.0, p.costModel.RequeueMs)
	assert.Equal(t, 0.5, p.costModel.FleetWeight)
	assert.Equal(t, 2, p.costModel.BusyQueueThreshold)
}

// busyQueueThreshold omitted or 0 defaults to 1.
func TestPluginFactory_CostModel_DefaultsBusyQueueThreshold(t *testing.T) {
	raw := `{"costModel": {"prefillMicrosecondsPerToken": 19, "transferMicrosecondsPerToken": 3.9}}`
	pl, err := PluginFactory("test", json.NewDecoder(strings.NewReader(raw)), nil)
	require.NoError(t, err)
	assert.Equal(t, 1, pl.(*Producer).costModel.BusyQueueThreshold)
}

// Without a costModel block the producer keeps the fixed-delta behavior.
func TestPluginFactory_NoCostModel_Nil(t *testing.T) {
	pl, err := PluginFactory("test", json.NewDecoder(strings.NewReader(`{"minCachedTokenDelta": 4}`)), nil)
	require.NoError(t, err)
	assert.Nil(t, pl.(*Producer).costModel)
}

// The transfer must be cheaper per token than the prefill, or no pull can ever win.
func TestPluginFactory_CostModel_RejectsTransferNotBelowPrefill(t *testing.T) {
	raw := `{"costModel": {"prefillMicrosecondsPerToken": 4, "transferMicrosecondsPerToken": 4}}`
	_, err := PluginFactory("test", json.NewDecoder(strings.NewReader(raw)), nil)
	require.Error(t, err)
}

// Negative constants are rejected.
func TestPluginFactory_CostModel_RejectsNegative(t *testing.T) {
	for _, raw := range []string{
		`{"costModel": {"prefillMicrosecondsPerToken": 19, "transferMicrosecondsPerToken": 3.9, "requeueMs": -1}}`,
		`{"costModel": {"prefillMicrosecondsPerToken": 19, "transferMicrosecondsPerToken": 3.9, "fleetWeight": -0.1}}`,
		`{"costModel": {"prefillMicrosecondsPerToken": 19, "transferMicrosecondsPerToken": 3.9, "busyQueueThreshold": -1}}`,
	} {
		_, err := PluginFactory("test", json.NewDecoder(strings.NewReader(raw)), nil)
		require.Error(t, err, raw)
	}
}

// New applies the busyQueueThreshold default for callers that bypass the
// factory: an empty queue must not count as busy.
func TestNew_CostModel_DefaultsBusyQueueThreshold(t *testing.T) {
	ctx := utils.NewTestContext(t)
	cm := rigCostModel()
	cm.BusyQueueThreshold = 0
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: cm})
	assert.Equal(t, 1, p.costModel.BusyQueueThreshold)
	assert.Equal(t, 0, cm.BusyQueueThreshold, "caller's config must not be mutated")

	req := costRequest(p, "req-new-default", 8192, 0)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 0)))

	assert.Equal(t, "10.0.0.2:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Both pods idle, 8K delta: gain 8192*(19-3.9) us = 124 ms > t0 15 ms -> pull.
func TestPreRequest_CostModel_IdlePull(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := costRequest(p, "req-idle", 8192, 0)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 0)))

	assert.Equal(t, "10.0.0.2:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Both pods idle, 512-token delta: gain 7.7 ms < t0 15 ms -> recompute.
func TestPreRequest_CostModel_IdleSmallDelta_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := costRequest(p, "req-small", 512, 0)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 0)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// Busy destination adds requeue: 8K gain 124 ms < 15 + 500 -> recompute.
func TestPreRequest_CostModel_BusyDestination_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := costRequest(p, "req-busy-d", 8192, 0)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 1)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// Busy destination, 64K delta: gain 990 ms > 515 -> pull.
func TestPreRequest_CostModel_BusyDestination_LongDelta_Pull(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := costRequest(p, "req-busy-d-long", 65536, 0)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 1)))

	assert.Equal(t, "10.0.0.2:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Busy source adds srcwait: 8K gain 124 ms < 15 + 350 -> recompute.
func TestPreRequest_CostModel_BusySource_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := costRequest(p, "req-busy-s", 8192, 1)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 0)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// busyQueueThreshold: a queue below the threshold does not count as busy.
func TestPreRequest_CostModel_BusyQueueThreshold(t *testing.T) {
	ctx := utils.NewTestContext(t)
	cm := rigCostModel()
	cm.BusyQueueThreshold = 3
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: cm})

	req := costRequest(p, "req-threshold", 8192, 2)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 2)))

	assert.Equal(t, "10.0.0.2:8080", req.Headers[routing.KVCacheSourceHeader])
}

// runningEndpoint builds a candidate with the given running-request count.
func runningEndpoint(p *Producer, name, address string, running, waiting int) scheduling.Endpoint {
	e := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: name},
		Address: address,
		Port:    "8080",
	}, &fwkdl.Metrics{RunningRequestsSize: running, WaitingQueueSize: waiting}, nil)
	e.Put(p.prefixMatchDataKey,
		attrprefix.NewPrefixCacheMatchInfo(0, 4, testBlockSize).WithCachedBlockCount(0))
	return e
}

// Fleet weight credits the prefill freed for each running request on the
// computing pod, with or without a waiting queue. 512-token delta on an idle
// queue with 2 running: gain 7.7 ms < 15 at w=0 (recompute) and
// 7.7 + 2*512*19 us = 27.2 ms > 15 at w=1 (pull).
func TestPreRequest_CostModel_FleetWeightScalesWithRunning(t *testing.T) {
	ctx := utils.NewTestContext(t)
	for _, tc := range []struct {
		w       float64
		running int
		pull    bool
	}{{0, 2, false}, {1, 2, true}, {1, 0, false}} {
		cm := rigCostModel()
		cm.FleetWeight = tc.w
		p := New("test", Config{MinCachedTokenDelta: 1, CostModel: cm})

		req := costRequest(p, "req-fleet", 512, 0)
		_ = p.PreRequest(ctx, req, decodeOnly(runningEndpoint(p, "pod-a", "10.0.0.1", tc.running, 0)))

		_, set := req.Headers[routing.KVCacheSourceHeader]
		assert.Equal(t, tc.pull, set, "fleetWeight=%v running=%d", tc.w, tc.running)
	}
}

// Busy destination: 8K with 4 running at w=1 gains 124 + 4*8192*19 us = 747 ms
// > 15 + 500 requeue (pull); at w=0 it is 124 < 515 (recompute).
func TestPreRequest_CostModel_FleetWeightOutweighsRequeue(t *testing.T) {
	ctx := utils.NewTestContext(t)
	for _, tc := range []struct {
		w    float64
		pull bool
	}{{0, false}, {1, true}} {
		cm := rigCostModel()
		cm.FleetWeight = tc.w
		p := New("test", Config{MinCachedTokenDelta: 1, CostModel: cm})

		req := costRequest(p, "req-fleet-busy", 8192, 0)
		_ = p.PreRequest(ctx, req, decodeOnly(runningEndpoint(p, "pod-a", "10.0.0.1", 4, 1)))

		_, set := req.Headers[routing.KVCacheSourceHeader]
		assert.Equal(t, tc.pull, set, "fleetWeight=%v", tc.w)
	}
}

// minCachedTokenDelta remains a floor under the cost model.
func TestPreRequest_CostModel_RespectsMinCachedTokenDelta(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 16384, CostModel: rigCostModel()})

	req := costRequest(p, "req-floor", 8192, 0)
	_ = p.PreRequest(ctx, req, decodeOnly(queuedEndpoint(p, "pod-a", "10.0.0.1", 0, 0)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// Produce stashes the sampled source's waiting-queue depth for PreRequest.
func TestProduce_StashesSourceWaitingQueue(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-wq"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{queuedEndpoint(p, "pod-b", "10.0.0.2", 3, 3)}))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, 3, best.waitingQueue)
}

// ---- cost-aware source pick ----

func bestHost(t *testing.T, p *Producer, req *scheduling.InferenceRequest) string {
	t.Helper()
	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	return best.hostPort
}

// With the cost model, an idle source one block short beats a busy source:
// a block of recompute costs milliseconds, the source wait hundreds.
func TestProduce_CostModel_PrefersIdleSourceOverBusyWithMoreCache(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := &scheduling.InferenceRequest{RequestID: "req-pick-idle"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		queuedEndpoint(p, "busy-more", "10.0.0.1", 8, 2),
		queuedEndpoint(p, "idle-less", "10.0.0.2", 7, 0),
	}))
	assert.Equal(t, "10.0.0.2:8080", bestHost(t, p, req))
}

// With the cost model, an idle source far outside the one-block band still
// wins over a busy one when its recompute shortfall is cheaper than the wait.
func TestProduce_CostModel_IdleSourceOutsideBandWins(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	// 8192 vs 4096 tokens: 4096 * 15.1 us = 62 ms of extra recompute < 350 ms source wait.
	req := &scheduling.InferenceRequest{RequestID: "req-pick-band"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		queuedEndpoint(p, "busy-8k", "10.0.0.1", 8192/testBlockSize, 1),
		queuedEndpoint(p, "idle-4k", "10.0.0.2", 4096/testBlockSize, 0),
	}))
	assert.Equal(t, "10.0.0.2:8080", bestHost(t, p, req))
}

// A busy source keeps winning when its extra cache is worth more than the wait.
func TestProduce_CostModel_BusySourceWithMuchMoreCacheWins(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	// 32768 vs 1024 tokens: 31744 * 15.1 us = 479 ms of extra recompute > 350 ms source wait.
	req := &scheduling.InferenceRequest{RequestID: "req-pick-busy"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		queuedEndpoint(p, "busy-32k", "10.0.0.1", 32768/testBlockSize, 1),
		queuedEndpoint(p, "idle-1k", "10.0.0.2", 1024/testBlockSize, 0),
	}))
	assert.Equal(t, "10.0.0.1:8080", bestHost(t, p, req))
}

// Both idle: the larger cache wins.
func TestProduce_CostModel_BothIdle_MoreCacheWins(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := &scheduling.InferenceRequest{RequestID: "req-pick-idle2"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		queuedEndpoint(p, "idle-4", "10.0.0.1", 4, 0),
		queuedEndpoint(p, "idle-8", "10.0.0.2", 8, 0),
	}))
	assert.Equal(t, "10.0.0.2:8080", bestHost(t, p, req))
}

// Equal-cost sources are spread across requests rather than herded.
func TestProduce_CostModel_EqualCostSpread(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	seen := map[string]bool{}
	for i := 0; i < 64; i++ {
		req := &scheduling.InferenceRequest{RequestID: fmt.Sprintf("req-spread-%d", i)}
		require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
			queuedEndpoint(p, "idle-a", "10.0.0.1", 8, 0),
			queuedEndpoint(p, "idle-b", "10.0.0.2", 8, 0),
		}))
		seen[bestHost(t, p, req)] = true
	}
	assert.Len(t, seen, 2)
}

// The stashed source keeps the chosen pod's own queue depth and cache count.
func TestProduce_CostModel_StashesChosenSourceState(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedTokenDelta: 1, CostModel: rigCostModel()})

	req := &scheduling.InferenceRequest{RequestID: "req-pick-state"}
	require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{
		queuedEndpoint(p, "busy-more", "10.0.0.1", 8, 2),
		queuedEndpoint(p, "idle-less", "10.0.0.2", 7, 0),
	}))
	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok)
	assert.Equal(t, 7*testBlockSize, best.cachedTokens)
	assert.Equal(t, 0, best.waitingQueue)
}
