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

package kvcachesource

import (
	"encoding/json"
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

// Factory defaults: delta 1, default PrefixCacheMatchInfo producer.
func TestPluginFactory_Defaults(t *testing.T) {
	p, err := PluginFactory("test", nil, nil)
	require.NoError(t, err)
	producer := p.(*Producer)
	assert.Equal(t, 1, producer.minCachedBlockDelta)
	assert.Equal(t, attrprefix.PrefixCacheMatchInfoDataKey.String(), producer.prefixMatchDataKey.String())
}

// Factory wires minCachedBlockDelta and binds the data key to the configured
// producer name.
func TestPluginFactory_WiresConfig(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(
		`{"prefixMatchInfoProducerName": "precise", "minCachedBlockDelta": 3}`))
	p, err := PluginFactory("test", dec, nil)
	require.NoError(t, err)
	producer := p.(*Producer)
	assert.Equal(t, 3, producer.minCachedBlockDelta)
	assert.Equal(t,
		attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("precise").String(),
		producer.prefixMatchDataKey.String())
}

// Factory rejects an explicit delta below 1.
func TestPluginFactory_RejectsZeroDelta(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(`{"minCachedBlockDelta": 0}`))
	_, err := PluginFactory("test", dec, nil)
	require.Error(t, err)
}

// Produce stashes the endpoint with the highest unweighted cached-block count.
func TestProduce_StashesBestMatchPeer(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-stash"}
	eps := []scheduling.Endpoint{
		endpoint(p, "pod-a", "10.0.0.1", 1),
		endpoint(p, "pod-b", "10.0.0.2", 3),
	}
	require.NoError(t, p.Produce(ctx, req, eps))

	best, ok := scheduling.ReadRequestAttribute[*bestMatchPeer](req, p.attrKey())
	require.True(t, ok, "expected best-match attribute to be stashed")
	assert.Equal(t, "10.0.0.2:8080", best.hostPort)
	assert.Equal(t, 3, best.cachedBlocks)
}

// No candidate holds any cached block: nothing to pull, no attribute.
func TestProduce_NoCachedBlocks_NoStash(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 1})

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
	p := New("test", Config{MinCachedBlockDelta: 1})

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

// Best peer exceeds the decode pod's cached blocks by >= delta: header set.
func TestPreRequest_SetsKVCacheSourceHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-hdr", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedBlocks: 3})

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1)))

	assert.Equal(t, "10.0.0.2:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Delta below threshold: header not set.
func TestPreRequest_DeltaBelowThreshold_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 2})

	req := &scheduling.InferenceRequest{RequestID: "req-low", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedBlocks: 2})

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 1)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// The chosen decode pod is itself the best match: header not set.
func TestPreRequest_BestIsChosen_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-self", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.1:8080", cachedBlocks: 2})

	p.PreRequest(ctx, req, decodeOnly(endpoint(p, "pod-a", "10.0.0.1", 2)))

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// P/D: the prefill pod computes the prefix; when it is the best match the
// header is not set even if the decode pod holds fewer blocks.
func TestPreRequest_PrefillProfile_BestIsPrefill_NoHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-pd-self", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.2:8080", cachedBlocks: 3})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":       {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-a", "10.0.0.1", 0)}},
			prefillProfile: {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-b", "10.0.0.2", 3)}},
		},
	}
	p.PreRequest(ctx, req, result)

	assert.NotContains(t, req.Headers, routing.KVCacheSourceHeader)
}

// P/D: a third pod out-caches the chosen prefill pod by >= delta: header set.
func TestPreRequest_PrefillProfile_HeaderFromThirdPod(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 1})

	req := &scheduling.InferenceRequest{RequestID: "req-pd-third", Headers: map[string]string{}}
	req.PutAttribute(p.attrKey(), &bestMatchPeer{hostPort: "10.0.0.3:8080", cachedBlocks: 4})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":       {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-a", "10.0.0.1", 0)}},
			prefillProfile: {TargetEndpoints: []scheduling.Endpoint{endpoint(p, "pod-b", "10.0.0.2", 1)}},
		},
	}
	p.PreRequest(ctx, req, result)

	assert.Equal(t, "10.0.0.3:8080", req.Headers[routing.KVCacheSourceHeader])
}

// Inbound (spoofed) header is removed even when no best-match attribute was
// stashed.
func TestPreRequest_DeletesInboundHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := New("test", Config{MinCachedBlockDelta: 1})

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
	p := New("test", Config{PrefixMatchInfoProducerName: "precise", MinCachedBlockDelta: 1})
	deps := p.Consumes()
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("precise")
	_, ok := deps.Required[key]
	assert.True(t, ok)
}
