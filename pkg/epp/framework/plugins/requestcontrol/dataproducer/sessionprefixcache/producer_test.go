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

package sessionprefixcache

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// testSessionProducer publishes a SessionCacheRequest for requests that carry
// a session header, with the prefix recorded for the request named by the
// previous header as the only candidate. It learns prefixes from KV events as
// its own consumer, independently of the producer under test.
type testSessionProducer struct {
	mu           sync.Mutex
	observations map[string]attrsession.SessionCachePrefix
	estimated    bool
}

var _ kvevents.EventConsumer = &testSessionProducer{}

func (*testSessionProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-session-producer", Name: "sessions"}
}

func (*testSessionProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{sessionKey(): attrsession.SessionCacheRequest{}}
}

func sessionKey() plugin.DataKey {
	return attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions")
}

func (m *testSessionProducer) Produce(_ context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if request.Headers["session"] == "" {
		return nil
	}
	lookup := attrsession.SessionCacheRequest{SessionID: request.RequestID, FullReport: true,
		TotalTokens: request.Body.TokenizedRequest.TokenCount()}
	m.mu.Lock()
	defer m.mu.Unlock()
	if prefix, ok := m.observations[request.Headers["previous"]]; ok {
		prefix.Exact = !m.estimated
		lookup.Prefixes = []attrsession.SessionCachePrefix{prefix}
	}
	request.PutAttribute(sessionKey(), lookup)
	return nil
}

func (m *testSessionProducer) ProcessEvents(_ context.Context, _ kvevents.EventSource, batch kvevents.EventBatch) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, event := range batch.Events {
		if ev, ok := event.(*kvevents.BlockStoredEvent); ok && ev.SessionID != nil {
			m.observations[*ev.SessionID] = attrsession.SessionCachePrefix{
				CacheNamespace: "model-v1", BlockHashes: append([]uint64(nil), ev.BlockHashes...), BlockSizeTokens: ev.BlockSize,
			}
		}
	}
	return nil
}

func (*testSessionProducer) Reset(context.Context, string) error { return nil }

func newTestProducer(t *testing.T) (*Producer, *testSessionProducer) {
	t.Helper()
	sessions := &testSessionProducer{observations: make(map[string]attrsession.SessionCachePrefix)}
	p, err := PluginFactory("cache", plugin.StrictDecoder(json.RawMessage(`{"sessionCacheRequestProducerName":"sessions","cacheNamespace":"model-v1"}`)),
		plugin.NewEppHandle(t.Context(), nil))
	require.NoError(t, err)
	producer := p.(*Producer)
	assert.Equal(t, map[plugin.DataKey]any{sessionKey(): attrsession.SessionCacheRequest{}}, producer.Consumes().Required)
	return producer, sessions
}

// observe delivers a batch to the producer under test and, when given, to the
// session producer, as the pool would to two independent consumers.
func observe(t *testing.T, p *Producer, sessions *testSessionProducer, source kvevents.EventSource, events ...kvevents.GenericEvent) {
	t.Helper()
	observeBatch(t, p, sessions, source, kvevents.EventBatch{Events: events})
}

func observeBatch(t *testing.T, p *Producer, sessions *testSessionProducer, source kvevents.EventSource, batch kvevents.EventBatch) {
	t.Helper()
	require.NoError(t, p.events.ProcessEvents(t.Context(), source, batch))
	if sessions != nil {
		require.NoError(t, sessions.ProcessEvents(t.Context(), source, batch))
	}
}

func freshEndpoints() []scheduling.Endpoint {
	return []scheduling.Endpoint{
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-a"}, Address: "10.0.0.1", Port: "8080"}, nil, nil),
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-b"}, Address: "10.0.0.2", Port: "8080"}, nil, nil),
	}
}

func sessionRequest(id, previous string) *scheduling.InferenceRequest {
	prompt := strings.Repeat("abcd", 32)
	return &scheduling.InferenceRequest{
		RequestID: id, TargetModel: "model", Headers: map[string]string{"session": "logical-session", "previous": previous},
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
			Payload:     fwkrh.PayloadMap{"model": "model", "prompt": prompt},
		},
	}
}

func lookupRequest(id string, lookup attrsession.SessionCacheRequest) *scheduling.InferenceRequest {
	req := sessionRequest(id, "")
	req.PutAttribute(sessionKey(), lookup)
	return req
}

func matchOn(t *testing.T, p *Producer, req *scheduling.InferenceRequest) []*attrprefix.PrefixCacheMatchInfo {
	t.Helper()
	endpoints := freshEndpoints()
	require.NoError(t, p.Produce(t.Context(), req, endpoints))
	infos := make([]*attrprefix.PrefixCacheMatchInfo, len(endpoints))
	for i, endpoint := range endpoints {
		value, ok := endpoint.Get(p.dk)
		require.True(t, ok)
		infos[i] = value.(*attrprefix.PrefixCacheMatchInfo)
	}
	return infos
}

func firstMatch(t *testing.T, p *Producer, req *scheduling.InferenceRequest) *attrprefix.PrefixCacheMatchInfo {
	t.Helper()
	return matchOn(t, p, req)[0]
}

func TestProducerTotalTokensBoundCoverageAndLoad(t *testing.T) {
	for _, inputCount := range []int{7200, 11018} {
		for _, resident := range []int{112, 93, 46, 0} {
			t.Run(fmt.Sprintf("input-%d/resident-%d", inputCount, resident), func(t *testing.T) {
				p, _ := newTestProducer(t)
				hashes := make([]uint64, 112)
				for i := range hashes {
					hashes[i] = uint64(i + 1)
				}
				observe(t, p, nil, kvevents.EventSource{Endpoint: "10.0.0.1:8080"},
					&kvevents.BlockStoredEvent{BlockHashes: hashes, BlockSize: 64, DeviceTier: "GPU"},
					&kvevents.BlockRemovedEvent{BlockHashes: hashes[resident:], DeviceTier: "GPU"})
				req := lookupRequest("request", attrsession.SessionCacheRequest{SessionID: "request", TotalTokens: 7200,
					Prefixes: []attrsession.SessionCachePrefix{{CacheNamespace: "model-v1", BlockHashes: hashes, BlockSizeTokens: 64}}})
				req.Body.TokenizedRequest = fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, inputCount)})
				endpoints := freshEndpoints()
				require.NoError(t, p.Produce(t.Context(), req, endpoints))
				value, _ := endpoints[0].Get(p.dk)
				info := value.(*attrprefix.PrefixCacheMatchInfo)
				require.Equal(t, resident*64, info.MatchBlocks())
				require.Zero(t, info.CachedBlockCount())
				total, present := info.TotalTokens()
				require.True(t, present)
				require.Equal(t, 7200, total)

				filter, err := prefixcacheaffinity.Factory("affinity", plugin.StrictDecoder(json.RawMessage(`{"prefixMatchInfoProducerName":"cache"}`)), nil)
				require.NoError(t, err)
				candidates := 2
				if resident >= 93 {
					candidates = 1
				}
				require.Len(t, filter.(*prefixcacheaffinity.Plugin).Filter(t.Context(), req, endpoints), candidates)

				load, err := inflightload.InFlightLoadProducerFactory("load", plugin.StrictDecoder(json.RawMessage(`{"prefixMatchInfoProducerName":"cache"}`)), testutils.NewTestHandle(t.Context()))
				require.NoError(t, err)
				require.NoError(t, load.(*inflightload.InFlightLoadProducer).Produce(t.Context(), req, endpoints))
				for i, ep := range endpoints {
					want := int64(7200)
					if i == 0 {
						want -= int64(resident * 64)
					}
					got, ok := ep.Get(attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName("load"))
					require.True(t, ok)
					require.Equal(t, want, got.(*attrconcurrency.UncachedRequestTokens).Tokens)
				}
			})
		}
	}
}

func TestProducerTokenBackends(t *testing.T) {
	for _, estimate := range []bool{false, true} {
		name := "render"
		if estimate {
			name = "estimate"
		}
		t.Run(name, func(t *testing.T) {
			p, sessions := newTestProducer(t)
			sessions.estimated = estimate
			var renderCalls atomic.Int32
			renderer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				renderCalls.Add(1)
				if r.URL.Path == "/chat/completions/render" {
					_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": []uint32{1}})
					return
				}
				_ = json.NewEncoder(w).Encode([]map[string]any{{"token_ids": make([]uint32, 32)}})
			}))
			defer renderer.Close()
			params := `{"estimate":{}}`
			if !estimate {
				encoded, err := json.Marshal(map[string]any{"modelName": "model", "vllm": map[string]any{"url": renderer.URL}})
				require.NoError(t, err)
				params = string(encoded)
			}
			rawTokens, err := tokenproducer.PluginFactory("tokens", plugin.StrictDecoder(json.RawMessage(params)), plugin.NewEppHandle(t.Context(), nil))
			require.NoError(t, err)
			tokens := rawTokens.(requestcontrol.DataProducer)

			first := sessionRequest("request-1", "")
			require.NoError(t, tokens.Produce(t.Context(), first, nil))
			require.NoError(t, sessions.Produce(t.Context(), first, nil))
			require.NoError(t, p.PreRequest(t.Context(), first, nil))
			payload := first.Body.Payload.(fwkrh.PayloadMap)
			assert.Equal(t, "request-1", payload["session_id"])
			assert.Equal(t, "full", payload["vllm_xargs"].(map[string]any)["kv_cache_report_mode"])
			assert.True(t, first.Body.Mutated)

			// Engine blocks are learned and queried without event tokens.
			observe(t, p, sessions, kvevents.EventSource{Endpoint: "10.0.0.1:8080", ModelName: "model", Sequence: 1},
				&kvevents.BlockStoredEvent{SessionID: ptr.To("request-1"), BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"})
			next := sessionRequest("request-2", "request-1")
			require.NoError(t, tokens.Produce(t.Context(), next, nil))
			require.NoError(t, sessions.Produce(t.Context(), next, nil))
			info := firstMatch(t, p, next)
			assert.Equal(t, 32, info.MatchBlocks())
			if estimate {
				assert.Zero(t, info.CachedBlockCount())
				assert.Empty(t, info.CachedBlocksByTier())
				assert.Zero(t, renderCalls.Load())
			} else {
				assert.Equal(t, 32, info.CachedBlockCount()*info.BlockSizeTokens())
				assert.GreaterOrEqual(t, renderCalls.Load(), int32(2))
			}
		})
	}
}

func TestProducerSharedBlocksForksAndEviction(t *testing.T) {
	p, sessions := newTestProducer(t)
	source := kvevents.EventSource{Endpoint: "10.0.0.1:8080"}
	for id, hashes := range map[string][]uint64{"session-a": {10, 20}, "session-b": {10, 20}, "fork": {10, 30}} {
		stored := &kvevents.BlockStoredEvent{SessionID: ptr.To(id), BlockHashes: hashes, BlockSize: 16, DeviceTier: "gpu"}
		for range 2 {
			observe(t, p, sessions, source, stored)
		}
	}
	lookup := func(previous string) *attrprefix.PrefixCacheMatchInfo {
		req := sessionRequest("next", previous)
		req.Body.TokenizedRequest = fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, 32)})
		require.NoError(t, sessions.Produce(t.Context(), req, nil))
		return firstMatch(t, p, req)
	}
	for _, id := range []string{"session-a", "session-b", "fork"} {
		assert.Equal(t, 32, lookup(id).MatchBlocks())
	}
	observe(t, p, sessions, source, &kvevents.BlockRemovedEvent{BlockHashes: []uint64{20}, DeviceTier: "GPU"})
	assert.Equal(t, 16, lookup("session-a").MatchBlocks())
	assert.Equal(t, 16, lookup("session-b").MatchBlocks())
	assert.Equal(t, 32, lookup("fork").MatchBlocks())
	require.Len(t, sessions.observations, 3)
	require.NoError(t, p.events.Reset(t.Context(), source.Endpoint))
	assert.Zero(t, lookup("fork").MatchBlocks())
}

func TestProducerCrossWorkerLineage(t *testing.T) {
	p, sessions := newTestProducer(t)
	for i, endpoint := range []string{"10.0.0.1:8080", "10.0.0.2:8080"} {
		stored := &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"}
		if i == 0 {
			stored.SessionID = ptr.To("previous")
		}
		observe(t, p, sessions, kvevents.EventSource{Endpoint: endpoint}, stored)
	}
	req := sessionRequest("next", "previous")
	req.Body.TokenizedRequest = fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, 32)})
	require.NoError(t, sessions.Produce(t.Context(), req, nil))
	check := func(expected ...int) {
		t.Helper()
		for i, info := range matchOn(t, p, req) {
			assert.Equal(t, expected[i], info.CachedBlockCount(), "endpoint %d", i)
		}
	}
	check(32, 32)
	a := kvevents.EventSource{Endpoint: "10.0.0.1:8080"}
	observe(t, p, sessions, a, &kvevents.BlockRemovedEvent{BlockHashes: []uint64{20}, DeviceTier: "GPU"})
	check(16, 32)
	require.NoError(t, p.events.Reset(t.Context(), a.Endpoint))
	check(0, 32)
	assert.Equal(t, []uint64{10, 20}, sessions.observations["previous"].BlockHashes)

	b := kvevents.EventSource{Endpoint: "10.0.0.2:8080"}
	observe(t, p, sessions, b, &kvevents.AllBlocksClearedEvent{})
	check(0, 0)
	observe(t, p, sessions, b, &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"})
	check(0, 32)
}

func TestProducerBlockSizeIsolation(t *testing.T) {
	p, _ := newTestProducer(t)
	observe(t, p, nil, kvevents.EventSource{Endpoint: "10.0.0.2:8080"},
		&kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 32, DeviceTier: "GPU"})
	req := lookupRequest("request", attrsession.SessionCacheRequest{SessionID: "request", TotalTokens: 32,
		Prefixes: []attrsession.SessionCachePrefix{{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 20}, BlockSizeTokens: 16, Exact: true}}})
	for _, info := range matchOn(t, p, req) {
		assert.Zero(t, info.MatchBlocks())
	}
}

func TestProducerScopeAndBranchIsolation(t *testing.T) {
	p, _ := newTestProducer(t)
	for _, observation := range []struct {
		endpoint string
		rank     *int
		group    *int
		hashes   []uint64
	}{
		{endpoint: "10.0.0.1:8080", rank: ptr.To(0), hashes: []uint64{10, 20}},
		{endpoint: "10.0.0.1:8080", rank: ptr.To(1), hashes: []uint64{40}},
		{endpoint: "10.0.0.2:8080", group: ptr.To(0), hashes: []uint64{10, 30}},
		{endpoint: "10.0.0.2:8080", group: ptr.To(1), hashes: []uint64{50}},
	} {
		store := &kvevents.BlockStoredEvent{BlockHashes: observation.hashes, BlockSize: 16, DeviceTier: "GPU",
			GroupIdx: observation.group, KVCacheSpecKind: kvevents.KVCacheSpecKindFullAttention}
		observeBatch(t, p, nil, kvevents.EventSource{Endpoint: observation.endpoint},
			kvevents.EventBatch{DataParallelRank: observation.rank, Events: []kvevents.GenericEvent{store}})
	}
	req := lookupRequest("request", attrsession.SessionCacheRequest{SessionID: "request", TotalTokens: 48,
		Prefixes: []attrsession.SessionCachePrefix{
			{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 20, 40}, BlockSizeTokens: 16, Exact: true},
			{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 30, 50}, BlockSizeTokens: 16, Exact: true},
		}})
	check := func() {
		t.Helper()
		for _, info := range matchOn(t, p, req) {
			assert.Equal(t, 32, info.CachedBlockCount())
		}
	}
	check()
	// Removals in another rank or group of the same endpoint do not apply.
	for _, removal := range []struct {
		endpoint string
		rank     *int
		group    *int
		hash     uint64
	}{
		{endpoint: "10.0.0.1:8080", rank: ptr.To(1), hash: 20},
		{endpoint: "10.0.0.2:8080", group: ptr.To(1), hash: 30},
	} {
		ev := &kvevents.BlockRemovedEvent{BlockHashes: []uint64{removal.hash}, DeviceTier: "GPU", GroupIdx: removal.group}
		observeBatch(t, p, nil, kvevents.EventSource{Endpoint: removal.endpoint},
			kvevents.EventBatch{DataParallelRank: removal.rank, Events: []kvevents.GenericEvent{ev}})
	}
	check()
}

func TestProducerStampEnvelope(t *testing.T) {
	p, _ := newTestProducer(t)
	for _, args := range []any{map[string]any{"custom": 7.0}, json.RawMessage(`{"custom":7}`)} {
		req := sessionRequest("request", "")
		content := json.RawMessage(`[{"z":1,"a":2}]`)
		req.Body.Payload = fwkrh.PayloadMap{"messages": content, "vllm_xargs": args, "session_id": "external-logical-id"}
		req.PutAttribute(sessionKey(), attrsession.SessionCacheRequest{SessionID: "producer-id", FullReport: true})
		require.NoError(t, p.PreRequest(t.Context(), req, nil))
		payload := req.Body.Payload.(fwkrh.PayloadMap)
		assert.Equal(t, content, payload["messages"])
		assert.Equal(t, "producer-id", payload["session_id"])
		encoded, err := json.Marshal(payload["vllm_xargs"])
		require.NoError(t, err)
		assert.JSONEq(t, `{"custom":7,"kv_cache_report_mode":"full"}`, string(encoded))
	}
	for _, payload := range []fwkrh.RequestPayload{fwkrh.RawPayload(`{}`), fwkrh.PayloadMap{"vllm_xargs": "invalid"}} {
		req := sessionRequest("request", "")
		req.Body.Payload = payload
		req.PutAttribute(sessionKey(), attrsession.SessionCacheRequest{SessionID: "producer-id"})
		require.Error(t, p.PreRequest(t.Context(), req, nil))
		assert.False(t, req.Body.Mutated)
	}
	req := sessionRequest("request", "")
	require.NoError(t, p.PreRequest(t.Context(), req, nil))
	require.NoError(t, p.Produce(t.Context(), req, nil))
	assert.False(t, req.Body.Mutated)
}

func TestProducerStampPreservesLargeArguments(t *testing.T) {
	p, _ := newTestProducer(t)
	req := sessionRequest("request", "")
	req.Body.Payload = fwkrh.PayloadMap{"vllm_xargs": json.RawMessage(`{"custom":9007199254740993}`)}
	req.PutAttribute(sessionKey(), attrsession.SessionCacheRequest{SessionID: "request"})
	require.NoError(t, p.PreRequest(t.Context(), req, nil))
	encoded, err := json.Marshal(req.Body.Payload)
	require.NoError(t, err)
	assert.Contains(t, string(encoded), `"custom":9007199254740993`)
}

func TestProducerConfiguration(t *testing.T) {
	for name, params := range map[string]string{
		"missing producer name": `{}`,
		"sglang events":         `{"sessionCacheRequestProducerName":"sessions","cacheNamespace":"model-v1","kvEventsConfig":{"engineType":"sglang"}}`,
		"global socket":         `{"sessionCacheRequestProducerName":"sessions","cacheNamespace":"model-v1","kvEventsConfig":{"zmqEndpoint":"tcp://localhost:5557"}}`,
		"no discovery":          `{"sessionCacheRequestProducerName":"sessions","cacheNamespace":"model-v1","kvEventsConfig":{"discoverPods":false}}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := PluginFactory("cache", plugin.StrictDecoder(json.RawMessage(params)), plugin.NewEppHandle(t.Context(), nil))
			require.Error(t, err)
		})
	}
}

func TestProducerRejectsRedis(t *testing.T) {
	server := miniredis.RunT(t)
	params := fmt.Sprintf(`{"sessionCacheRequestProducerName":"sessions","cacheNamespace":"model-v1","indexConfig":{"redisConfig":{"address":%q}}}`, server.Addr())
	_, err := PluginFactory("cache", plugin.StrictDecoder(json.RawMessage(params)), plugin.NewEppHandle(t.Context(), nil))
	require.ErrorContains(t, err, "redisConfig")
	assert.Zero(t, server.CommandCount(), "reject Redis before opening a connection")
}

func TestProducerCoverageAndEventScope(t *testing.T) {
	p, _ := newTestProducer(t)
	source := kvevents.EventSource{Endpoint: "10.0.0.1:8080"}
	lookup := attrsession.SessionCacheRequest{SessionID: "request", TotalTokens: 32,
		Prefixes: []attrsession.SessionCachePrefix{{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 20}, BlockSizeTokens: 16, Exact: true}}}
	store := &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16,
		DeviceTier: "GPU", GroupIdx: ptr.To(0), KVCacheSpecKind: kvevents.KVCacheSpecKindFullAttention}
	batch := kvevents.EventBatch{DataParallelRank: ptr.To(1)}
	for _, kind := range []string{"remote", "owned", "cpu", "swa", "local"} {
		t.Run(kind, func(t *testing.T) {
			ev := *store
			switch kind {
			case "remote":
				ev.Locality = "REMOTE"
			case "owned":
				ev.Ownership = "offloader"
			case "cpu":
				ev.DeviceTier = "CPU"
			case "swa":
				ev.KVCacheSpecKind = kvevents.KVCacheSpecKindSlidingWindow
			case "local":
				ev.Locality = "LOCAL"
			}
			batch.Events = []kvevents.GenericEvent{&ev}
			observeBatch(t, p, nil, source, batch)
			info := firstMatch(t, p, lookupRequest("request", lookup))
			if kind == "local" {
				assert.Equal(t, 32, info.CachedBlockCount())
			} else {
				assert.Zero(t, info.MatchBlocks())
			}
		})
	}
	lookup.TotalTokens = 31
	assert.Equal(t, 16, firstMatch(t, p, lookupRequest("request", lookup)).CachedBlockCount())
	lookup.TotalTokens = 15
	assert.Zero(t, firstMatch(t, p, lookupRequest("request", lookup)).CachedBlockCount())
	lookup.TotalTokens = 32
	batch.Events = []kvevents.GenericEvent{&kvevents.BlockRemovedEvent{BlockHashes: []uint64{10}, DeviceTier: "GPU", GroupIdx: ptr.To(0), Locality: "REMOTE"}}
	observeBatch(t, p, nil, source, batch)
	assert.Equal(t, 32, firstMatch(t, p, lookupRequest("request", lookup)).CachedBlockCount())
	batch.Events = []kvevents.GenericEvent{&kvevents.AllBlocksClearedEvent{}}
	observeBatch(t, p, nil, source, batch)
	assert.Zero(t, firstMatch(t, p, lookupRequest("request", lookup)).MatchBlocks())
}

func TestProducerIndependentReplicas(t *testing.T) {
	first, _ := newTestProducer(t)
	second, _ := newTestProducer(t)
	source := kvevents.EventSource{Endpoint: "10.0.0.1:8080"}
	store := &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"}
	req := lookupRequest("next", attrsession.SessionCacheRequest{SessionID: "next", TotalTokens: 32,
		Prefixes: []attrsession.SessionCachePrefix{{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 20}, BlockSizeTokens: 16, Exact: true}}})
	for _, p := range []*Producer{first, second} {
		observe(t, p, nil, source, store)
		assert.Equal(t, 32, firstMatch(t, p, req).CachedBlockCount())
	}
	require.NoError(t, second.events.Reset(t.Context(), source.Endpoint))
	assert.Equal(t, 32, firstMatch(t, first, req).CachedBlockCount())
	assert.Zero(t, firstMatch(t, second, req).CachedBlockCount())
	observe(t, second, nil, source, store)
	assert.Equal(t, 32, firstMatch(t, second, req).CachedBlockCount())
}

func TestProducerCacheNamespace(t *testing.T) {
	p, _ := newTestProducer(t)
	observe(t, p, nil, kvevents.EventSource{Endpoint: "10.0.0.1:8080"},
		&kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"})
	for _, namespace := range []string{"model-v1", "different-model", ""} {
		t.Run(namespace, func(t *testing.T) {
			req := lookupRequest("request", attrsession.SessionCacheRequest{SessionID: "request", TotalTokens: 32,
				Prefixes: []attrsession.SessionCachePrefix{{CacheNamespace: namespace, BlockHashes: []uint64{10, 20}, BlockSizeTokens: 16, Exact: true}}})
			info := firstMatch(t, p, req)
			expected := 0
			if namespace == "model-v1" {
				expected = 32
			}
			assert.Equal(t, expected, info.MatchBlocks())
			assert.Equal(t, expected, info.CachedBlockCount())
		})
	}
	_, err := PluginFactory("cache", plugin.StrictDecoder(json.RawMessage(`{"sessionCacheRequestProducerName":"sessions"}`)), plugin.NewEppHandle(t.Context(), nil))
	require.ErrorContains(t, err, "cacheNamespace is required")
}

func TestProducerDataDependencies(t *testing.T) {
	p, sessions := newTestProducer(t)
	// Expose only the request producer methods, with no event consumer contract.
	manager := struct{ requestcontrol.DataProducer }{sessions}
	scorer, err := prefix.New(t.Context(), "scorer", "cache")
	require.NoError(t, err)
	order, err := datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{scorer, p, manager})
	require.NoError(t, err)
	require.Len(t, order, 3)
	assert.Less(t, slices.Index(order, manager.TypedName().String()), slices.Index(order, p.TypedName().String()))
	assert.Less(t, slices.Index(order, p.TypedName().String()), slices.Index(order, scorer.TypedName().String()))
	_, err = datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{p, scorer})
	require.Error(t, err, "the named session producer is required")
}
