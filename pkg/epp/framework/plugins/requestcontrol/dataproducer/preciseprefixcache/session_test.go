package preciseprefixcache

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// testSessionManager selects an explicitly named prior request. Production
// managers can resolve continuations using their own identity or content data.
type testSessionManager struct {
	mu           sync.Mutex
	observations map[string]fwkrc.SessionCachePrefix
	resets       []string
	estimated    bool
	namespaces   map[string]string
}

func (*testSessionManager) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-session-manager", Name: "sessions"}
}

func (*testSessionManager) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{sessionTestKey(): fwkrc.SessionCacheRequest{}}
}

func sessionTestKey() plugin.DataKey {
	return fwkrc.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions")
}

func (m *testSessionManager) Produce(_ context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if request.Headers["session"] == "" {
		return nil
	}
	lookup := fwkrc.SessionCacheRequest{Stamp: request.RequestID, FullReport: true,
		TotalTokens: request.Body.TokenizedRequest.TokenCount()}
	m.mu.Lock()
	defer m.mu.Unlock()
	if prefix, ok := m.observations[request.Headers["previous"]]; ok {
		prefix.Exact = !m.estimated
		lookup.Prefixes = []fwkrc.SessionCachePrefix{prefix}
	}
	request.PutAttribute(sessionTestKey(), lookup)
	return nil
}

func (m *testSessionManager) ProcessEvents(_ context.Context, source kvevents.EventSource, batch kvevents.EventBatch) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, event := range batch.Events {
		if ev, ok := event.(*kvevents.BlockStoredEvent); ok && ev.SessionID != nil && m.CacheNamespace(source, ev.GroupIdx) != "" {
			m.observations[*ev.SessionID] = fwkrc.SessionCachePrefix{
				CacheNamespace: m.CacheNamespace(source, ev.GroupIdx),
				BlockHashes:    append([]uint64(nil), ev.BlockHashes...), BlockSizeTokens: ev.BlockSize,
			}
		}
	}
	return nil
}

func (m *testSessionManager) CacheNamespace(source kvevents.EventSource, _ *int) string {
	return m.namespaces[source.Endpoint]
}

func (m *testSessionManager) Reset(_ context.Context, endpoint string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.resets = append(m.resets, endpoint)
	return nil
}

func newSessionProducer(t *testing.T) (*Producer, *testSessionManager) {
	t.Helper()
	handle := plugin.NewEppHandle(t.Context(), nil)
	manager := &testSessionManager{observations: make(map[string]fwkrc.SessionCachePrefix),
		namespaces: map[string]string{"10.0.0.1:8080": "model-v1", "10.0.0.2:8080": "model-v1"}}
	handle.AddPlugin("sessions", manager)
	p, err := PluginFactory("cache", plugin.StrictDecoder(json.RawMessage(`{"sessionManager":"sessions"}`)), handle)
	require.NoError(t, err)
	producer := p.(*Producer)
	assert.Equal(t, map[plugin.DataKey]any{sessionTestKey(): fwkrc.SessionCacheRequest{}}, producer.Consumes().Required)
	return producer, manager
}

func sessionRequest(stamp, previous string) *scheduling.InferenceRequest {
	prompt := strings.Repeat("abcd", 32)
	return &scheduling.InferenceRequest{
		RequestID: stamp, TargetModel: "model", Headers: map[string]string{"session": "logical-session", "previous": previous},
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
			Payload:     fwkrh.PayloadMap{"model": "model", "prompt": prompt},
		},
	}
}

func sessionMatch(t *testing.T, p *Producer, req *scheduling.InferenceRequest) *attrprefix.PrefixCacheMatchInfo {
	t.Helper()
	endpoints := freshEndpoints()
	require.NoError(t, p.Produce(t.Context(), req, endpoints))
	value, ok := endpoints[0].Get(p.dk)
	require.True(t, ok)
	return value.(*attrprefix.PrefixCacheMatchInfo)
}

func TestSessionCacheTokenBackends(t *testing.T) {
	for _, estimate := range []bool{false, true} {
		name := "render"
		if estimate {
			name = "estimate"
		}
		t.Run(name, func(t *testing.T) {
			p, manager := newSessionProducer(t)
			manager.estimated = estimate
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
			tokens := rawTokens.(fwkrc.DataProducer)
			first := sessionRequest("request-1", "")
			require.NoError(t, tokens.Produce(t.Context(), first, nil))
			require.NoError(t, manager.Produce(t.Context(), first, nil))
			require.NoError(t, p.PreRequest(t.Context(), first, nil))
			payload := first.Body.Payload.(fwkrh.PayloadMap)
			assert.Equal(t, "request-1", payload["session_id"])
			assert.Equal(t, "full", payload["vllm_xargs"].(map[string]any)["kv_cache_report_mode"])
			assert.True(t, first.Body.Mutated)

			source := kvevents.EventSource{Endpoint: "10.0.0.1:8080", ModelName: "model", Sequence: 1}
			stored := &kvevents.BlockStoredEvent{SessionID: ptr.To("request-1"), BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"}
			// No event tokens are needed to learn or query engine blocks.
			require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), source, kvevents.EventBatch{Events: []kvevents.GenericEvent{stored}}))
			next := sessionRequest("request-2", "request-1")
			require.NoError(t, tokens.Produce(t.Context(), next, nil))
			require.NoError(t, manager.Produce(t.Context(), next, nil))
			info := sessionMatch(t, p, next)
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

func TestSessionCacheSharedBlocksForksAndEviction(t *testing.T) {
	p, manager := newSessionProducer(t)
	source := kvevents.EventSource{Endpoint: "10.0.0.1:8080"}
	for stamp, hashes := range map[string][]uint64{"session-a": {10, 20}, "session-b": {10, 20}, "fork": {10, 30}} {
		stored := &kvevents.BlockStoredEvent{SessionID: ptr.To(stamp), BlockHashes: hashes, BlockSize: 16, DeviceTier: "gpu"}
		for range 2 {
			require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), source, kvevents.EventBatch{Events: []kvevents.GenericEvent{stored}}))
		}
	}
	lookup := func(stamp string) *attrprefix.PrefixCacheMatchInfo {
		req := sessionRequest("next", stamp)
		req.Body.TokenizedRequest = fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, 32)})
		require.NoError(t, manager.Produce(t.Context(), req, nil))
		return sessionMatch(t, p, req)
	}
	for _, stamp := range []string{"session-a", "session-b", "fork"} {
		assert.Equal(t, 32, lookup(stamp).MatchBlocks())
	}
	remove := &kvevents.BlockRemovedEvent{BlockHashes: []uint64{20}, DeviceTier: "GPU"}
	require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), source, kvevents.EventBatch{Events: []kvevents.GenericEvent{remove}}))
	assert.Equal(t, 16, lookup("session-a").MatchBlocks())
	assert.Equal(t, 16, lookup("session-b").MatchBlocks())
	assert.Equal(t, 32, lookup("fork").MatchBlocks())
	require.Len(t, manager.observations, 3)
	require.NoError(t, p.sessionEvents.Reset(t.Context(), source.Endpoint))
	assert.Zero(t, lookup("fork").MatchBlocks())
	assert.Equal(t, []string{source.Endpoint}, manager.resets)
}

func TestSessionCacheCrossWorkerLineage(t *testing.T) {
	p, manager := newSessionProducer(t)
	for i, endpoint := range []string{"10.0.0.1:8080", "10.0.0.2:8080"} {
		stored := &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"}
		if i == 0 {
			stored.SessionID = ptr.To("previous")
		}
		require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), kvevents.EventSource{Endpoint: endpoint},
			kvevents.EventBatch{Events: []kvevents.GenericEvent{stored}}))
	}
	req := sessionRequest("next", "previous")
	req.Body.TokenizedRequest = fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, 32)})
	require.NoError(t, manager.Produce(t.Context(), req, nil))
	check := func(expected ...int) {
		t.Helper()
		endpoints := freshEndpoints()
		require.NoError(t, p.Produce(t.Context(), req, endpoints))
		for i, endpoint := range endpoints {
			value, ok := endpoint.Get(p.dk)
			require.True(t, ok)
			assert.Equal(t, expected[i], value.(*attrprefix.PrefixCacheMatchInfo).CachedBlockCount(), endpoint.GetMetadata().Address)
		}
	}
	check(32, 32)
	a := kvevents.EventSource{Endpoint: "10.0.0.1:8080"}
	remove := &kvevents.BlockRemovedEvent{BlockHashes: []uint64{20}, DeviceTier: "GPU"}
	require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), a, kvevents.EventBatch{Events: []kvevents.GenericEvent{remove}}))
	check(16, 32)
	require.NoError(t, p.sessionEvents.Reset(t.Context(), a.Endpoint))
	check(0, 32)
	assert.Equal(t, []string{a.Endpoint}, manager.resets)
	assert.Equal(t, []uint64{10, 20}, manager.observations["previous"].BlockHashes)

	b := kvevents.EventSource{Endpoint: "10.0.0.2:8080"}
	require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), b,
		kvevents.EventBatch{Events: []kvevents.GenericEvent{&kvevents.AllBlocksClearedEvent{}}}))
	check(0, 0)
	store := &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16, DeviceTier: "GPU"}
	require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), b, kvevents.EventBatch{Events: []kvevents.GenericEvent{store}}))
	check(0, 32)
}

func TestSessionCacheNamespaceAndBlockSizeIsolation(t *testing.T) {
	for _, tc := range []struct {
		name      string
		namespace string
		blockSize int
	}{
		{name: "incompatible namespace", namespace: "model-v2", blockSize: 16},
		{name: "unconfigured source", blockSize: 16},
		{name: "different block size", namespace: "model-v1", blockSize: 32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, manager := newSessionProducer(t)
			manager.namespaces["10.0.0.2:8080"] = tc.namespace
			store := &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: tc.blockSize, DeviceTier: "GPU"}
			require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), kvevents.EventSource{Endpoint: "10.0.0.2:8080"},
				kvevents.EventBatch{Events: []kvevents.GenericEvent{store}}))
			req := sessionRequest("stamp", "")
			req.PutAttribute(sessionTestKey(), fwkrc.SessionCacheRequest{Stamp: "stamp", TotalTokens: 32,
				Prefixes: []fwkrc.SessionCachePrefix{{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 20}, BlockSizeTokens: 16, Exact: true}}})
			endpoints := freshEndpoints()
			require.NoError(t, p.Produce(t.Context(), req, endpoints))
			for _, endpoint := range endpoints {
				value, ok := endpoint.Get(p.dk)
				require.True(t, ok)
				assert.Zero(t, value.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
			}
		})
	}
}

func TestSessionCacheLocationAndBranchIsolation(t *testing.T) {
	p, _ := newSessionProducer(t)
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
		require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), kvevents.EventSource{Endpoint: observation.endpoint},
			kvevents.EventBatch{DataParallelRank: observation.rank, Events: []kvevents.GenericEvent{store}}))
	}
	req := sessionRequest("stamp", "")
	req.PutAttribute(sessionTestKey(), fwkrc.SessionCacheRequest{Stamp: "stamp", TotalTokens: 48,
		Prefixes: []fwkrc.SessionCachePrefix{
			{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 20, 40}, BlockSizeTokens: 16, Exact: true},
			{CacheNamespace: "model-v1", BlockHashes: []uint64{10, 30, 50}, BlockSizeTokens: 16, Exact: true},
		}})
	check := func() {
		t.Helper()
		endpoints := freshEndpoints()
		require.NoError(t, p.Produce(t.Context(), req, endpoints))
		for _, endpoint := range endpoints {
			value, ok := endpoint.Get(p.dk)
			require.True(t, ok)
			assert.Equal(t, 32, value.(*attrprefix.PrefixCacheMatchInfo).CachedBlockCount())
		}
	}
	check()
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
		require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), kvevents.EventSource{Endpoint: removal.endpoint},
			kvevents.EventBatch{DataParallelRank: removal.rank, Events: []kvevents.GenericEvent{ev}}))
	}
	check()
}

func TestSessionCacheStampEnvelope(t *testing.T) {
	p, _ := newSessionProducer(t)
	for _, args := range []any{map[string]any{"custom": 7.0}, json.RawMessage(`{"custom":7}`)} {
		req := sessionRequest("request", "")
		content := json.RawMessage(`[{"z":1,"a":2}]`)
		req.Body.Payload = fwkrh.PayloadMap{"messages": content, "vllm_xargs": args, "session_id": "external-logical-id"}
		req.PutAttribute(sessionTestKey(), fwkrc.SessionCacheRequest{Stamp: "manager-stamp", FullReport: true})
		require.NoError(t, p.PreRequest(t.Context(), req, nil))
		payload := req.Body.Payload.(fwkrh.PayloadMap)
		assert.Equal(t, content, payload["messages"])
		assert.Equal(t, "manager-stamp", payload["session_id"])
		encoded, err := json.Marshal(payload["vllm_xargs"])
		require.NoError(t, err)
		assert.JSONEq(t, `{"custom":7,"kv_cache_report_mode":"full"}`, string(encoded))
	}
	for _, payload := range []fwkrh.RequestPayload{fwkrh.RawPayload(`{}`), fwkrh.PayloadMap{"vllm_xargs": "invalid"}} {
		req := sessionRequest("request", "")
		req.Body.Payload = payload
		req.PutAttribute(sessionTestKey(), fwkrc.SessionCacheRequest{Stamp: "manager-stamp"})
		require.Error(t, p.PreRequest(t.Context(), req, nil))
		assert.False(t, req.Body.Mutated)
	}
	req := sessionRequest("request", "")
	require.NoError(t, p.PreRequest(t.Context(), req, nil))
	require.NoError(t, p.Produce(t.Context(), req, nil))
	assert.False(t, req.Body.Mutated)
}

func TestSessionCacheStampPreservesLargeArguments(t *testing.T) {
	p, _ := newSessionProducer(t)
	req := sessionRequest("stamp", "")
	req.Body.Payload = fwkrh.PayloadMap{"vllm_xargs": json.RawMessage(`{"custom":9007199254740993}`)}
	req.PutAttribute(sessionTestKey(), fwkrc.SessionCacheRequest{Stamp: "stamp"})
	require.NoError(t, p.PreRequest(t.Context(), req, nil))
	encoded, err := json.Marshal(req.Body.Payload)
	require.NoError(t, err)
	assert.Contains(t, string(encoded), `"custom":9007199254740993`)
}

func TestSessionCacheConfiguration(t *testing.T) {
	_, err := PluginFactory("cache", plugin.StrictDecoder(json.RawMessage(`{"sessionManager":"missing"}`)), plugin.NewEppHandle(t.Context(), nil))
	require.ErrorContains(t, err, "missing")
	for _, params := range []string{
		`{"sessionManager":"sessions","speculativeIndexing":true}`,
		`{"sessionManager":"sessions","kvEventsConfig":{"engineType":"sglang"}}`,
		`{"sessionManager":"sessions","kvEventsConfig":{"zmqEndpoint":"tcp://localhost:5557"}}`,
	} {
		handle := plugin.NewEppHandle(t.Context(), nil)
		handle.AddPlugin("sessions", &testSessionManager{})
		_, err := PluginFactory("cache", plugin.StrictDecoder(json.RawMessage(params)), handle)
		require.Error(t, err)
	}
}

func TestSessionCacheCoverageAndEventScope(t *testing.T) {
	p, _ := newSessionProducer(t)
	source := kvevents.EventSource{Endpoint: "10.0.0.1:8080"}
	lookup := fwkrc.SessionCacheRequest{Stamp: "stamp", TotalTokens: 32, Prefixes: []fwkrc.SessionCachePrefix{{
		CacheNamespace: "model-v1", BlockHashes: []uint64{10, 20}, BlockSizeTokens: 16, Exact: true,
	}}}
	req := sessionRequest("stamp", "")
	store := &kvevents.BlockStoredEvent{BlockHashes: []uint64{10, 20}, BlockSize: 16,
		DeviceTier: "GPU", GroupIdx: ptr.To(0), KVCacheSpecKind: kvevents.KVCacheSpecKindFullAttention}
	batch := kvevents.EventBatch{DataParallelRank: ptr.To(1), Events: []kvevents.GenericEvent{store}}
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
			require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), source, batch))
			req.PutAttribute(sessionTestKey(), lookup)
			if kind == "local" {
				assert.Equal(t, 32, sessionMatch(t, p, req).CachedBlockCount())
			} else {
				assert.Zero(t, sessionMatch(t, p, req).MatchBlocks())
			}
		})
	}
	lookup.TotalTokens = 31
	req.PutAttribute(sessionTestKey(), lookup)
	assert.Equal(t, 16, sessionMatch(t, p, req).CachedBlockCount())
	lookup.TotalTokens = 15
	req.PutAttribute(sessionTestKey(), lookup)
	assert.Zero(t, sessionMatch(t, p, req).CachedBlockCount())
	lookup.TotalTokens = 32
	lookup.Prefixes[0].CacheNamespace = "model-v2"
	req.PutAttribute(sessionTestKey(), lookup)
	assert.Zero(t, sessionMatch(t, p, req).MatchBlocks())
	lookup.Prefixes[0].CacheNamespace = "model-v1"
	req.PutAttribute(sessionTestKey(), lookup)
	remove := &kvevents.BlockRemovedEvent{BlockHashes: []uint64{10}, DeviceTier: "GPU", GroupIdx: ptr.To(0), Locality: "REMOTE"}
	batch.Events = []kvevents.GenericEvent{remove}
	require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), source, batch))
	assert.Equal(t, 32, sessionMatch(t, p, req).CachedBlockCount())
	batch.Events = []kvevents.GenericEvent{&kvevents.AllBlocksClearedEvent{}}
	require.NoError(t, p.sessionEvents.ProcessEvents(t.Context(), source, batch))
	assert.Zero(t, sessionMatch(t, p, req).MatchBlocks())
}
