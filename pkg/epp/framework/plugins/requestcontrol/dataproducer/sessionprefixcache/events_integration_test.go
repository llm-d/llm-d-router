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
	"encoding/binary"
	"net"
	"testing"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

func TestIndependentSessionEventSubscriptions(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	address := "tcp://" + listener.Addr().String()
	require.NoError(t, listener.Close())
	publisher := zmq4.NewPub(ctx)
	require.NoError(t, publisher.Listen(address))
	defer publisher.Close()
	cfg := kvevents.DefaultConfig()
	cfg.PodDiscoveryConfig.SocketPort = port
	cfg.PodDiscoveryConfig.ReplaySocketPort = -1
	p, err := New(ctx, "cache", PluginConfig{CacheNamespace: "model-v1", SessionCacheRequestProducerName: "sessions", IndexConfig: kvblock.DefaultIndexConfig(), KVEventsConfig: cfg})
	require.NoError(t, err)
	sessions := &testSessionProducer{observations: make(map[string]attrsession.SessionCachePrefix)}
	pool, err := kvevents.NewConsumerPool(cfg, engineadapter.NewVLLMAdapter(), sessions)
	require.NoError(t, err)
	pool.Start(ctx)
	defer pool.Shutdown(ctx)
	manager := kvevents.NewSubscriberManager(pool)
	defer manager.Shutdown(ctx)
	subscriptions, err := kvevents.NewEndpointSubscriptions(ctx, cfg, manager)
	require.NoError(t, err)
	metadata := &fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "engine"}, Address: "127.0.0.1", Port: "8080"}
	endpoint := fwkdl.NewEndpoint(metadata, nil)
	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}))
	require.NoError(t, subscriptions.Ensure(ctx, "external/engine", metadata.Address, metadata.Port, 0))

	store := func(hashes ...uint64) map[string]any {
		return map[string]any{"type": "BlockStored", "block_hashes": hashes, "block_size": 16, "token_ids": []uint32{}, "medium": "GPU", "session_id": "first"}
	}
	// Each event reaches separate pools: one learns session paths, the other residency.
	publishUntilMatch := func(sequence uint64, event map[string]any, want int) {
		t.Helper()
		payload, err := msgpack.Marshal([]any{0.0, []any{event}, nil})
		require.NoError(t, err)
		seq := make([]byte, 8)
		binary.BigEndian.PutUint64(seq, sequence)
		require.Eventually(t, func() bool {
			if err := publisher.Send(zmq4.NewMsgFrom([]byte("kv@127.0.0.1:8080@model"), seq, payload)); err != nil {
				return false
			}
			sessions.mu.Lock()
			learned := len(sessions.observations["first"].BlockHashes)
			sessions.mu.Unlock()
			if learned == 0 {
				return false
			}
			req := sessionRequest("next", "first")
			req.Body.TokenizedRequest = fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, 32)})
			if err := sessions.Produce(ctx, req, nil); err != nil {
				return false
			}
			ep := scheduling.NewEndpoint(metadata, nil, nil)
			if err := p.Produce(ctx, req, []scheduling.Endpoint{ep}); err != nil {
				return false
			}
			value, ok := ep.Get(p.dk)
			return ok && value.(*attrprefix.PrefixCacheMatchInfo).CachedBlockCount() == want
		}, 5*time.Second, 20*time.Millisecond)
	}
	publishUntilMatch(1, store(10, 20), 32)
	publishUntilMatch(2, map[string]any{"type": "BlockRemoved", "block_hashes": []uint64{20}, "medium": "GPU"}, 16)
	sessions.mu.Lock()
	learned := len(sessions.observations["first"].BlockHashes)
	sessions.mu.Unlock()
	require.Equal(t, 2, learned, "eviction preserves the manager's logical lineage")
	publishUntilMatch(3, store(10, 20), 32)
	publishUntilMatch(4, map[string]any{"type": "AllBlocksCleared"}, 0)
	publishUntilMatch(5, store(10, 20), 32)
	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: endpoint}))
	publishUntilMatch(6, store(10, 20), 0)
}
