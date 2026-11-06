// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package preciseprefixcache

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

type eventTestHandle struct {
	plugin.Handle
	ctx context.Context
}

func (h eventTestHandle) Context() context.Context { return h.ctx }

func TestExternalEventsRequireExplicitConfiguration(t *testing.T) {
	for _, external := range []bool{false, true} {
		config := `{}`
		if external {
			config = `{"externalEvents":true}`
		}
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		instance, err := PluginFactory("cache", json.NewDecoder(strings.NewReader(config)), eventTestHandle{ctx: ctx})
		require.NoError(t, err)
		consumer, ok := instance.(kvevents.EventConsumer)
		require.Equal(t, external, ok)
		if !external {
			continue
		}
		p := instance.(*externalEventsProducer)
		require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: newEndpoint("pod", "127.0.0.1")}))
		subscribers, _ := p.subscribersManager.GetActiveSubscribers()
		require.Empty(t, subscribers, "shared consumers must not open their own subscribers")
		source := kvevents.EventSource{Endpoint: "127.0.0.1:8080", ModelName: "model"}
		stored := &kvevents.BlockStoredEvent{BlockHashes: []uint64{123}, Tokens: make([]uint32, 64), BlockSize: 64}
		require.NoError(t, consumer.ProcessEvents(ctx, source, kvevents.EventBatch{Events: []kvevents.GenericEvent{stored}}))
		index := p.kvCacheIndexer.KVBlockIndex()
		key, err := index.GetRequestKey(ctx, kvblock.BlockHash(123))
		require.NoError(t, err)
		hits, err := index.Lookup(ctx, []kvblock.BlockHash{key}, nil)
		require.NoError(t, err)
		require.Len(t, hits[key], 1)
		require.NoError(t, consumer.Reset(ctx, source))
		hits, err = index.Lookup(ctx, []kvblock.BlockHash{key}, nil)
		require.NoError(t, err)
		require.Empty(t, hits[key])
	}
}
