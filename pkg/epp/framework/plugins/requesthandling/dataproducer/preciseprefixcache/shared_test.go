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

type recordingSource struct {
	consumers map[string]kvevents.EventConsumer
}

func (s *recordingSource) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "kv-events-source", Name: "events"}
}

func (s *recordingSource) AddConsumer(owner plugin.TypedName, consumer kvevents.EventConsumer) error {
	s.consumers[owner.String()] = consumer
	return nil
}

type plainPlugin struct{}

func (plainPlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "other", Name: "other"}
}

func newHandle(t *testing.T, plugins map[string]plugin.Plugin) plugin.Handle {
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	handle := plugin.NewEppHandle(ctx, nil)
	for name, p := range plugins {
		handle.AddPlugin(name, p)
	}
	return handle
}

func TestKVEventsSourceRefBindsTheIndex(t *testing.T) {
	source := &recordingSource{consumers: map[string]kvevents.EventConsumer{}}
	handle := newHandle(t, map[string]plugin.Plugin{"events": source})
	instance, err := PluginFactory("cache", json.NewDecoder(strings.NewReader(`{"kvEventsSourceRef":"events"}`)), handle)
	require.NoError(t, err)
	p := instance.(*Producer)
	consumer := source.consumers[p.TypedName().String()]
	require.NotNil(t, consumer)

	ctx := handle.Context()
	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: newEndpoint("pod", "127.0.0.1")}))
	subscribers, _ := p.subscribersManager.GetActiveSubscribers()
	require.Empty(t, subscribers, "a producer bound to a source opens no subscribers of its own")

	eventSource := kvevents.EventSource{Endpoint: "127.0.0.1:8080", ModelName: "model"}
	stored := &kvevents.BlockStoredEvent{BlockHashes: []uint64{123}, Tokens: make([]uint32, 64), BlockSize: 64}
	require.NoError(t, consumer.ProcessEvents(ctx, eventSource, kvevents.EventBatch{Events: []kvevents.GenericEvent{stored}}))
	tokens, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	keys, err := tokens.TokensToKVBlockKeys(kvblock.EmptyBlockHash, stored.Tokens, eventSource.ModelName, nil)
	require.NoError(t, err)
	require.NotEmpty(t, keys)
	index := p.kvCacheIndexer.KVBlockIndex()
	hits, err := index.Lookup(ctx, []kvblock.BlockHash{keys[0]}, nil)
	require.NoError(t, err)
	require.Len(t, hits[keys[0]], 1)
	require.NoError(t, consumer.Reset(ctx, eventSource))
	hits, err = index.Lookup(ctx, []kvblock.BlockHash{keys[0]}, nil)
	require.NoError(t, err)
	require.Empty(t, hits[keys[0]])
}

func TestKVEventsSourceRefFailsStartupWhenMisconfigured(t *testing.T) {
	handle := newHandle(t, map[string]plugin.Plugin{
		"other":  plainPlugin{},
		"events": &recordingSource{consumers: map[string]kvevents.EventConsumer{}},
	})
	for name, tc := range map[string]struct{ config, want string }{
		"unknown plugin":         {`{"kvEventsSourceRef":"missing"}`, "names no configured plugin"},
		"wrong plugin type":      {`{"kvEventsSourceRef":"other"}`, "a kv-events-source is required"},
		"own transport settings": {`{"kvEventsSourceRef":"events","kvEventsConfig":{"topicFilter":"kv@"}}`, "mutually exclusive"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := PluginFactory("cache", json.NewDecoder(strings.NewReader(tc.config)), handle)
			require.ErrorContains(t, err, tc.want)
		})
	}
}
