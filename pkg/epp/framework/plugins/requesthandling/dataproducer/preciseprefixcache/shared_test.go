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
	srckvevents "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

type plainPlugin struct{}

func (plainPlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "other", Name: "other"}
}

// newHandle returns a handle holding a kv-events-source named "events" and a
// plugin of another type named "other".
func newHandle(t *testing.T) (plugin.Handle, *srckvevents.Source) {
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	cfg := kvevents.DefaultConfig()
	cfg.PodDiscoveryConfig.PodLabelSelector = ""
	source, err := srckvevents.New(ctx, "events", cfg)
	require.NoError(t, err)
	handle := plugin.NewEppHandle(ctx, nil)
	handle.AddPlugin("events", source)
	handle.AddPlugin("other", plainPlugin{})
	return handle, source
}

func TestKVEventsSourceRefBindsTheIndex(t *testing.T) {
	handle, source := newHandle(t)
	instance, err := PluginFactory("cache", json.NewDecoder(strings.NewReader(`{"kvEventsSourceRef":"events"}`)), handle)
	require.NoError(t, err)
	p := instance.(*Producer)
	require.NotNil(t, p.sharedEvents)
	state, err := source.DumpState()
	require.NoError(t, err)
	require.Contains(t, string(state), `"consumers":["`+p.TypedName().String()+`"]`)

	ctx := handle.Context()
	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: newEndpoint("pod", "127.0.0.1")}))
	require.Nil(t, p.subscribersManager, "a producer bound to a source opens no subscribers of its own")

	eventSource := kvevents.EventSource{Endpoint: "127.0.0.1:8080", ModelName: "model"}
	stored := &kvevents.BlockStoredEvent{BlockHashes: []uint64{123}, Tokens: make([]uint32, 64), BlockSize: 64}
	require.NoError(t, p.sharedEvents.ProcessEvents(ctx, eventSource, kvevents.EventBatch{Events: []kvevents.GenericEvent{stored}}))
	tokens, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	keys, err := tokens.TokensToKVBlockKeys(kvblock.EmptyBlockHash, stored.Tokens, eventSource.ModelName, nil)
	require.NoError(t, err)
	require.NotEmpty(t, keys)
	index := p.kvCacheIndexer.KVBlockIndex()
	hits, err := index.Lookup(ctx, []kvblock.BlockHash{keys[0]}, nil)
	require.NoError(t, err)
	require.Len(t, hits[keys[0]], 1)
	require.NoError(t, p.sharedEvents.Reset(ctx, eventSource))
	hits, err = index.Lookup(ctx, []kvblock.BlockHash{keys[0]}, nil)
	require.NoError(t, err)
	require.Empty(t, hits[keys[0]])
}

func TestKVEventsSourceRefFailsStartupWhenMisconfigured(t *testing.T) {
	handle, _ := newHandle(t)
	for name, tc := range map[string]struct{ config, want string }{
		"unknown plugin":         {`{"kvEventsSourceRef":"missing"}`, "no plugin with the name 'missing'"},
		"wrong plugin type":      {`{"kvEventsSourceRef":"other"}`, "not an instance of *kvevents.Source"},
		"own transport settings": {`{"kvEventsSourceRef":"events","kvEventsConfig":{"topicFilter":"kv@"}}`, "mutually exclusive"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := PluginFactory("cache", json.NewDecoder(strings.NewReader(tc.config)), handle)
			require.ErrorContains(t, err, tc.want)
		})
	}
}
