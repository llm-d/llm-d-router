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
	"github.com/vmihailenco/msgpack/v5"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	srckvevents "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
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

func TestSGLangDecodeEventsThroughSharedSource(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)

	handle := plugin.NewEppHandle(ctx, nil)
	srcInstance, err := srckvevents.Factory("decode-events", json.NewDecoder(strings.NewReader(`{
		"engineType": "sglang",
		"podDiscoveryConfig": {
			"podLabelSelector": "llm-d.ai/role=decode",
			"socketPort": 5557
		}
	}`)), handle)
	require.NoError(t, err)
	source := srcInstance.(*srckvevents.Source)
	handle.AddPlugin("decode-events", source)

	prodInstance, err := PluginFactory("decode-cache", json.NewDecoder(strings.NewReader(`{
		"kvEventsSourceRef": "decode-events",
		"tokenProcessorConfig": {"blockSizeTokens": 16}
	}`)), handle)
	require.NoError(t, err)
	producer := prodInstance.(*Producer)

	adapter := engineadapter.NewSGLangAdapter()
	publishSGLang := func(endpoint string, events ...map[string]any) {
		t.Helper()
		rawEvents := make([]any, len(events))
		for i, ev := range events {
			rawEvents[i] = ev
		}
		payload, marshalErr := msgpack.Marshal([]any{1.0, rawEvents, nil})
		require.NoError(t, marshalErr)
		podID, modelName, batch, parseErr := adapter.ParseMessage(&kvevents.RawMessage{
			SourceEndpoint: endpoint,
			Topic:          "kv@test-model",
			Payload:        payload,
		})
		require.NoError(t, parseErr)
		require.NoError(t, source.ProcessEvents(ctx, kvevents.EventSource{
			Endpoint:  podID,
			ModelName: modelName,
		}, batch))
	}

	plainPrompt := make([]uint32, 32)
	for i := range uint32(32) {
		plainPrompt[i] = 100 + i
	}
	// Decode pod 10.0.0.1:8080 (--disaggregation-decode-enable-radix-cache with HiCache)
	// keeps block 0 in GPU and backs up blocks 0..1 to CPU_PINNED, salted.
	publishSGLang("10.0.0.1:8080",
		map[string]any{
			"type":         "BlockStored",
			"block_hashes": []int64{1001},
			"token_ids":    plainPrompt[:16],
			"block_size":   16,
			"medium":       "GPU",
			"cache_salt":   "tenant-a",
			"session_id":   "sess-1",
		},
		map[string]any{
			"type":         "BlockStored",
			"block_hashes": []int64{1001},
			"token_ids":    plainPrompt[:16],
			"block_size":   16,
			"medium":       "CPU_PINNED",
			"cache_salt":   "tenant-a",
			"session_id":   "sess-1",
		},
		map[string]any{
			"type":              "BlockStored",
			"block_hashes":      []int64{1002},
			"parent_block_hash": int64(1001),
			"token_ids":         plainPrompt[16:32],
			"block_size":        16,
			"medium":            "CPU_PINNED",
			"cache_salt":        "tenant-a",
			"session_id":        "sess-1",
		},
	)

	dataKey := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("decode-cache")
	endpoints := freshEndpoints()
	saltedReq := &scheduling.InferenceRequest{
		RequestID:   "req-salted",
		TargetModel: "test-model",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				CacheSalt: "tenant-a",
				Prompts:   []fwkrequest.PromptTokens{{TokenIDs: plainPrompt}},
			},
		},
	}
	require.NoError(t, producer.Produce(ctx, saltedReq, endpoints))
	raw, ok := endpoints[0].Get(dataKey)
	require.True(t, ok)
	info := raw.(*attrprefix.PrefixCacheMatchInfo)
	require.Equal(t, 16, info.BlockSizeTokens())
	require.Equal(t, 2, info.CachedBlockCount())
	require.Equal(t, map[string]int{"gpu": 1, "cpu_pinned": 2}, info.CachedBlocksByTier())
	// 1.0 (gpu) + 0.8 (cpu_pinned) floors to 1 block.
	require.Equal(t, 1, info.MatchBlocks())

	unsaltedEndpoints := freshEndpoints()
	unsaltedReq := &scheduling.InferenceRequest{
		RequestID:   "req-unsalted",
		TargetModel: "test-model",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: plainPrompt}},
			},
		},
	}
	require.NoError(t, producer.Produce(ctx, unsaltedReq, unsaltedEndpoints))
	raw, ok = unsaltedEndpoints[0].Get(dataKey)
	require.True(t, ok)
	require.Equal(t, 0, raw.(*attrprefix.PrefixCacheMatchInfo).CachedBlockCount())

	// Decode pod 10.0.0.2:8080 runs EAGLE3 at --page-size 4: two 4-bigram pages
	// span 9 raw tokens (positions 0..3 with lookahead 4, and 4..7 with lookahead 8).
	bigramTokens := []uint32{10, 20, 30, 40, 50, 60, 70, 80, 90}
	pairs := func(slice []uint32) [][]uint32 {
		out := make([][]uint32, len(slice)-1)
		for i := range out {
			out[i] = []uint32{slice[i], slice[i+1]}
		}
		return out
	}
	publishSGLang("10.0.0.2:8080",
		map[string]any{
			"type":         "BlockStored",
			"block_hashes": []int64{2001},
			"token_ids":    pairs(bigramTokens[0:5]),
			"block_size":   4,
			"medium":       "GPU",
			"session_id":   "sess-2",
		},
		map[string]any{
			"type":              "BlockStored",
			"block_hashes":      []int64{2002},
			"parent_block_hash": int64(2001),
			"token_ids":         pairs(bigramTokens[4:9]),
			"block_size":        4,
			"medium":            "GPU",
			"session_id":        "sess-2",
		},
	)

	// 9 tokens provide the lookahead for both 4-bigram pages (8 cached tokens).
	fullBigramEndpoints := freshEndpoints()
	require.NoError(t, producer.Produce(ctx, &scheduling.InferenceRequest{
		RequestID:   "req-bigram-9",
		TargetModel: "test-model",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: bigramTokens}},
			},
		},
	}, fullBigramEndpoints))
	raw, ok = fullBigramEndpoints[1].Get(dataKey)
	require.True(t, ok)
	bigramInfo := raw.(*attrprefix.PrefixCacheMatchInfo)
	require.Equal(t, 1, bigramInfo.BlockSizeTokens())
	require.Equal(t, 8, bigramInfo.CachedBlockCount())
	require.Equal(t, map[string]int{"gpu": 8}, bigramInfo.CachedBlocksByTier())

	// 8 tokens span only 7 bigrams, so only the first 4-bigram page matches.
	eightTokenEndpoints := freshEndpoints()
	require.NoError(t, producer.Produce(ctx, &scheduling.InferenceRequest{
		RequestID:   "req-bigram-8",
		TargetModel: "test-model",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: bigramTokens[:8]}},
			},
		},
	}, eightTokenEndpoints))
	raw, ok = eightTokenEndpoints[1].Get(dataKey)
	require.True(t, ok)
	require.Equal(t, 4, raw.(*attrprefix.PrefixCacheMatchInfo).CachedBlockCount())
}
