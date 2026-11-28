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

package tokenizer

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// messageCountIDs returns one token ID per message, so a longer conversation's
// IDs extend the shorter one's: the prefix property the cache relies on.
func messageCountIDs(payload fwkrequest.RequestPayload) []uint32 {
	chain, ok := chatPrefixChainFromPayload(payload)
	if !ok {
		return nil
	}
	ids := make([]uint32, len(chain.nodes)-1)
	for i := range ids {
		ids[i] = uint32(i)
	}
	return ids
}

func countingMock(calls *atomic.Int32) *mockTokenizer {
	return &mockTokenizer{renderChatFunc: func(payload fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
		calls.Add(1)
		return messageCountIDs(payload), nil, nil
	}}
}

const (
	chainTurn1 = `{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"u1"}]}`
	chainTurn2 = `{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"u1"},{"role":"assistant","content":"a1"},{"role":"user","content":"u2"}]}`
)

func TestChatPrefixChainSharedPrefix(t *testing.T) {
	c1, ok := chatPrefixChainFromBytes([]byte(chainTurn1))
	require.True(t, ok)
	c2, ok := chatPrefixChainFromBytes([]byte(chainTurn2))
	require.True(t, ok)

	require.Len(t, c1.nodes, 3) // root + 2 messages
	require.Len(t, c2.nodes, 5) // root + 4 messages
	// Turn 2's node over its first 2 messages equals turn 1's full node.
	assert.Equal(t, c1.nodes[2], c2.nodes[2])
	assert.NotEqual(t, c1.nodes[2], c2.nodes[4])
	assert.True(t, c1.textOnly)
	assert.True(t, c2.textOnly)
}

func TestChatPrefixChainSaltFields(t *testing.T) {
	base := `{"model":"m","temperature":0.1,"messages":[{"role":"user","content":"u"}]}`
	hot := `{"model":"m","temperature":0.9,"messages":[{"role":"user","content":"u"}]}`
	tools := `{"model":"m","tools":[{"type":"function","function":{"name":"f"}}],"messages":[{"role":"user","content":"u"}]}`
	toolsReordered := `{"tools":[{"type":"function","function":{"name":"f"}}],"model":"m","messages":[{"role":"user","content":"u"}]}`

	cb, ok := chatPrefixChainFromBytes([]byte(base))
	require.True(t, ok)
	ch, ok := chatPrefixChainFromBytes([]byte(hot))
	require.True(t, ok)
	ct, ok := chatPrefixChainFromBytes([]byte(tools))
	require.True(t, ok)
	cr, ok := chatPrefixChainFromBytes([]byte(toolsReordered))
	require.True(t, ok)

	// Generation-only fields do not salt the chain.
	assert.Equal(t, cb.nodes[0], ch.nodes[0])
	// Template-affecting fields do.
	assert.NotEqual(t, cb.nodes[0], ct.nodes[0])
	// Envelope key order does not.
	assert.Equal(t, ct.nodes[0], cr.nodes[0])
}

func TestChatPrefixChainRejectsNonChat(t *testing.T) {
	_, ok := chatPrefixChainFromBytes([]byte(`{"model":"m","prompt":"hello"}`))
	assert.False(t, ok, "completions prompt has no messages array")

	_, ok = chatPrefixChainFromBytes([]byte(`{"model":"m","messages":"not-an-array"}`))
	assert.False(t, ok)

	_, ok = chatPrefixChainFromBytes([]byte(`{"model":"m","messages":["plain-string"]}`))
	assert.False(t, ok, "messages must be objects")

	_, ok = chatPrefixChainFromBytes([]byte(`[1,2,3]`))
	assert.False(t, ok)
}

func TestChatPrefixChainMultimodalMarker(t *testing.T) {
	mm := `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,x"}}]}]}`
	c, ok := chatPrefixChainFromBytes([]byte(mm))
	require.True(t, ok)
	assert.False(t, c.textOnly)

	// Escaped marker text inside a message does not count as multimodal.
	escaped := `{"model":"m","messages":[{"role":"user","content":"she said {\"type\":\"image_url\"} out loud"}]}`
	c, ok = chatPrefixChainFromBytes([]byte(escaped))
	require.True(t, ok)
	assert.True(t, c.textOnly)
}

func TestPrefixCachePartialHitServesPrefixAndWarms(t *testing.T) {
	var calls atomic.Int32
	r := newPrefixCachedRenderer(countingMock(&calls), "test",
		&prefixCacheConfig{Enabled: true, AsyncWarm: &asyncWarmConfig{Concurrency: 1, Timeout: "5s"}})
	ctx := context.Background()
	turn1 := fwkrequest.RawPayload(chainTurn1)
	turn2 := fwkrequest.RawPayload(chainTurn2)

	// Turn 1 misses and renders synchronously.
	ids, _, err := r.RenderChat(ctx, turn1)
	require.NoError(t, err)
	assert.Equal(t, []uint32{0, 1}, ids)
	assert.Equal(t, int32(1), calls.Load())

	// Turn 2 partially hits: the cached prefix is published without a
	// synchronous render.
	ids, _, err = r.RenderChat(ctx, turn2)
	require.NoError(t, err)
	assert.Equal(t, []uint32{0, 1}, ids, "partial hit publishes the cached prefix tokens")
	assert.Equal(t, int32(1), calls.Load(), "no synchronous render on partial hit")

	// The background warm renders the full prompt and stores it.
	chain, ok := chatPrefixChainFromPayload(turn2)
	require.True(t, ok)
	assert.Eventually(t, func() bool {
		return r.cache.lookup(chain.nodes[len(chain.nodes)-1]) != nil
	}, 2*time.Second, 5*time.Millisecond)
	assert.Equal(t, int32(2), calls.Load())

	// An identical request now full-hits.
	ids, _, err = r.RenderChat(ctx, turn2)
	require.NoError(t, err)
	assert.Equal(t, []uint32{0, 1, 2, 3}, ids)
	assert.Equal(t, int32(2), calls.Load(), "full hit does not render")
}

func TestPrefixCacheFullHitReturnsStoredTokens(t *testing.T) {
	var calls atomic.Int32
	r := newPrefixCachedRenderer(countingMock(&calls), "test", &prefixCacheConfig{Enabled: true})
	payload := fwkrequest.RawPayload(chainTurn1)

	_, _, err := r.RenderChat(context.Background(), payload)
	require.NoError(t, err)

	ids, _, err := r.RenderChat(context.Background(), payload)
	require.NoError(t, err)
	assert.Equal(t, []uint32{0, 1}, ids)
	assert.Equal(t, int32(1), calls.Load())
}

func TestPrefixCacheRenderErrorNotStored(t *testing.T) {
	boom := errors.New("render down")
	var attempts atomic.Int32
	mock := &mockTokenizer{renderChatFunc: func(_ fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
		attempts.Add(1)
		return nil, nil, boom
	}}
	r := newPrefixCachedRenderer(mock, "test", &prefixCacheConfig{Enabled: true})
	payload := fwkrequest.RawPayload(chainTurn1)

	_, _, err := r.RenderChat(context.Background(), payload)
	assert.ErrorIs(t, err, boom)

	// Not stored, so the retry renders again.
	_, _, err = r.RenderChat(context.Background(), payload)
	assert.ErrorIs(t, err, boom)
	assert.Equal(t, int32(2), attempts.Load())
}

func TestPrefixCacheEviction(t *testing.T) {
	var calls atomic.Int32
	r := newPrefixCachedRenderer(countingMock(&calls), "test",
		&prefixCacheConfig{Enabled: true, MaxEntries: 1, AsyncWarm: &asyncWarmConfig{Concurrency: 1}})
	other := `{"model":"m","messages":[{"role":"system","content":"other"},{"role":"user","content":"q"}]}`

	_, _, err := r.RenderChat(context.Background(), fwkrequest.RawPayload(chainTurn1))
	require.NoError(t, err)
	// A different conversation (no shared node) evicts the first entry.
	_, _, err = r.RenderChat(context.Background(), fwkrequest.RawPayload(other))
	require.NoError(t, err)
	assert.Equal(t, int32(2), calls.Load())

	// The evicted conversation renders again.
	_, _, err = r.RenderChat(context.Background(), fwkrequest.RawPayload(chainTurn1))
	require.NoError(t, err)
	assert.Equal(t, int32(3), calls.Load())
}

func TestPrefixCacheSkipsWarmForMultimodal(t *testing.T) {
	var calls atomic.Int32
	r := newPrefixCachedRenderer(countingMock(&calls), "test",
		&prefixCacheConfig{Enabled: true, AsyncWarm: &asyncWarmConfig{Concurrency: 2, Timeout: "5s"}})
	mmTurn2 := `{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"u1"},{"role":"user","content":[{"type":"text","text":"img"},{"type":"image_url","image_url":{"url":"data:image/png;base64,x"}}]}]}`

	_, _, err := r.RenderChat(context.Background(), fwkrequest.RawPayload(chainTurn1))
	require.NoError(t, err)

	// Partial hit over the text prefix; the multimodal suffix suppresses the
	// background warm, so the inner renderer is not called again.
	ids, _, err := r.RenderChat(context.Background(), fwkrequest.RawPayload(mmTurn2))
	require.NoError(t, err)
	assert.Equal(t, []uint32{0, 1}, ids)
	assert.Never(t, func() bool { return calls.Load() >= 2 }, 200*time.Millisecond, 10*time.Millisecond,
		"multimodal payloads must not warm in the background")
}

func TestPrefixCacheRenderPassthrough(t *testing.T) {
	var calls atomic.Int32
	mock := countingMock(&calls)
	renderCalls := atomic.Int32{}
	mock.renderFunc = func(payload fwkrequest.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
		renderCalls.Add(1)
		return [][]uint32{messageCountIDs(payload)}, nil, nil
	}
	r := newPrefixCachedRenderer(mock, "test", &prefixCacheConfig{Enabled: true})

	// Completions rendering is not chained and passes through.
	arrays, _, err := r.Render(context.Background(), fwkrequest.RawPayload(chainTurn1))
	require.NoError(t, err)
	assert.Equal(t, [][]uint32{{0, 1}}, arrays)
	assert.Equal(t, int32(1), renderCalls.Load())
	assert.Zero(t, calls.Load(), "RenderChat must not be invoked")
}

func TestPrefixCacheConcurrentAccess(t *testing.T) {
	var calls atomic.Int32
	r := newPrefixCachedRenderer(countingMock(&calls), "test",
		&prefixCacheConfig{Enabled: true, MaxEntries: 8})
	payloads := []fwkrequest.RequestPayload{
		fwkrequest.RawPayload(chainTurn1),
		fwkrequest.RawPayload(chainTurn2),
		fwkrequest.RawPayload(`{"model":"m","messages":[{"role":"system","content":"a"},{"role":"user","content":"x"}]}`),
		fwkrequest.RawPayload(`{"model":"m","messages":[{"role":"system","content":"b"},{"role":"user","content":"y"}]}`),
	}
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, _, _ = r.RenderChat(context.Background(), payloads[i%len(payloads)])
		}(i)
	}
	wg.Wait()
}

func TestPrefixCacheConfigParseAndDefaults(t *testing.T) {
	raw := `{"prefixCache":{"enabled":true,"maxEntries":5,"maxBytes":1048576,"asyncWarm":{"concurrency":2,"timeout":"5s"}}}`
	var cfg vllmConfig
	require.NoError(t, json.Unmarshal([]byte(raw), &cfg))
	require.NotNil(t, cfg.PrefixCache)
	assert.True(t, cfg.PrefixCache.Enabled)
	assert.Equal(t, 5, cfg.PrefixCache.MaxEntries)
	assert.Equal(t, int64(1048576), cfg.PrefixCache.MaxBytes)
	assert.Equal(t, 2, cfg.PrefixCache.AsyncWarm.Concurrency)
	assert.Equal(t, "5s", cfg.PrefixCache.AsyncWarm.Timeout)

	d := cfg.PrefixCache.withDefaults()
	assert.Equal(t, 5, d.MaxEntries)
	assert.Equal(t, int64(1048576), d.MaxBytes)
	assert.Equal(t, 5*time.Second, d.AsyncWarm.timeout)

	empty := (&prefixCacheConfig{}).withDefaults()
	assert.Equal(t, defaultPrefixCacheMaxEntries, empty.MaxEntries)
	assert.Equal(t, int64(defaultPrefixCacheMaxBytes), empty.MaxBytes)
	assert.Equal(t, defaultAsyncWarmConcurrency, empty.AsyncWarm.Concurrency)
	assert.Equal(t, defaultAsyncWarmTimeout, empty.AsyncWarm.timeout)
}

func TestNewPluginPrefixCacheWiring(t *testing.T) {
	ctx := context.Background()

	p, err := NewPlugin(ctx, "tok", &tokenizerPluginConfig{
		ModelName: "m",
		VLLM:      &vllmConfig{PrefixCache: &prefixCacheConfig{Enabled: true}},
	})
	require.NoError(t, err)
	rb, ok := p.backend.(renderBackend)
	require.True(t, ok)
	_, wrapped := rb.tk.(*prefixCachedRenderer)
	assert.True(t, wrapped, "enabled prefixCache wraps the renderer")
	// The wrapper keeps the render timeout delegation intact.
	assert.Equal(t, defaultHTTPRenderMMTimeout, p.ProduceTimeout())

	p2, err := NewPlugin(ctx, "tok", &tokenizerPluginConfig{ModelName: "m", VLLM: &vllmConfig{}})
	require.NoError(t, err)
	rb2, ok := p2.backend.(renderBackend)
	require.True(t, ok)
	_, wrapped2 := rb2.tk.(*prefixCachedRenderer)
	assert.False(t, wrapped2, "disabled prefixCache leaves the renderer unwrapped")
}
