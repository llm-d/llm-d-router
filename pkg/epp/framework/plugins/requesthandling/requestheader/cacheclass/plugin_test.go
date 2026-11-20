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

package cacheclass

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func newPlugin(t *testing.T, cfg string) *Plugin {
	t.Helper()
	var dec *json.Decoder
	if cfg != "" {
		dec = json.NewDecoder(bytes.NewBufferString(cfg))
	}
	p, err := Factory("cache-classifier", dec, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	return p.(*Plugin)
}

func chatRequest(id, prefix, suffix string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Raw: prefix + suffix}}},
			},
		},
	}
}

func classOf(t *testing.T, r *fwksched.InferenceRequest) *CacheClass {
	t.Helper()
	c, ok := fwksched.ReadRequestAttribute[*CacheClass](r, CacheClassKey)
	if !ok {
		t.Fatalf("no CacheClass attribute on request %s", r.RequestID)
	}
	return c
}

func TestTypeIsCacheClassifier(t *testing.T) {
	if Type != "cache-classifier" {
		t.Fatalf("Type = %q, want cache-classifier", Type)
	}
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	if p.TypedName().Type != Type {
		t.Fatalf("TypedName.Type = %q, want %q", p.TypedName().Type, Type)
	}
}

func TestClassifiesMissThenHitAfterPreRequest(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 1}`)
	prefix := strings.Repeat("shared conversation context. ", 400)

	first := chatRequest("r1", prefix, " question one")
	if err := p.RequestHeader(ctx, first); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if c := classOf(t, first); c.Hit {
		t.Fatalf("first sighting should be a miss, got ratio %v", c.MatchRatio)
	}
	if got := first.Objectives.Priority; got != 0 {
		t.Fatalf("priority = %d, want 0 unchanged", got)
	}

	if err := p.PreRequest(ctx, first, nil); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}

	second := chatRequest("r2", prefix, " question two")
	if err := p.RequestHeader(ctx, second); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	c := classOf(t, second)
	if !c.Hit {
		t.Fatalf("same prefix should hit after indexing, ratio %v (%d/%d)",
			c.MatchRatio, c.MatchedBlocks, c.TotalBlocks)
	}
	if !c.Ready {
		t.Fatal("class should be Ready after warmupBlocks=1 and one indexed request")
	}
}

func TestDynamicAttributeSeesIndexUpdatesWhileQueued(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0, "minMatchedTokens": 0, "hitMatchRatio": 0.3}`)
	prefix := strings.Repeat("shared conversation context. ", 400)

	queued := chatRequest("queued", prefix, " waiting")
	if err := p.RequestHeader(ctx, queued); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if classOf(t, queued).Hit {
		t.Fatal("queued request should start as miss")
	}

	admitted := chatRequest("admitted", prefix, " done")
	_ = p.RequestHeader(ctx, admitted)
	_ = p.PreRequest(ctx, admitted, nil)

	if !classOf(t, queued).Hit {
		t.Fatal("dynamic attribute should see index update while request still queued")
	}
}

func TestRequestAbandonedClearsPending(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	r := chatRequest("r1", strings.Repeat("x", 8000), " y")
	if err := p.RequestHeader(ctx, r); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if _, ok := p.pending.Load(r.RequestID); !ok {
		t.Fatal("expected pending entry")
	}
	p.RequestAbandoned(ctx, r)
	if _, ok := p.pending.Load(r.RequestID); ok {
		t.Fatal("pending entry should be cleared on abandon")
	}
	// Idempotent.
	p.RequestAbandoned(ctx, r)
}

func TestLookupDoesNotRefreshLRUOrder(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0, "indexCapacity": 2, "minMatchedTokens": 0, "hitMatchRatio": 0.01}`)
	// Fill capacity with two distinct prefixes, then classify without admitting.
	// Peek must not refresh recent-ness (Keys order oldest→newest stays put).
	a := strings.Repeat("aaaaaaaaaaaaaaaa ", 200)
	b := strings.Repeat("bbbbbbbbbbbbbbbb ", 200)

	ra := chatRequest("a", a, "1")
	_ = p.RequestHeader(ctx, ra)
	_ = p.PreRequest(ctx, ra, nil)
	rb := chatRequest("b", b, "1")
	_ = p.RequestHeader(ctx, rb)
	_ = p.PreRequest(ctx, rb, nil)

	keysBefore := p.index.Keys()
	rc := chatRequest("c", a, "2")
	_ = p.RequestHeader(ctx, rc)
	keysAfter := p.index.Keys()
	if len(keysBefore) != len(keysAfter) {
		t.Fatalf("Peek classify changed index len %d -> %d", len(keysBefore), len(keysAfter))
	}
	for i := range keysBefore {
		if keysBefore[i] != keysAfter[i] {
			t.Fatalf("Peek classify refreshed LRU order at %d", i)
		}
	}
}

func TestPreRequestRecordsZeroMatchedTokensBeforeSelfInsert(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	r := chatRequest("cold", strings.Repeat("unique cold prefix content. ", 400), " q")
	_ = p.RequestHeader(ctx, r)
	_ = p.PreRequest(ctx, r, nil)
	v, ok := p.dispatchedMatchedTokens.Load(r.RequestID)
	if !ok {
		t.Fatal("expected dispatch matched tokens")
	}
	matched, ok := v.(int)
	if !ok || matched != 0 {
		t.Fatalf("first cold dispatch MatchedTokens = %v, want 0 (predict before self-insert)", v)
	}
}

func TestResponseBodyRecordsMatchedVsCached(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	prefix := strings.Repeat("shared conversation context. ", 400)
	r := chatRequest("r1", prefix, " q")
	_ = p.RequestHeader(ctx, r)
	_ = p.PreRequest(ctx, r, nil)
	if _, ok := p.dispatchedMatchedTokens.Load(r.RequestID); !ok {
		t.Fatal("expected dispatch matched tokens")
	}
	p.ResponseBody(ctx, r, &fwkrh.Response{
		EndOfStream: true,
		Usage: fwkrequest.Usage{
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 100},
		},
	}, nil)
	if _, ok := p.dispatchedMatchedTokens.Load(r.RequestID); ok {
		t.Fatal("dispatch matched tokens should be cleared after response")
	}
}

func TestResponseBodySkipsWhenCachedUsageAbsent(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	r := chatRequest("r1", strings.Repeat("shared conversation context. ", 400), " q")
	_ = p.RequestHeader(ctx, r)
	_ = p.PreRequest(ctx, r, nil)
	p.ResponseBody(ctx, r, &fwkrh.Response{EndOfStream: true}, nil)
	if _, ok := p.dispatchedMatchedTokens.Load(r.RequestID); ok {
		t.Fatal("dispatch state should still be deleted when usage details are absent")
	}
}

func TestMatchedTokensCapsPartialFinalBlock(t *testing.T) {
	ctx := context.Background()
	// Length not divisible by block size so the final block is partial.
	prefix := strings.Repeat("a", 961*4)
	p := newPlugin(t, `{"warmupBlocks": 0, "blockSizeTokens": 64, "minMatchedTokens": 0, "hitMatchRatio": 0.01}`)
	seed := chatRequest("seed", prefix, "")
	_ = p.RequestHeader(ctx, seed)
	_ = p.PreRequest(ctx, seed, nil)

	r := chatRequest("hit", prefix, "")
	_ = p.RequestHeader(ctx, r)
	c := classOf(t, r)
	uncapped := c.MatchedBlocks * 64
	if c.MatchedBlocks < 2 {
		t.Fatalf("expected multiple blocks, got %d", c.MatchedBlocks)
	}
	if c.MatchedTokens >= uncapped {
		t.Fatalf("MatchedTokens = %d, want < %d (cap partial final block)", c.MatchedTokens, uncapped)
	}
	if c.MatchedTokens <= (c.MatchedBlocks-1)*64 {
		t.Fatalf("MatchedTokens = %d, want above full preceding blocks", c.MatchedTokens)
	}
}

func TestWarmupNotReadyUntilIndexFilled(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 100000}`)
	prefix := strings.Repeat("shared conversation context. ", 400)
	r := chatRequest("r1", prefix, " q")
	if err := p.RequestHeader(ctx, r); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if classOf(t, r).Ready {
		t.Fatal("class must not be Ready before warmup")
	}
}

func TestFactoryRejectsInvalidParams(t *testing.T) {
	cases := map[string]string{
		"block size zero":      `{"blockSizeTokens": 0}`,
		"hit match ratio zero": `{"hitMatchRatio": 0}`,
		"warmup negative":      `{"warmupBlocks": -1}`,
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Factory("cache-classifier", json.NewDecoder(bytes.NewBufferString(cfg)), nil)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestProducesDeclaresCacheClassKey(t *testing.T) {
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	if _, ok := p.Produces()[CacheClassKey]; !ok {
		t.Fatal("Produces must declare CacheClassKey")
	}
	_ = fwkplugin.TypedName{}
}
