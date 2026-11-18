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
	"time"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func newPlugin(t *testing.T, cfg string) *Plugin {
	t.Helper()
	var dec *json.Decoder
	if cfg != "" {
		dec = json.NewDecoder(bytes.NewBufferString(cfg))
	}
	p, err := Factory("cache-class", dec, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	return p.(*Plugin)
}

func chatRequest(id, prefix, suffix string, priority int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Raw: prefix + suffix}}},
			},
		},
		Objectives: fwksched.RequestObjectives{Priority: priority},
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

func TestClassifiesMissThenHitAfterPreRequest(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 1}`)
	prefix := strings.Repeat("shared conversation context. ", 400)

	first := chatRequest("r1", prefix, " question one", 0)
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

	second := chatRequest("r2", prefix, " question two", 0)
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

func TestWarmupNotReadyUntilIndexFilled(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 100000}`)
	prefix := strings.Repeat("shared conversation context. ", 400)
	r := chatRequest("r1", prefix, " q", 0)
	if err := p.RequestHeader(ctx, r); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	c := classOf(t, r)
	if c.Ready {
		t.Fatal("class must not be Ready before warmup")
	}
}

func TestMinMatchedTokensForcesMiss(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0, "minMatchedTokens": 100000, "hitMatchRatio": 0.01}`)
	prefix := strings.Repeat("shared conversation context. ", 400)
	first := chatRequest("r1", prefix, " a", 0)
	_ = p.RequestHeader(ctx, first)
	_ = p.PreRequest(ctx, first, nil)
	second := chatRequest("r2", prefix, " b", 0)
	_ = p.RequestHeader(ctx, second)
	if classOf(t, second).Hit {
		t.Fatal("hitMatchRatio alone must not mark a hit below minMatchedTokens")
	}
}

func TestFactoryRejectsInvalidParams(t *testing.T) {
	cases := map[string]string{
		"block size zero":          `{"blockSizeTokens": 0}`,
		"hit match ratio zero":     `{"hitMatchRatio": 0}`,
		"hit match ratio over one": `{"hitMatchRatio": 1.1}`,
		"min matched negative":     `{"minMatchedTokens": -1}`,
		"warmup negative":          `{"warmupBlocks": -1}`,
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Factory("cache-class", json.NewDecoder(bytes.NewBufferString(cfg)), nil)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestProducesDeclaresCacheClassKey(t *testing.T) {
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	prod := p.Produces()
	if _, ok := prod[CacheClassKey]; !ok {
		t.Fatal("Produces must declare CacheClassKey")
	}
	_ = fwkplugin.TypedName{}
}

func TestSweepTurnsAwayUnclaimed(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0}`)
	p.pendingTTL = time.Millisecond
	r := chatRequest("r1", strings.Repeat("x", 8000), " y", 0)
	if err := p.RequestHeader(ctx, r); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	time.Sleep(2 * time.Millisecond)
	if got := p.sweep(time.Now()); got != 1 {
		t.Fatalf("sweep = %d, want 1", got)
	}
	if _, ok := p.pending.Load(r.RequestID); ok {
		t.Fatal("pending entry should be removed")
	}
}

func TestLookupPromptsIndependent(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 0, "minMatchedTokens": 0, "hitMatchRatio": 0.3}`)
	warm := strings.Repeat("warm prefix content here. ", 200)
	cold := strings.Repeat("totally different cold text. ", 200)

	w := chatRequest("w1", warm, " turn", 0)
	_ = p.RequestHeader(ctx, w)
	_ = p.PreRequest(ctx, w, nil)

	// Completions-style multi-prompt is exercised via two separate chat requests: the
	// independent-prompt path is covered by lookupPrompts unit behavior through classify.
	hit := chatRequest("w2", warm, " again", 0)
	_ = p.RequestHeader(ctx, hit)
	if !classOf(t, hit).Hit {
		t.Fatal("warm prefix should hit")
	}
	miss := chatRequest("c1", cold, " q", 0)
	_ = p.RequestHeader(ctx, miss)
	if classOf(t, miss).Hit {
		t.Fatal("cold prefix should miss")
	}
}
