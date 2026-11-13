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
	"fmt"
	"strings"
	"testing"
	"time"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
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

// chatRequest builds a chat request whose prompt is prefix + suffix, so two requests sharing
// prefix share their leading block hashes.
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

// A prefix the plugin has never indexed must classify as a miss, and an identical prefix
// classifies as a hit only after a request carrying it is actually admitted (PreRequest).
func TestClassifiesMissThenHitAfterPreRequest(t *testing.T) {
	ctx := context.Background()
	// warmupBlocks 1: this test is about classification, not the cold-start guard.
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
	if got := second.Objectives.Priority; got != 0 {
		t.Fatalf("priority = %d, want 0 unchanged", got)
	}
	if !c.Ready {
		t.Fatal("warm index should mark the class Ready")
	}
}

// A request rejected before PreRequest must not index itself, otherwise a flood of rejected
// cold traffic would score as warm on retry.
func TestRejectedRequestDoesNotIndex(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, "")
	prefix := strings.Repeat("cold traffic body. ", 400)

	first := chatRequest("r1", prefix, " a", 0)
	if err := p.RequestHeader(ctx, first); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	// No PreRequest: the request was rejected at admission.

	second := chatRequest("r2", prefix, " b", 0)
	if err := p.RequestHeader(ctx, second); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if c := classOf(t, second); c.Hit {
		t.Fatalf("unadmitted prefix must not be indexed, got ratio %v", c.MatchRatio)
	}
}

// Customer priority is not a cache-class channel.
func TestPriorityStaysAtCustomerTier(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 1}`)
	req := chatRequest("r1", strings.Repeat("body text. ", 400), " q", 3)
	if err := p.RequestHeader(ctx, req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if got := req.Objectives.Priority; got != 3 {
		t.Fatalf("priority = %d, want 3 unchanged", got)
	}
	if c := classOf(t, req); c.OriginalPriority != 3 {
		t.Fatalf("OriginalPriority = %d, want 3", c.OriginalPriority)
	}
}

// Nothing hashable must fail open: the request keeps the band it arrived with.
func TestUnhashableRequestFailsOpen(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, "")
	req := &fwksched.InferenceRequest{RequestID: "r1", Objectives: fwksched.RequestObjectives{Priority: 7}}
	if err := p.RequestHeader(ctx, req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if got := req.Objectives.Priority; got != 7 {
		t.Fatalf("priority = %d, want 7 unchanged", got)
	}
	if _, ok := fwksched.ReadRequestAttribute[*CacheClass](req, CacheClassKey); ok {
		t.Fatal("unhashable request should carry no CacheClass attribute")
	}
}

func TestFactoryRejectsInvalidConfig(t *testing.T) {
	for name, cfg := range map[string]string{
		"shed threshold above one": `{"shedThreshold": 1.5}`,
		"ratio above one":          `{"hitMatchRatio": 1.5}`,
		"ratio at zero":            `{"hitMatchRatio": 0}`,
		"zero capacity":            `{"indexCapacity": 0}`,
		"negative blocks":          `{"maxPrefixBlocks": -1}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Factory("x", json.NewDecoder(bytes.NewBufferString(cfg)), nil); err == nil {
				t.Fatalf("expected error for %s", cfg)
			}
		})
	}
}

var _ fwkplugin.Plugin = &Plugin{}

// A short prompt can be fully cached and still carry almost no committed prefill. The
// investment floor must keep it out of the protected band even at ratio 1.0.
func TestInvestmentFloorDemotesTrivialHits(t *testing.T) {
	ctx := context.Background()
	// blockSizeTokens 64 => a 4-block match is 256 tokens, below a 4096-token floor.
	p := newPlugin(t, `{"minMatchedTokens": 4096, "warmupBlocks": 1}`)
	short := strings.Repeat("hi there. ", 30)

	first := chatRequest("r1", short, " a", 0)
	if err := p.RequestHeader(ctx, first); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if err := p.PreRequest(ctx, first, nil); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}

	second := chatRequest("r2", short, " a", 0)
	if err := p.RequestHeader(ctx, second); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	c := classOf(t, second)
	if c.MatchRatio < 0.9 {
		t.Fatalf("precondition: expected a high ratio re-send, got %v", c.MatchRatio)
	}
	if c.Hit {
		t.Fatalf("ratio %.2f with only %d cached tokens must not count as a hit",
			c.MatchRatio, c.MatchedTokens)
	}
}

// The same prefix clears the floor once it is long enough to represent real investment.
func TestInvestmentFloorAdmitsSubstantialHits(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"minMatchedTokens": 1024}`)
	long := strings.Repeat("substantial shared conversation context. ", 600)

	first := chatRequest("r1", long, " a", 0)
	if err := p.RequestHeader(ctx, first); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if err := p.PreRequest(ctx, first, nil); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}

	second := chatRequest("r2", long, " b", 0)
	if err := p.RequestHeader(ctx, second); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	c := classOf(t, second)
	if !c.Hit {
		t.Fatalf("substantial cached prefix should hit: ratio %.2f, %d cached tokens",
			c.MatchRatio, c.MatchedTokens)
	}
}

// A classified request that is never dispatched must have its entry reclaimed, otherwise the
// map grows without bound under exactly the overload this plugin measures.
func TestSweepReclaimsRejectedRequests(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, "")
	p.pendingTTL = 50 * time.Millisecond

	for i := 0; i < 5; i++ {
		req := chatRequest(fmt.Sprintf("r%d", i), strings.Repeat("rejected body. ", 400), " q", 0)
		if err := p.RequestHeader(ctx, req); err != nil {
			t.Fatalf("RequestHeader: %v", err)
		}
	}
	// None reached PreRequest: all were shed at admission.
	if n := pendingLen(p); n != 5 {
		t.Fatalf("pending = %d, want 5 before sweep", n)
	}
	if swept := p.sweep(time.Now()); swept != 0 {
		t.Fatalf("swept %d entries that are not yet stale", swept)
	}
	if swept := p.sweep(time.Now().Add(time.Second)); swept != 5 {
		t.Fatalf("swept = %d, want 5", swept)
	}
	if n := pendingLen(p); n != 0 {
		t.Fatalf("pending = %d, want 0 after sweep", n)
	}
}

// A dispatched request claims its own entry, so the sweep must not double-count it.
func TestSweepLeavesDispatchedRequestsAlone(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, "")
	p.pendingTTL = 50 * time.Millisecond

	req := chatRequest("r1", strings.Repeat("dispatched body. ", 400), " q", 0)
	if err := p.RequestHeader(ctx, req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if err := p.PreRequest(ctx, req, nil); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}
	if swept := p.sweep(time.Now().Add(time.Second)); swept != 0 {
		t.Fatalf("swept = %d, want 0 -- entry was already claimed", swept)
	}
}

func pendingLen(p *Plugin) int {
	n := 0
	p.pending.Range(func(_, _ any) bool { n++; return true })
	return n
}

// A just-restarted router has an empty index, so everything looks like a miss. Until the
// index is warm the class is not Ready and the dispatch gate must not hold it.
func TestColdIndexIsNotReady(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 1000000}`) // never warm for this test
	req := chatRequest("r1", strings.Repeat("some conversation. ", 400), " q", 5)
	if err := p.RequestHeader(ctx, req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if got := req.Objectives.Priority; got != 5 {
		t.Fatalf("priority = %d, want 5 unchanged", got)
	}
	c := classOf(t, req)
	if c.Ready {
		t.Fatal("cold index must not mark the class Ready")
	}
}

// Once the index is warm a repeat prefix is a Ready hit and priority is still the tier.
func TestWarmIndexMarksReadyHit(t *testing.T) {
	ctx := context.Background()
	p := newPlugin(t, `{"warmupBlocks": 1, "minMatchedTokens": 0}`)
	prefix := strings.Repeat("warm conversation body. ", 400)

	seed := chatRequest("r0", prefix, " a", 0)
	if err := p.RequestHeader(ctx, seed); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if err := p.PreRequest(ctx, seed, nil); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}
	if p.index.Len() < 1 {
		t.Fatal("precondition: index should hold blocks after PreRequest")
	}

	req := chatRequest("r1", prefix, " b", 0)
	if err := p.RequestHeader(ctx, req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	c := classOf(t, req)
	if !c.Hit || !c.Ready {
		t.Fatalf("re-sent prefix should be a Ready hit, hit=%v ready=%v", c.Hit, c.Ready)
	}
	if got := req.Objectives.Priority; got != 0 {
		t.Fatalf("priority = %d, want 0 unchanged", got)
	}
}

func TestAllowHoldsReadyMissAboveThreshold(t *testing.T) {
	p := newPlugin(t, `{"shedThreshold": 0.85, "warmupBlocks": 0}`)
	miss := &CacheClass{Hit: false, Ready: true}
	hit := &CacheClass{Hit: true, Ready: true}
	cold := &CacheClass{Hit: false, Ready: false}

	if !p.Allow(context.Background(), itemWithClass(miss), 0.5) {
		t.Fatal("miss below threshold should dispatch")
	}
	if p.Allow(context.Background(), itemWithClass(miss), 0.85) {
		t.Fatal("ready miss at threshold should stay queued")
	}
	if !p.Allow(context.Background(), itemWithClass(hit), 0.99) {
		t.Fatal("hit should dispatch above threshold")
	}
	if !p.Allow(context.Background(), itemWithClass(cold), 0.99) {
		t.Fatal("class that is not Ready should dispatch")
	}
	if got, want := FlowID("tenant", miss), FlowID("tenant", hit); got == want {
		t.Fatal("hit and miss must not share a flow id")
	}
	if got, want := FlowID("tenant", cold), "tenant"; got != want {
		t.Fatalf("not-ready class flow id = %q, want %q", got, want)
	}
	if got, want := ParentFlowID(FlowID("tenant", miss)), "tenant"; got != want {
		t.Fatalf("ParentFlowID(miss) = %q, want %q", got, want)
	}
	if got, want := ParentFlowID(FlowID("tenant", hit)), "tenant"; got != want {
		t.Fatalf("ParentFlowID(hit) = %q, want %q", got, want)
	}

	req := &fwksched.InferenceRequest{RequestID: "rewrite"}
	req.PutAttribute(CacheClassKey, miss)
	if got, want := p.RewriteFlowID("tenant", req), FlowID("tenant", miss); got != want {
		t.Fatalf("RewriteFlowID = %q, want %q", got, want)
	}
	if got := p.RewriteFlowID("tenant", &fwksched.InferenceRequest{RequestID: "bare"}); got != "tenant" {
		t.Fatalf("RewriteFlowID without class = %q, want tenant", got)
	}

	if _, ok := p.Produces()[CacheClassKey]; !ok {
		t.Fatal("Produces() must declare CacheClassKey")
	}
}

func itemWithClass(class *CacheClass) *fwkfcmocks.MockQueueItemAccessor {
	req := &fwksched.InferenceRequest{RequestID: "r"}
	req.PutAttribute(CacheClassKey, class)
	return &fwkfcmocks.MockQueueItemAccessor{
		OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{InferenceRequestV: req},
	}
}
