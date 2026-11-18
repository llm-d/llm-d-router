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

// Package cacheclass classifies a request as a prefix-cache hit or miss before admission
// control runs and attaches that class as a request attribute for metrics and downstream
// consumers. It does not change priority, fairness identity, or flow-control dispatch.
package cacheclass

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

const (
	// Type is the registered plugin type.
	Type = "cache-class-router"

	// A pseudo-token is 4 bytes (tokenizer.bytesPerToken), so 64 pseudo-tokens covers 256 bytes
	// of prompt text -- coarse enough to keep the index small, fine enough that two requests
	// sharing a conversation prefix agree on many leading blocks.
	defaultBlockSizeTokens = 64
	// Caps work per request. 256 blocks covers ~64KB of prompt; a shared prefix longer than
	// that is already decisively a hit.
	defaultMaxPrefixBlocks = 256
	// Distinct prefix blocks tracked pool-wide.
	defaultIndexCapacity = 100_000
	// Fraction of the prompt that must be indexed for the request to count as a hit.
	defaultHitMatchRatio = 0.3
	// Cached pseudo-tokens a request must carry before its hit is worth labeling as such.
	defaultMinMatchedTokens = 1024
	// Age after which an unclaimed entry is treated as shed: attributed as turned-away and
	// removed. Admitted requests claim their entry in PreRequest before this elapses.
	defaultPendingTTL = 2 * time.Minute
	// How often unclaimed entries are swept.
	defaultSweepInterval = 30 * time.Second
	// Indexed blocks required before a class is Ready. The index starts empty, so every
	// request looks like a miss until admitted traffic fills it.
	defaultWarmupBlocks = 5000
)

// CacheClassKey carries the classification to any downstream plugin that wants it (scorers,
// metrics, capacity, sharing) without re-deriving it.
var CacheClassKey = fwkplugin.NewDataKey("CacheClassKey", Type)

// CacheClass is the verdict attached to the request.
type CacheClass struct {
	// MatchRatio is matched blocks over total blocks, in [0,1].
	MatchRatio float64
	// Hit reports whether MatchRatio met the configured threshold.
	Hit bool
	// MatchedBlocks and TotalBlocks are the raw counts behind MatchRatio.
	MatchedBlocks int
	TotalBlocks   int
	// MatchedTokens is the cached prefix length in tokens -- the prefill investment this
	// request would reuse.
	MatchedTokens int
	// Ready is false until the index has passed warmup. Downstream consumers that act on
	// Hit should treat !Ready as unclassified.
	Ready bool
	// OriginalPriority is the customer tier observed at classification. The plugin does not
	// rewrite it.
	OriginalPriority int
}

// Parameters is the JSON configuration for the plugin.
type Parameters struct {
	// BlockSizeTokens is the pseudo-token block granularity. Optional.
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
	// MaxPrefixBlocks caps blocks hashed per request. Optional.
	MaxPrefixBlocks int `json:"maxPrefixBlocks,omitempty"`
	// IndexCapacity is the number of prefix blocks retained pool-wide. Optional.
	IndexCapacity int `json:"indexCapacity,omitempty"`
	// HitMatchRatio is the match fraction at or above which a request counts as a hit.
	// Optional; must be in (0,1].
	HitMatchRatio float64 `json:"hitMatchRatio,omitempty"`
	// MinMatchedTokens is the cached-token floor below which a request is treated as a miss
	// regardless of its match ratio. Optional; 0 disables the floor.
	MinMatchedTokens *int `json:"minMatchedTokens,omitempty"`
	// WarmupBlocks is how many blocks the index must hold before a class becomes Ready.
	// Optional; 0 disables the guard.
	WarmupBlocks *int `json:"warmupBlocks,omitempty"`
}

// Plugin implements pre-admission cache classification.
type Plugin struct {
	typedName fwkplugin.TypedName

	blockSizeTokens  int
	maxPrefixBlocks  int
	hitMatchRatio    float64
	minMatchedTokens int

	warmupBlocks  int
	pendingTTL    time.Duration
	sweepInterval time.Duration

	// index holds prefix block hashes believed resident somewhere in the pool. It is
	// pool-wide rather than per-endpoint on purpose: classification answers whether a
	// prefix was recently admitted, not which endpoint holds it.
	index *lru.Cache[prefixhash.BlockHash, struct{}]

	// pending carries each classified request from admission through to PreRequest, so the
	// index is updated only for requests that were actually admitted and routed. Unclaimed
	// entries are swept after pendingTTL and counted as turned away for metrics.
	pending sync.Map
}

var (
	_ fwkrh.RequestHeaderProcessor = &Plugin{}
	_ fwkrh.PreRequest             = &Plugin{}
)

// Factory builds the plugin from JSON configuration.
func Factory(name string, raw *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{
		BlockSizeTokens: defaultBlockSizeTokens,
		MaxPrefixBlocks: defaultMaxPrefixBlocks,
		IndexCapacity:   defaultIndexCapacity,
		HitMatchRatio:   defaultHitMatchRatio,
	}
	minMatched := defaultMinMatchedTokens
	warmup := defaultWarmupBlocks
	if raw != nil {
		if err := raw.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for '%s' plugin: %w", Type, err)
		}
	}
	if params.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("plugin '%s': blockSizeTokens must be > 0, got %d", Type, params.BlockSizeTokens)
	}
	if params.MaxPrefixBlocks <= 0 {
		return nil, fmt.Errorf("plugin '%s': maxPrefixBlocks must be > 0, got %d", Type, params.MaxPrefixBlocks)
	}
	if params.IndexCapacity <= 0 {
		return nil, fmt.Errorf("plugin '%s': indexCapacity must be > 0, got %d", Type, params.IndexCapacity)
	}
	if params.HitMatchRatio <= 0 || params.HitMatchRatio > 1 {
		return nil, fmt.Errorf("plugin '%s': hitMatchRatio must be in (0,1], got %g", Type, params.HitMatchRatio)
	}
	if params.MinMatchedTokens != nil {
		if *params.MinMatchedTokens < 0 {
			return nil, fmt.Errorf("plugin '%s': minMatchedTokens must be >= 0, got %d", Type, *params.MinMatchedTokens)
		}
		minMatched = *params.MinMatchedTokens
	}
	if params.WarmupBlocks != nil {
		if *params.WarmupBlocks < 0 {
			return nil, fmt.Errorf("plugin '%s': warmupBlocks must be >= 0, got %d", Type, *params.WarmupBlocks)
		}
		warmup = *params.WarmupBlocks
	}

	idx, err := lru.New[prefixhash.BlockHash, struct{}](params.IndexCapacity)
	if err != nil {
		return nil, fmt.Errorf("plugin '%s': failed to create index: %w", Type, err)
	}

	p := &Plugin{
		typedName:        fwkplugin.TypedName{Type: Type, Name: name},
		blockSizeTokens:  params.BlockSizeTokens,
		maxPrefixBlocks:  params.MaxPrefixBlocks,
		hitMatchRatio:    params.HitMatchRatio,
		minMatchedTokens: minMatched,
		warmupBlocks:     warmup,
		pendingTTL:       defaultPendingTTL,
		sweepInterval:    defaultSweepInterval,
		index:            idx,
	}
	// Tests construct the plugin with a nil handle and drive sweep() directly, so the
	// handle-dependent wiring must not be fatal in its absence.
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
		initSeries(p.typedName.Name, p.typedName.Type)
		go p.runJanitor(handle.Context())
	}
	return p, nil
}

// TypedName returns the plugin type and instance name.
func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Produces declares the CacheClass request attribute written by RequestHeader.
func (p *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{CacheClassKey: &CacheClass{}}
}

// RequestHeader classifies the request and attaches CacheClass. It does not change priority.
//
// It fails open on any problem deriving hashes and leaves the request without a class.
func (p *Plugin) RequestHeader(ctx context.Context, request *fwksched.InferenceRequest) error {
	logger := log.FromContext(ctx)

	perPrompt := p.blockHashes(ctx, request)
	matched, total := p.lookupPrompts(perPrompt)
	if total == 0 {
		return nil
	}

	ratio := float64(matched) / float64(total)
	matchedTokens := matched * p.blockSizeTokens
	hit := ratio >= p.hitMatchRatio && matchedTokens >= p.minMatchedTokens
	ready := p.warmupBlocks == 0 || p.index.Len() >= p.warmupBlocks
	original := request.Objectives.Priority

	request.PutAttribute(CacheClassKey, &CacheClass{
		MatchRatio:       ratio,
		Hit:              hit,
		MatchedBlocks:    matched,
		TotalBlocks:      total,
		MatchedTokens:    matchedTokens,
		Ready:            ready,
		OriginalPriority: original,
	})

	var flat []prefixhash.BlockHash
	for _, hs := range perPrompt {
		flat = append(flat, hs...)
	}
	p.pending.Store(request.RequestID, &pendingEntry{
		hashes:        flat,
		matchedTokens: matchedTokens,
		hit:           hit,
		at:            time.Now(),
	})

	recordClassification(p.typedName, hit, ratio)

	logger.V(logutil.TRACE).Info("Classified request for admission",
		"requestID", request.RequestID, "matchRatio", ratio, "hit", hit, "ready", ready,
		"matchedBlocks", matched, "totalBlocks", total, "matchedTokens", matchedTokens,
		"priority", original, "indexSize", p.index.Len())
	return nil
}

// PreRequest records the request's prefix blocks as resident, now that it has actually been
// admitted and routed. Indexing here rather than at classification time keeps the index a
// record of what the pool was asked to serve, not of what it turned away.
func (p *Plugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	v, ok := p.pending.LoadAndDelete(request.RequestID)
	if !ok {
		return nil
	}
	entry, ok := v.(*pendingEntry)
	if !ok {
		return nil
	}
	for _, h := range entry.hashes {
		p.index.Add(h, struct{}{})
	}
	recordIndexSize(p.typedName, p.index.Len())
	return nil
}

// blockHashes derives content-stable prefix block hashes without a tokenizer round-trip.
func (p *Plugin) blockHashes(ctx context.Context, request *fwksched.InferenceRequest) [][]prefixhash.BlockHash {
	if request == nil || request.Body == nil {
		return nil
	}
	est, err := tokenizer.EstimatePromptTokens(ctx, request.Body)
	if err != nil || est == nil || est.TokenCount() == 0 {
		if err != nil {
			log.FromContext(ctx).V(logutil.DEBUG).Info("Cache classification skipped, estimate failed",
				"requestID", request.RequestID, "err", err)
		}
		return nil
	}

	shadowBody := *request.Body
	shadowBody.TokenizedRequest = est
	shadow := *request
	shadow.Body = &shadowBody

	return prefixhash.GetBlockHashes(ctx, &shadow, p.blockSizeTokens, p.maxPrefixBlocks)
}

// lookupPrompts matches each prompt's prefix independently and sums matched/total blocks.
func (p *Plugin) lookupPrompts(perPrompt [][]prefixhash.BlockHash) (matched, total int) {
	for _, hashes := range perPrompt {
		m, t := p.lookup(hashes)
		matched += m
		total += t
	}
	return matched, total
}

// lookup returns the length of the longest indexed prefix, in blocks, and the total block
// count. It stops at the first gap: prefix caching is contiguous from the start of the
// prompt.
func (p *Plugin) lookup(hashes []prefixhash.BlockHash) (matched, total int) {
	for _, h := range hashes {
		if _, ok := p.index.Get(h); !ok {
			break
		}
		matched++
	}
	return matched, len(hashes)
}

type pendingEntry struct {
	hashes        []prefixhash.BlockHash
	matchedTokens int
	hit           bool
	at            time.Time
}

func (p *Plugin) runJanitor(ctx context.Context) {
	ticker := time.NewTicker(p.sweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.sweep(time.Now())
		}
	}
}

// sweep reclaims entries for requests that never reached PreRequest and attributes what they
// were carrying for metrics.
func (p *Plugin) sweep(now time.Time) (swept int) {
	p.pending.Range(func(k, v any) bool {
		entry, ok := v.(*pendingEntry)
		if !ok {
			p.pending.Delete(k)
			return true
		}
		if now.Sub(entry.at) < p.pendingTTL {
			return true
		}
		p.pending.Delete(k)
		swept++
		recordTurnedAway(p.typedName, entry.hit, entry.matchedTokens)
		return true
	})
	return swept
}
