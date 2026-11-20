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
//
// The admission-time index is pool-wide on this EPP replica and separate from the
// approximate/precise prefix indexes used for placement scoring.
package cacheclass

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

const (
	// Type is the registered plugin type.
	Type = "cache-classifier"

	// A pseudo-token is 4 bytes (tokenizer.bytesPerToken), so 64 pseudo-tokens covers 256 bytes
	// of prompt text -- coarse enough to keep the index small, fine enough that two requests
	// sharing a conversation prefix agree on many leading blocks.
	defaultBlockSizeTokens = 64
	// Caps work per request. 256 blocks covers ~64KB of prompt; a shared prefix longer than
	// that is already decisively a hit.
	defaultMaxPrefixBlocks = 256
	// Distinct prefix blocks tracked pool-wide on this replica.
	defaultIndexCapacity = 100_000
	// Fraction of the prompt that must be indexed for the request to count as a hit.
	defaultHitMatchRatio = 0.3
	// Cached pseudo-tokens a request must carry before its hit is worth labeling as such.
	defaultMinMatchedTokens = 1024
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
	BlockSizeTokens  int     `json:"blockSizeTokens,omitempty"`
	MaxPrefixBlocks  int     `json:"maxPrefixBlocks,omitempty"`
	IndexCapacity    int     `json:"indexCapacity,omitempty"`
	HitMatchRatio    float64 `json:"hitMatchRatio,omitempty"`
	MinMatchedTokens *int    `json:"minMatchedTokens,omitempty"`
	WarmupBlocks     *int    `json:"warmupBlocks,omitempty"`
}

// Plugin implements pre-admission cache classification.
type Plugin struct {
	typedName fwkplugin.TypedName

	blockSizeTokens  int
	maxPrefixBlocks  int
	hitMatchRatio    float64
	minMatchedTokens int
	warmupBlocks     int

	// index holds prefix block hashes believed resident somewhere in the pool on this
	// replica. It is separate from the approximate/precise prefix indexes used for
	// placement.
	index *lru.Cache[prefixhash.BlockHash, struct{}]

	// pending carries hashes from RequestHeader until PreRequest or RequestAbandoned.
	pending sync.Map
	// dispatchedMatchedTokens stores MatchedTokens observed at PreRequest (dispatch) so
	// ResponseBody can compare them to engine cached_tokens.
	dispatchedMatchedTokens sync.Map
}

var (
	_ fwkrh.RequestHeaderProcessor = &Plugin{}
	_ fwkrh.PreRequest             = &Plugin{}
	_ fwkrh.RequestAbandoned       = &Plugin{}
	_ fwkrh.ResponseBodyProcessor  = &Plugin{}
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
		index:            idx,
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
		initSeries(p.typedName.Name, p.typedName.Type)
	}
	return p, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{CacheClassKey: &CacheClass{}}
}

// RequestHeader stages prefix hashes and publishes a DynamicRequestAttribute so readers
// (including a request still in the FC queue) observe the current index on every read.
func (p *Plugin) RequestHeader(ctx context.Context, request *fwksched.InferenceRequest) error {
	logger := log.FromContext(ctx)

	perPrompt, promptTokens := p.blockHashes(ctx, request)
	totalBlocks := 0
	for _, hs := range perPrompt {
		totalBlocks += len(hs)
	}
	if totalBlocks == 0 {
		return nil
	}

	original := request.Objectives.Priority
	prompts := perPrompt
	lengths := promptTokens
	request.PutAttribute(CacheClassKey, &fwksched.DynamicRequestAttribute{
		Get: func() any {
			return p.classFromPrompts(prompts, lengths, original)
		},
	})

	initial := p.classFromPrompts(prompts, lengths, original)
	p.pending.Store(request.RequestID, &pendingEntry{
		perPrompt:     prompts,
		promptTokens:  lengths,
		matchedTokens: initial.MatchedTokens,
		hit:           initial.Hit,
	})
	recordClassification(p.typedName, initial.Hit, initial.MatchRatio)

	logger.V(logutil.TRACE).Info("Classified request for admission",
		"requestID", request.RequestID, "matchRatio", initial.MatchRatio, "hit", initial.Hit,
		"ready", initial.Ready, "matchedBlocks", initial.MatchedBlocks,
		"totalBlocks", initial.TotalBlocks, "matchedTokens", initial.MatchedTokens,
		"priority", original, "indexSize", p.index.Len())
	return nil
}

// RequestAbandoned drops pending state when the request leaves before PreRequest.
func (p *Plugin) RequestAbandoned(_ context.Context, request *fwksched.InferenceRequest) {
	if request == nil {
		return
	}
	v, ok := p.pending.LoadAndDelete(request.RequestID)
	if !ok {
		return
	}
	entry, ok := v.(*pendingEntry)
	if !ok {
		return
	}
	recordTurnedAway(p.typedName, entry.hit, entry.matchedTokens)
}

// PreRequest records the dispatch-time prediction, then indexes admitted hashes.
func (p *Plugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	v, ok := p.pending.LoadAndDelete(request.RequestID)
	if !ok {
		return nil
	}
	entry, ok := v.(*pendingEntry)
	if !ok {
		return nil
	}
	// Predict against the index as it stood for this request; adding first would
	// make a cold request look like a full match against its own hashes.
	class := p.classFromPrompts(entry.perPrompt, entry.promptTokens, request.Objectives.Priority)
	p.dispatchedMatchedTokens.Store(request.RequestID, class.MatchedTokens)
	for _, hs := range entry.perPrompt {
		for _, h := range hs {
			p.index.Add(h, struct{}{})
		}
	}
	recordIndexSize(p.typedName, p.index.Len())
	return nil
}

// ResponseBody compares MatchedTokens captured at dispatch with engine cached_tokens.
func (p *Plugin) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	response *fwkrh.Response,
	_ *fwkdl.EndpointMetadata,
) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	v, ok := p.dispatchedMatchedTokens.LoadAndDelete(request.RequestID)
	if !ok {
		return
	}
	matched, ok := v.(int)
	if !ok {
		return
	}
	// Streams without usage details are not a zero-cache observation.
	if response.Usage.PromptTokenDetails == nil {
		return
	}
	recordMatchedVsCached(p.typedName, matched, response.Usage.PromptTokenDetails.CachedTokens)
}

func (p *Plugin) classFromPrompts(perPrompt [][]prefixhash.BlockHash, promptTokens []int, originalPriority int) *CacheClass {
	matched, total, matchedTokens := p.lookupPrompts(perPrompt, promptTokens)
	if total == 0 {
		return &CacheClass{OriginalPriority: originalPriority}
	}
	ratio := float64(matched) / float64(total)
	hit := ratio >= p.hitMatchRatio && matchedTokens >= p.minMatchedTokens
	ready := p.warmupBlocks == 0 || p.index.Len() >= p.warmupBlocks
	return &CacheClass{
		MatchRatio:       ratio,
		Hit:              hit,
		MatchedBlocks:    matched,
		TotalBlocks:      total,
		MatchedTokens:    matchedTokens,
		Ready:            ready,
		OriginalPriority: originalPriority,
	}
}

func (p *Plugin) blockHashes(ctx context.Context, request *fwksched.InferenceRequest) ([][]prefixhash.BlockHash, []int) {
	if request == nil || request.Body == nil {
		return nil, nil
	}
	est, err := tokenizer.EstimatePromptTokens(ctx, request.Body)
	if err != nil || est == nil || est.TokenCount() == 0 {
		if err != nil {
			log.FromContext(ctx).V(logutil.DEBUG).Info("Cache classification skipped, estimate failed",
				"requestID", request.RequestID, "err", err)
		}
		return nil, nil
	}

	shadowBody := *request.Body
	shadowBody.TokenizedRequest = est
	shadow := *request
	shadow.Body = &shadowBody

	return prefixhash.GetBlockHashesWithPromptTokens(ctx, &shadow, p.blockSizeTokens, p.maxPrefixBlocks)
}

func (p *Plugin) lookupPrompts(perPrompt [][]prefixhash.BlockHash, promptTokens []int) (matched, total, matchedTokens int) {
	for i, hashes := range perPrompt {
		m, t := p.lookup(hashes)
		matched += m
		total += t
		tokens := m * p.blockSizeTokens
		if i < len(promptTokens) && promptTokens[i] < tokens {
			tokens = promptTokens[i]
		}
		matchedTokens += tokens
	}
	return matched, total, matchedTokens
}

// lookup returns the longest indexed prefix length without refreshing LRU recent-ness.
func (p *Plugin) lookup(hashes []prefixhash.BlockHash) (matched, total int) {
	for _, h := range hashes {
		if _, ok := p.index.Peek(h); !ok {
			break
		}
		matched++
	}
	return matched, len(hashes)
}

type pendingEntry struct {
	perPrompt     [][]prefixhash.BlockHash
	promptTokens  []int
	matchedTokens int
	hit           bool
}
