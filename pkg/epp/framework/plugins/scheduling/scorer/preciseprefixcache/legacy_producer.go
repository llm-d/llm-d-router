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
	"fmt"
	"sort"

	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// promptTokenizer is the subset of tokenization.Pool the wrapper uses.
// Narrow interface so tests can inject a stub without standing up a UDS
// tokenizer.
type promptTokenizer interface {
	Tokenize(renderReq *tokenizerTypes.RenderChatRequest, prompt string) ([]uint32, *tokenization.MultiModalFeatures)
}

// legacyProducer wraps the public Producer with a prompt-string fallback
// driven by indexerConfig.tokenizersPoolConfig, restoring the heavy
// scorer's "tokenize the prompt myself" behaviour. The pool is owned by
// the wrapper; the embedded Producer stays tokens-only and is unaware of
// the fallback.
type legacyProducer struct {
	*preciseproducer.Producer
	tokenizerPool promptTokenizer
}

// newLegacyProducer strips tokenizersPoolConfig off cfg before handing it
// to preciseproducer.New (so the embedded indexer doesn't build a second
// pool), then constructs the wrapper-owned pool when requested.
func newLegacyProducer(ctx context.Context, name string, cfg preciseproducer.PluginConfig) (*legacyProducer, error) {
	var poolCfg *tokenization.Config
	if cfg.IndexerConfig != nil {
		//nolint:staticcheck // SA1019: legacy config still flows through here.
		if cfg.IndexerConfig.TokenizersPoolConfig != nil {
			poolCfg = cfg.IndexerConfig.TokenizersPoolConfig
			//nolint:staticcheck // SA1019
			cfg.IndexerConfig.TokenizersPoolConfig = nil
		}
	}

	inner, err := preciseproducer.New(ctx, name, cfg)
	if err != nil {
		return nil, err
	}

	lp := &legacyProducer{Producer: inner}
	if poolCfg != nil {
		pool, err := tokenization.NewTokenizationPool(ctx, poolCfg)
		if err != nil {
			return nil, fmt.Errorf("legacy tokenization pool: %w", err)
		}
		// Pool.Tokenize enqueues tasks and blocks on a result channel that
		// only the worker goroutines feed; Run starts them and stops them
		// on ctx cancellation.
		go pool.Run(ctx)
		lp.tokenizerPool = pool
	}
	return lp, nil
}

// Consumes drops the TokenizedPrompt dependency when the wrapper owns a
// tokenizer pool, since the prompt-fallback path tokenizes the request
// itself and no upstream token-producer is required.
func (lp *legacyProducer) Consumes() plugin.DataDependencies {
	if lp.tokenizerPool != nil {
		return plugin.DataDependencies{}
	}
	return lp.Producer.Consumes()
}

// Produce tokenizes the request prompt via the wrapper-owned pool when
// no TokenizedPrompt is set, then delegates to the embedded Producer.
func (lp *legacyProducer) Produce(ctx context.Context,
	request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint,
) error {
	if lp.tokenizerPool != nil && needsLegacyTokenization(request) {
		lp.tokenizeRequest(request)
	}
	return lp.Producer.Produce(ctx, request, endpoints)
}

func needsLegacyTokenization(request *scheduling.InferenceRequest) bool {
	if request == nil || request.Body == nil {
		return false
	}
	if tp := request.Body.TokenizedPrompt; tp != nil && len(tp.TokenIDs) > 0 {
		return false
	}
	return request.Body.Completions != nil || request.Body.ChatCompletions != nil
}

func (lp *legacyProducer) tokenizeRequest(request *scheduling.InferenceRequest) {
	var renderReq *tokenizerTypes.RenderChatRequest
	var prompt string
	switch {
	case request.Body.ChatCompletions != nil:
		renderReq = tokenizer.ChatCompletionsToRenderChatRequest(request.Body.ChatCompletions)
	case request.Body.Completions != nil:
		prompt = request.Body.Completions.Prompt.Raw
	default:
		return
	}

	tokens, mmFeatures := lp.tokenizerPool.Tokenize(renderReq, prompt)
	if len(tokens) == 0 {
		return
	}

	request.Body.TokenizedPrompt = &fwkrh.TokenizedPrompt{
		TokenIDs:           tokens,
		MultiModalFeatures: flattenMMFeatures(mmFeatures),
		CacheSalt:          tokenizer.CacheSaltFromBody(request.Body),
	}
}

// flattenMMFeatures regroups the kvcache map-shaped multimodal metadata
// into the upstream flat list expected on TokenizedPrompt, sorted by
// placeholder offset so consumers see items in prompt order.
func flattenMMFeatures(src *tokenization.MultiModalFeatures) []fwkrh.MultiModalFeature {
	if src == nil || len(src.MMHashes) == 0 {
		return nil
	}
	var items []fwkrh.MultiModalFeature
	for modality, hashes := range src.MMHashes {
		ranges, ok := src.MMPlaceholders[modality]
		if !ok {
			continue
		}
		n := len(hashes)
		if len(ranges) < n {
			n = len(ranges)
		}
		for i := 0; i < n; i++ {
			items = append(items, fwkrh.MultiModalFeature{
				Modality: fwkrh.Modality(modality),
				Hash:     hashes[i],
				Offset:   ranges[i].Offset,
				Length:   ranges[i].Length,
			})
		}
	}
	if len(items) == 0 {
		return nil
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Offset < items[j].Offset })
	return items
}
