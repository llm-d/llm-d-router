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

	kvctok "github.com/llm-d/llm-d-kv-cache/pkg/tokenization"

	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// LegacyPool wraps the deprecated tokenization pool (sourced from
// llm-d-kv-cache) and exposes an in-tree-typed Tokenize so callers stay
// decoupled from the external module.
//
// Deprecated: tokenize externally and use the tokens-in scoring API.
type LegacyPool struct {
	//nolint:staticcheck // SA1019: the deprecated pool is intentionally retained for the legacy path
	pool *kvctok.Pool
}

// NewLegacyPool builds the deprecated tokenization pool from its config.
func NewLegacyPool(ctx context.Context, cfg *kvctok.Config) (*LegacyPool, error) {
	//nolint:staticcheck // SA1019: deprecated pool retained for the legacy path
	pool, err := kvctok.NewTokenizationPool(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return &LegacyPool{pool: pool}, nil
}

// Run starts the pool workers and blocks until ctx is cancelled.
func (p *LegacyPool) Run(ctx context.Context) { p.pool.Run(ctx) }

// Tokenize renders and tokenizes the request, converting the external results
// to the in-tree tokenization types.
func (p *LegacyPool) Tokenize(renderReq *tokenizerTypes.RenderChatRequest, prompt string) ([]uint32, *tokenization.MultiModalFeatures) {
	extReq, err := toExternalRenderChatRequest(renderReq)
	if err != nil {
		return nil, nil
	}
	tokens, mmf := p.pool.Tokenize(extReq, prompt)
	return tokens, toInTreeMMFeatures(mmf)
}
