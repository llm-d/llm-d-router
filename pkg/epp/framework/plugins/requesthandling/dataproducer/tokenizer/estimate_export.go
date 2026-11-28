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

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

// EstimatePromptTokens packs a request body's prompt bytes into pseudo-tokens using the
// tokenizer-free estimate backend, handling every body shape (chat, Anthropic messages,
// completions, embeddings, pre-tokenized input) the same way the configured backend would.
//
// It exists for consumers that run before the token-producer DataProducer and cannot afford
// its /render round-trip -- notably admission-time callers, which need a stable content hash
// rather than engine-accurate token IDs. The IDs are suitable for content-locality hashing
// only; they never match engine KV blocks, so they must not be fed to engine-correlated
// consumers such as the precise prefix cache. Callers pair this with their own index, built
// from the same pseudo-tokens, so the hashes are self-consistent.
//
// The returned TokenizedRequest is freshly allocated and never aliases body.
func EstimatePromptTokens(ctx context.Context, body *fwkrequest.InferenceRequestBody) (*fwkrequest.TokenizedRequest, error) {
	if body == nil {
		return nil, nil
	}
	b := estimateBackend{img: newImageEstimator(nil), vid: newVideoEstimator(nil)}
	tp, err := b.produce(ctx, body)
	if err != nil || tp == nil {
		return tp, err
	}
	// Match the token-producer: prefix hashing reads TokenizedRequest.CacheSalt.
	tp.CacheSalt = CacheSaltFromBody(body)
	return tp, nil
}
