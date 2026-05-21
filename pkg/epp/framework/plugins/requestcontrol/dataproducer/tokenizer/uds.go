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

	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

// udsTokenizerAdapter adapts the kvc UdsTokenizer (which has no ctx in its
// signature) to the local ctx-aware tokenizer interface. The ctx is unused by
// the underlying gRPC client, but accepted for interface uniformity.
type udsTokenizerAdapter struct {
	t *tokenization.UdsTokenizer
}

func newUDSTokenizer(ctx context.Context, cfg *tokenization.UdsTokenizerConfig, modelName string) (*udsTokenizerAdapter, error) {
	uds, err := tokenization.NewUdsTokenizer(ctx, cfg, modelName)
	if err != nil {
		return nil, err
	}
	return &udsTokenizerAdapter{t: uds}, nil
}

func (a *udsTokenizerAdapter) Render(_ context.Context, prompt string) ([]uint32, []tokenizerTypes.Offset, error) {
	return a.t.Render(prompt)
}

func (a *udsTokenizerAdapter) RenderChat(_ context.Context, req *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return a.t.RenderChat(req)
}
