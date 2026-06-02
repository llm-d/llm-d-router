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
	"errors"
	"fmt"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// tokenInputProducer turns a request body into a TokenizedPrompt. Backends vary
// in fidelity (render vs estimate); callers never branch on which produced it.
type tokenInputProducer interface {
	produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error)
}

// renderBackend produces real token IDs and owns protocol dispatch, including
// the pre-tokenized (Generate) passthrough.
type renderBackend struct {
	tk tokenizer
}

func (b renderBackend) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error) {
	switch {
	case body.Completions != nil:
		tokenIDs, _, err := b.tk.Render(ctx, body.Completions.Prompt.Raw)
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}
		return &fwkrh.TokenizedPrompt{TokenIDs: tokenIDs}, nil
	case body.ChatCompletions != nil:
		tokenIDs, mmFeatures, err := b.tk.RenderChat(ctx, ChatCompletionsToRenderChatRequest(body.ChatCompletions))
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}
		return &fwkrh.TokenizedPrompt{TokenIDs: tokenIDs, MultiModalFeatures: convertMMFeaturesToUpstream(mmFeatures)}, nil
	case body.Generate != nil:
		return &fwkrh.TokenizedPrompt{
			TokenIDs:           body.Generate.TokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features),
		}, nil
	default:
		return nil, errors.New("unsupported request body type, skipping tokenization")
	}
}

// cacheSaltFromBody returns the cache salt from whichever protocol is populated.
// The protocol switch lives in the producer so consumers read only
// TokenizedPrompt.CacheSalt.
func cacheSaltFromBody(body *fwkrh.InferenceRequestBody) string {
	switch {
	case body.Conversations != nil:
		return body.Conversations.CacheSalt
	case body.Responses != nil:
		return body.Responses.CacheSalt
	case body.ChatCompletions != nil:
		return body.ChatCompletions.CacheSalt
	case body.Messages != nil:
		return body.Messages.CacheSalt
	case body.Completions != nil:
		return body.Completions.CacheSalt
	case body.Embeddings != nil:
		return body.Embeddings.CacheSalt
	case body.Generate != nil:
		return body.Generate.CacheSalt
	default:
		return ""
	}
}
