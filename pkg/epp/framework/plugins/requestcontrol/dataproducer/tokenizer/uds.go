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
	"encoding/json"
	"errors"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// udsTokenizerAdapter adapts the kvc UdsTokenizer (which has no ctx in its
// signature) to the local ctx-aware tokenizer interface. Both ctx and the
// RequestPayload-to-RenderChatRequest conversion are handled in the adapter.
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

func (a *udsTokenizerAdapter) Render(_ context.Context, payload fwkrh.RequestPayload) ([]uint32, []tokenizerTypes.Offset, error) {
	pm, ok := payload.AsMap()
	if !ok {
		return nil, nil, errors.New("UDS tokenizer requires a parsed PayloadMap")
	}
	prompt, ok := pm["prompt"].(string)
	if !ok {
		return nil, nil, errors.New("UDS tokenizer requires string prompt")
	}
	return a.t.Render(prompt)
}

func (a *udsTokenizerAdapter) RenderChat(_ context.Context, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	pm, ok := payload.AsMap()
	if !ok {
		return nil, nil, errors.New("UDS tokenizer requires a parsed PayloadMap")
	}
	req, err := renderChatRequestFromPayload(pm)
	if err != nil {
		return nil, nil, err
	}
	return a.t.RenderChat(req)
}

func renderChatRequestFromPayload(pm fwkrh.PayloadMap) (*tokenizerTypes.RenderChatRequest, error) {
	data, err := json.Marshal(pm)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}
	var chat fwkrh.ChatCompletionsRequest
	if err := json.Unmarshal(data, &chat); err != nil {
		return nil, fmt.Errorf("unmarshal chat request: %w", err)
	}
	return ChatCompletionsToRenderChatRequest(&chat), nil
}
