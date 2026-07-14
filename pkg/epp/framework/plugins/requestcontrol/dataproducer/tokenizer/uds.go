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

	kvctok "github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	kvctoktypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// udsTokenizerAdapter adapts the deprecated UdsTokenizer (still sourced from
// llm-d-kv-cache) to the local ctx-aware renderer interface. It bridges the
// external module's tokenization types to the in-tree ones.
type udsTokenizerAdapter struct {
	t *kvctok.UdsTokenizer
}

func newUDSTokenizer(ctx context.Context, cfg *kvctok.UdsTokenizerConfig, modelName string) (*udsTokenizerAdapter, error) {
	uds, err := kvctok.NewUdsTokenizer(ctx, cfg, modelName)
	if err != nil {
		return nil, err
	}
	return &udsTokenizerAdapter{t: uds}, nil
}

func (a *udsTokenizerAdapter) Render(_ context.Context, payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
	pm, ok := payload.AsMap()
	if !ok {
		return nil, nil, errors.New("UDS tokenizer requires a parsed PayloadMap")
	}
	prompt, ok := pm["prompt"].(string)
	if !ok {
		return nil, nil, errors.New("UDS tokenizer requires string prompt")
	}
	tokenIDs, offsets, err := a.t.Render(prompt)
	if err != nil {
		return nil, nil, err
	}
	return [][]uint32{tokenIDs}, [][]tokenizerTypes.Offset{toInTreeOffsets(offsets)}, nil
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
	tokenIDs, mmf, err := a.t.RenderChat(req)
	if err != nil {
		return nil, nil, err
	}
	return tokenIDs, toInTreeMMFeatures(mmf), nil
}

func renderChatRequestFromPayload(pm fwkrh.PayloadMap) (*kvctoktypes.RenderChatRequest, error) {
	data, err := json.Marshal(pm)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}
	var chat fwkrh.ChatCompletionsRequest
	if err := json.Unmarshal(data, &chat); err != nil {
		return nil, fmt.Errorf("unmarshal chat request: %w", err)
	}
	return toExternalRenderChatRequest(ChatCompletionsToRenderChatRequest(&chat))
}

// The deprecated UDS tokenizer exchanges llm-d-kv-cache's tokenization types.
// These helpers bridge to the in-tree types; the structs are identical, so a
// JSON round-trip is a faithful projection.

func toExternalRenderChatRequest(in *tokenizerTypes.RenderChatRequest) (*kvctoktypes.RenderChatRequest, error) {
	if in == nil {
		//nolint:nilnil // nil in -> nil out is the intended passthrough
		return nil, nil
	}
	data, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("marshal render request: %w", err)
	}
	var out kvctoktypes.RenderChatRequest
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("unmarshal render request: %w", err)
	}
	return &out, nil
}

func toInTreeOffsets(in []kvctoktypes.Offset) []tokenizerTypes.Offset {
	if in == nil {
		return nil
	}
	out := make([]tokenizerTypes.Offset, len(in))
	for i, o := range in {
		out[i] = tokenizerTypes.Offset(o)
	}
	return out
}

func toInTreeMMFeatures(in *kvctok.MultiModalFeatures) *tokenization.MultiModalFeatures {
	if in == nil {
		return nil
	}
	out := &tokenization.MultiModalFeatures{MMHashes: in.MMHashes}
	if in.MMPlaceholders != nil {
		out.MMPlaceholders = make(map[string][]kvblock.PlaceholderRange, len(in.MMPlaceholders))
		for k, ranges := range in.MMPlaceholders {
			conv := make([]kvblock.PlaceholderRange, len(ranges))
			for i, r := range ranges {
				conv[i] = kvblock.PlaceholderRange{Offset: r.Offset, Length: r.Length}
			}
			out.MMPlaceholders[k] = conv
		}
	}
	return out
}
