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

package vllm

import (
	"encoding/json"
	"io"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// PrepareEncode adds vLLM token and multimodal fields to a per-image request.
func PrepareEncode(reqCtx *pipeline.RequestContext, body map[string]any, entry pipeline.MultimodalEntry, format gateway.RequestFormat) {
	tokenIDs := buildEncodeTokenIDs(reqCtx.TokenIDs, entry)
	features := map[string]any{
		"mm_hashes":       map[string][]string{modalityImage: {entry.Hash}},
		"mm_placeholders": map[string][]any{modalityImage: {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
	}
	if format == gateway.FormatChatCompletions {
		body["tokens"] = map[string]any{"token_ids": tokenIDs, "features": features}
	} else {
		features["kwargs_data"] = mmKwargsField([]string{entry.KwargsData})
		body["token_ids"] = tokenIDs
		body["features"] = features
	}
}

// ReadEncodeResponse extracts encoder-cache transfer parameters from a JSON response.
func ReadEncodeResponse(body io.Reader) (any, error) {
	var response encodeResponse
	if err := json.NewDecoder(body).Decode(&response); err != nil {
		return nil, err
	}
	return response.ECTransferParams, nil
}

func buildEncodeTokenIDs(fullTokenIDs []int, entry pipeline.MultimodalEntry) []int {
	bos := 1
	placeholderTokenID := 0
	if len(fullTokenIDs) > 0 {
		bos = fullTokenIDs[0]
		// Only the upper bound is checked here; offset >= 0 is guaranteed for all
		// paths, either by extractMultimodalEntries (generate) or by the trusted
		// render-service response (chat/completions). A negative offset would
		// index out of range.
		if entry.Placeholder.Offset < len(fullTokenIDs) {
			placeholderTokenID = fullTokenIDs[entry.Placeholder.Offset]
		}
	}

	tokenIDs := make([]int, 1+entry.Placeholder.Length)
	tokenIDs[0] = bos
	for j := 1; j <= entry.Placeholder.Length; j++ {
		tokenIDs[j] = placeholderTokenID
	}
	return tokenIDs
}

type encodeResponse struct {
	// ECTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	ECTransferParams any `json:"ec_transfer_params"`
}
