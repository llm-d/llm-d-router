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
	"encoding/binary"
	"encoding/json"
	"errors"
	"strconv"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// bytesPerToken matches the scorer's averageCharactersPerToken, so a block of N
// pseudo-tokens covers the same input bytes as an N-token raw-byte block.
const bytesPerToken = 4

const blockTypeText = "text"

// estimateBackend packs request bytes into pseudo-tokens with no real tokenizer.
// The IDs suit content-locality hashing only; they never match engine KV blocks,
// so pairing this backend with the engine-correlated scorer yields misses, not bad routes.
type estimateBackend struct {
	img imageEstimator
}

func (b estimateBackend) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error) {
	// Pre-tokenized inputs are already real tokens; pass them through unchanged
	// rather than byte-estimating. Token-ID inputs are valid for generate,
	// /v1/completions, and /v1/embeddings.
	switch {
	case body.Generate != nil:
		return &fwkrh.TokenizedPrompt{
			TokenIDs:           body.Generate.TokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features),
		}, nil
	case body.Completions != nil && len(body.Completions.Prompt.TokenIDs) > 0:
		return &fwkrh.TokenizedPrompt{TokenIDs: body.Completions.Prompt.TokenIDs}, nil
	case body.Embeddings != nil && len(body.Embeddings.Input.TokenIDs) > 0:
		return &fwkrh.TokenizedPrompt{TokenIDs: body.Embeddings.Input.TokenIDs}, nil
	}

	raw, features, err := b.estimateBytes(ctx, body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.TokenizedPrompt{TokenIDs: packBytes(raw), MultiModalFeatures: features}, nil
}

// estimateBytes serializes a non-pre-tokenized request body to a byte stream
// and, for protocols that carry multimodal assets, the features that describe
// them. Coverage matches the protocols the approximate prefix-cache scorer
// handles.
func (b estimateBackend) estimateBytes(ctx context.Context, body *fwkrh.InferenceRequestBody) ([]byte, []fwkrh.MultiModalFeature, error) {
	switch {
	case body.ChatCompletions != nil:
		raw, features := b.chatCompletionsBytes(body.ChatCompletions)
		return raw, features, nil
	case body.Messages != nil:
		raw, features := b.messagesBytes(body.Messages)
		log.FromContext(ctx).V(logutil.DEBUG).Info("Anthropic messages prefix-cache estimation",
			"messageCount", len(body.Messages.Messages),
			"rawBytes", len(raw),
			"mmFeatureCount", len(features),
			"mmFeatures", features,
		)
		return raw, features, nil
	case body.Conversations != nil:
		raw, err := json.Marshal(body.Conversations.Items)
		return raw, nil, err
	case body.Responses != nil:
		var combined []map[string]any
		if body.Responses.Instructions != nil {
			combined = append(combined, map[string]any{"instructions": body.Responses.Instructions})
		}
		if body.Responses.Tools != nil {
			combined = append(combined, map[string]any{"tools": body.Responses.Tools})
		}
		combined = append(combined, map[string]any{"input": body.Responses.Input})
		raw, err := json.Marshal(combined)
		return raw, nil, err
	case body.Completions != nil:
		return []byte(body.Completions.Prompt.PlainText()), nil, nil
	case body.Embeddings != nil:
		raw, err := json.Marshal(body.Embeddings.Input)
		return raw, nil, err
	default:
		return nil, nil, errors.New("unsupported request body type, skipping estimation")
	}
}

// chatCompletionsBytes flattens roles + text into pseudo-token bytes, folding
// multimodal assets in on aligned boundaries. Each asset occupies N placeholder
// pseudo-tokens (its content hash repeated N times) so it carries weight in the
// stream, and is reported as a MultiModalFeature with its token offset and span.
func (b estimateBackend) chatCompletionsBytes(chat *fwkrh.ChatCompletionsRequest) ([]byte, []fwkrh.MultiModalFeature) {
	var out []byte
	var features []fwkrh.MultiModalFeature
	if len(chat.Tools) > 0 {
		if raw, err := json.Marshal(chat.Tools); err == nil {
			out = append(out, raw...)
		}
	}
	for _, msg := range chat.Messages {
		out, features = b.appendChatMessage(out, features, msg)
	}
	return out, features
}

// appendChatMessage flattens a single chat-completions message into the byte
// stream, recording multimodal placeholders on aligned boundaries.
func (b estimateBackend) appendChatMessage(out []byte, features []fwkrh.MultiModalFeature, msg fwkrh.Message) ([]byte, []fwkrh.MultiModalFeature) {
	if msg.Role != "" {
		out = append(out, []byte(msg.Role)...)
	}
	if msg.Content.Raw != "" {
		out = append(out, []byte(msg.Content.Raw)...)
		return out, features
	}
	for _, block := range msg.Content.Structured {
		switch block.Type {
		case blockTypeText:
			out = append(out, []byte(block.Text)...)
		case "image_url":
			out, features = appendMMAsset(out, features, block.ImageURL.URL, b.img.placeholderCount(block.ImageURL.URL))
		case "video_url":
			out, features = appendMMAsset(out, features, block.VideoURL.URL, assetPlaceholderCount(len(block.VideoURL.URL)))
		case "input_audio", "audio_url":
			data := block.InputAudio.Data + block.InputAudio.Format
			out, features = appendMMAsset(out, features, data, assetPlaceholderCount(len(data)))
		}
	}
	return out, features
}

// messagesBytes flattens an Anthropic /v1/messages request into pseudo-token
// bytes.
func (b estimateBackend) messagesBytes(req *fwkrh.MessagesRequest) ([]byte, []fwkrh.MultiModalFeature) {
	var out []byte
	var features []fwkrh.MultiModalFeature
	if len(req.Tools) > 0 {
		if raw, err := json.Marshal(req.Tools); err == nil {
			out = append(out, raw...)
		}
	}
	// The system field accepts only text -- a string or an array of text blocks.
	// See https://docs.anthropic.com/en/api/messages#body-system.
	if req.System.Raw != "" {
		out = append(out, []byte(req.System.Raw)...)
	} else {
		for _, block := range req.System.Structured {
			if block.Type == blockTypeText {
				out = append(out, []byte(block.Text)...)
			}
		}
	}
	for _, msg := range req.Messages {
		if msg.Role != "" {
			out = append(out, []byte(msg.Role)...)
		}
		if msg.Content.Raw != "" {
			out = append(out, []byte(msg.Content.Raw)...)
			continue
		}
		for _, block := range msg.Content.Structured {
			switch block.Type {
			case blockTypeText:
				out = append(out, []byte(block.Text)...)
			case "image":
				if content, count := b.img.placeholderForAnthropicImage(block.Source); content != "" {
					out, features = appendMMAsset(out, features, content, count)
				}
			}
		}
	}
	return out, features
}

// appendMMAsset aligns out to a token boundary, appends count placeholder
// pseudo-tokens derived from a stable content hash, and records the matching
// feature. Modality is always ModalityImage: it is the only defined modality
// const, and detection/scoring need only a non-empty, stably-hashed feature.
func appendMMAsset(out []byte, features []fwkrh.MultiModalFeature, content string, count int) ([]byte, []fwkrh.MultiModalFeature) {
	out = align(out)
	offset := len(out) / bytesPerToken

	sum := xxhash.Sum64String(content)
	token := make([]byte, bytesPerToken)
	binary.LittleEndian.PutUint32(token, uint32(sum))
	for i := 0; i < count; i++ {
		out = append(out, token...)
	}

	features = append(features, fwkrh.MultiModalFeature{
		Modality: fwkrh.ModalityImage,
		Hash:     strconv.FormatUint(sum, 16),
		Offset:   offset,
		Length:   count,
	})
	return out, features
}

// assetPlaceholderCount derives a deterministic placeholder count (>= 1) from an
// asset's byte length for modalities without a dedicated estimator.
func assetPlaceholderCount(dataLen int) int {
	if n := (dataLen + bytesPerToken - 1) / bytesPerToken; n > 0 {
		return n
	}
	return 1
}

// packBytes packs bytes into little-endian uint32 tokens (zero-padded tail).
// Reinterpreting them reproduces the input, so locality keys are unchanged.
func packBytes(raw []byte) []uint32 {
	if len(raw) == 0 {
		return nil
	}
	raw = align(raw)
	out := make([]uint32, len(raw)/bytesPerToken)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(raw[i*bytesPerToken:])
	}
	return out
}

// align zero-pads b up to a bytesPerToken boundary.
func align(b []byte) []byte {
	if r := len(b) % bytesPerToken; r != 0 {
		b = append(b, make([]byte, bytesPerToken-r)...)
	}
	return b
}
