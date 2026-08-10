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
	"strings"
	"unicode/utf8"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
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
	vid videoEstimator
}

// parseMMMetadataHeaders reads the x-llm-d-* request headers into an mmMetadata.
// Only video is populated today; image and audio parsing slot in here later.
func parseMMMetadataHeaders(headers map[string]string) mmMetadata {
	return mmMetadata{video: parseVideoMetadataHeaders(headers)}
}

// mmMetadataCtxKey keys the request-scoped mmMetadata on the context, carrying it
// from Plugin.Produce (which holds request.Headers) to the estimate backend
// without widening the shared tokenInputProducer.produce signature.
type mmMetadataCtxKey struct{}

// withMMMetadata returns ctx carrying meta.
func withMMMetadata(ctx context.Context, meta mmMetadata) context.Context {
	return context.WithValue(ctx, mmMetadataCtxKey{}, meta)
}

// mmMetadataFromContext returns the mmMetadata on ctx, or the zero value.
func mmMetadataFromContext(ctx context.Context) mmMetadata {
	meta, _ := ctx.Value(mmMetadataCtxKey{}).(mmMetadata)
	return meta
}

// parseVideoMetadataHeaders reads the x-llm-d-video- request headers into a
// videoMetadata, using metadata.GetLowerCaseHeaderValue so aliases resolve the
// same way as the SLO headers. Missing or malformed values leave their field zero
// so the estimator falls back per field to config and then defaults.
func parseVideoMetadataHeaders(headers map[string]string) videoMetadata {
	var meta videoMetadata
	if s, ok := metadata.GetLowerCaseHeaderValue(headers, metadata.VideoFPSHeaderKey); ok {
		if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
			meta.fps = v
		}
	}
	if s, ok := metadata.GetLowerCaseHeaderValue(headers, metadata.VideoDurationHeaderKey); ok {
		if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
			meta.duration = v
		}
	}
	if s, ok := metadata.GetLowerCaseHeaderValue(headers, metadata.VideoResolutionHeaderKey); ok {
		meta.width, meta.height = parseResolution(s)
	}
	return meta
}

// parseResolution splits a "WIDTHxHEIGHT" value into pixel dimensions, returning
// zeros when the value is empty or malformed.
func parseResolution(s string) (width, height int) {
	i := strings.IndexAny(s, "xX")
	if i <= 0 {
		return 0, 0
	}
	w, err := strconv.Atoi(strings.TrimSpace(s[:i]))
	if err != nil || w <= 0 {
		return 0, 0
	}
	h, err := strconv.Atoi(strings.TrimSpace(s[i+1:]))
	if err != nil || h <= 0 {
		return 0, 0
	}
	return w, h
}

func (b estimateBackend) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error) {
	// Pre-tokenized inputs are already real tokens; pass them through unchanged
	// rather than byte-estimating. Token-ID inputs are valid for generate,
	// /v1/completions, and /v1/embeddings.
	switch {
	case body.Generate != nil:
		return &fwkrh.TokenizedPrompt{
			PerPromptTokens:    [][]uint32{body.Generate.TokenIDs},
			MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features),
		}, nil
	case body.Completions != nil && len(body.Completions.Prompt.TokenIDs) > 0:
		return &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{body.Completions.Prompt.TokenIDs}}, nil
	case body.Embeddings != nil && len(body.Embeddings.Input.TokenIDs) > 0:
		return &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{body.Embeddings.Input.TokenIDs}}, nil
	}

	// Chat and Anthropic messages fold multimodal placeholders into the stream
	// and report them as features.
	if body.ChatCompletions != nil {
		tokens, features := b.chatCompletionsTokens(body.ChatCompletions, mmMetadataFromContext(ctx))
		return &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}, MultiModalFeatures: features}, nil
	}
	if body.Messages != nil {
		tokens, features, rawBytes := b.messagesTokens(body.Messages)
		log.FromContext(ctx).V(logutil.DEBUG).Info("Anthropic messages prefix-cache estimation",
			"messageCount", len(body.Messages.Messages),
			"rawBytes", rawBytes,
			"tokenCount", len(tokens),
			"mmFeatureCount", len(features),
			"mmFeatures", features,
		)
		return &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}, MultiModalFeatures: features}, nil
	}

	if body.Completions != nil && len(body.Completions.Prompt.Strings) > 1 {
		return estimateMultiStringCompletions(body.Completions)
	}

	raw, err := estimateBytes(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{packBytes(raw)}}, nil
}

func estimateMultiStringCompletions(req *fwkrh.CompletionsRequest) (*fwkrh.TokenizedPrompt, error) {
	allTokenIDs := make([][]uint32, 0, len(req.Prompt.Strings))
	for _, s := range req.Prompt.Strings {
		ids := packBytes([]byte(s))
		allTokenIDs = append(allTokenIDs, ids)
	}
	return &fwkrh.TokenizedPrompt{PerPromptTokens: allTokenIDs}, nil
}

// estimateBytes serializes the user input of a non-chat request body to a byte
// stream. Coverage matches the protocols the approximate prefix-cache scorer
// handles. The chat path is handled separately to emit multimodal features.
func estimateBytes(body *fwkrh.InferenceRequestBody) ([]byte, error) {
	switch {
	case body.Conversations != nil:
		return json.Marshal(body.Conversations.Items)
	case body.Responses != nil:
		var combined []map[string]any
		if body.Responses.Instructions != nil {
			combined = append(combined, map[string]any{"instructions": body.Responses.Instructions})
		}
		if body.Responses.Tools != nil {
			combined = append(combined, map[string]any{"tools": body.Responses.Tools})
		}
		combined = append(combined, map[string]any{"input": body.Responses.Input})
		return json.Marshal(combined)
	case body.Completions != nil:
		return []byte(body.Completions.Prompt.PlainText()), nil
	case body.Embeddings != nil:
		return json.Marshal(body.Embeddings.Input)
	default:
		return nil, errors.New("unsupported request body type, skipping estimation")
	}
}

// estimatedTokenStream keeps text packing and multimodal placeholder packing
// separate so UTF-8 padding cannot shift feature offsets or reinterpret hash bytes.
type estimatedTokenStream struct {
	tokens   []uint32
	text     []byte
	features []fwkrh.MultiModalFeature
	rawBytes int
}

func (s *estimatedTokenStream) appendText(raw []byte) {
	s.text = append(s.text, raw...)
	s.rawBytes += len(raw)
}

func (s *estimatedTokenStream) flushText() {
	s.tokens = append(s.tokens, packBytes(s.text)...)
	s.text = s.text[:0]
}

func (s *estimatedTokenStream) appendMMAsset(modality fwkrh.Modality, content string, count int) {
	s.flushText()
	offset := len(s.tokens)
	sum := xxhash.Sum64String(content)
	for i := 0; i < count; i++ {
		s.tokens = append(s.tokens, uint32(sum))
	}
	s.rawBytes += count * bytesPerToken
	s.features = append(s.features, fwkrh.MultiModalFeature{
		Modality: modality,
		Hash:     strconv.FormatUint(sum, 16),
		Offset:   offset,
		Length:   count,
	})
}

func (s *estimatedTokenStream) finish() ([]uint32, []fwkrh.MultiModalFeature, int) {
	s.flushText()
	return s.tokens, s.features, s.rawBytes
}

// chatCompletionsTokens flattens roles + text into pseudo-tokens and inserts
// multimodal placeholders on token boundaries.
func (b estimateBackend) chatCompletionsTokens(chat *fwkrh.ChatCompletionsRequest, meta mmMetadata) ([]uint32, []fwkrh.MultiModalFeature) {
	var stream estimatedTokenStream
	if len(chat.Tools) > 0 {
		if raw, err := json.Marshal(chat.Tools); err == nil {
			stream.appendText(raw)
		}
	}
	for _, msg := range chat.Messages {
		b.appendChatMessage(&stream, msg, meta)
	}
	tokens, features, _ := stream.finish()
	return tokens, features
}

func (b estimateBackend) appendChatMessage(stream *estimatedTokenStream, msg fwkrh.Message, meta mmMetadata) {
	if msg.Role != "" {
		stream.appendText([]byte(msg.Role))
	}
	if msg.Content.Raw != "" {
		stream.appendText([]byte(msg.Content.Raw))
		return
	}
	for _, block := range msg.Content.Structured {
		switch block.Type {
		case blockTypeText:
			stream.appendText([]byte(block.Text))
		case "image_url":
			stream.appendMMAsset(fwkrh.ModalityImage, block.ImageURL.URL, b.img.placeholderCount(block.ImageURL.URL))
		case "video_url":
			stream.appendMMAsset(fwkrh.ModalityVideo, block.VideoURL.URL, b.vid.placeholderCount(meta.video))
		case "input_audio", "audio_url":
			data := block.InputAudio.Data + block.InputAudio.Format
			stream.appendMMAsset(fwkrh.ModalityAudio, data, assetPlaceholderCount(len(data)))
		}
	}
}

// messagesTokens flattens an Anthropic /v1/messages request into pseudo-tokens.
func (b estimateBackend) messagesTokens(req *fwkrh.MessagesRequest) ([]uint32, []fwkrh.MultiModalFeature, int) {
	var stream estimatedTokenStream
	if len(req.Tools) > 0 {
		if raw, err := json.Marshal(req.Tools); err == nil {
			stream.appendText(raw)
		}
	}
	// The system field accepts only text -- a string or an array of text blocks.
	// See https://docs.anthropic.com/en/api/messages#body-system.
	if req.System.Raw != "" {
		stream.appendText([]byte(req.System.Raw))
	} else {
		for _, block := range req.System.Structured {
			if block.Type == blockTypeText {
				stream.appendText([]byte(block.Text))
			}
		}
	}
	for _, msg := range req.Messages {
		if msg.Role != "" {
			stream.appendText([]byte(msg.Role))
		}
		if msg.Content.Raw != "" {
			stream.appendText([]byte(msg.Content.Raw))
			continue
		}
		for _, block := range msg.Content.Structured {
			switch block.Type {
			case blockTypeText:
				stream.appendText([]byte(block.Text))
			case "image":
				if content, count := b.img.placeholderForAnthropicImage(block.Source); content != "" {
					stream.appendMMAsset(fwkrh.ModalityImage, content, count)
				}
			}
		}
	}
	return stream.finish()
}

// assetPlaceholderCount derives a deterministic placeholder count (>= 1) from an
// asset's byte length for modalities without a dedicated estimator.
func assetPlaceholderCount(dataLen int) int {
	if n := (dataLen + bytesPerToken - 1) / bytesPerToken; n > 0 {
		return n
	}
	return 1
}

// utf8CharSize returns the next valid UTF-8 character's byte length. Invalid
// and truncated encodings consume one byte so packing always makes progress.
func utf8CharSize(raw []byte) int {
	_, size := utf8.DecodeRune(raw)
	return size
}

// packBytes packs bytes into little-endian uint32 pseudo-tokens respecting
// UTF-8 character boundaries. Unfilled trailing bytes are naturally zero-padded.
func packBytes(raw []byte) []uint32 {
	if len(raw) == 0 {
		return nil
	}
	out := make([]uint32, 0, (len(raw)+bytesPerToken-1)/bytesPerToken)
	var slot [bytesPerToken]byte
	pos := 0

	for len(raw) > 0 {
		size := utf8CharSize(raw)
		if pos+size > bytesPerToken {
			out = append(out, binary.LittleEndian.Uint32(slot[:]))
			slot = [bytesPerToken]byte{}
			pos = 0
		}
		copy(slot[pos:], raw[:size])
		pos += size
		raw = raw[size:]
	}
	if pos > 0 {
		out = append(out, binary.LittleEndian.Uint32(slot[:]))
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
