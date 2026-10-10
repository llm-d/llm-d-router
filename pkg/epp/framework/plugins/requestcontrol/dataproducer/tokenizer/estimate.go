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

	"github.com/cespare/xxhash/v2"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// Content-block types read by the estimate backend.
const (
	blockTypeText       = "text"
	blockTypeImage      = "image"
	blockTypeImageURL   = "image_url"
	blockTypeThinking   = "thinking"
	blockTypeToolUse    = "tool_use"
	blockTypeToolResult = "tool_result"
)

const (
	// Per-request billing hashes must not defeat prefix caching.
	anthropicBillingHeaderPrefix = "x-anthropic-billing-header"
	// The base64 image fallback matches vLLM's Anthropic conversion.
	defaultImageMediaType = "image/jpeg"
)

func anthropicSystemText(ac fwkrh.AnthropicContent) string {
	if ac.Raw != "" {
		return ac.Raw
	}
	var sb strings.Builder
	for _, block := range ac.Structured {
		if block.Type == blockTypeText && block.Text != "" && !strings.HasPrefix(block.Text, anthropicBillingHeaderPrefix) {
			sb.WriteString(block.Text)
		}
	}
	return sb.String()
}

func appendImageBlock(blocks []tokenizerTypes.ContentBlock, src *fwkrh.AnthropicImageSource) []tokenizerTypes.ContentBlock {
	if url := anthropicImageToURL(src); url != "" {
		blocks = append(blocks, tokenizerTypes.ContentBlock{
			Type:     blockTypeImageURL,
			ImageURL: tokenizerTypes.ImageBlock{URL: url},
		})
	}
	return blocks
}

func anthropicToolResultContent(b fwkrh.AnthropicContentBlock) (string, []tokenizerTypes.ContentBlock) {
	if b.Content.Raw != "" {
		return b.Content.Raw, nil
	}
	var parts []string
	var imageBlocks []tokenizerTypes.ContentBlock
	for _, item := range b.Content.Structured {
		switch item.Type {
		case blockTypeText:
			parts = append(parts, item.Text)
		case blockTypeImage:
			imageBlocks = appendImageBlock(imageBlocks, item.Source)
		}
	}
	return strings.Join(parts, "\n"), imageBlocks
}

func anthropicImageToURL(src *fwkrh.AnthropicImageSource) string {
	if src == nil {
		return ""
	}
	if src.Type == "url" || src.URL != "" {
		return src.URL
	}
	if src.Data == "" {
		return ""
	}
	mediaType := src.MediaType
	if mediaType == "" {
		mediaType = defaultImageMediaType
	}
	return "data:" + mediaType + ";base64," + src.Data
}

// estimateBackend packs request bytes into pseudo-tokens with no real tokenizer.
// The IDs suit content-locality hashing only; they never match engine KV blocks,
// so pairing this backend with the engine-correlated scorer yields misses, not bad routes.
type estimateBackend struct {
	img imageEstimator
	vid videoEstimator
	aud audioEstimator
}

// parseMMMetadataHeaders reads the x-llm-d-* request headers into an mmMetadata.
// Video and audio are populated from headers; image parsing follows the same
// pattern when its headers are added.
func parseMMMetadataHeaders(headers map[string]string) mmMetadata {
	return mmMetadata{
		video: parseVideoMetadataHeaders(headers),
		audio: parseAudioMetadataHeaders(headers),
	}
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

// parseAudioMetadataHeaders reads the x-llm-d-audio- request headers into an
// audioMetadata, using metadata.GetLowerCaseHeaderValue so aliases resolve the
// same way as the SLO headers. Missing or malformed values leave their field zero
// so the estimator falls back per field to config and then defaults.
func parseAudioMetadataHeaders(headers map[string]string) audioMetadata {
	var meta audioMetadata
	if s, ok := metadata.GetLowerCaseHeaderValue(headers, metadata.AudioDurationHeaderKey); ok {
		if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
			meta.duration = v
		}
	}
	if s, ok := metadata.GetLowerCaseHeaderValue(headers, metadata.AudioBytesPerSecondHeaderKey); ok {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			meta.bytesPerSecond = v
		}
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

func (b estimateBackend) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	// Pre-tokenized inputs are already real tokens; pass them through unchanged
	// rather than byte-estimating. Token-ID inputs are valid for generate,
	// /v1/completions, and /v1/embeddings.
	switch {
	case body.Generate != nil && len(body.Generate.TokenIDs) > 0:
		return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
			TokenIDs:           body.Generate.TokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features),
		}}}, nil
	case body.Completions != nil && len(body.Completions.Prompt.TokenIDs) > 0:
		return fwkrh.NewTokenizedRequest(body.Completions.Prompt.TokenIDs), nil
	case body.Embeddings != nil && len(body.Embeddings.Input.TokenIDs) > 0:
		return fwkrh.NewTokenizedRequest(body.Embeddings.Input.TokenIDs), nil
	}

	// Chat and Anthropic messages fold multimodal placeholders into the stream
	// and report them as features.
	if body.ChatCompletions != nil {
		raw, features := b.chatCompletionsBytes(body.ChatCompletions, mmMetadataFromContext(ctx))
		return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
			TokenIDs:           packBytes(raw),
			MultiModalFeatures: features,
		}}}, nil
	}
	if body.Messages != nil {
		raw, features := b.messagesBytes(body.Messages)
		tokens := packBytes(raw)
		log.FromContext(ctx).V(logutil.DEBUG).Info("Anthropic messages prefix-cache estimation",
			"messageCount", len(body.Messages.Messages),
			"rawBytes", len(raw),
			"tokenCount", len(tokens),
			"mmFeatureCount", len(features),
			"mmFeatures", features,
		)
		return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
			TokenIDs:           tokens,
			MultiModalFeatures: features,
		}}}, nil
	}
	if body.Responses != nil {
		raw, features := b.responsesBytes(body.Responses)
		return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
			TokenIDs:           packBytes(raw),
			MultiModalFeatures: features,
		}}}, nil
	}

	if body.Completions != nil && len(body.Completions.Prompt.Strings) > 1 {
		return estimateMultiStringCompletions(body.Completions)
	}

	raw, err := estimateBytes(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: packBytes(raw)}}}, nil
}

func estimateMultiStringCompletions(req *fwkrh.CompletionsRequest) (*fwkrh.TokenizedRequest, error) {
	prompts := make([]fwkrh.PromptTokens, 0, len(req.Prompt.Strings))
	for _, s := range req.Prompt.Strings {
		ids := packBytes([]byte(s))
		prompts = append(prompts, fwkrh.PromptTokens{TokenIDs: ids})
	}
	return &fwkrh.TokenizedRequest{Prompts: prompts}, nil
}

// estimateBytes serializes the user input of a non-chat request body to a byte
// stream. Coverage matches the protocols the approximate prefix-cache scorer
// handles. The chat path is handled separately to emit multimodal features.
func estimateBytes(body *fwkrh.InferenceRequestBody) ([]byte, error) {
	switch {
	case body.Conversations != nil:
		return json.Marshal(body.Conversations.Items)
	case body.Completions != nil:
		return []byte(body.Completions.Prompt.PlainText()), nil
	case body.Embeddings != nil:
		return json.Marshal(body.Embeddings.Input)
	case body.Generate != nil && body.Generate.Text != "":
		return []byte(body.Generate.Text), nil
	default:
		return nil, errors.New("unsupported request body type, skipping estimation")
	}
}

// chatCompletionsBytes flattens roles + text into pseudo-token bytes, folding
// multimodal assets in on aligned boundaries. Each asset occupies N placeholder
// pseudo-tokens (its content hash repeated N times) so it carries weight in the
// stream, and is reported as a MultiModalFeature with its token offset and span.
func (b estimateBackend) chatCompletionsBytes(chat *fwkrh.ChatCompletionsRequest, meta mmMetadata) ([]byte, []fwkrh.MultiModalFeature) {
	var out []byte
	var features []fwkrh.MultiModalFeature
	if len(chat.Tools) > 0 {
		if raw, err := json.Marshal(chat.Tools); err == nil {
			out = append(out, raw...)
		}
	}
	for _, msg := range chat.Messages {
		out, features = b.appendChatMessage(out, features, msg, meta)
	}
	return out, features
}

// appendChatMessage flattens a single chat-completions message into the byte
// stream, recording multimodal placeholders on aligned boundaries. Assistant
// tool_calls follow the content, matching tool_use blocks in messagesBytes and
// function_call items in appendResponsesItem.
func (b estimateBackend) appendChatMessage(out []byte, features []fwkrh.MultiModalFeature, msg fwkrh.Message, meta mmMetadata) ([]byte, []fwkrh.MultiModalFeature) {
	if msg.Role != "" {
		out = append(out, []byte(msg.Role)...)
	}
	if msg.Content.Raw != "" {
		out = append(out, []byte(msg.Content.Raw)...)
		return appendChatToolCalls(out, msg.ToolCalls), features
	}
	for _, block := range msg.Content.Structured {
		switch block.Type {
		case blockTypeText:
			out = append(out, []byte(block.Text)...)
		case "image_url":
			out, features = appendMMAsset(out, features, fwkrh.ModalityImage, block.ImageURL.URL, b.img.placeholderCount(block.ImageURL.URL))
		case "video_url":
			out, features = appendMMAsset(out, features, fwkrh.ModalityVideo, block.VideoURL.URL, b.vid.placeholderCount(meta.video))
		case "audio_url":
			// A clip carried by URL has no payload to read a duration from.
			out, features = appendMMAsset(out, features, fwkrh.ModalityAudio, block.AudioURL.URL, b.aud.placeholderCount("", meta.audio))
		case "input_audio":
			data := block.InputAudio.Data + block.InputAudio.Format
			out, features = appendMMAsset(out, features, fwkrh.ModalityAudio, data, b.aud.placeholderCount(block.InputAudio.Data, meta.audio))
		}
	}
	return appendChatToolCalls(out, msg.ToolCalls), features
}

// appendChatToolCalls flattens an assistant message's tool_calls: the function
// name and its arguments. Arguments are a JSON string on the wire; an object is
// accepted too and contributes its JSON encoding. A call of any other shape
// contributes nothing.
func appendChatToolCalls(out []byte, toolCalls []any) []byte {
	for _, call := range toolCalls {
		c, ok := call.(map[string]any)
		if !ok {
			continue
		}
		fn, ok := c["function"].(map[string]any)
		if !ok {
			continue
		}
		if name, ok := fn["name"].(string); ok {
			out = append(out, []byte(name)...)
		}
		switch args := fn["arguments"].(type) {
		case string:
			out = append(out, []byte(args)...)
		case map[string]any:
			if raw, err := json.Marshal(args); err == nil {
				out = append(out, raw...)
			}
		}
	}
	return out
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
	if sys := anthropicSystemText(req.System); sys != "" {
		out = append(out, []byte(sys)...)
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
			case blockTypeImage:
				if content, count := b.img.placeholderForAnthropicImage(block.Source); content != "" {
					out, features = appendMMAsset(out, features, fwkrh.ModalityImage, content, count)
				}
			case blockTypeThinking:
				out = append(out, []byte(block.Thinking)...)
			case blockTypeToolUse:
				out = append(out, []byte(block.ID)...)
				out = append(out, []byte(block.Name)...)
				out = append(out, block.Input...)
			case blockTypeToolResult:
				text, imageBlocks := anthropicToolResultContent(block)
				out = append(out, []byte(text)...)
				for _, img := range imageBlocks {
					url := img.ImageURL.URL
					out, features = appendMMAsset(out, features, fwkrh.ModalityImage, url, b.img.placeholderCount(url))
				}
			}
		}
	}
	return out, features
}

// responsesBytes flattens a /v1/responses request into pseudo-token bytes,
// folding multimodal placeholders in on aligned boundaries. Tools and a
// string Instructions contribute their bytes ahead of Input, matching
// messagesBytes. Input is a string or an array of items (see
// appendResponsesItem for per-item coverage).
func (b estimateBackend) responsesBytes(r *fwkrh.ResponsesRequest) ([]byte, []fwkrh.MultiModalFeature) {
	var out []byte
	var features []fwkrh.MultiModalFeature
	if r.Tools != nil {
		if raw, err := json.Marshal(r.Tools); err == nil {
			out = append(out, raw...)
		}
	}
	if sys, ok := r.Instructions.(string); ok && sys != "" {
		out = append(out, []byte(sys)...)
	}
	return b.appendResponsesInput(out, features, r.Input)
}

// appendResponsesInput flattens the Responses Input field: a plain string, or
// an array of items.
func (b estimateBackend) appendResponsesInput(out []byte, features []fwkrh.MultiModalFeature, input any) ([]byte, []fwkrh.MultiModalFeature) {
	switch v := input.(type) {
	case string:
		out = append(out, []byte(v)...)
	case []any:
		for _, item := range v {
			out, features = b.appendResponsesItem(out, features, item)
		}
	}
	return out, features
}

// appendResponsesItem flattens one Input item for the byte estimate. An item
// shaped like a plain chat message ({"role": ..., "content": ...}, with an
// optional "type": "message") flattens through content. function_call,
// function_call_output, and reasoning items flatten their own text-bearing
// fields rather than contributing nothing: an agentic Responses turn is
// mostly these item types, and undercounting their length feeds a too-short
// estimate into the P/D disaggregation decider, context-length admission,
// and prefix-hash scoring, all of which read this backend's token count. An
// item of any other type contributes nothing, since this code does not know
// its shape.
func (b estimateBackend) appendResponsesItem(out []byte, features []fwkrh.MultiModalFeature, item any) ([]byte, []fwkrh.MultiModalFeature) {
	m, ok := item.(map[string]any)
	if !ok {
		return out, features
	}
	switch t, _ := m["type"].(string); t {
	case "", responsesItemTypeMessage:
		if role, ok := m["role"].(string); ok {
			out = append(out, []byte(role)...)
		}
		return b.appendResponsesContent(out, features, m["content"])
	case "function_call":
		if name, ok := m["name"].(string); ok {
			out = append(out, []byte(name)...)
		}
		if args, ok := m["arguments"].(string); ok {
			out = append(out, []byte(args)...)
		}
	case "function_call_output":
		out = appendResponsesOutputText(out, m["output"])
	case "reasoning":
		out = appendResponsesSummaryText(out, m["summary"])
	}
	return out, features
}

// appendResponsesOutputText flattens a function_call_output item's "output"
// field: a plain string, or an array of content parts shaped like
// {"type": "output_text", "text": ...}.
func appendResponsesOutputText(out []byte, output any) []byte {
	switch v := output.(type) {
	case string:
		return append(out, []byte(v)...)
	case []any:
		for _, part := range v {
			if p, ok := part.(map[string]any); ok {
				if text, ok := p["text"].(string); ok {
					out = append(out, []byte(text)...)
				}
			}
		}
	}
	return out
}

// appendResponsesSummaryText flattens a reasoning item's "summary" field: an
// array of {"type": "summary_text", "text": ...} parts.
func appendResponsesSummaryText(out []byte, summary any) []byte {
	parts, ok := summary.([]any)
	if !ok {
		return out
	}
	for _, part := range parts {
		if p, ok := part.(map[string]any); ok {
			if text, ok := p["text"].(string); ok {
				out = append(out, []byte(text)...)
			}
		}
	}
	return out
}

// appendResponsesContent flattens an Input item's "content" field: a plain
// string, or an array of content parts.
func (b estimateBackend) appendResponsesContent(out []byte, features []fwkrh.MultiModalFeature, content any) ([]byte, []fwkrh.MultiModalFeature) {
	switch v := content.(type) {
	case string:
		out = append(out, []byte(v)...)
	case []any:
		for _, part := range v {
			out, features = b.appendResponsesContentPart(out, features, part)
		}
	}
	return out, features
}

// appendResponsesContentPart flattens one content part. input_image carries
// its URL as a bare string field, unlike chat completions' nested
// {"image_url": {"url": ...}} shape. An input_audio part contributes
// nothing: the Responses input content union does not define it, so the
// model server refuses a request carrying one.
func (b estimateBackend) appendResponsesContentPart(out []byte, features []fwkrh.MultiModalFeature, part any) ([]byte, []fwkrh.MultiModalFeature) {
	p, ok := part.(map[string]any)
	if !ok {
		return out, features
	}
	switch p["type"] {
	case "input_text", "output_text":
		if text, ok := p["text"].(string); ok {
			out = append(out, []byte(text)...)
		}
	case "input_image":
		if url, ok := p["image_url"].(string); ok && url != "" {
			out, features = appendMMAsset(out, features, fwkrh.ModalityImage, url, b.img.placeholderCount(url))
		}
	}
	return out, features
}

// appendMMAsset aligns out to a token boundary, appends count placeholder
// pseudo-tokens derived from a stable content hash, and records the matching
// feature under modality so labels agree with the vllm backend.
func appendMMAsset(out []byte, features []fwkrh.MultiModalFeature, modality fwkrh.Modality, content string, count int) ([]byte, []fwkrh.MultiModalFeature) {
	out = align(out)
	offset := len(out) / fwkrh.BytesPerToken

	sum := xxhash.Sum64String(content)
	token := make([]byte, fwkrh.BytesPerToken)
	binary.LittleEndian.PutUint32(token, uint32(sum)) //#nosec G115 -- intentional hash truncation to build a placeholder token, not an overflow
	for i := 0; i < count; i++ {
		out = append(out, token...)
	}

	features = append(features, fwkrh.MultiModalFeature{
		Modality: modality,
		Hash:     strconv.FormatUint(sum, 16),
		Offset:   offset,
		Length:   count,
	})
	return out, features
}

// packBytes packs bytes into little-endian uint32 tokens (zero-padded tail).
// Reinterpreting them reproduces the input, so locality keys are unchanged.
func packBytes(raw []byte) []uint32 {
	if len(raw) == 0 {
		return nil
	}
	raw = align(raw)
	out := make([]uint32, len(raw)/fwkrh.BytesPerToken)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(raw[i*fwkrh.BytesPerToken:])
	}
	return out
}

// align zero-pads b up to a fwkrh.BytesPerToken boundary.
func align(b []byte) []byte {
	if r := len(b) % fwkrh.BytesPerToken; r != 0 {
		b = append(b, make([]byte, fwkrh.BytesPerToken-r)...)
	}
	return b
}
