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

// Package tokenizer provides a DataProducer plugin that tokenizes the request
// prompt and publishes the result on InferenceRequestBody.TokenizedRequest for
// downstream consumers (scorers, filters, other data producers).
package tokenizer

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	kvctok "github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/observability/multimodal"
	rcplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol"
)

type tokenizer interface {
	Render(ctx context.Context, payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error)
	RenderChat(ctx context.Context, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error)
}

const (
	// PluginType is the canonical type name used to register the plugin.
	PluginType = "token-producer"

	// LegacyPluginType is the previous type name. Existing YAML configs that
	// reference it continue to work. Will be removed in a future release.
	//
	// Deprecated: use PluginType ("token-producer") instead.
	LegacyPluginType = "tokenizer"

	tokenizedPromptKeyID = "TokenizedPrompt"

	// anthropicBillingHeaderPrefix marks Claude Code system text that carries a
	// per-request hash; vLLM strips it server-side, so the tokenizer must too.
	anthropicBillingHeaderPrefix = "x-anthropic-billing-header"

	// defaultImageMediaType fills in Anthropic base64 image sources with no
	// media_type, matching vLLM's conversion.
	defaultImageMediaType = "image/jpeg"
)

// Content-block types the Anthropic Messages conversion reads and emits.
const (
	blockTypeText             = "text"
	blockTypeImage            = "image"
	blockTypeImageURL         = "image_url"
	blockTypeThinking         = "thinking"
	blockTypeRedactedThinking = "redacted_thinking"
	blockTypeToolUse          = "tool_use"
	blockTypeToolResult       = "tool_result"
)

// Backend identifiers reported on the tokenize span.
const (
	backendUDS      = "uds"
	backendVLLM     = "vllm"
	backendEstimate = "estimate"
)

// resultSkippedNoTokens marks a tokenize span whose backend returned no tokens,
// distinguishing it from a span missing attributes for any other reason.
const resultSkippedNoTokens = "skipped_no_tokens"

var TokenizedPromptDataKey = plugin.NewDataKey(tokenizedPromptKeyID, PluginType)

// tokenizerPluginConfig holds the configuration for the tokenizer plugin.
//
// Backend selection: `vllm` or `modelName` selects the vLLM HTTP /render
// backend; `udsTokenizerConfig` selects the deprecated gRPC-over-UDS backend;
// `estimate` selects the tokenizer-free byte-packing backend, which is also the
// zero-config default when no backend is set.
type tokenizerPluginConfig struct {
	// TokenizerConfig configures the deprecated gRPC-over-UDS backend.
	//
	// Deprecated: the UDS tokenizer backend is deprecated and will be removed
	// in a future release. Migrate to the `vllm` HTTP /render backend.
	TokenizerConfig kvctok.UdsTokenizerConfig `json:"udsTokenizerConfig,omitempty"`
	// VLLM configures the vLLM /render backend.
	VLLM *vllmConfig `json:"vllm,omitempty"`
	// Estimate selects the tokenizer-free byte-packing backend; mutually
	// exclusive with 'vllm'/'udsTokenizerConfig' and needs no 'modelName'.
	Estimate *estimateConfig `json:"estimate,omitempty"`
	// ModelName is the name of the model whose tokenizer should be loaded.
	ModelName string `json:"modelName"`
}

// estimateConfig configures the estimation backend. Multimodal image and video
// estimation are the only tunables; an empty config uses built-in defaults.
type estimateConfig struct {
	// Image tunes multimodal image placeholder-token estimation.
	Image *imageEstimateConfig `json:"image,omitempty"`
	// Video tunes multimodal video placeholder-token estimation.
	Video *videoEstimateConfig `json:"video,omitempty"`
}

// imageEstimateConfig tunes how an image's placeholder-token count is estimated.
// Empty fields fall back to built-in defaults (dynamic mode, 640x360, factor 1024).
type imageEstimateConfig struct {
	// Mode selects "dynamic" (width*height/factor) or "static" (a constant count).
	Mode string `json:"mode,omitempty"`
	// DefaultResolution is the fallback resolution for dynamic mode when an
	// image's dimensions cannot be decoded.
	DefaultResolution *resolution `json:"defaultResolution,omitempty"`
	// Static configures the static (constant per-image) mode.
	Static *staticImageConfig `json:"static,omitempty"`
	// Dynamic configures the dynamic (pixels/factor) mode.
	Dynamic *dynamicImageConfig `json:"dynamic,omitempty"`
}

// staticImageConfig is the static-mode parameter.
type staticImageConfig struct {
	// StaticToken is the per-image placeholder count.
	StaticToken int `json:"staticToken,omitempty"`
}

// dynamicImageConfig is the dynamic-mode parameter.
type dynamicImageConfig struct {
	// Factor maps pixels to placeholder tokens (width*height/factor).
	Factor int `json:"factor,omitempty"`
}

// resolution is an image or video-frame width/height in pixels.
type resolution struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// videoEstimateConfig tunes how a video's placeholder-token count is estimated:
// min(frames * tokensPerFrame, maxVideoTokens). Empty fields fall back to
// built-in defaults. qwen3 is dynamic tokens-per-frame + sampled frames; gemma4
// is static tokens-per-frame + strided frames. Duration and resolution are not
// decoded from the video; they come from these fields.
type videoEstimateConfig struct {
	// DefaultResolution is the per-frame resolution used for dynamic
	// tokens-per-frame.
	DefaultResolution *resolution `json:"defaultResolution,omitempty"`
	// DefaultDuration is the video length in seconds used for frame counting.
	DefaultDuration float64 `json:"defaultDuration,omitempty"`
	// TokensPerFrame configures the per-frame placeholder count.
	TokensPerFrame *tokensPerFrameConfig `json:"tokensPerFrame,omitempty"`
	// Frames configures how many frames are sampled from the video.
	Frames *framesConfig `json:"frames,omitempty"`
	// MaxVideoTokens caps the total placeholder count. Zero means uncapped.
	MaxVideoTokens int `json:"maxVideoTokens,omitempty"`
}

// tokensPerFrameConfig configures the per-frame placeholder count.
type tokensPerFrameConfig struct {
	// Mode selects "dynamic" (width*height/factor) or "static" (a constant count).
	Mode string `json:"mode,omitempty"`
	// Static configures the static (constant per-frame) mode.
	Static *tokensPerFrameStaticMode `json:"static,omitempty"`
	// Dynamic configures the dynamic (pixels/factor) mode.
	Dynamic *tokensPerFrameDynamicMode `json:"dynamic,omitempty"`
}

// tokensPerFrameStaticMode is the static-mode parameter.
type tokensPerFrameStaticMode struct {
	// NumTokensPerFrame is the per-frame placeholder count.
	NumTokensPerFrame int `json:"numTokensPerFrame,omitempty"`
}

// tokensPerFrameDynamicMode is the dynamic-mode parameter.
type tokensPerFrameDynamicMode struct {
	// Factor maps a frame's pixels to placeholder tokens (width*height/factor).
	Factor int `json:"factor,omitempty"`
}

// framesConfig configures how many frames are counted from a video. MinFrames
// and MaxFrames clamp the count in both modes; the mode sub-structs hold the
// mode-specific knobs.
type framesConfig struct {
	// Mode selects "sampled" (duration*sampleFPS) or "strided"
	// (duration*sourceFPS/frameStride).
	Mode string `json:"mode,omitempty"`
	// MinFrames floors the frame count. Zero means no floor.
	MinFrames int `json:"minFrames,omitempty"`
	// MaxFrames caps the frame count. Zero means uncapped.
	MaxFrames int `json:"maxFrames,omitempty"`
	// Sampled configures the sampled (duration*sampleFPS) mode.
	Sampled *framesSampledMode `json:"sampled,omitempty"`
	// Strided configures the strided (duration*sourceFPS/frameStride) mode.
	Strided *framesStridedMode `json:"strided,omitempty"`
}

// framesSampledMode configures the sampled frame-count mode.
type framesSampledMode struct {
	// SampleFPS is the sampling rate.
	SampleFPS float64 `json:"sampleFPS,omitempty"`
	// TemporalPatchSize merges every N sampled frames into one token group,
	// modeling temporal patch merging (e.g. qwen3-vl uses 2). Values < 2 apply
	// no merging.
	TemporalPatchSize int `json:"temporalPatchSize,omitempty"`
}

// framesStridedMode configures the strided frame-count mode.
type framesStridedMode struct {
	// DefaultSourceFPS is the fallback source frame rate, used when the
	// x-llm-d-video-fps header is absent.
	DefaultSourceFPS float64 `json:"defaultSourceFPS,omitempty"`
	// FrameStride keeps every Nth source frame.
	FrameStride int `json:"frameStride,omitempty"`
}

// PluginFactory is the factory function for the tokenizer plugin.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	config := tokenizerPluginConfig{}

	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}

	estimate := config.Estimate != nil
	uds := config.TokenizerConfig.IsEnabled()
	vllm := config.VLLM != nil || config.ModelName != ""
	if (estimate && (uds || vllm)) || (uds && vllm) {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: only one of 'estimate', 'vllm', or 'udsTokenizerConfig' may be set", PluginType)
	}
	// modelName is required only by the real-tokenizer backends; the zero-config
	// path selects the estimate backend, which needs none.
	if (uds || vllm) && config.ModelName == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: 'modelName' must be specified", PluginType)
	}
	if config.Estimate != nil && config.Estimate.Image != nil {
		if m := config.Estimate.Image.Mode; m != "" && m != imageModeDynamic && m != imageModeStatic {
			return nil, fmt.Errorf("invalid configuration for '%s' plugin: estimate.image.mode must be %q or %q", PluginType, imageModeDynamic, imageModeStatic)
		}
	}
	if config.Estimate != nil && config.Estimate.Video != nil {
		vid := config.Estimate.Video
		if vid.TokensPerFrame != nil {
			if m := vid.TokensPerFrame.Mode; m != "" && m != videoTPFModeDynamic && m != videoTPFModeStatic {
				return nil, fmt.Errorf("invalid configuration for '%s' plugin: estimate.video.tokensPerFrame.mode must be %q or %q", PluginType, videoTPFModeDynamic, videoTPFModeStatic)
			}
		}
		if vid.Frames != nil {
			if m := vid.Frames.Mode; m != "" && m != videoFramesModeSampled && m != videoFramesModeStrided {
				return nil, fmt.Errorf("invalid configuration for '%s' plugin: estimate.video.frames.mode must be %q or %q", PluginType, videoFramesModeSampled, videoFramesModeStrided)
			}
		}
	}

	p, err := NewPlugin(handle.Context(), name, &config)
	if err != nil {
		return nil, err
	}

	return p, nil
}

// LegacyPluginFactory wraps PluginFactory for the deprecated `tokenizer` type
// name. It logs a one-time-per-instantiation deprecation warning and delegates
// to PluginFactory. Will be removed when LegacyPluginType is removed.
//
// Deprecated: register PluginType ("token-producer") instead.
func LegacyPluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	log.FromContext(handle.Context()).Info(
		"DEPRECATION: plugin type '"+LegacyPluginType+"' is deprecated; use '"+PluginType+"' instead",
		"pluginName", name,
	)
	return PluginFactory(name, rawParameters, handle)
}

// NewPlugin constructs the configured backend: udsTokenizerConfig (deprecated),
// vllm /render (selected by 'vllm' or 'modelName'), or estimate byte-packing
// (the default when no backend is set).
func NewPlugin(ctx context.Context, name string, config *tokenizerPluginConfig) (*Plugin, error) {
	var backend tokenInputProducer
	var backendName string
	switch {
	case config.TokenizerConfig.IsEnabled():
		log.FromContext(ctx).Info(
			"DEPRECATION: the 'udsTokenizerConfig' parameter is deprecated and will be removed in a future release; set the 'vllm' parameter instead (see plugin README)",
			"pluginType", PluginType,
		)
		uds, err := newUDSTokenizer(ctx, &config.TokenizerConfig, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize UDS tokenizer for '%s' plugin - %w", PluginType, err)
		}
		backend = renderBackend{tk: uds}
		backendName = backendUDS
	case config.VLLM != nil || config.ModelName != "":
		cfg := config.VLLM
		if cfg == nil {
			cfg = &vllmConfig{}
		}
		renderer, err := newVLLMHTTPRenderer(cfg, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize vLLM HTTP renderer for '%s' plugin - %w", PluginType, err)
		}
		backend = renderBackend{tk: renderer}
		backendName = backendVLLM
	default:
		backend = estimateBackend{img: newImageEstimator(config.Estimate), vid: newVideoEstimator(config.Estimate)}
		backendName = backendEstimate
	}

	p := &Plugin{
		typedName:   plugin.TypedName{Type: PluginType, Name: name},
		backend:     backend,
		backendName: backendName,
		dk:          TokenizedPromptDataKey.WithNonEmptyProducerName(name),
	}
	if w, ok := backend.(warmer); ok {
		go w.warmup(ctx)
	}
	return p, nil
}

// Plugin tokenizes the prompt in the incoming request and writes the result to
// InferenceRequestBody.TokenizedRequest for downstream DataProducer / scoring plugins.
type Plugin struct {
	typedName plugin.TypedName
	backend   tokenInputProducer
	// backendName identifies the configured backend on the tokenize span.
	backendName string
	dk          plugin.DataKey
}

// compile-time assertions.
var (
	_ requestcontrol.DataProducer         = &Plugin{}
	_ requestcontrol.TimeoutAwareProducer = &Plugin{}
)

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces returns the data keys this plugin produces.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: fwkrh.TokenizedRequest{}}
}

// ProduceTimeout surfaces the backend's render timeout when it manages one, so
// the director extends the data-producer budget past its default. Returns 0 to
// keep the default (e.g. the estimate backend, which is in-memory).
func (p *Plugin) ProduceTimeout() time.Duration {
	if ta, ok := p.backend.(timeoutAware); ok {
		return ta.produceTimeout()
	}
	return 0
}

// Produce derives the request's TokenizedRequest via the configured backend and
// stores it on the body. Skips when one is already present; errors propagate to
// the Director, which logs and continues.
//
// The tokenize span opens below the already-tokenized skip path, so it is
// emitted only when the backend is actually invoked.
func (p *Plugin) Produce(ctx context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if request.Body == nil {
		return errors.New("request body is nil")
	}
	if request.Body.TokenizedRequest != nil {
		// A parser (e.g. vLLM gRPC) may pre-populate tokens without a salt;
		// ensure cache-salt isolation still applies on the skip path.
		if request.Body.TokenizedRequest.CacheSalt == "" {
			request.Body.TokenizedRequest.CacheSalt = CacheSaltFromBody(request.Body)
		}
		return nil
	}

	ctx = withMMMetadata(ctx, parseMMMetadataHeaders(request.Headers))

	ctx, span := tracing.Tracer(rcplugins.TracerScope).Start(ctx, "tokenize",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()
	// On the default (tracing-disabled) path Start returns a non-recording span;
	// skip attribute construction, which walks the request's multimodal features.
	tracingActive := span.IsRecording()
	if tracingActive {
		attrs := []attribute.KeyValue{
			attribute.String("llm_d.epp.token_producer.backend", p.backendName),
		}
		if request.TargetModel != "" {
			attrs = append(attrs, attribute.String("gen_ai.request.model", request.TargetModel))
		}
		if request.RequestID != "" {
			attrs = append(attrs, attribute.String("gen_ai.request.id", request.RequestID))
		}
		span.SetAttributes(attrs...)
	}

	tp, err := p.backend.produce(ctx, request.Body)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return err
	}
	if tp == nil || tp.TokenCount() == 0 {
		if tracingActive {
			span.SetAttributes(attribute.String("llm_d.epp.token_producer.result", resultSkippedNoTokens))
		}
		return nil
	}
	tp.CacheSalt = CacheSaltFromBody(request.Body)
	request.Body.TokenizedRequest = tp

	if tracingActive {
		span.SetAttributes(append(mmobs.SpanAttributes(request),
			attribute.Int("llm_d.epp.token_producer.token_count", tp.TokenCount()),
		)...)
	}
	return nil
}

// ChatCompletionsToRenderChatRequest converts a ChatCompletionsRequest to a
// tokenization RenderChatRequest, including multimodal content blocks.
func ChatCompletionsToRenderChatRequest(chat *fwkrh.ChatCompletionsRequest) *tokenizerTypes.RenderChatRequest {
	conversation := make([]tokenizerTypes.Conversation, 0, len(chat.Messages))
	for _, msg := range chat.Messages {
		conv := tokenizerTypes.Conversation{
			Role:      msg.Role,
			Content:   &tokenizerTypes.Content{Raw: msg.Content.Raw},
			ToolCalls: msg.ToolCalls,
		}
		for _, block := range msg.Content.Structured {
			conv.Content.Structured = append(conv.Content.Structured, tokenizerTypes.ContentBlock{
				Type:     block.Type,
				Text:     block.Text,
				ImageURL: tokenizerTypes.ImageBlock{URL: block.ImageURL.URL},
			})
		}
		conversation = append(conversation, conv)
	}

	return &tokenizerTypes.RenderChatRequest{
		Conversation:              conversation,
		Tools:                     chat.Tools,
		Documents:                 chat.Documents,
		ChatTemplate:              chat.ChatTemplate,
		ReturnAssistantTokensMask: chat.ReturnAssistantTokensMask,
		ContinueFinalMessage:      chat.ContinueFinalMessage,
		AddGenerationPrompt:       chat.AddGenerationPrompt,
		ChatTemplateKWArgs:        chat.ChatTemplateKWArgs,
	}
}

// MessagesToRenderChatRequest converts an Anthropic MessagesRequest into the
// OpenAI chat shape vLLM builds when serving /v1/messages, so the render
// backend and the server apply the identical chat-template pipeline to the
// same request and prefix-cache blocks line up.
func MessagesToRenderChatRequest(msg *fwkrh.MessagesRequest) *tokenizerTypes.RenderChatRequest {
	conversation := make([]tokenizerTypes.Conversation, 0, 1+len(msg.Messages))

	if sys := anthropicSystemText(msg.System); sys != "" {
		conversation = append(conversation, tokenizerTypes.Conversation{
			Role:    "system",
			Content: &tokenizerTypes.Content{Raw: sys},
		})
	}

	for _, m := range msg.Messages {
		if m.Role == "system" {
			// Not valid Anthropic input; tolerated the way vLLM does.
			if text := anthropicSystemText(m.Content); text != "" {
				conversation = append(conversation, tokenizerTypes.Conversation{
					Role:    "system",
					Content: &tokenizerTypes.Content{Raw: text},
				})
			}
			continue
		}
		conversation = appendAnthropicMessage(conversation, m)
	}

	return &tokenizerTypes.RenderChatRequest{
		Conversation: conversation,
		Tools:        convertAnthropicTools(msg.Tools),
	}
}

// anthropicSystemText joins the system prompt's text blocks with no
// separator, dropping Claude Code's per-request billing header so it does not
// defeat prefix caching (mirrors vLLM).
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

// appendAnthropicMessage appends the conversations for one message. User
// tool_result blocks append their tool messages immediately, so those precede
// the user message that carried them - the order vLLM emits.
func appendAnthropicMessage(conversation []tokenizerTypes.Conversation, m fwkrh.AnthropicMessage) []tokenizerTypes.Conversation {
	if m.Content.Raw != "" {
		return append(conversation, tokenizerTypes.Conversation{
			Role:    m.Role,
			Content: &tokenizerTypes.Content{Raw: m.Content.Raw},
		})
	}

	var contentBlocks []tokenizerTypes.ContentBlock
	var toolCalls []any
	var reasoning strings.Builder
	for _, b := range m.Content.Structured {
		switch b.Type {
		case blockTypeText:
			if b.Text != "" {
				contentBlocks = append(contentBlocks, tokenizerTypes.ContentBlock{Type: blockTypeText, Text: b.Text})
			}
		case blockTypeImage:
			contentBlocks = appendImageBlock(contentBlocks, b.Source)
		case blockTypeThinking:
			reasoning.WriteString(b.Thinking)
		case blockTypeRedactedThinking:
			// Opaque safety-filtered reasoning; parses but contributes no tokens.
		case blockTypeToolUse:
			toolCalls = append(toolCalls, anthropicToolCall(b))
		case blockTypeToolResult:
			if m.Role == "user" {
				conversation = appendAnthropicToolResult(conversation, b)
			} else {
				text, _ := anthropicToolResultContent(b)
				contentBlocks = append(contentBlocks, tokenizerTypes.ContentBlock{
					Type: blockTypeText,
					Text: "Tool result: " + text,
				})
			}
		}
	}

	conv := tokenizerTypes.Conversation{Role: m.Role}
	if reasoning.Len() > 0 {
		conv.Reasoning = reasoning.String()
	}
	conv.ToolCalls = toolCalls
	switch {
	case len(contentBlocks) == 1 && contentBlocks[0].Type == blockTypeText:
		conv.Content = &tokenizerTypes.Content{Raw: contentBlocks[0].Text}
	case len(contentBlocks) > 0:
		conv.Content = &tokenizerTypes.Content{Structured: contentBlocks}
	}
	// A user message reduced to bare tool_results has no content of its own;
	// its tool messages were already appended above.
	if m.Role == "user" && conv.Content == nil {
		return conversation
	}
	return append(conversation, conv)
}

// appendImageBlock maps an Anthropic image source to an OpenAI image_url
// content block; sources that resolve to no URL are dropped.
func appendImageBlock(blocks []tokenizerTypes.ContentBlock, src *fwkrh.AnthropicImageSource) []tokenizerTypes.ContentBlock {
	if url := anthropicImageToURL(src); url != "" {
		blocks = append(blocks, tokenizerTypes.ContentBlock{
			Type:     blockTypeImageURL,
			ImageURL: tokenizerTypes.ImageBlock{URL: url},
		})
	}
	return blocks
}

// anthropicToolCall converts a tool_use block into an OpenAI function tool
// call. Arguments are CPython json.dumps formatted (separators, ASCII
// escaping, wire key order) - the exact string vLLM renders into the prompt.
func anthropicToolCall(b fwkrh.AnthropicContentBlock) map[string]any {
	id := b.ID
	if id == "" {
		// vLLM falls back to call_<unix-time>; a fixed stand-in only keeps the
		// rendered length stable (ids are generated in practice).
		id = "call_0000000000"
	}
	return map[string]any{
		"id":   id,
		"type": "function",
		"function": map[string]any{
			"name":      b.Name,
			"arguments": pythonArguments(b.Input),
		},
	}
}

// pythonArguments renders tool_use input as json.dumps(input or {}): absent,
// null, and empty-object inputs all render as "{}".
func pythonArguments(raw json.RawMessage) string {
	switch string(bytes.TrimSpace(raw)) {
	case "", "null", "{}":
		return "{}"
	}
	if out, err := pythonDumps(raw); err == nil {
		return out
	}
	return "{}"
}

// appendAnthropicToolResult appends a tool-role message for a user
// tool_result block; images in the result follow as their own user message.
func appendAnthropicToolResult(conversation []tokenizerTypes.Conversation, b fwkrh.AnthropicContentBlock) []tokenizerTypes.Conversation {
	text, imageBlocks := anthropicToolResultContent(b)
	conversation = append(conversation, tokenizerTypes.Conversation{
		Role:       "tool",
		ToolCallID: b.ToolUseID,
		Content:    &tokenizerTypes.Content{Raw: text},
	})
	if len(imageBlocks) > 0 {
		conversation = append(conversation, tokenizerTypes.Conversation{
			Role:    "user",
			Content: &tokenizerTypes.Content{Structured: imageBlocks},
		})
	}
	return conversation
}

// anthropicToolResultContent splits a tool_result's content into its text
// (block texts joined with newlines) and image blocks.
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

// convertAnthropicTools rewrites Anthropic tool definitions into OpenAI
// function tools. input_schema passes through as raw JSON so the template's
// re-serialization keeps the wire key order.
func convertAnthropicTools(tools []fwkrh.AnthropicTool) []any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		var schema json.RawMessage = bytes.TrimSpace(t.InputSchema)
		if len(schema) == 0 || bytes.Equal(schema, []byte("null")) {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		fn := map[string]any{
			"name":       t.Name,
			"parameters": schema,
		}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		if t.Strict != nil {
			fn["strict"] = *t.Strict
		}
		if t.DeferLoading != nil {
			fn["defer_loading"] = *t.DeferLoading
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

// anthropicImageToURL converts an Anthropic image source to an OpenAI-shaped
// URL. Sources carrying a URL pass it through (URL sources, and sources
// missing a type); base64 sources become data URIs, with an image/jpeg media
// type when absent. Sources with neither a URL nor data yield "" so the
// caller drops the block.
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

// convertMMFeaturesToUpstream flattens the kv-cache map-shaped multimodal
// metadata into a flat list sorted by placeholder offset so consumers see
// items in prompt order. Returns nil when no content is present.
func convertMMFeaturesToUpstream(src *tokenization.MultiModalFeatures) []fwkrh.MultiModalFeature {
	if src == nil || len(src.MMHashes) == 0 {
		return nil
	}

	var items []fwkrh.MultiModalFeature
	for modality, hashes := range src.MMHashes {
		ranges, ok := src.MMPlaceholders[modality]
		if !ok {
			continue
		}
		n := len(hashes)
		if len(ranges) < n {
			n = len(ranges)
		}
		for i := 0; i < n; i++ {
			items = append(items, fwkrh.MultiModalFeature{
				Modality: fwkrh.Modality(modality),
				Hash:     hashes[i],
				Offset:   ranges[i].Offset,
				Length:   ranges[i].Length,
			})
		}
	}
	if len(items) == 0 {
		return nil
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Offset < items[j].Offset })
	return items
}

// ConvertMMFeaturesFromUpstream regroups the flat list of multimodal features
// back into the kv-cache map-shape expected by kvblock.ComputeBlockExtraFeatures.
func ConvertMMFeaturesFromUpstream(features []fwkrh.MultiModalFeature) (map[string][]string, map[string][]kvblock.PlaceholderRange) {
	if len(features) == 0 {
		return nil, nil
	}
	hashes := make(map[string][]string)
	ranges := make(map[string][]kvblock.PlaceholderRange)
	for _, f := range features {
		k := string(f.Modality)
		hashes[k] = append(hashes[k], f.Hash)
		ranges[k] = append(ranges[k], kvblock.PlaceholderRange{
			Offset: f.Offset,
			Length: f.Length,
		})
	}
	return hashes, ranges
}
