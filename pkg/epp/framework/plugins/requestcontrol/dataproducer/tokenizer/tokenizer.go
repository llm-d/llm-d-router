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
// prompt and publishes the result on InferenceRequestBody.TokenizedPrompt for
// downstream consumers (scorers, filters, other data producers).
package tokenizer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	kvctok "github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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
)

var TokenizedPromptDataKey = plugin.NewDataKey(tokenizedPromptKeyID, PluginType)

type tokenizerBackend string

const (
	tokenizerBackendEstimate tokenizerBackend = "estimate"
	tokenizerBackendVLLM     tokenizerBackend = "vllm"
	tokenizerBackendUDS      tokenizerBackend = "uds"
)

// tokenizerPluginConfig holds the configuration for the tokenizer plugin.
type tokenizerPluginConfig struct {
	// Backend explicitly selects "estimate", "vllm", or "uds". When omitted,
	// the plugin preserves legacy behavior by inferring the backend from
	// existing config fields.
	Backend tokenizerBackend `json:"backend,omitempty"`
	// TokenizerConfig configures the deprecated gRPC-over-UDS backend.
	//
	// Deprecated: the UDS tokenizer backend is deprecated and will be removed
	// in a future release. Migrate to the `vllm` HTTP /render backend.
	TokenizerConfig kvctok.UdsTokenizerConfig `json:"udsTokenizerConfig,omitempty"`
	// VLLM configures the vLLM /render backend.
	VLLM *vllmConfig `json:"vllm,omitempty"`
	// Estimate configures the tokenizer-free byte-packing backend. When
	// backend is omitted, it selects the estimate backend and needs no
	// 'modelName'.
	Estimate *estimateConfig `json:"estimate,omitempty"`
	// ModelName is the name of the model whose tokenizer should be loaded.
	ModelName string `json:"modelName"`
}

// estimateConfig configures the estimation backend. Multimodal image estimation
// is the only tunable; an empty config uses built-in defaults.
type estimateConfig struct {
	// Image tunes multimodal image placeholder-token estimation.
	Image *imageEstimateConfig `json:"image,omitempty"`
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

// resolution is an image width/height in pixels.
type resolution struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// PluginFactory is the factory function for the tokenizer plugin.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	config := tokenizerPluginConfig{}

	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}

	p, err := NewPlugin(handle.Context(), name, &config)
	if err != nil {
		return nil, err
	}

	return p, nil
}

func (config *tokenizerPluginConfig) validateAndSelectBackend() (tokenizerBackend, error) {
	selected, err := config.selectBackend()
	if err != nil {
		return "", err
	}

	switch selected {
	case tokenizerBackendVLLM:
		if config.ModelName == "" {
			return "", fmt.Errorf("invalid configuration for '%s' plugin: 'modelName' must be specified", PluginType)
		}
	case tokenizerBackendUDS:
		if config.ModelName == "" {
			return "", fmt.Errorf("invalid configuration for '%s' plugin: 'modelName' must be specified", PluginType)
		}
		if !config.TokenizerConfig.IsEnabled() {
			return "", fmt.Errorf("invalid configuration for '%s' plugin: 'udsTokenizerConfig' must be specified when backend is %q", PluginType, tokenizerBackendUDS)
		}
	case tokenizerBackendEstimate:
		if config.Estimate != nil && config.Estimate.Image != nil {
			if m := config.Estimate.Image.Mode; m != "" && m != imageModeDynamic && m != imageModeStatic {
				return "", fmt.Errorf("invalid configuration for '%s' plugin: estimate.image.mode must be %q or %q", PluginType, imageModeDynamic, imageModeStatic)
			}
		}
	}

	return selected, nil
}

func (config *tokenizerPluginConfig) selectBackend() (tokenizerBackend, error) {
	if config.Backend != "" {
		switch config.Backend {
		case tokenizerBackendEstimate, tokenizerBackendVLLM, tokenizerBackendUDS:
			return config.Backend, nil
		default:
			return "", fmt.Errorf("invalid configuration for '%s' plugin: backend must be one of %q, %q, or %q", PluginType, tokenizerBackendEstimate, tokenizerBackendVLLM, tokenizerBackendUDS)
		}
	}

	estimate := config.Estimate != nil
	uds := config.TokenizerConfig.IsEnabled()
	vllm := config.VLLM != nil || config.ModelName != ""
	if (estimate && (uds || vllm)) || (uds && vllm) {
		return "", fmt.Errorf("invalid configuration for '%s' plugin: only one of 'estimate', 'vllm', or 'udsTokenizerConfig' may be set", PluginType)
	}
	switch {
	case uds:
		return tokenizerBackendUDS, nil
	case vllm:
		return tokenizerBackendVLLM, nil
	default:
		return tokenizerBackendEstimate, nil
	}
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

// NewPlugin constructs the configured backend. The explicit backend selector
// wins when set; otherwise legacy config inference is used.
func NewPlugin(ctx context.Context, name string, config *tokenizerPluginConfig) (*Plugin, error) {
	if config == nil {
		config = &tokenizerPluginConfig{}
	}
	selected, err := config.validateAndSelectBackend()
	if err != nil {
		return nil, err
	}

	var backend tokenInputProducer
	switch selected {
	case tokenizerBackendUDS:
		log.FromContext(ctx).Info(
			"DEPRECATION: the 'udsTokenizerConfig' parameter is deprecated and will be removed in a future release; set the 'vllm' parameter instead (see plugin README)",
			"pluginType", PluginType,
		)
		uds, err := newUDSTokenizer(ctx, &config.TokenizerConfig, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize UDS tokenizer for '%s' plugin - %w", PluginType, err)
		}
		backend = renderBackend{tk: uds}
	case tokenizerBackendVLLM:
		cfg := config.VLLM
		if cfg == nil {
			cfg = &vllmConfig{}
		}
		renderer, err := newVLLMHTTPRenderer(cfg, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize vLLM HTTP renderer for '%s' plugin - %w", PluginType, err)
		}
		backend = renderBackend{tk: renderer}
	case tokenizerBackendEstimate:
		backend = estimateBackend{img: newImageEstimator(config.Estimate)}
	}

	p := &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
		backend:   backend,
		dk:        TokenizedPromptDataKey.WithNonEmptyProducerName(name),
	}
	if w, ok := backend.(warmer); ok {
		go w.warmup(ctx)
	}
	return p, nil
}

// Plugin tokenizes the prompt in the incoming request and writes the result to
// InferenceRequestBody.TokenizedPrompt for downstream DataProducer / scoring plugins.
type Plugin struct {
	typedName plugin.TypedName
	backend   tokenInputProducer
	dk        plugin.DataKey
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
	return map[plugin.DataKey]any{p.dk: fwkrh.TokenizedPrompt{}}
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

// Produce derives the request's TokenizedPrompt via the configured backend and
// stores it on the body. Skips when one is already present; errors propagate to
// the Director, which logs and continues.
func (p *Plugin) Produce(ctx context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if request.Body == nil {
		return errors.New("request body is nil")
	}
	if request.Body.TokenizedPrompt != nil {
		// A parser (e.g. vLLM gRPC) may pre-populate tokens without a salt;
		// ensure cache-salt isolation still applies on the skip path.
		if request.Body.TokenizedPrompt.CacheSalt == "" {
			request.Body.TokenizedPrompt.CacheSalt = CacheSaltFromBody(request.Body)
		}
		return nil
	}

	tp, err := p.backend.produce(ctx, request.Body)
	if err != nil {
		return err
	}
	if tp == nil || tp.TokenCount() == 0 {
		return nil
	}
	tp.CacheSalt = CacheSaltFromBody(request.Body)
	request.Body.TokenizedPrompt = tp
	return nil
}

// ChatCompletionsToRenderChatRequest converts a ChatCompletionsRequest to a
// tokenization RenderChatRequest, including multimodal content blocks.
func ChatCompletionsToRenderChatRequest(chat *fwkrh.ChatCompletionsRequest) *tokenizerTypes.RenderChatRequest {
	conversation := make([]tokenizerTypes.Conversation, 0, len(chat.Messages))
	for _, msg := range chat.Messages {
		conv := tokenizerTypes.Conversation{
			Role:      msg.Role,
			Content:   tokenizerTypes.Content{Raw: msg.Content.Raw},
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

// MessagesToRenderChatRequest converts an Anthropic MessagesRequest to a
// tokenization RenderChatRequest for vLLM /render endpoint with System, Message and Tools set in RenderChatRequest only.
func MessagesToRenderChatRequest(msg *fwkrh.MessagesRequest) *tokenizerTypes.RenderChatRequest {
	conversation := make([]tokenizerTypes.Conversation, 0, 1+len(msg.Messages))

	if msg.System.Raw != "" || len(msg.System.Structured) > 0 {
		conversation = append(conversation, tokenizerTypes.Conversation{
			Role:    "system",
			Content: convertAnthropicContent(msg.System),
		})
	}

	for _, m := range msg.Messages {
		conversation = append(conversation, tokenizerTypes.Conversation{
			Role:    m.Role, // role: user, assistant, system
			Content: convertAnthropicContent(m.Content),
		})
	}

	return &tokenizerTypes.RenderChatRequest{
		Conversation: conversation,
		Tools:        msg.Tools,
	}
}

// convertAnthropicContent converts an AnthropicContent to the kv-cache tokenizer Content type
// mapping Anthropic image blocks to OpenAI-shaped image_url blocks.
func convertAnthropicContent(ac fwkrh.AnthropicContent) tokenizerTypes.Content {
	if ac.Raw != "" {
		return tokenizerTypes.Content{Raw: ac.Raw}
	}
	blocks := make([]tokenizerTypes.ContentBlock, 0, len(ac.Structured))
	for _, b := range ac.Structured {
		switch b.Type {
		case "text":
			blocks = append(blocks, tokenizerTypes.ContentBlock{
				Type: "text",
				Text: b.Text,
			})
		case "image":
			if url := anthropicImageToURL(b.Source); url != "" {
				blocks = append(blocks, tokenizerTypes.ContentBlock{
					Type:     "image_url",
					ImageURL: tokenizerTypes.ImageBlock{URL: url},
				})
			}
		}
	}
	return tokenizerTypes.Content{Structured: blocks}
}

// anthropicImageToURL converts an Anthropic image source to an OpenAI-shaped URL.
// Base64 sources become data URIs; URL sources pass through.
func anthropicImageToURL(src *fwkrh.AnthropicImageSource) string {
	if src == nil {
		return ""
	}
	if src.URL != "" {
		return src.URL
	}
	if src.Data != "" {
		return "data:" + src.MediaType + ";base64," + src.Data
	}
	return ""
}

// convertMMFeaturesToUpstream flattens the kv-cache map-shaped multimodal
// metadata into the upstream flat list, sorted by placeholder offset so
// consumers see items in prompt order. Returns nil when no content is present.
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
