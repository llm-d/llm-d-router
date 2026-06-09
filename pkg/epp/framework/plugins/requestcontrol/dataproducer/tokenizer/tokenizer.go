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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type tokenizer interface {
	Render(ctx context.Context, payload fwkrh.RequestPayload) ([]uint32, []tokenizerTypes.Offset, error)
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
	TokenizerConfig tokenization.UdsTokenizerConfig `json:"udsTokenizerConfig,omitempty"`
	// VLLM configures the vLLM /render backend.
	VLLM *vllmConfig `json:"vllm,omitempty"`
	// Estimate selects the tokenizer-free byte-packing backend; mutually
	// exclusive with 'vllm'/'udsTokenizerConfig' and needs no 'modelName'.
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
	default:
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
			request.Body.TokenizedPrompt.CacheSalt = cacheSaltFromBody(request.Body)
		}
		return nil
	}

	tp, err := p.backend.produce(ctx, request.Body)
	if err != nil {
		return err
	}
	tp.CacheSalt = cacheSaltFromBody(request.Body)

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
