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
	Render(ctx context.Context, prompt string) ([]uint32, []tokenizerTypes.Offset, error)
	RenderChat(ctx context.Context, req *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error)
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
// The default backend is `vllm` (HTTP /render). `udsTokenizerConfig` is the
// deprecated gRPC-over-UDS backend, selected only when explicitly enabled. An
// empty configuration falls back to `vllm` with its default endpoint.
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

// estimateConfig selects the estimation backend; no parameters yet.
type estimateConfig struct{}

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
	vllm := config.VLLM != nil
	if (estimate && (uds || vllm)) || (uds && vllm) {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: only one of 'estimate', 'vllm', or 'udsTokenizerConfig' may be set", PluginType)
	}
	if !estimate && config.ModelName == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: 'modelName' must be specified", PluginType)
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

// NewPlugin constructs the configured backend: estimate (byte-packing),
// udsTokenizerConfig (deprecated), or vllm /render (default).
func NewPlugin(ctx context.Context, name string, config *tokenizerPluginConfig) (*Plugin, error) {
	var backend tokenInputProducer
	switch {
	case config.Estimate != nil:
		backend = estimateBackend{}
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
	default:
		cfg := config.VLLM
		if cfg == nil {
			cfg = &vllmConfig{}
		}
		renderer, err := newVLLMHTTPRenderer(cfg, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize vLLM HTTP renderer for '%s' plugin - %w", PluginType, err)
		}
		backend = renderBackend{tk: renderer}
	}

	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
		backend:   backend,
		dk:        TokenizedPromptDataKey.WithNonEmptyProducerName(name),
	}, nil
}

// Plugin tokenizes the prompt in the incoming request and writes the result to
// InferenceRequestBody.TokenizedPrompt for downstream DataProducer / scoring plugins.
type Plugin struct {
	typedName plugin.TypedName
	backend   tokenInputProducer
	dk        plugin.DataKey
}

// compile-time assertion.
var _ requestcontrol.DataProducer = &Plugin{}

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces returns the data keys this plugin produces.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: fwkrh.TokenizedPrompt{}}
}

// Produce derives the request's TokenizedPrompt via the configured backend and
// stores it on the body. Skips when one is already present; errors propagate to
// the Director, which logs and continues.
func (p *Plugin) Produce(ctx context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if request.Body == nil {
		return errors.New("request body is nil")
	}
	if request.Body.TokenizedPrompt != nil {
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
			Role:    msg.Role,
			Content: tokenizerTypes.Content{Raw: msg.Content.Raw},
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
