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
	"slices"
	"sort"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
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

	// TokenizedPromptKey is the data key advertised by this plugin to indicate
	// that it produces tokenized prompt data on InferenceRequestBody.TokenizedPrompt.
	TokenizedPromptKey = "TokenizedPrompt"
)

// tokenizerPluginConfig holds the configuration for the tokenizer plugin.
//
// The default backend is `vllm` (HTTP /render). `udsTokenizerConfig` is the
// legacy gRPC-over-UDS backend, selected only when explicitly enabled. An
// empty configuration falls back to `vllm` with its default endpoint.
type tokenizerPluginConfig struct {
	// TokenizerConfig configures the legacy gRPC-over-UDS backend.
	TokenizerConfig tokenization.UdsTokenizerConfig `json:"udsTokenizerConfig,omitempty"`
	// VLLM configures the vLLM /render backend.
	VLLM *vllmConfig `json:"vllm,omitempty"`
	// ModelName is the name of the model whose tokenizer should be loaded.
	ModelName string `json:"modelName"`
}

// PluginFactory is the factory function for the tokenizer plugin.
func PluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	config := tokenizerPluginConfig{}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}

	if config.ModelName == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: 'modelName' must be specified", PluginType)
	}
	if config.VLLM != nil && config.TokenizerConfig.IsEnabled() {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: only one of 'udsTokenizerConfig' or 'vllm' may be set", PluginType)
	}

	p, err := NewPlugin(handle.Context(), &config)
	if err != nil {
		return nil, err
	}

	return p.WithName(name), nil
}

// LegacyPluginFactory wraps PluginFactory for the deprecated `tokenizer` type
// name. It logs a one-time-per-instantiation deprecation warning and delegates
// to PluginFactory. Will be removed when LegacyPluginType is removed.
//
// Deprecated: register PluginType ("token-producer") instead.
func LegacyPluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	log.FromContext(handle.Context()).Info(
		"DEPRECATION: plugin type '"+LegacyPluginType+"' is deprecated; use '"+PluginType+"' instead",
		"pluginName", name,
	)
	return PluginFactory(name, rawParameters, handle)
}

// NewPlugin creates a new tokenizer plugin instance and constructs the
// configured backend. vllm is the default; udsTokenizerConfig is selected
// only when explicitly enabled (its socketFile is set).
func NewPlugin(ctx context.Context, config *tokenizerPluginConfig) (*Plugin, error) {
	var tk tokenizer
	switch {
	case config.TokenizerConfig.IsEnabled():
		uds, err := newUDSTokenizer(ctx, &config.TokenizerConfig, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize UDS tokenizer for '%s' plugin - %w", PluginType, err)
		}
		tk = uds
	default:
		cfg := config.VLLM
		if cfg == nil {
			cfg = &vllmConfig{}
		}
		renderer, err := newVLLMHTTPRenderer(cfg, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize vLLM HTTP renderer for '%s' plugin - %w", PluginType, err)
		}
		tk = renderer
	}

	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType},
		tokenizer: tk,
	}, nil
}

// Plugin tokenizes the prompt in the incoming request and writes the result to
// InferenceRequestBody.TokenizedPrompt for downstream DataProducer / scoring plugins.
type Plugin struct {
	typedName plugin.TypedName
	tokenizer tokenizer
}

// compile-time assertion.
var _ requestcontrol.DataProducer = &Plugin{}

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// Produces returns the data keys this plugin produces.
func (p *Plugin) Produces() map[string]any {
	return map[string]any{TokenizedPromptKey: fwkrh.TokenizedPrompt{}}
}

// Produce tokenizes the request prompt and stores the result on
// InferenceRequestBody.TokenizedPrompt (TokenIDs + MultiModalFeatures in flat shape).
// Returns an error when tokenization fails; the caller (Director) decides the
// policy (currently: log and continue). If the request already carries a
// TokenizedPrompt, tokenization is skipped.
func (p *Plugin) Produce(ctx context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	tp, err := p.tokenize(ctx, request)
	if err != nil {
		return err
	}

	if tp == nil || len(tp.TokenIDs) == 0 {
		return nil
	}
	request.Body.TokenizedPrompt = tp
	return nil
}

// tokenize extracts token IDs and optional multimodal features from the request.
// Returns the existing TokenizedPrompt unchanged if one is already set.
// Returns a non-nil error if the request body is nil, has an unsupported type,
// or if the tokenizer fails.
func (p *Plugin) tokenize(ctx context.Context, request *scheduling.InferenceRequest) (*fwkrh.TokenizedPrompt, error) {
	logger := log.FromContext(ctx).WithName(p.typedName.String())
	traceLogger := logger.V(logging.TRACE)

	if request == nil || request.Body == nil {
		return nil, errors.New("request body is nil")
	}

	if request.Body.TokenizedPrompt != nil {
		traceLogger.Info("TokenizedPrompt already present, skipping")
		return request.Body.TokenizedPrompt, nil
	}

	traceLogger.Info("Request body present",
		"hasCompletions", request.Body.Completions != nil,
		"hasChatCompletions", request.Body.ChatCompletions != nil)

	var allTokenIDs [][]uint32
	var mmFeatures *tokenization.MultiModalFeatures
	var err error

	switch {
	case request.Body.Completions != nil:
		prompt := request.Body.Completions.Prompt
		if len(prompt.Strings) > 0 {
			allTokenIDs = make([][]uint32, 0, len(prompt.Strings))
			for i, promptStr := range prompt.Strings {
				traceLogger.Info("Calling Render for completions string", "promptIndex", i, "promptLength", len(promptStr))
				sTokenIDs, _, renderErr := p.tokenizer.Render(ctx, promptStr)
				if renderErr != nil {
					logger.Error(renderErr, "String tokenization failed, skipping tokenized prompt",
						"promptIndex", i, "promptLength", len(promptStr))
					return nil, fmt.Errorf("tokenization failed: %w", renderErr)
				}
				allTokenIDs = append(allTokenIDs, sTokenIDs)
			}
		} else {
			traceLogger.Info("Calling Render for completions", "prompt", prompt)
			var tokenIDs []uint32
			tokenIDs, _, err = p.tokenizer.Render(ctx, prompt.PlainText())
			if err == nil {
				allTokenIDs = [][]uint32{tokenIDs}
			}
		}
	case request.Body.ChatCompletions != nil:
		renderReq := ChatCompletionsToRenderChatRequest(request.Body.ChatCompletions)
		traceLogger.Info("Calling RenderChat for chat completions", "messageCount", len(request.Body.ChatCompletions.Messages))
		var tokenIDs []uint32
		tokenIDs, mmFeatures, err = p.tokenizer.RenderChat(ctx, renderReq)
		if err == nil {
			allTokenIDs = [][]uint32{tokenIDs}
		}
	default:
		return nil, errors.New("unsupported request body type, skipping tokenization")
	}

	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}

	if !hasAnyTokenIDs(allTokenIDs) {
		return nil, nil
	}

	totalTokens := 0
	for _, ids := range allTokenIDs {
		totalTokens += len(ids)
	}
	traceLogger.Info("Tokenization succeeded", "tokenCount", totalTokens, "promptCount", len(allTokenIDs))

	tp := &fwkrh.TokenizedPrompt{
		TokenIDs:           slices.Concat(allTokenIDs...),
		MultiModalFeatures: convertMMFeaturesToUpstream(mmFeatures),
	}
	if len(allTokenIDs) > 1 {
		tp.PerPromptTokens = allTokenIDs
	}
	return tp, nil
}

func hasAnyTokenIDs(tokenGroups [][]uint32) bool {
	for _, tokens := range tokenGroups {
		if len(tokens) > 0 {
			return true
		}
	}
	return false
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
