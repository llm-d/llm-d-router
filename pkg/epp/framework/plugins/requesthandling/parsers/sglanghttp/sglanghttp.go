/*
Copyright 2026 The Kubernetes Authors.

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

// Package sglanghttp provides a request parser for SGLang HTTP endpoints that are
// not part of the OpenAI-compatible API surface — specifically
// /generate.
package sglanghttp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	// SGLangHTTPParserType is the canonical type name used to register the plugin.
	SGLangHTTPParserType = "sglanghttp-parser"

	// generatePathSuffix is the SGLang native generate API path.
	generatePathSuffix = "generate"

	streamingRespPrefix = "data: "
	streamingDoneMarker = "[DONE]"
	contentTypeHeader   = "content-type"
	eventStreamType     = "text/event-stream"
)

// compile-time type validation
var (
	_ fwkrh.Parser = &SGLangHTTPParser{}
)

// SGLangHTTPParser implements fwkrh.Parser for SGLang's native /generate
// endpoint. Only pre-tokenized prompts are supported.
type SGLangHTTPParser struct {
	typedName fwkplugin.TypedName
}

// NewSGLangHTTPParser creates a new SGLangHTTPParser.
func NewSGLangHTTPParser() *SGLangHTTPParser {
	return &SGLangHTTPParser{
		typedName: fwkplugin.TypedName{
			Type: SGLangHTTPParserType,
			Name: SGLangHTTPParserType,
		},
	}
}

// SGLangHTTPParserPluginFactory is the factory function used to register the plugin.
func SGLangHTTPParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewSGLangHTTPParser().WithName(name), nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *SGLangHTTPParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// WithName sets the plugin instance name.
func (p *SGLangHTTPParser) WithName(name string) *SGLangHTTPParser {
	p.typedName.Name = name
	return p
}

func (p *SGLangHTTPParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths:     []string{generatePathSuffix},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
}

// sgLangGenerateWire is the subset of /generate fields this parser reads.
type sgLangGenerateWire struct {
	InputIDs       json.RawMessage `json:"input_ids"`
	ExtraKey       json.RawMessage `json:"extra_key"`
	ImageData      json.RawMessage `json:"image_data"`
	VideoData      json.RawMessage `json:"video_data"`
	AudioData      json.RawMessage `json:"audio_data"`
	SamplingParams json.RawMessage `json:"sampling_params"`
	Stream         bool            `json:"stream"`
}

type sgLangSamplingParams struct {
	MaxNewTokens *int64 `json:"max_new_tokens"`
}

// ParseRequest handles /generate and rejects other paths.
// It returns a ParseResult containing the decoded InferenceRequestBody.
func (p *SGLangHTTPParser) ParseRequest(_ context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := strings.TrimSuffix(strings.TrimSpace(request.GetRequestPath(headers)), "/")
	if path != generatePathSuffix && path != "/"+generatePathSuffix {
		return nil, fmt.Errorf("unsupported path: %s", path)
	}
	return p.parseGenerateRequest(body)
}

// parseGenerateRequest decodes a /generate body into an InferenceRequestBody.
// input_ids are required. Multimodal inputs are not supported.
func (p *SGLangHTTPParser) parseGenerateRequest(rawBody []byte) (*fwkrh.ParseResult, error) {
	var wire sgLangGenerateWire
	if err := json.Unmarshal(rawBody, &wire); err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}

	if hasMultimodalData(wire.ImageData) || hasMultimodalData(wire.VideoData) || hasMultimodalData(wire.AudioData) {
		return nil, errors.New("unsupported generate request: multimodal inputs are not supported; only pre-tokenized prompts are supported")
	}
	if !hasJSONValue(wire.InputIDs) {
		return nil, errors.New("invalid generate request: input_ids must be provided")
	}

	cacheSalt, err := parseCacheSalt(wire.ExtraKey)
	if err != nil {
		return nil, fmt.Errorf("unsupported generate request: %w", err)
	}
	batches, err := parseInputIDs(wire.InputIDs)
	if err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}
	// Single prompt: TokenIDs for the producer to copy. Batch: TokenizedPrompt,
	// because TokenIDs cannot hold [][]uint32 (producer skips if already set).
	result := &fwkrh.InferenceRequestBody{
		Generate:        &fwkrh.GenerateRequest{CacheSalt: cacheSalt},
		Payload:         fwkrh.RawPayload(rawBody),
		MaxOutputTokens: maxOutputTokens(wire.SamplingParams),
		Stream:          wire.Stream,
	}
	if len(batches) == 1 {
		result.Generate.TokenIDs = batches[0]
	} else {
		result.TokenizedPrompt = &fwkrh.TokenizedPrompt{
			PerPromptTokens: batches,
			CacheSalt:       cacheSalt,
		}
	}

	return &fwkrh.ParseResult{Body: result, SkipResponseProcessing: false}, nil
}

func hasJSONValue(data json.RawMessage) bool {
	return len(data) > 0 && strings.TrimSpace(string(data)) != "null"
}

func hasMultimodalData(data json.RawMessage) bool {
	if !hasJSONValue(data) {
		return false
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return true
	}
	switch typed := value.(type) {
	case string:
		return typed != ""
	case []any:
		return len(typed) > 0
	default:
		return value != nil
	}
}

func parseCacheSalt(data json.RawMessage) (string, error) {
	if !hasJSONValue(data) {
		return "", nil
	}
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return "", errors.New("extra_key must be a string")
	}
	return value, nil
}

func parseInputIDs(data json.RawMessage) ([][]uint32, error) {
	var single []uint32
	if err := json.Unmarshal(data, &single); err == nil {
		if len(single) == 0 {
			return nil, errors.New("input_ids cannot be empty")
		}
		return [][]uint32{single}, nil
	}

	var batches [][]uint32
	if err := json.Unmarshal(data, &batches); err != nil {
		return nil, errors.New("input_ids must be an array of uint32 integers or arrays of uint32 integers")
	}
	if len(batches) == 0 {
		return nil, errors.New("input_ids cannot be empty")
	}
	for i, row := range batches {
		if len(row) == 0 {
			return nil, fmt.Errorf("input_ids[%d] must be a non-empty array of integers", i)
		}
	}
	return batches, nil
}

func maxOutputTokens(data json.RawMessage) *int64 {
	if !hasJSONValue(data) {
		return nil
	}

	var single sgLangSamplingParams
	if err := json.Unmarshal(data, &single); err != nil ||
		single.MaxNewTokens == nil || *single.MaxNewTokens < 0 {
		return nil
	}
	return single.MaxNewTokens
}

// sgLangResponse is the /generate response wire format (non-streaming).
// Usage lives under meta_info, not under usage like the OpenAI format.
type sgLangResponse struct {
	MetaInfo struct {
		PromptTokens     *int `json:"prompt_tokens"`
		CompletionTokens *int `json:"completion_tokens"`
		CachedTokens     *int `json:"cached_tokens"`
	} `json:"meta_info"`
}

// ParseResponse extracts token usage from a /generate response.
func (p *SGLangHTTPParser) ParseResponse(_ context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}
	if isEventStream(headers) {
		return &fwkrh.ParsedResponse{Usage: extractStreamingUsage(body)}, nil
	}
	usage, err := extractNonStreamingUsage(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.ParsedResponse{Usage: usage}, nil
}

func isEventStream(headers map[string]string) bool {
	for key, value := range headers {
		if strings.EqualFold(key, contentTypeHeader) &&
			strings.Contains(strings.ToLower(value), eventStreamType) {
			return true
		}
	}
	return false
}

// extractStreamingUsage reads usage from SGLang SSE data lines.
// Streaming SSE chunks carry cumulative meta_info; the last parseable chunk is used.
func extractStreamingUsage(body []byte) *fwkrh.Usage {
	text := strings.TrimSpace(string(body))
	var last *fwkrh.Usage
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		data, ok := strings.CutPrefix(line, streamingRespPrefix)
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == streamingDoneMarker {
			continue
		}
		if usage, err := parseMetaInfoUsage([]byte(data)); err == nil && usage != nil {
			last = usage
		}
	}
	return last
}

// extractNonStreamingUsage reads usage from a single SGLang response object or
// sums usage across a batch response array.
func extractNonStreamingUsage(body []byte) (*fwkrh.Usage, error) {
	text := strings.TrimSpace(string(body))
	if strings.HasPrefix(text, "[") {
		return extractBatchUsage(body)
	}
	return parseMetaInfoUsage(body)
}

func extractBatchUsage(body []byte) (*fwkrh.Usage, error) {
	var responses []json.RawMessage
	if err := json.Unmarshal(body, &responses); err != nil {
		return nil, err
	}
	total := &fwkrh.Usage{}
	found := false
	for _, response := range responses {
		usage, err := parseMetaInfoUsage(response)
		if err != nil {
			return nil, err
		}
		if usage == nil {
			continue
		}
		found = true
		addUsage(total, usage)
	}
	if !found {
		return nil, nil //nolint:nilnil
	}
	return total, nil
}

func parseMetaInfoUsage(data []byte) (*fwkrh.Usage, error) {
	var resp sgLangResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	if resp.MetaInfo.PromptTokens == nil && resp.MetaInfo.CompletionTokens == nil {
		return nil, nil //nolint:nilnil
	}
	u := &fwkrh.Usage{
		PromptTokens:     intValue(resp.MetaInfo.PromptTokens),
		CompletionTokens: intValue(resp.MetaInfo.CompletionTokens),
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	if resp.MetaInfo.CachedTokens != nil {
		u.PromptTokenDetails = &fwkrh.PromptTokenDetails{
			CachedTokens: *resp.MetaInfo.CachedTokens,
		}
	}
	return u, nil
}

func intValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

// addUsage accumulates per-prompt usage into a request-level total.
func addUsage(total, usage *fwkrh.Usage) {
	total.PromptTokens += usage.PromptTokens
	total.CompletionTokens += usage.CompletionTokens
	total.TotalTokens += usage.TotalTokens
	if usage.PromptTokenDetails != nil {
		if total.PromptTokenDetails == nil {
			total.PromptTokenDetails = &fwkrh.PromptTokenDetails{}
		}
		total.PromptTokenDetails.CachedTokens += usage.PromptTokenDetails.CachedTokens
	}
}
