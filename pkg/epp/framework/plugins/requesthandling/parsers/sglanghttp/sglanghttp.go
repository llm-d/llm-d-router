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

// Package sglanghttp provides a request parser for SGLang HTTP endpoints that are
// not part of the OpenAI-compatible API surface — specifically
// /generate.
package sglanghttp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	// SGLangHTTPParserType is the canonical type name used to register the plugin.
	SGLangHTTPParserType = "sglanghttp-parser"

	// generatePathSuffix is the SGLang native generate API path.
	generatePathSuffix = "generate"
)

// compile-time type validation
var (
	_ fwkrh.Parser            = &SGLangHTTPParser{}
	_ fwkrh.ModelNameRewriter = &SGLangHTTPParser{}
	_ fwkrh.PriorityRewriter  = &SGLangHTTPParser{}
)

// SGLangHTTPParser implements fwkrh.Parser for SGLang's native /generate
// endpoint. Pre-tokenized prompts and raw text prompts are supported.
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
	CacheSalt      json.RawMessage `json:"cache_salt"`
	Text           json.RawMessage `json:"text"`
	ExtraKey       json.RawMessage `json:"extra_key"`
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
// input_ids or text are required. Multimodal inputs are not supported.
func (p *SGLangHTTPParser) parseGenerateRequest(rawBody []byte) (*fwkrh.ParseResult, error) {
	var wire sgLangGenerateWire
	if err := json.Unmarshal(rawBody, &wire); err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}

	hasInputIDs := hasJSONValue(wire.InputIDs)
	hasText := hasJSONValue(wire.Text)
	if !hasInputIDs && !hasText {
		return nil, errors.New("invalid generate request: input_ids or text must be provided")
	}

	cacheSalt, err := parseCacheSalt("cache_salt", wire.CacheSalt)
	if err != nil {
		return nil, fmt.Errorf("unsupported generate request: %w", err)
	}
	extraKey, err := parseCacheSalt("extra_key", wire.ExtraKey)
	if err != nil {
		return nil, fmt.Errorf("unsupported generate request: %w", err)
	}
	cacheSalt, err = parsers.ComposeSGLangCacheSalt(extraKey, cacheSalt)
	if err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}

	var tokenIDs []uint32
	var text string
	if hasInputIDs {
		tokenIDs, err = parseInputIDs(wire.InputIDs)
		if err != nil {
			return nil, fmt.Errorf("invalid generate request: %w", err)
		}
	} else {
		text, err = parseText(wire.Text)
		if err != nil {
			return nil, fmt.Errorf("invalid generate request: %w", err)
		}
	}

	// Keep the full body as a map so priority can be injected; when nothing is
	// mutated, repackage still forwards the original bytes unchanged. input_ids
	// stays raw because decoding it into []any allocates per token.
	bodyMap, err := parserutil.UnmarshalMapWithRawField(rawBody, "input_ids")
	if err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}

	body := &request.InferenceRequestBody{
		Generate: &request.GenerateRequest{
			TokenIDs:  tokenIDs,
			Text:      text,
			CacheSalt: cacheSalt,
		},
		Payload:         request.PayloadMap(bodyMap),
		MaxOutputTokens: maxOutputTokens(wire.SamplingParams),
		Stream:          wire.Stream,
	}
	if model, ok := bodyMap["model"].(string); ok {
		body.Model = model
	}
	return &fwkrh.ParseResult{Body: body, SkipResponseProcessing: false}, nil
}

// RewriteModelName writes the resolved model into the request payload map.
func (p *SGLangHTTPParser) RewriteModelName(payload request.MarshalablePayload, model string) (request.MarshalablePayload, error) {
	m, ok := payload.(request.PayloadMap)
	if !ok {
		return payload, nil
	}
	m["model"] = model
	return m, nil
}

// RewritePriority removes any client-supplied priority from the SGLang /generate
// payload and writes the resolved EPP priority. The director only calls this when
// priority propagation is enabled; see parsers.RewritePriority for the
// cross-backend priority semantics.
func (p *SGLangHTTPParser) RewritePriority(ctx fwkrh.PriorityRewriteContext, payload request.MarshalablePayload, priority int) (request.MarshalablePayload, bool, error) {
	return parsers.RewritePriority(ctx, payload, priority)
}

func hasJSONValue(data json.RawMessage) bool {
	return len(data) > 0 && strings.TrimSpace(string(data)) != "null"
}

func parseCacheSalt(field string, data json.RawMessage) (string, error) {
	if !hasJSONValue(data) {
		return "", nil
	}
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return "", fmt.Errorf("%s must be a string", field)
	}
	return value, nil
}

func parseText(data json.RawMessage) (string, error) {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return "", errors.New("text must be a string")
	}
	if len(value) == 0 {
		return "", errors.New("text cannot be empty")
	}
	return value, nil
}

func parseInputIDs(data json.RawMessage) ([]uint32, error) {
	if ids, ok := request.ParseCanonicalTokenIDs(data); ok {
		return ids, nil
	}

	var ids []uint32
	if err := json.Unmarshal(data, &ids); err != nil {
		return nil, errors.New("input_ids must be an array of uint32 integers")
	}
	if len(ids) == 0 {
		return nil, errors.New("input_ids cannot be empty")
	}
	return ids, nil
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

// sgLangResponse is the /generate response wire format.
// Usage lives under meta_info, not under usage like the OpenAI format.
type sgLangResponse struct {
	MetaInfo struct {
		// FinishReason is parsed to identify the last meta_info of a finished stream;
		// null on intermediate chunks, non-null on the final chunk
		FinishReason     json.RawMessage `json:"finish_reason"`
		PromptTokens     *int            `json:"prompt_tokens"`
		CompletionTokens *int            `json:"completion_tokens"`
		CachedTokens     *int            `json:"cached_tokens"`
	} `json:"meta_info"`
}

// ParseResponse extracts token usage from a /generate response.
func (p *SGLangHTTPParser) ParseResponse(_ context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}
	if isEventStream(headers) {
		return parseStreamingResponse(body), nil
	}
	usage, stopReason, err := parseGenerateResponse(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.ParsedResponse{Usage: usage, StopReason: stopReason}, nil
}

// parseGenerateResponse reads usage and finish_reason in one decode. With
// sampling_params.n above 1 the response is an array: samples share one prompt,
// so completion tokens are summed and the first recognized finish reason wins.
func parseGenerateResponse(body []byte) (*request.Usage, fwkrh.StopReason, error) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var samples []json.RawMessage
		if err := json.Unmarshal(body, &samples); err != nil {
			return nil, "", err
		}
		var usage *request.Usage
		stopReason := fwkrh.StopReason("")
		sawUnknown := false
		for _, sample := range samples {
			var resp sgLangResponse
			if err := json.Unmarshal(sample, &resp); err != nil {
				return nil, "", err
			}
			if u := metaInfoUsage(resp); u != nil {
				if usage == nil {
					usage = u
				} else {
					usage.CompletionTokens += u.CompletionTokens
				}
			}
			switch reason := sglangStopReason(resp.MetaInfo.FinishReason); reason {
			case "":
			case fwkrh.StopReasonUnknown:
				sawUnknown = true
			default:
				if stopReason == "" {
					stopReason = reason
				}
			}
		}
		if stopReason == "" && sawUnknown {
			stopReason = fwkrh.StopReasonUnknown
		}
		if usage != nil {
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		}
		return usage, stopReason, nil
	}

	var resp sgLangResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, "", err
	}
	return metaInfoUsage(resp), sglangStopReason(resp.MetaInfo.FinishReason), nil
}

func isEventStream(headers map[string]string) bool {
	for key, value := range headers {
		if strings.EqualFold(key, reqcommon.HeaderContentType) &&
			strings.Contains(strings.ToLower(value), reqcommon.MediaTypeEventStream) {
			return true
		}
	}
	return false
}

// parseStreamingResponse reads the terminal chunk's usage and finish reason in
// one pass. Only the final chunk carries a non-null finish_reason.
func parseStreamingResponse(body []byte) *fwkrh.ParsedResponse {
	for line := range bytes.SplitSeq(body, []byte("\n")) {
		data, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte(reqcommon.SSEDataPrefix))
		if !ok {
			continue
		}
		data = bytes.TrimSpace(data)
		if bytes.Equal(data, []byte(reqcommon.SSEDoneMarker)) {
			continue
		}
		var resp sgLangResponse
		if err := json.Unmarshal(data, &resp); err != nil {
			continue
		}
		if !hasJSONValue(resp.MetaInfo.FinishReason) {
			continue
		}
		return &fwkrh.ParsedResponse{
			Usage:      metaInfoUsage(resp),
			StopReason: sglangStopReason(resp.MetaInfo.FinishReason),
		}
	}
	return &fwkrh.ParsedResponse{}
}

func sglangStopReason(raw json.RawMessage) fwkrh.StopReason {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	if trimmed[0] == '"' {
		var reason string
		if err := json.Unmarshal(trimmed, &reason); err == nil {
			if reason == "" {
				return ""
			}
			return parsers.NormalizeStopReason(reason)
		}
		return fwkrh.StopReasonUnknown
	}
	var reason struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(trimmed, &reason); err != nil {
		return fwkrh.StopReasonUnknown
	}
	if reason.Type == "" {
		return ""
	}
	return parsers.NormalizeStopReason(reason.Type)
}

func metaInfoUsage(resp sgLangResponse) *request.Usage {
	if resp.MetaInfo.PromptTokens == nil && resp.MetaInfo.CompletionTokens == nil {
		return nil
	}
	u := &request.Usage{
		PromptTokens:     intValue(resp.MetaInfo.PromptTokens),
		CompletionTokens: intValue(resp.MetaInfo.CompletionTokens),
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	if resp.MetaInfo.CachedTokens != nil {
		u.PromptTokenDetails = &request.PromptTokenDetails{
			CachedTokens: *resp.MetaInfo.CachedTokens,
		}
	}
	return u
}

func intValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}
