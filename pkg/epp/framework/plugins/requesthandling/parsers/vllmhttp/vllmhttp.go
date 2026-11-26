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

// Package vllmhttp provides a request parser for vLLM HTTP endpoints that are
// not part of the OpenAI-compatible API surface — specifically
// /inference/v1/generate.
package vllmhttp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"k8s.io/utils/ptr"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	// VllmHTTPParserType is the canonical type name used to register the plugin.
	VllmHTTPParserType = "vllmhttp-parser"

	// generatePathSuffix is the vLLM custom generate API path.
	generatePathSuffix = "inference/v1/generate"
)

// compile-time type validation
var (
	_ fwkrh.Parser            = &VllmHTTPParser{}
	_ fwkrh.ModelNameRewriter = &VllmHTTPParser{}
	_ fwkrh.PriorityRewriter  = &VllmHTTPParser{}
)

// VllmHTTPParser implements fwkrh.Parser for vLLM HTTP endpoints. It handles
// /inference/v1/generate and delegates response parsing to an embedded
// OpenAI parser.
type VllmHTTPParser struct {
	typedName fwkplugin.TypedName
	openai    *openai.OpenAIParser
}

// NewVllmHTTPParser creates a new VllmHTTPParser.
func NewVllmHTTPParser() *VllmHTTPParser {
	return &VllmHTTPParser{
		typedName: fwkplugin.TypedName{
			Type: VllmHTTPParserType,
			Name: VllmHTTPParserType,
		},
		openai: openai.NewOpenAIParser(),
	}
}

// VllmHTTPParserPluginFactory is the factory function used to register the plugin.
func VllmHTTPParserPluginFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	plugin, err := openai.OpenAIParserPluginFactory(name, parameters, nil)
	if err != nil {
		return nil, err
	}
	openAIParser, ok := plugin.(*openai.OpenAIParser)
	if !ok {
		return nil, fmt.Errorf("openai parser factory returned %T, want *openai.OpenAIParser", plugin)
	}
	parser := NewVllmHTTPParser().WithName(name)
	parser.openai = openAIParser
	return parser, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *VllmHTTPParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// WithName sets the plugin instance name.
func (p *VllmHTTPParser) WithName(name string) *VllmHTTPParser {
	p.typedName.Name = name
	return p
}

func (p *VllmHTTPParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths:     []string{generatePathSuffix},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
}

// ParseRequest handles /inference/v1/generate with an optional trailing slash
// and rejects other paths.
func (p *VllmHTTPParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	if request.MatchPathSuffix(request.GetRequestPath(headers), generatePathSuffix) {
		return p.parseGenerateRequest(body)
	}
	return nil, fmt.Errorf("unsupported path: %s", request.GetRequestPath(headers))
}

// ParseResponse delegates to the OpenAI parser. /inference/v1/generate
// responses share the OpenAI usage shape, so the same extractor works.
func (p *VllmHTTPParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, isStreaming bool) (*fwkrh.ParsedResponse, error) {
	return p.openai.ParseResponse(ctx, body, headers, isStreaming)
}

// RewriteModelName delegates to the OpenAI parser; the generate body shares the payload map format.
func (p *VllmHTTPParser) RewriteModelName(payload request.MarshalablePayload, model string) (request.MarshalablePayload, error) {
	return p.openai.RewriteModelName(payload, model)
}

// RewritePriority delegates to the OpenAI-compatible map rewriter; the generate
// body shares the same top-level priority field.
func (p *VllmHTTPParser) RewritePriority(ctx fwkrh.PriorityRewriteContext, payload request.MarshalablePayload, priority int) (request.MarshalablePayload, bool, error) {
	return p.openai.RewritePriority(ctx, payload, priority)
}

// parseGenerateRequest decodes a /inference/v1/generate body into an
// InferenceRequestBody. Token IDs are required; everything else is optional.
func (p *VllmHTTPParser) parseGenerateRequest(rawBody []byte) (*fwkrh.ParseResult, error) {
	bodyMap, err := parserutil.UnmarshalMapWithRawField(rawBody, "token_ids")
	if err != nil {
		return nil, err
	}

	var generate request.GenerateRequest
	if err := json.Unmarshal(rawBody, &generate); err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}
	if len(generate.TokenIDs) == 0 {
		return nil, errors.New("invalid generate request: must have non-empty token_ids field")
	}

	body := &request.InferenceRequestBody{
		Generate: &generate,
		Payload:  request.PayloadMap(bodyMap),
	}
	if model, ok := bodyMap["model"].(string); ok {
		body.Model = model
	}
	// max_tokens lives under sampling_params in the generate wire format, which
	// OutputTokenLimit resolves for the generate API.
	if n, ok := reqcommon.OutputTokenLimit(bodyMap, reqcommon.APITypeVLLMGenerate); ok {
		body.MaxOutputTokens = ptr.To(int64(n))
	}
	stream, err := reqcommon.StreamRequested(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("invalid generate stream field: %w", err)
	}
	body.Stream = stream
	return &fwkrh.ParseResult{Body: body, SkipResponseProcessing: false}, nil
}
