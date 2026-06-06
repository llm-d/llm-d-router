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

// Package vllmhttp provides a request parser for vLLM HTTP endpoints that are
// not part of the OpenAI-compatible API surface — currently just
// /inference/v1/generate (the disaggregated Prefill/Decode API). All other
// paths are delegated to the embedded OpenAI parser, so a single
// vllmhttp-parser plugin instance covers both vLLM-specific and OpenAI-
// compatible HTTP traffic served by the same endpoint.
package vllmhttp

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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

const (
	// VllmHTTPParserType is the canonical type name used to register the plugin.
	VllmHTTPParserType = "vllmhttp-parser"

	// generatePathSuffix is the vLLM disaggregated Prefill/Decode API path.
	generatePathSuffix = "inference/v1/generate"
)

// compile-time type validation
var _ fwkrh.Parser = &VllmHTTPParser{}

// VllmHTTPParser implements fwkrh.Parser for vLLM HTTP endpoints. It handles
// /inference/v1/generate locally and delegates all other paths to an embedded
// OpenAI parser so that the same plugin can serve mixed traffic.
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
func VllmHTTPParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewVllmHTTPParser().WithName(name), nil
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

// ParseRequest handles /inference/v1/generate locally and delegates everything
// else to the embedded OpenAI parser.
func (p *VllmHTTPParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	if strings.HasSuffix(request.GetRequestPath(headers), generatePathSuffix) {
		return p.parseGenerateRequest(body)
	}
	return nil, fmt.Errorf("unsupported path: %s", request.GetRequestPath(headers))
}

// ParseResponse delegates to the OpenAI parser. /inference/v1/generate
// responses share the OpenAI usage shape, so the same extractor works.
func (p *VllmHTTPParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, isStreaming bool) (*fwkrh.ParsedResponse, error) {
	return p.openai.ParseResponse(ctx, body, headers, isStreaming)
}

// parseGenerateRequest decodes a /inference/v1/generate body into an
// InferenceRequestBody. Token IDs are required; everything else is optional.
func (p *VllmHTTPParser) parseGenerateRequest(rawBody []byte) (*fwkrh.ParseResult, error) {
	bodyMap := make(map[string]any)
	if err := json.Unmarshal(rawBody, &bodyMap); err != nil {
		return nil, err
	}

	var generate fwkrh.GenerateRequest
	if err := json.Unmarshal(rawBody, &generate); err != nil {
		return nil, fmt.Errorf("invalid generate request: %w", err)
	}
	if len(generate.TokenIDs) == 0 {
		return nil, errors.New("invalid generate request: must have non-empty token_ids field")
	}

	body := &fwkrh.InferenceRequestBody{
		Generate: &generate,
		Payload:  fwkrh.PayloadMap(bodyMap),
	}
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		body.Stream = true
	}
	return &fwkrh.ParseResult{Body: body, SkipResponseProcessing: false}, nil
}
