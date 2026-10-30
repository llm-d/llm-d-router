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

// Package systemone parses the System One structured decisions API
// (POST /v1/systemone). A request carries a state and a set of questions; the
// engine renders one prompt per question and answers each with a single label
// read, so the response holds probabilities instead of generated text.
package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const SystemOneParserType = "systemone-parser"

// compile-time type validation
var (
	_ fwkrh.Parser            = &SystemOneParser{}
	_ fwkrh.ModelNameRewriter = &SystemOneParser{}
	_ fwkrh.PriorityRewriter  = &SystemOneParser{}
)

type SystemOneParser struct {
	typedName fwkplugin.TypedName
}

func NewSystemOneParser() *SystemOneParser {
	return &SystemOneParser{
		typedName: fwkplugin.TypedName{
			Type: SystemOneParserType,
			Name: SystemOneParserType,
		},
	}
}

func (p *SystemOneParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *SystemOneParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths:     []string{request.SystemOnePathSuffix},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
}

func SystemOneParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewSystemOneParser().WithName(name), nil
}

func (p *SystemOneParser) WithName(name string) *SystemOneParser {
	p.typedName.Name = name
	return p
}

// ParseRequest handles /v1/systemone and rejects other paths.
func (p *SystemOneParser) ParseRequest(_ context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := request.GetRequestPath(headers)
	if !request.MatchPathSuffix(path, request.SystemOnePathSuffix) {
		return nil, fmt.Errorf("unsupported API endpoint: %s", path)
	}

	bodyMap, err := parserutil.UnmarshalEnvelope(body, "state")
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling request body: %w", err)
	}
	questions, ok := bodyMap["questions"].(json.RawMessage)
	if !ok || len(questions) == 0 || questions[0] != '{' {
		return nil, errors.New("invalid systemone request: questions must be a JSON object")
	}

	result := &fwkrh.InferenceRequestBody{
		Payload: fwkrh.PayloadMap(bodyMap),
		RawBody: body,
		// The engine renders one prompt per question, which the router cannot reproduce.
		RenderRequest: true,
	}
	if model, ok := bodyMap["model"].(string); ok {
		result.Model = model
	}
	return &fwkrh.ParseResult{Body: result, JSONValidated: true, SkipResponseProcessing: false}, nil
}

// RewriteModelName writes the resolved model into the request payload map.
func (p *SystemOneParser) RewriteModelName(payload fwkrh.MarshalablePayload, model string) (fwkrh.MarshalablePayload, error) {
	m, ok := payload.(fwkrh.PayloadMap)
	if !ok {
		return payload, nil
	}
	m["model"] = model
	return m, nil
}

// RewritePriority delegates to parsers.RewritePriority; the request declares a top-level priority.
func (p *SystemOneParser) RewritePriority(ctx fwkrh.PriorityRewriteContext, payload fwkrh.MarshalablePayload, priority int) (fwkrh.MarshalablePayload, bool, error) {
	return parsers.RewritePriority(ctx, payload, priority)
}

// ParseResponse reads usage from the response. The API has no streaming form,
// so the body is the complete JSON response.
func (p *SystemOneParser) ParseResponse(_ context.Context, body []byte, _ map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}
	var response struct {
		Usage *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, err
	}
	if response.Usage == nil {
		return nil, nil //nolint:nilnil
	}
	return &fwkrh.ParsedResponse{Usage: &fwkrh.Usage{
		PromptTokens:     response.Usage.InputTokens,
		CompletionTokens: response.Usage.OutputTokens,
		TotalTokens:      response.Usage.InputTokens + response.Usage.OutputTokens,
	}}, nil
}
