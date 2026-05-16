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

package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	grpcutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/protobuf/proto"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

const (
	VertexAIParserType = "vertexai-parser"

	chatCompletionsMethod     = "PredictionService/ChatCompletions"
	openAIChatCompletionsPath = "/chat/completions"
)

// compile-time type validation
var _ fwkrh.Parser = &VertexAIParser{}

// VertexAIParser implements the fwkrh.Parser interface for Vertex AI gRPC API
type VertexAIParser struct {
	typedName    fwkplugin.TypedName
	openAIParser *openai.OpenAIParser
}

// NewVertexAIParser creates a new VertexAIParser.
func NewVertexAIParser() *VertexAIParser {
	return &VertexAIParser{
		typedName: fwkplugin.TypedName{
			Type: VertexAIParserType,
			Name: VertexAIParserType,
		},
		openAIParser: openai.NewOpenAIParser(),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *VertexAIParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *VertexAIParser) SupportedAppProtocols() []v1.AppProtocol {
	return []v1.AppProtocol{v1.AppProtocolH2C}
}

func VertexAIParserPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewVertexAIParser().WithName(name), nil
}

func (p *VertexAIParser) WithName(name string) *VertexAIParser {
	p.typedName.Name = name
	return p
}

// ParseRequest parses the gRPC request body and headers and returns an InferenceRequestBody.
// It handles Vertex AI ChatCompletions requests by unmarshaling the gRPC payload into a
// ChatCompletionsRequest protobuf message. This message embeds an HttpBody containing the
// actual request payload as an OpenAI-compatible JSON string. The parser extracts this JSON
// data and delegates the parsing to the OpenAI parser.
// See: https://github.com/googleapis/googleapis/blob/89c3153888201c9e80bc5ec78d6ffca0debe6b52/google/cloud/aiplatform/v1beta1/prediction_service.proto#L235
func (p *VertexAIParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := headers[parsers.MethodPathKey]

	switch {
	case strings.HasSuffix(path, chatCompletionsMethod):
		parsedPayload, err := grpcutil.ParseGrpcPayload(body)
		if err != nil {
			return nil, fmt.Errorf("invalid or unsupported gRPC payload: %w", err)
		}

		req := &aiplatformpb.ChatCompletionsRequest{}
		if err := proto.Unmarshal(parsedPayload, req); err != nil {
			return nil, fmt.Errorf("unmarshaling ChatCompletionsRequest: %w", err)
		}

		httpBody := req.GetHttpBody()
		if httpBody == nil {
			return nil, errors.New("ChatCompletionsRequest has no HttpBody")
		}
		jsonBytes := httpBody.GetData()

		// Use OpenAI parser to parse the JSON payload
		// Clone headers and set path to /chat/completions to make OpenAI parser recognize it
		headersCopy := maps.Clone(headers)
		headersCopy[parsers.MethodPathKey] = openAIChatCompletionsPath
		parseResult, err := p.openAIParser.ParseRequest(ctx, jsonBytes, headersCopy)
		if err != nil {
			return nil, fmt.Errorf("parsing ChatCompletionsRequest: %w", err)
		}

		inferenceRequestBody := parseResult.Body
		inferenceRequestBody.Payload = fwkrh.PayloadProto{Message: req}
		return &fwkrh.ParseResult{Body: inferenceRequestBody, Skip: parseResult.Skip}, nil

	default:
		return &fwkrh.ParseResult{Skip: true}, nil
	}
}

// ParseResponse parses the response body and returns a ParsedResponse
func (p *VertexAIParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		// Nothing to parse
		return nil, nil //nolint:nilnil
	}

	parsedPayload, err := grpcutil.ParseGrpcPayload(body)
	if err != nil {
		return nil, fmt.Errorf("parsing gRPC payload: %w", err)
	}

	respMsg := &httpbody.HttpBody{}
	if err := proto.Unmarshal(parsedPayload, respMsg); err != nil {
		return nil, fmt.Errorf("unmarshaling HttpBody response: %w", err)
	}
	jsonBytes := respMsg.GetData()

	return p.openAIParser.ParseResponse(ctx, jsonBytes, headers, false)
}
