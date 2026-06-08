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

	// chatCompletionsMethod is the gRPC method path suffix for Vertex AI's OpenAI-compatible
	// ChatCompletions service (maps to aiplatformpb.ChatCompletionsRequest).
	// See: https://github.com/googleapis/googleapis/blob/89c3153888201c9e80bc5ec78d6ffca0debe6b52/google/cloud/aiplatform/v1beta1/prediction_service.proto#L234 for definition.
	chatCompletionsMethod = "PredictionService/ChatCompletions"
	// streamRawPredictServiceMethod is the gRPC method path suffix for Vertex AI's flexible,
	// low-level raw prediction streaming service (maps to aiplatformpb.StreamRawPredictRequest).
	// See: https://github.com/googleapis/googleapis/blob/89c3153888201c9e80bc5ec78d6ffca0debe6b52/google/cloud/aiplatform/v1beta1/prediction_service.proto#L84 for definition.
	streamRawPredictServiceMethod = "PredictionService/StreamRawPredict"
	// rawPredictServiceMethod is the gRPC method path suffix for Vertex AI's flexible,
	// low-level raw prediction service (maps to aiplatformpb.RawPredictRequest).
	// See: https://github.com/googleapis/googleapis/blob/89c3153888201c9e80bc5ec78d6ffca0debe6b52/google/cloud/aiplatform/v1beta1/prediction_service.proto#L71 for definition.
	rawPredictServiceMethod = "PredictionService/RawPredict"
	// openAIChatCompletionsPath is the standard OpenAI endpoint path for Chat Completions,
	// used to route extracted JSON payloads to the OpenAI parser.
	openAIChatCompletionsPath = "/chat/completions"
	// openAIResponsesPath is the OpenAI-compatible path for raw responses, used to route
	// extracted StreamRawPredict JSON payloads to the OpenAI parser.
	openAIResponsesPath = "/responses"
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

func (p *VertexAIParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths: []string{
			chatCompletionsMethod,
			streamRawPredictServiceMethod,
			rawPredictServiceMethod,
		},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C},
	}
}

func VertexAIParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
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
func (p *VertexAIParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := headers[parsers.MethodPathKey]

	switch {
	case strings.HasSuffix(path, chatCompletionsMethod):
		return p.parseVertexRequest(ctx, body, headers, &aiplatformpb.ChatCompletionsRequest{}, "ChatCompletionsRequest", openAIChatCompletionsPath)

	case strings.HasSuffix(path, streamRawPredictServiceMethod):
		return p.parseVertexRequest(ctx, body, headers, &aiplatformpb.StreamRawPredictRequest{}, "StreamRawPredictRequest", openAIResponsesPath)

	case strings.HasSuffix(path, rawPredictServiceMethod):
		return p.parseVertexRequest(ctx, body, headers, &aiplatformpb.RawPredictRequest{}, "RawPredictRequest", openAIResponsesPath)

	default:
		return &fwkrh.ParseResult{
			Body: &fwkrh.InferenceRequestBody{
				Payload: fwkrh.RawPayload(body),
			},
			SkipResponseProcessing: true,
		}, nil
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
		return nil, fmt.Errorf("parsing gRPC response payload: %w", err)
	}

	respMsg := &httpbody.HttpBody{}
	if err := proto.Unmarshal(parsedPayload, respMsg); err != nil {
		return nil, fmt.Errorf("unmarshaling HttpBody response: %w", err)
	}
	jsonBytes := respMsg.GetData()

	return p.openAIParser.ParseResponse(ctx, jsonBytes, headers, false)
}

type httpBodyMessage interface {
	proto.Message
	GetHttpBody() *httpbody.HttpBody
}

// parseVertexRequest is a generic helper to parse Vertex AI gRPC requests that wrap an HttpBody payload.
func (p *VertexAIParser) parseVertexRequest(ctx context.Context, body []byte, headers map[string]string, req httpBodyMessage, typeName string, targetPath string) (*fwkrh.ParseResult, error) {
	parsedPayload, err := grpcutil.ParseGrpcPayload(body)
	if err != nil {
		return nil, fmt.Errorf("invalid or unsupported gRPC payload: %w", err)
	}

	if err := proto.Unmarshal(parsedPayload, req); err != nil {
		return nil, fmt.Errorf("unmarshaling %s: %w", typeName, err)
	}

	httpBody := req.GetHttpBody()
	if httpBody == nil {
		return nil, fmt.Errorf("%s has no HttpBody", typeName)
	}
	jsonBytes := httpBody.GetData()

	// Use OpenAI parser to parse the JSON payload
	// Clone headers and set path to targetPath to make OpenAI parser recognize it
	headersCopy := maps.Clone(headers)
	headersCopy[parsers.MethodPathKey] = targetPath
	parseResult, err := p.openAIParser.ParseRequest(ctx, jsonBytes, headersCopy)
	if err != nil {
		return nil, fmt.Errorf("parsing %s: %w", typeName, err)
	}

	inferenceRequestBody := parseResult.Body
	inferenceRequestBody.Payload = fwkrh.PayloadProto{Message: req}
	return &fwkrh.ParseResult{Body: inferenceRequestBody, SkipResponseProcessing: parseResult.SkipResponseProcessing}, nil
}
