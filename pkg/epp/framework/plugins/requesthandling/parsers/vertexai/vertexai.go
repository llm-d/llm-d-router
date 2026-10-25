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

package vertexai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
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
	// openAICompletionsPath is the OpenAI endpoint path for text completions, used
	// when a REST payload carries a "prompt" field.
	openAICompletionsPath = "/completions"

	// The REST path suffixes below are what a Vertex AI dedicated endpoint sends
	// when it forwards traffic over HTTP rather than gRPC, e.g.
	// /v1/projects/P/locations/L/endpoints/E:predict. The gRPC method suffixes above
	// never appear on those requests, so each REST form needs its own claim.
	// ":streamRawPredict" is not matched by a ":rawPredict" claim, since its trailing
	// characters are "mRawPredict".
	predictPathSuffix          = ":predict"
	rawPredictPathSuffix       = ":rawPredict"
	streamRawPredictPathSuffix = ":streamRawPredict"
	chatCompletionsPathSuffix  = "/chat/completions"
	// invokePathClaim uses the registry's segment-wildcard form: Vertex invoke
	// routing appends the container's own path after /invoke/, so the request has
	// no fixed suffix to match on.
	invokePathClaim = "invoke/*"
	// invokePathSegment is how ParseRequest recognizes the paths invokePathClaim
	// matches.
	invokePathSegment = "/invoke/"

	// grpcContentTypePrefix covers application/grpc and its +proto/+json variants.
	grpcContentTypePrefix = "application/grpc"
	// sseDataPrefix opens an SSE event; the space after the colon is optional.
	sseDataPrefix = "data:"
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
			// gRPC service methods.
			chatCompletionsMethod,
			streamRawPredictServiceMethod,
			rawPredictServiceMethod,
			// REST paths, as sent by Vertex AI dedicated endpoints.
			predictPathSuffix,
			rawPredictPathSuffix,
			streamRawPredictPathSuffix,
			chatCompletionsPathSuffix,
			invokePathClaim,
		},
		// h2c serves the gRPC methods and http the REST paths. Without http, EPP
		// refuses to serve a pool whose backends speak HTTP/1.1.
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
}

func VertexAIParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewVertexAIParser().WithName(name), nil
}

func (p *VertexAIParser) WithName(name string) *VertexAIParser {
	p.typedName.Name = name
	return p
}

// ParseRequest parses the request body and headers and returns an InferenceRequestBody.
//
// For the gRPC methods it unmarshals the gRPC payload into the matching Vertex AI request
// proto (e.g. ChatCompletionsRequest). That message embeds an HttpBody containing the actual
// request payload as an OpenAI-compatible JSON string, which the parser extracts and
// delegates to the OpenAI parser.
//
// For the REST paths a Vertex AI dedicated endpoint sends, the body is the payload itself;
// see parseVertexRESTRequest.
func (p *VertexAIParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	// The query string is not part of the method; e.g. ?alt=sse would otherwise defeat
	// every REST suffix below.
	path := request.GetRequestPath(headers)

	switch {
	case strings.HasSuffix(path, chatCompletionsMethod):
		return p.parseVertexRequest(ctx, body, headers, &aiplatformpb.ChatCompletionsRequest{}, "ChatCompletionsRequest", openAIChatCompletionsPath)

	case strings.HasSuffix(path, streamRawPredictServiceMethod):
		return p.parseVertexRequest(ctx, body, headers, &aiplatformpb.StreamRawPredictRequest{}, "StreamRawPredictRequest", openAIResponsesPath)

	case strings.HasSuffix(path, rawPredictServiceMethod):
		return p.parseVertexRequest(ctx, body, headers, &aiplatformpb.RawPredictRequest{}, "RawPredictRequest", openAIResponsesPath)

	case strings.HasSuffix(path, chatCompletionsPathSuffix):
		return p.parseVertexRESTRequest(ctx, body, headers, openAIChatCompletionsPath), nil

	case strings.HasSuffix(path, predictPathSuffix),
		strings.HasSuffix(path, rawPredictPathSuffix),
		strings.HasSuffix(path, streamRawPredictPathSuffix),
		strings.Contains(path, invokePathSegment):
		return p.parseVertexRESTRequest(ctx, body, headers, ""), nil

	default:
		return &fwkrh.ParseResult{
			Body: &fwkrh.InferenceRequestBody{
				Payload: fwkrh.RawPayload(body),
			},
			SkipResponseProcessing: true,
		}, nil
	}
}

// ParseResponse parses the response body and returns a ParsedResponse.
//
// A gRPC response is a gRPC-framed HttpBody wrapping the model server's JSON. A REST
// response is that JSON (or SSE stream) as is, so it goes straight to the OpenAI parser.
func (p *VertexAIParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endOfStream bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		// Nothing to parse
		return nil, nil //nolint:nilnil
	}

	if !isGRPCResponse(headers, body) {
		return p.openAIParser.ParseResponse(ctx, body, headers, endOfStream)
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

// parseVertexRESTRequest handles a Vertex AI REST request, as forwarded by a dedicated
// endpoint, where the body is the payload itself rather than a gRPC-framed Vertex proto.
// When targetPath is empty, the OpenAI route is chosen from the payload's shape.
//
// Parsing is best effort. A dedicated endpoint may front any container -- a TensorFlow
// model served {"instances": [...]} is not OpenAI-shaped -- so a payload the OpenAI parser
// cannot read is forwarded untouched, and still scheduled, instead of failing the request.
func (p *VertexAIParser) parseVertexRESTRequest(ctx context.Context, body []byte, headers map[string]string, targetPath string) *fwkrh.ParseResult {
	if targetPath == "" {
		targetPath = resolveOpenAIPath(body)
	}
	if targetPath == "" {
		return rawPassthrough(body)
	}

	headersCopy := maps.Clone(headers)
	if headersCopy == nil {
		headersCopy = make(map[string]string, 1)
	}
	headersCopy[parsers.MethodPathKey] = targetPath
	parseResult, err := p.openAIParser.ParseRequest(ctx, body, headersCopy)
	if err != nil || parseResult == nil || parseResult.Body == nil {
		return rawPassthrough(body)
	}
	return parseResult
}

// resolveOpenAIPath picks the OpenAI path matching the shape of a payload, or "" when it
// matches none: messages -> /chat/completions, prompt -> /completions, input -> /responses.
func resolveOpenAIPath(payload []byte) string {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(payload, &probe); err != nil {
		return ""
	}
	switch {
	case probe["messages"] != nil:
		return openAIChatCompletionsPath
	case probe["prompt"] != nil:
		return openAICompletionsPath
	case probe["input"] != nil:
		return openAIResponsesPath
	default:
		return ""
	}
}

// rawPassthrough returns the request unparsed. Scheduling still runs, but EPP does not
// interpret or rewrite the body and stops intercepting after the request phase.
func rawPassthrough(body []byte) *fwkrh.ParseResult {
	return &fwkrh.ParseResult{
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.RawPayload(body),
		},
		SkipResponseProcessing: true,
	}
}

// isGRPCResponse reports whether a response is gRPC-framed. The content type is
// authoritative when present; without one, a body that opens like JSON or an SSE event is
// REST, since a gRPC frame always opens with a 0x00 or 0x01 flag byte.
func isGRPCResponse(headers map[string]string, body []byte) bool {
	for key, value := range headers {
		if strings.EqualFold(key, reqcommon.HeaderContentType) {
			return strings.HasPrefix(strings.ToLower(strings.TrimSpace(value)), grpcContentTypePrefix)
		}
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) > 0 && (trimmed[0] == '{' || trimmed[0] == '[') {
		return false
	}
	return !bytes.HasPrefix(trimmed, []byte(sseDataPrefix))
}
