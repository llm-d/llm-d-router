/*
Copyright 2025 The Kubernetes Authors.

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

package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requesthandling/parsers"
)

const (
	AnthropicParserType = "anthropic-parser"

	messagesAPI = "messages"

	streamingRespPrefix = "data: "

	contentType     = "content-type"
	eventStreamType = "text/event-stream"
)

// compile-time type validation
var _ fwkrh.Parser = &AnthropicParser{}

type AnthropicParser struct {
	typedName fwkplugin.TypedName
}

func NewAnthropicParser() *AnthropicParser {
	return &AnthropicParser{
		typedName: fwkplugin.TypedName{
			Type: AnthropicParserType,
			Name: AnthropicParserType,
		},
	}
}

func (p *AnthropicParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *AnthropicParser) SupportedAppProtocols() []v1.AppProtocol {
	return []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP}
}

func AnthropicParserPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewAnthropicParser().WithName(name), nil
}

func (p *AnthropicParser) WithName(name string) *AnthropicParser {
	p.typedName.Name = name
	return p
}

func (p *AnthropicParser) ParseRequest(_ context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	bodyMap := make(map[string]any)
	if err := json.Unmarshal(body, &bodyMap); err != nil {
		return nil, fmt.Errorf("error unmarshaling request body: %w", err)
	}

	path := getRequestPath(headers)
	if !strings.HasSuffix(path, "/"+messagesAPI) {
		return nil, fmt.Errorf("unsupported API endpoint: %s", path)
	}

	var messagesReq fwkrh.MessagesRequest
	if err := json.Unmarshal(body, &messagesReq); err != nil {
		return nil, fmt.Errorf("error parsing messages request: %w", err)
	}
	if len(messagesReq.Messages) == 0 {
		return nil, errors.New("invalid messages request: must have at least one message")
	}

	result := &fwkrh.InferenceRequestBody{
		Messages: &messagesReq,
		Payload:  fwkrh.PayloadMap(bodyMap),
	}
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		result.Stream = true
	}

	return &fwkrh.ParseResult{Body: result, Skip: false}, nil
}

func (p *AnthropicParser) ParseResponse(_ context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}

	isStream := false
	for k, v := range headers {
		if strings.ToLower(k) == contentType && strings.Contains(strings.ToLower(v), eventStreamType) {
			isStream = true
			break
		}
	}
	if isStream {
		return p.parseStreamResponse(body)
	}

	usage, err := extractUsage(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.ParsedResponse{Usage: usage}, nil
}

func getRequestPath(headers map[string]string) string {
	if path := headers[parsers.MethodPathKey]; path != "" {
		return path
	}
	if path := headers["x-original-path"]; path != "" {
		return path
	}
	if path := headers["x-forwarded-path"]; path != "" {
		return path
	}
	return ""
}

func extractUsage(responseBytes []byte) (*fwkrh.Usage, error) {
	var responseBody map[string]any
	if err := json.Unmarshal(responseBytes, &responseBody); err != nil {
		return nil, err
	}

	usg, ok := responseBody["usage"].(map[string]any)
	if !ok {
		return nil, nil //nolint:nilnil
	}

	usage := fwkrh.Usage{}
	if v, ok := usg["input_tokens"].(float64); ok {
		usage.PromptTokens = int(v)
	}
	if v, ok := usg["output_tokens"].(float64); ok {
		usage.CompletionTokens = int(v)
	}
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens

	if v, ok := usg["cache_read_input_tokens"].(float64); ok {
		usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{
			CachedTokens: int(v),
		}
	}

	return &usage, nil
}

// Anthropic SSE streaming format:
//
//	event: message_start
//	data: {"type":"message_start","message":{"usage":{"input_tokens":25},...}}
//
//	event: message_delta
//	data: {"type":"message_delta","delta":{...},"usage":{"output_tokens":15}}
//
//	event: message_stop
//	data: {"type":"message_stop"}
func (p *AnthropicParser) parseStreamResponse(chunk []byte) (*fwkrh.ParsedResponse, error) {
	usage := extractUsageStreaming(string(chunk))
	return &fwkrh.ParsedResponse{Usage: usage}, nil
}

func extractUsageStreaming(responseText string) *fwkrh.Usage {
	var result *fwkrh.Usage

	lines := strings.Split(responseText, "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, streamingRespPrefix) {
			continue
		}
		content := strings.TrimPrefix(line, streamingRespPrefix)

		var event struct {
			Type    string `json:"type"`
			Message struct {
				Usage map[string]any `json:"usage"`
			} `json:"message"`
			Usage map[string]any `json:"usage"`
		}
		if err := json.Unmarshal([]byte(content), &event); err != nil {
			continue
		}

		switch event.Type {
		case "message_start":
			if event.Message.Usage != nil {
				if result == nil {
					result = &fwkrh.Usage{}
				}
				if v, ok := event.Message.Usage["input_tokens"].(float64); ok {
					result.PromptTokens = int(v)
				}
				if v, ok := event.Message.Usage["cache_read_input_tokens"].(float64); ok {
					result.PromptTokenDetails = &fwkrh.PromptTokenDetails{
						CachedTokens: int(v),
					}
				}
			}
		case "message_delta":
			if event.Usage != nil {
				if result == nil {
					result = &fwkrh.Usage{}
				}
				if v, ok := event.Usage["output_tokens"].(float64); ok {
					result.CompletionTokens = int(v)
				}
			}
		}
	}

	if result != nil {
		result.TotalTokens = result.PromptTokens + result.CompletionTokens
	}

	return result
}
