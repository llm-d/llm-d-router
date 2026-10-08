/*
Copyright 2025 The llm-d Authors.

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
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	AnthropicParserType = "anthropic-parser"

	messagesAPI    = "messages"
	countTokensAPI = "messages/count_tokens"
)

// compile-time type validation
var (
	_ fwkrh.Parser            = &AnthropicParser{}
	_ fwkrh.ModelNameRewriter = &AnthropicParser{}
)

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

func (p *AnthropicParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths:     []string{messagesAPI, countTokensAPI, messagesAPI + "/render"},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
}

func AnthropicParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewAnthropicParser().WithName(name), nil
}

func (p *AnthropicParser) WithName(name string) *AnthropicParser {
	p.typedName.Name = name
	return p
}

func (p *AnthropicParser) ParseRequest(_ context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := request.GetRequestPath(headers)
	if request.MatchPathSuffix(path, messagesAPI+"/render") {
		return parserutil.ParseRenderRequest(body)
	}

	countTokens := request.MatchPathSuffix(path, countTokensAPI)
	if !countTokens && !request.MatchPathSuffix(path, messagesAPI) {
		return nil, fmt.Errorf("unsupported API endpoint: %s", path)
	}

	bodyMap, err := parserutil.UnmarshalEnvelope(body, "system")
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling request body: %w", err)
	}

	result := &fwkrh.InferenceRequestBody{
		Payload:         fwkrh.PayloadMap(bodyMap),
		RawBody:         body,
		MaxOutputTokens: fwkrh.MaxOutputTokensFromPayload(bodyMap, "max_tokens"),
	}
	if model, ok := bodyMap["model"].(string); ok {
		result.Model = model
	}

	// count_tokens delegates token counting to the server and passes its response
	// through, so only the envelope is read: Messages stays nil to keep the token
	// producers out, while the model still resolves and rewrites. The server
	// requires model, and a JSON null unmarshals into an envelope with no fields,
	// so the field is validated here as messages is on the path below.
	if countTokens {
		if result.Model == "" {
			return nil, errors.New("invalid count_tokens request: must have a model")
		}
		return &fwkrh.ParseResult{Body: result, SkipResponseProcessing: true}, nil
	}

	var messagesReq fwkrh.MessagesRequest
	if err := json.Unmarshal(body, &messagesReq); err != nil {
		return nil, fmt.Errorf("error parsing messages request: %w", err)
	}
	if len(messagesReq.Messages) == 0 {
		return nil, errors.New("invalid messages request: must have at least one message")
	}
	result.Messages = &messagesReq
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		result.Stream = true
	}

	return &fwkrh.ParseResult{Body: result, SkipResponseProcessing: false}, nil
}

// RewriteModelName writes the resolved model into the request payload map.
func (p *AnthropicParser) RewriteModelName(payload fwkrh.MarshalablePayload, model string) (fwkrh.MarshalablePayload, error) {
	m, ok := payload.(fwkrh.PayloadMap)
	if !ok {
		return payload, nil
	}
	m["model"] = model
	return m, nil
}

func (p *AnthropicParser) ParseResponse(_ context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}

	isStream := false
	for k, v := range headers {
		if strings.ToLower(k) == reqcommon.HeaderContentType && strings.Contains(strings.ToLower(v), request.MediaTypeEventStream) {
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
	applyInputTokens(&usage, usg)
	if v, ok := jsonInt(usg, "output_tokens"); ok {
		usage.CompletionTokens = v
	}
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens

	return &usage, nil
}

// applyInputTokens copies the input counts of an Anthropic usage block into usage.
// The Messages API reports input across three additive fields, where input_tokens
// counts only the tokens that were neither read from nor written to the prompt
// cache, so the input the server processed is their sum. CachedTokens keeps the
// cache_read_input_tokens subset that Usage documents. A block carrying none of
// the three leaves usage untouched, so a message_delta reporting only output
// tokens does not erase what message_start reported.
func applyInputTokens(usage *fwkrh.Usage, usg map[string]any) {
	input, inputOK := jsonInt(usg, "input_tokens")
	read, readOK := jsonInt(usg, "cache_read_input_tokens")
	creation, creationOK := jsonInt(usg, "cache_creation_input_tokens")
	if !inputOK && !readOK && !creationOK {
		return
	}
	usage.PromptTokens = input + read + creation
	if readOK {
		usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{CachedTokens: read}
	}
}

func jsonInt(m map[string]any, key string) (int, bool) {
	v, ok := m[key].(float64)
	return int(v), ok
}

// Anthropic SSE streaming format:
//
//	event: message_start
//	data: {"type":"message_start","message":{"usage":{"input_tokens":25},...}}
//
//	event: message_delta
//	data: {"type":"message_delta","delta":{...},"usage":{"input_tokens":25,"output_tokens":15}}
//
//	event: message_stop
//	data: {"type":"message_stop"}
func (p *AnthropicParser) parseStreamResponse(chunk []byte) (*fwkrh.ParsedResponse, error) {
	usage := extractUsageStreaming(chunk)
	return &fwkrh.ParsedResponse{Usage: usage}, nil
}

func extractUsageStreaming(responseBytes []byte) *fwkrh.Usage {
	var result *fwkrh.Usage

	lines := bytes.SplitSeq(responseBytes, []byte("\n"))
	for line := range lines {
		content, ok := bytes.CutPrefix(line, []byte(reqcommon.SSEDataPrefix))
		// Safe because only message_start/message_delta carry usage, both with a literal "usage" key.
		if !ok || !bytes.Contains(content, []byte("usage")) {
			continue
		}

		var event struct {
			Type    string `json:"type"`
			Message struct {
				Usage map[string]any `json:"usage"`
			} `json:"message"`
			Usage map[string]any `json:"usage"`
		}
		if err := json.Unmarshal(content, &event); err != nil {
			continue
		}

		switch event.Type {
		case "message_start":
			if event.Message.Usage != nil {
				if result == nil {
					result = &fwkrh.Usage{}
				}
				applyInputTokens(result, event.Message.Usage)
			}
		case "message_delta":
			if event.Usage != nil {
				if result == nil {
					result = &fwkrh.Usage{}
				}
				// The delta counts are cumulative and authoritative over message_start.
				applyInputTokens(result, event.Usage)
				if v, ok := jsonInt(event.Usage, "output_tokens"); ok {
					result.CompletionTokens = v
				}
			}
		}
	}

	if result != nil {
		result.TotalTokens = result.PromptTokens + result.CompletionTokens
	}

	return result
}
