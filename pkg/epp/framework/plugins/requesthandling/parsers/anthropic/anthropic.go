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

	result := &request.InferenceRequestBody{
		Payload:         request.PayloadMap(bodyMap),
		RawBody:         body,
		MaxOutputTokens: request.MaxOutputTokensFromPayload(bodyMap, "max_tokens"),
	}
	if model, ok := bodyMap["model"].(string); ok {
		result.Model = model
	}

	// count_tokens delegates token counting to the server and passes its response
	// through, so only the envelope is read: Messages stays nil, while the model
	// still resolves and rewrites.
	if countTokens {
		return &fwkrh.ParseResult{Body: result, JSONValidated: true, SkipResponseProcessing: true}, nil
	}

	var messagesReq request.MessagesRequest
	if err := json.Unmarshal(body, &messagesReq); err != nil {
		var typeErr *json.UnmarshalTypeError
		if errors.As(err, &typeErr) && typeErr.Field != "" {
			err = &fwkrh.RequestFieldError{Field: typeErr.Field, Err: err}
		}
		return nil, fmt.Errorf("error parsing messages request: %w", err)
	}
	if len(messagesReq.Messages) == 0 {
		return nil, errors.New("invalid messages request: must have at least one message")
	}
	result.Messages = &messagesReq
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		result.Stream = true
	}

	return &fwkrh.ParseResult{Body: result, JSONValidated: true, SkipResponseProcessing: false}, nil
}

// RewriteModelName writes the resolved model into the request payload map.
func (p *AnthropicParser) RewriteModelName(payload request.MarshalablePayload, model string) (request.MarshalablePayload, error) {
	m, ok := payload.(request.PayloadMap)
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
	if isEventStream(headers) {
		return parseStreamResponse(body), nil
	}

	usage, err := extractUsage(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.ParsedResponse{Usage: usage}, nil
}

func isEventStream(headers map[string]string) bool {
	for k, v := range headers {
		if !strings.EqualFold(k, reqcommon.HeaderContentType) {
			continue
		}
		mediaType, _, _ := strings.Cut(v, ";")
		if strings.EqualFold(strings.TrimSpace(mediaType), reqcommon.MediaTypeEventStream) {
			return true
		}
	}
	return false
}

func extractUsage(responseBytes []byte) (*request.Usage, error) {
	var responseBody map[string]any
	if err := json.Unmarshal(responseBytes, &responseBody); err != nil {
		return nil, err
	}

	usg, ok := responseBody[reqcommon.FieldUsage].(map[string]any)
	if !ok {
		return nil, nil //nolint:nilnil
	}

	var tokens inputTokens
	tokens.add(usg)
	var usage request.Usage
	tokens.apply(&usage)
	usage.CompletionTokens = jsonInt(usg, reqcommon.FieldOutputTokens)
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens

	return &usage, nil
}

// inputTokens accumulates the three Messages API input counts across the events
// of one chunk. input_tokens counts only the tokens that were neither read from
// nor written to the prompt cache, so the prompt the server processed is
// input_tokens + cache_read_input_tokens + cache_creation_input_tokens.
//
// Each bucket is updated only by a positive value so a later event that omits
// or zeroes a bucket keeps the count an earlier event reported.
type inputTokens struct {
	input         int
	cacheRead     int
	cacheCreation int
	hasRead       bool
}

func (t *inputTokens) add(usg map[string]any) {
	if input := jsonInt(usg, reqcommon.FieldInputTokens); input > 0 {
		t.input = input
	}
	if read, ok := jsonIntPresent(usg, reqcommon.FieldCacheReadInputTokens); ok {
		t.hasRead = true
		if read > 0 {
			t.cacheRead = read
		}
	}
	if creation := jsonInt(usg, reqcommon.FieldCacheCreationInputTokens); creation > 0 {
		t.cacheCreation = creation
	}
}

func (t inputTokens) apply(usage *request.Usage) {
	if total := t.input + t.cacheRead + t.cacheCreation; total > 0 {
		usage.PromptTokens = total
	}
	// An explicit zero records a cache miss in PromptTokenDetails. The metrics
	// layer only records request_cached_tokens when PromptTokenDetails is set,
	// so dropping the zero leaves the misses out of the histogram and inflates
	// the reported hit rate.
	if t.hasRead {
		usage.PromptTokenDetails = &request.PromptTokenDetails{CachedTokens: t.cacheRead}
	}
}

func jsonInt(m map[string]any, key string) int {
	v, _ := m[key].(float64)
	return int(v)
}

// jsonIntPresent reports the integer stored under key and whether the key was
// present, so a caller can tell an explicit zero from a missing field.
func jsonIntPresent(m map[string]any, key string) (int, bool) {
	v, ok := m[key].(float64)
	return int(v), ok
}

// parseStreamResponse reads the usage of one stream chunk. message_start and
// message_delta often arrive in separate chunks, and message_delta may omit the
// cache buckets, so the chunk also reports its buckets in DisjointInputTokens
// for the caller to fold across chunks.
//
// Anthropic SSE streaming format:
//
//	event: message_start
//	data: {"type":"message_start","message":{"usage":{"input_tokens":25},...}}
//
//	event: message_delta
//	data: {"type":"message_delta","delta":{...},"usage":{"input_tokens":30,"output_tokens":15}}
//
//	event: message_stop
//	data: {"type":"message_stop"}
func parseStreamResponse(chunk []byte) *fwkrh.ParsedResponse {
	resp := &fwkrh.ParsedResponse{}
	var tokens inputTokens
	completion := 0
	hasUsage := false
	for line := range bytes.SplitSeq(chunk, []byte("\n")) {
		content, ok := bytes.CutPrefix(line, []byte(reqcommon.SSEDataPrefix))
		if !ok {
			continue
		}
		resp.StreamedEvents++
		// Safe because only message_start/message_delta carry usage, both with a literal "usage" key.
		if !bytes.Contains(content, []byte(reqcommon.FieldUsage)) {
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
		case reqcommon.MessageEventStart:
			if event.Message.Usage != nil {
				hasUsage = true
				tokens = inputTokens{}
				tokens.add(event.Message.Usage)
			}
		case reqcommon.MessageEventDelta:
			if event.Usage != nil {
				hasUsage = true
				tokens.add(event.Usage)
				completion = jsonInt(event.Usage, reqcommon.FieldOutputTokens)
			}
		}
	}
	if !hasUsage {
		return resp
	}

	usage := &request.Usage{CompletionTokens: completion}
	tokens.apply(usage)
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	resp.Usage = usage
	resp.DisjointInputTokens = &fwkrh.DisjointInputTokens{
		Uncached:      tokens.input,
		CacheRead:     tokens.cacheRead,
		CacheCreation: tokens.cacheCreation,
	}
	return resp
}
