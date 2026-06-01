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

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
)

const (
	OpenAIParserType = "openai-parser"

	conversationsAPI   = "conversations"
	responsesAPI       = "responses"
	chatCompletionsAPI = "chat/completions"
	completionsAPI     = "completions"
	embeddingsAPI      = "embeddings"

	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"

	contentType = "content-type"
	// The base media type for Server-Sent Events. We check for this substring
	// to account for optional parameters like "; charset=utf-8" often appended by proxies.
	eventStreamType = "text/event-stream"

	usageField               = "usage"
	promptTokensField        = "prompt_tokens"
	inputTokensField         = "input_tokens"
	completionTokensField    = "completion_tokens"
	outputTokensField        = "output_tokens"
	promptTokensDetailsField = "prompt_tokens_details"
	inputTokensDetailsField  = "input_tokens_details"
	cachedTokensField        = "cached_tokens"
	totalTokensField         = "total_tokens"
)

// compile-time type validation
var _ fwkrh.Parser = &OpenAIParser{}

// OpenAIParser implements the fwkrh.Parser interface for OpenAI API
// https://developers.openai.com/api/reference/overview
type OpenAIParser struct {
	typedName fwkplugin.TypedName
}

// NewOpenAIParser creates a new OpenAIParser.
func NewOpenAIParser() *OpenAIParser {
	return &OpenAIParser{
		typedName: fwkplugin.TypedName{
			Type: OpenAIParserType,
			Name: OpenAIParserType,
		},
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *OpenAIParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *OpenAIParser) SupportedAppProtocols() []v1.AppProtocol {
	return []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP}
}

func OpenAIParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewOpenAIParser().WithName(name), nil
}

func (p *OpenAIParser) WithName(name string) *OpenAIParser {
	p.typedName.Name = name
	return p
}

// ParseRequest parses the request body and headers and returns a map representation.
func (p *OpenAIParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	bodyMap := make(map[string]any)
	if err := json.Unmarshal(body, &bodyMap); err != nil {
		return nil, fmt.Errorf("error unmarshaling request bodyMap: %w", err)
	}
	extractedBody, err := extractRequestBody(body, headers)
	if err != nil {
		return nil, err
	}
	extractedBody.Payload = fwkrh.PayloadMap(bodyMap)
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		extractedBody.Stream = true
	}
	return &fwkrh.ParseResult{Body: extractedBody, Skip: false}, nil
}

// ParseResponse extracts usage metadata from the provider's response.
// It automatically detects and handles both standard JSON responses and SSE streams.
func (p *OpenAIParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		// An empty body can occur during streaming; for instance, Envoy proxies
		// may emit a trailing empty body with the EndOfStream flag set to true.
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

func (p *OpenAIParser) parseStreamResponse(chunk []byte) (*fwkrh.ParsedResponse, error) {
	usage := extractUsageStreaming(string(chunk))
	return &fwkrh.ParsedResponse{
		Usage: usage,
	}, nil
}

// GetRequestPath extracts the request path from headers with fallback priority
func GetRequestPath(headers map[string]string) string {
	// Try primary path header
	if path := headers[parsers.MethodPathKey]; path != "" {
		return path
	}

	// Try fallback headers
	if path := headers["x-original-path"]; path != "" {
		return path
	}

	if path := headers["x-forwarded-path"]; path != "" {
		return path
	}

	// Default to completions API for backward compatibility with existing clients and integration tests
	return "/v1/completions"
}

// determineAPITypeFromPath determines the API type based on the request path.
// Note: path strings have already been cleaned and normalized by the gateway/proxy layer
// (no trailing slashes, query parameters, or additional suffix strings at this point).
// The suffix-based matching supports both standard OpenAI paths (e.g. /v1/chat/completions)
// and provider-specific paths (e.g. Vertex AI's /v1/projects/.../chat/completions).
func determineAPITypeFromPath(path string) string {
	if strings.HasSuffix(path, "/conversations") {
		return conversationsAPI
	}
	if strings.HasSuffix(path, "/responses") {
		return responsesAPI
	}
	if strings.HasSuffix(path, "/chat/completions") {
		return chatCompletionsAPI
	}
	if strings.HasSuffix(path, "/completions") {
		return completionsAPI
	}
	if strings.HasSuffix(path, "/embeddings") {
		return embeddingsAPI
	}

	// Default to completions API for backward compatibility with existing clients and integration tests
	return completionsAPI
}

// extractRequestBody extracts the InferenceRequestBody from the given request body map using path-based detection.
func extractRequestBody(rawBody []byte, headers map[string]string) (*fwkrh.InferenceRequestBody, error) {
	// Determine API type from request path
	path := GetRequestPath(headers)
	apiType := determineAPITypeFromPath(path)

	switch apiType {
	case conversationsAPI:
		var conversations fwkrh.ConversationsRequest
		if err := json.Unmarshal(rawBody, &conversations); err == nil && len(conversations.Items) > 0 {
			return &fwkrh.InferenceRequestBody{Conversations: &conversations}, nil
		}
		return nil, errors.New("invalid conversations request: must have items field")

	case responsesAPI:
		var responses fwkrh.ResponsesRequest
		if err := json.Unmarshal(rawBody, &responses); err == nil && responses.Input != nil {
			return &fwkrh.InferenceRequestBody{Responses: &responses}, nil
		}
		return nil, errors.New("invalid responses request: must have input field")

	case chatCompletionsAPI:
		var chatCompletions fwkrh.ChatCompletionsRequest
		if err := json.Unmarshal(rawBody, &chatCompletions); err == nil {
			if err = validateChatCompletionsMessages(chatCompletions.Messages); err == nil {
				return &fwkrh.InferenceRequestBody{ChatCompletions: &chatCompletions}, nil
			}
		}
		return nil, errors.New("invalid chat completions request: must have valid messages field")

	case completionsAPI:
		var completions fwkrh.CompletionsRequest
		if err := json.Unmarshal(rawBody, &completions); err == nil && !completions.Prompt.IsEmpty() {
			return &fwkrh.InferenceRequestBody{Completions: &completions}, nil
		}
		return nil, errors.New("invalid completions request: must have prompt field")

	case embeddingsAPI:
		var embeddings fwkrh.EmbeddingsRequest
		if err := json.Unmarshal(rawBody, &embeddings); err == nil && !embeddings.Input.IsEmpty() {
			return &fwkrh.InferenceRequestBody{Embeddings: &embeddings}, nil
		}
		return nil, errors.New("invalid embeddings request: must have input field")
	default:
		return nil, errors.New("unsupported API endpoint")
	}
}

func validateChatCompletionsMessages(messages []fwkrh.Message) error {
	if len(messages) == 0 {
		return errors.New("chat-completions request must have at least one message")
	}
	return nil
}

// toInt coerces a JSON-decoded number-ish value into an int. JSON numbers
// land as float64 after json.Unmarshal into map[string]any; some
// non-conforming providers emit strings. Anything else is ignored so that
// usage extraction stays best-effort rather than panicking.
func toInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	case string:
		if f, err := strconv.ParseFloat(n, 64); err == nil {
			return int(f)
		}
	}
	return 0
}

func extractUsage(responseBytes []byte) (*fwkrh.Usage, error) {
	var responseErr error
	var responseBody map[string]any
	responseErr = json.Unmarshal(responseBytes, &responseBody)
	if responseErr != nil {
		return nil, responseErr
	}
	usg, ok := responseBody[usageField].(map[string]any)
	if !ok {
		return nil, nil //nolint:nilnil
	}

	usage := fwkrh.Usage{}

	// Chat/Completions APIs use prompt_tokens. Responses/Conversations APIs use input_tokens.
	for _, inputTokens := range []string{promptTokensField, inputTokensField} {
		if v, ok := usg[inputTokens]; ok && v != nil {
			usage.PromptTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use completion_tokens. Responses/Conversations APIs use output_tokens.
	for _, outputTokens := range []string{completionTokensField, outputTokensField} {
		if v, ok := usg[outputTokens]; ok && v != nil {
			usage.CompletionTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use prompt_tokens_details. Responses/Conversations APIs use input_tokens_details.
	for _, details := range []string{promptTokensDetailsField, inputTokensDetailsField} {
		if detailsMap, ok := usg[details].(map[string]any); ok {
			if cachedTokens, ok := detailsMap[cachedTokensField]; ok {
				usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{
					CachedTokens: toInt(cachedTokens),
				}
			}
		}
	}

	// total_tokens field name is consistent across all API types.
	if v, ok := usg[totalTokensField]; ok && v != nil {
		usage.TotalTokens = toInt(v)
	}

	return &usage, nil
}

// Example message if "stream_options": {"include_usage": "true"} is included in the request:
// data: {"id":"...","object":"text_completion","created":1739400043,"model":"small-segment-lora-0","choices":[],
// "usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
//
// data: [DONE]
//
// Noticed that vLLM returns two entries in one response.
// We need to strip the `data:` prefix and next Data: [DONE] from the message to fetch response data.
//
// If include_usage is not included in the request, `data: [DONE]` is returned separately, which
// indicates end of streaming.
//
// For ResponsesAPI streaming, usage is nested in the response object:
//
//	event: response.completed
//	data: {"response":{"usage":{"input_tokens":31,..},...},"type":"response.completed"}
//
// It extracts usage from events with type="response.completed".
func extractUsageStreaming(responseText string) *fwkrh.Usage {

	var streamResponse struct {
		Usage    *fwkrh.Usage `json:"usage"`
		Response struct {
			Usage  map[string]any `json:"usage"`
			Object string         `json:"object"`
		} `json:"response"`
		Type string `json:"type"`
	}

	lines := strings.SplitSeq(responseText, "\n")
	for line := range lines {
		if !strings.HasPrefix(line, streamingRespPrefix) {
			continue
		}
		content := strings.TrimPrefix(line, streamingRespPrefix)
		if content == "[DONE]" {
			continue
		}

		byteSlice := []byte(content)
		if err := json.Unmarshal(byteSlice, &streamResponse); err != nil {
			continue
		}
		// Standard ChatCompletion / vLLM usage format
		if streamResponse.Usage != nil {
			return streamResponse.Usage
		}
		// Responses API streaming format
		if streamResponse.Response.Usage != nil && streamResponse.Type == "response.completed" {
			// Convert map[string]any to JSON and parse
			jsonBytes, _ := json.Marshal(map[string]any{
				"usage":  streamResponse.Response.Usage,
				"object": streamResponse.Response.Object,
			})
			if usage, err := extractUsage(jsonBytes); err == nil && usage != nil {
				return usage
			}
		}
	}
	return nil
}
