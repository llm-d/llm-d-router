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

	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	OpenAIParserType = "openai-parser"

	conversationsAPI   = "conversations"
	responsesAPI       = "responses"
	chatCompletionsAPI = "chat/completions"
	completionsAPI     = "completions"
	embeddingsAPI      = "embeddings"
	// imagesGenerationsAPI is the OpenAI-compatible image generation endpoint/
	imagesGenerationsAPI = "images/generations"

	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"

	contentType = "content-type"
	// The base media type for Server-Sent Events. We check for this substring
	// to account for optional parameters like "; charset=utf-8" often appended by proxies.
	eventStreamType = "text/event-stream"

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
var (
	_ fwkrh.Parser            = &OpenAIParser{}
	_ fwkrh.ModelNameRewriter = &OpenAIParser{}
)

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

func (p *OpenAIParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths: []string{
			chatCompletionsAPI,
			completionsAPI,
			embeddingsAPI,
			responsesAPI,
			conversationsAPI,
			chatCompletionsAPI + "/render",
			completionsAPI + "/render",
			imagesGenerationsAPI,
		},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
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
	apiType := determineAPITypeFromPath(request.GetRequestPath(headers))
	extractedBody, err := extractRequestBody(apiType, body)
	if err != nil {
		return nil, err
	}
	extractedBody.Payload = fwkrh.PayloadMap(bodyMap)
	if model, ok := bodyMap["model"].(string); ok {
		extractedBody.Model = model
	}
	extractedBody.MaxOutputTokens = maxOutputTokensForAPI(apiType, bodyMap)
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		extractedBody.Stream = true
	}
	return &fwkrh.ParseResult{Body: extractedBody, SkipResponseProcessing: false}, nil
}

// RewriteModelName writes the resolved model into the request payload map.
func (p *OpenAIParser) RewriteModelName(payload fwkrh.MarshalablePayload, model string) (fwkrh.MarshalablePayload, error) {
	m, ok := payload.(fwkrh.PayloadMap)
	if !ok {
		return payload, nil
	}
	m["model"] = model
	return m, nil
}

// maxOutputTokensForAPI normalizes the per-API output-token cap field into a
// single value, applying each API's field name and precedence. Endpoints with no
// output-token concept (conversations, embeddings) return nil.
func maxOutputTokensForAPI(apiType string, bodyMap map[string]any) *int64 {
	switch apiType {
	case chatCompletionsAPI:
		return fwkrh.MaxOutputTokensFromPayload(bodyMap, "max_completion_tokens", "max_tokens")
	case completionsAPI:
		return fwkrh.MaxOutputTokensFromPayload(bodyMap, "max_tokens")
	case responsesAPI:
		return fwkrh.MaxOutputTokensFromPayload(bodyMap, "max_output_tokens")
	default:
		return nil
	}
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

// determineAPITypeFromPath determines the API type based on the request path.
// The suffix-based matching supports both standard OpenAI paths (e.g. /v1/chat/completions)
// and provider-specific paths (e.g. Vertex AI's /v1/projects/.../chat/completions).
// Sub-paths /render under chat-completions and completions share the parent's body schema.
func determineAPITypeFromPath(path string) string {
	if request.MatchPathSuffix(path, "/conversations") {
		return conversationsAPI
	}
	if request.MatchPathSuffix(path, "/responses") {
		return responsesAPI
	}
	if request.MatchPathSuffix(path, "/chat/completions") ||
		request.MatchPathSuffix(path, "/chat/completions/render") {
		return chatCompletionsAPI
	}
	if request.MatchPathSuffix(path, "/completions") ||
		request.MatchPathSuffix(path, "/completions/render") {
		return completionsAPI
	}
	if request.MatchPathSuffix(path, "/embeddings") {
		return embeddingsAPI
	}
	if request.MatchPathSuffix(path, "/images/generations") {
		return imagesGenerationsAPI
	}

	// Default to completions API for backward compatibility with existing clients and integration tests
	return completionsAPI
}

// extractRequestBody extracts the InferenceRequestBody from the given raw body
// for the already-resolved API type.
func extractRequestBody(apiType string, rawBody []byte) (*fwkrh.InferenceRequestBody, error) {
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

	case imagesGenerationsAPI:
		var images fwkrh.ImagesGenerationsRequest
		if err := json.Unmarshal(rawBody, &images); err == nil && images.Prompt != "" {
			return &fwkrh.InferenceRequestBody{Images: &images}, nil
		}
		return nil, errors.New("invalid images generations request: must have prompt field")
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
	var responseBody struct {
		Usage map[string]any `json:"usage"`
	}
	err := json.Unmarshal(responseBytes, &responseBody)
	if err != nil {
		return nil, err
	}
	if responseBody.Usage == nil {
		return nil, nil //nolint:nilnil
	}

	usage := fwkrh.Usage{}

	// Chat/Completions APIs use prompt_tokens. Responses/Conversations APIs use input_tokens.
	for _, inputTokens := range []string{promptTokensField, inputTokensField} {
		if v, ok := responseBody.Usage[inputTokens]; ok && v != nil {
			usage.PromptTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use completion_tokens. Responses/Conversations APIs use output_tokens.
	for _, outputTokens := range []string{completionTokensField, outputTokensField} {
		if v, ok := responseBody.Usage[outputTokens]; ok && v != nil {
			usage.CompletionTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use prompt_tokens_details. Responses/Conversations APIs use input_tokens_details.
	for _, details := range []string{promptTokensDetailsField, inputTokensDetailsField} {
		if detailsMap, ok := responseBody.Usage[details].(map[string]any); ok {
			if cachedTokens, ok := detailsMap[cachedTokensField]; ok {
				usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{
					CachedTokens: toInt(cachedTokens),
				}
			}
		}
	}

	// total_tokens field name is consistent across all API types.
	if v, ok := responseBody.Usage[totalTokensField]; ok && v != nil {
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
			Usage json.RawMessage `json:"usage"` // Delay JSON decoding until we know we have usage data
		} `json:"response"`
		Type string `json:"type"`
	}

	lines := strings.SplitSeq(responseText, "\n")
	for line := range lines {
		content, ok := strings.CutPrefix(line, streamingRespPrefix)
		if !ok {
			continue
		}
		// When the stream is terminated with [DONE] or there's not any usage data, skip the line
		if content == "[DONE]" || !strings.Contains(content, "usage") {
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
		if len(streamResponse.Response.Usage) > 0 && streamResponse.Type == "response.completed" {
			jsonBytes, _ := json.Marshal(map[string]any{
				"usage": streamResponse.Response.Usage,
			})
			if usage, err := extractUsage(jsonBytes); err == nil && usage != nil {
				return usage
			}
		}
	}
	return nil
}
