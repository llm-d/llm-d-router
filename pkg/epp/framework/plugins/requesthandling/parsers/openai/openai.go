/*
Copyright 2025 The Kubernetes Authors.
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

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"mime/multipart"
	"reflect"
	"strconv"
	"strings"

	"github.com/cespare/xxhash/v2"
	"k8s.io/utils/ptr"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	OpenAIParserType = "openai-parser"

	conversationsAPI   = "conversations"
	responsesAPI       = "responses"
	chatCompletionsAPI = "chat/completions"
	completionsAPI     = "completions"

	formFieldNumInferenceSteps = "num_inference_steps"
	embeddingsAPI              = "embeddings"
	// imagesGenerationsAPI is the OpenAI-compatible image generation endpoint/
	imagesGenerationsAPI = "images/generations"
	// imagesEditsAPI is the OpenAI-compatible image edit (image-to-image) endpoint.
	// Requests are multipart/form-data.
	imagesEditsAPI = "images/edits"
	// videosAPI is the OpenAI-compatible vLLM-Omni asynchronous video generation
	// endpoint. Requests are multipart/form-data. Status and content requests
	// (GET /v1/videos/{id} and /v1/videos/{id}/content) have no body, so they are
	// routed to a random endpoint. With more than one replica, a poll can reach a
	// pod that does not hold the job, so the pool needs a single replica or
	// session affinity.
	videosAPI = "videos"
	// videosSyncAPI is the vLLM-Omni synchronous video generation endpoint, which
	// takes the same form as videosAPI and returns raw video bytes.
	videosSyncAPI  = "videos/sync"
	audioSpeechAPI = "audio/speech"
	// audioGenerateAPI is the vLLM Omni text-to-audio generation endpoint (e.g.
	// Stable Audio), distinct from audioSpeechAPI's text-to-speech.
	audioGenerateAPI = "audio/generate"
	// audioTranscriptionsAPI and audioTranslationsAPI are the OpenAI-compatible
	// speech-to-text endpoints. Both take multipart/form-data with the same
	// scalar fields, so one parser serves both.
	audioTranscriptionsAPI = "audio/transcriptions"
	audioTranslationsAPI   = "audio/translations"

	octetStreamType = "application/octet-stream"

	textCompletionObject = "text_completion"
	finishReasonStop     = "stop"
	finishReasonLength   = "length"
	// maxCompletionResponseBytes bounds the non-streamed completions body whose text is
	// returned in ParsedResponse.Completion. Larger bodies are not decoded for text.
	maxCompletionResponseBytes = 1 << 20

	// Text to speech api response format:
	// https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/#response-format
	vllmOmniInputTokensHeader  = "x-vllm-omni-input-tokens"  //#nosec G101 -- HTTP header name, not a credential
	vllmOmniOutputTokensHeader = "x-vllm-omni-output-tokens" //#nosec G101 -- HTTP header name, not a credential
	vllmOmniTotalTokensHeader  = "x-vllm-omni-total-tokens"  //#nosec G101 -- HTTP header name, not a credential
)

// compile-time type validation
var (
	_ fwkrh.Parser            = &OpenAIParser{}
	_ fwkrh.ModelNameRewriter = &OpenAIParser{}
	_ fwkrh.PriorityRewriter  = &OpenAIParser{}
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
			imagesEditsAPI,
			videosAPI,
			videosSyncAPI,
			audioSpeechAPI,
			audioTranscriptionsAPI,
			audioTranslationsAPI,
			request.DecisionsPathSuffix,
			audioGenerateAPI,
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
	path := request.GetRequestPath(headers)
	if request.MatchPathSuffix(path, chatCompletionsAPI+"/render") || request.MatchPathSuffix(path, completionsAPI+"/render") {
		return parserutil.ParseRenderRequest(body)
	}
	apiType := determineAPITypeFromPath(path)
	if apiType == imagesEditsAPI {
		return parseImagesEditsRequest(body, headers)
	}
	if apiType == videosAPI || apiType == videosSyncAPI {
		return parseVideosRequest(body, headers)
	}
	if apiType == audioTranscriptionsAPI || apiType == audioTranslationsAPI {
		return parseTranscriptionsRequest(apiType, body, headers)
	}
	if apiType == request.DecisionsPathSuffix {
		return parseDecisionsRequest(body)
	}
	extractedBody, err := extractRequestBody(apiType, body)
	if err != nil {
		return nil, fmt.Errorf("error extracting request body: %w", err)
	}

	var bodyMap request.PayloadMap
	if rawFields, ok := envelopeRawFields[apiType]; ok {
		var payload map[string]any
		payload, err = parserutil.UnmarshalEnvelope(body, rawFields...)
		bodyMap = request.PayloadMap(payload)
		extractedBody.RawBody = body
	} else if rawField := tokenInputField(extractedBody); rawField == "" {
		bodyMap = make(request.PayloadMap)
		err = parserutil.Unmarshal(body, &bodyMap)
	} else {
		var payload map[string]any
		payload, err = parserutil.UnmarshalMapWithRawField(body, rawField)
		bodyMap = request.PayloadMap(payload)
	}
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling request bodyMap: %w", err)
	}

	extractedBody.Payload = bodyMap
	if model, ok := bodyMap[reqcommon.FieldModel].(string); ok {
		extractedBody.Model = model
	}
	extractedBody.MaxOutputTokens = maxOutputTokensForAPI(apiType, bodyMap)
	extractedBody.Stream = isStreamingRequest(apiType, bodyMap)
	return &fwkrh.ParseResult{Body: extractedBody, JSONValidated: true, SkipResponseProcessing: false}, nil
}

// envelopeRawFields lists the APIs whose payload map holds only the routing
// envelope, and the string fields each keeps opaque. Prompt content is decoded
// once into the typed body; the original bytes are forwarded unless mutated.
var envelopeRawFields = map[string][]string{
	chatCompletionsAPI: {reqcommon.FieldPrompt},
	completionsAPI:     {reqcommon.FieldPrompt},
	responsesAPI:       {reqcommon.FieldInput, reqcommon.FieldInstructions},
	// input and instructions are decoded once into the typed body; ref_audio_2
	// can be a large inline data URI, and the object and array conditioning
	// fields stay raw on their own shape.
	audioSpeechAPI: {reqcommon.FieldInput, reqcommon.FieldInstructions, "ref_audio", "ref_audio_2"},
}

func isStreamingRequest(apiType string, bodyMap map[string]any) bool {
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		return true
	}
	if apiType != audioSpeechAPI {
		return false
	}
	streamFormat, _ := bodyMap["stream_format"].(string)
	return streamFormat == "sse" || streamFormat == "audio"
}

func tokenInputField(body *request.InferenceRequestBody) string {
	switch {
	case body.Embeddings != nil && len(body.Embeddings.Input.TokenIDs) > 0:
		return "input"
	default:
		return ""
	}
}

// RewriteModelName writes the resolved model into the request payload map.
func (p *OpenAIParser) RewriteModelName(payload request.MarshalablePayload, model string) (request.MarshalablePayload, error) {
	m, ok := payload.(request.PayloadMap)
	if !ok {
		return payload, nil
	}
	m[reqcommon.FieldModel] = model
	return m, nil
}

// RewritePriority removes any client-supplied priority from the
// OpenAI-compatible request payload and writes the resolved EPP priority. The
// director only calls this when priority propagation is enabled; see
// parsers.RewritePriority for the cross-backend priority semantics.
// Decisions requests are left unchanged: the Decisions schema declares no priority and rejects unknown fields.
func (p *OpenAIParser) RewritePriority(ctx fwkrh.PriorityRewriteContext, payload request.MarshalablePayload, priority int) (request.MarshalablePayload, bool, error) {
	if determineAPITypeFromPath(ctx.RequestPath) == request.DecisionsPathSuffix {
		return payload, false, nil
	}
	return parsers.RewritePriority(ctx, payload, priority)
}

// maxOutputTokensForAPI reads the output-token cap through the shared per-API
// field table. Endpoints with no output-token concept (conversations,
// embeddings) return nil.
func maxOutputTokensForAPI(apiType string, bodyMap map[string]any) *int64 {
	var limitAPI reqcommon.APIType
	switch apiType {
	case chatCompletionsAPI:
		limitAPI = reqcommon.APITypeChatCompletions
	case completionsAPI:
		limitAPI = reqcommon.APITypeCompletions
	case responsesAPI:
		limitAPI = reqcommon.APITypeResponses
	case audioSpeechAPI:
		limitAPI = reqcommon.APITypeAudioSpeech
	default:
		return nil
	}
	if n, ok := reqcommon.OutputTokenLimit(bodyMap, limitAPI); ok {
		return ptr.To(int64(n))
	}
	return nil
}

// ParseResponse extracts usage metadata from JSON, SSE, and binary audio and video responses.
func (p *OpenAIParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endOfStream bool) (*fwkrh.ParsedResponse, error) {
	mediaType := responseMediaType(headers)
	if strings.HasPrefix(mediaType, "audio/") || strings.HasPrefix(mediaType, "video/") || mediaType == octetStreamType {
		if !endOfStream {
			return &fwkrh.ParsedResponse{}, nil
		}
		return &fwkrh.ParsedResponse{Usage: extractUsageHeaders(headers)}, nil
	}
	if len(body) == 0 {
		// An empty body can occur during streaming; for instance, Envoy proxies
		// may emit a trailing empty body with the EndOfStream flag set to true.
		return nil, nil //nolint:nilnil
	}
	if mediaType == reqcommon.MediaTypeEventStream {
		return p.parseStreamResponse(body)
	}

	// The transcription response formats text, srt and vtt carry no usage,
	// and parsing them as JSON would only log an error per response. srt
	// arrives as application/x-subrip and vtt as text/vtt.
	if mediaType == "text/plain" || mediaType == "text/vtt" || mediaType == "application/x-subrip" {
		return nil, nil //nolint:nilnil
	}

	return extractResponse(body)
}

// responseChoice holds the fields of one Chat Completions or Completions
// choice that response parsing reads.
type responseChoice struct {
	Text         *string `json:"text"`
	FinishReason string  `json:"finish_reason"`
}

// responseOutputItem holds the type of one Responses API output item.
type responseOutputItem struct {
	Type string `json:"type"`
}

type responseIncompleteDetails struct {
	Reason string `json:"reason"`
}

// completionChoices holds the fields of a completions response that
// ParsedResponse.Completion is built from.
type completionChoices struct {
	Object  string           `json:"object"`
	Choices []responseChoice `json:"choices"`
}

// stopReasonFromChoices aggregates response choices into one stop reason. A
// tool call in any choice wins; otherwise a normal termination in any choice
// yields turn_ended; otherwise unknown. Responses with mixed choices therefore
// report the signal that matters most for join detection.
func stopReasonFromChoices(choices []responseChoice) fwkrh.StopReason {
	result := fwkrh.StopReason("")
	seen := false
	for _, choice := range choices {
		reason := strings.TrimSpace(choice.FinishReason)
		if reason == "" {
			continue
		}
		seen = true
		switch parsers.NormalizeStopReason(reason) {
		case fwkrh.StopReasonToolCall:
			return fwkrh.StopReasonToolCall
		case fwkrh.StopReasonTurnEnded:
			result = fwkrh.StopReasonTurnEnded
		default:
			result = fwkrh.StopReasonUnknown
		}
	}
	if !seen {
		return ""
	}
	return result
}

// responsesStopReason maps a Responses API status and output list into the
// shared stop-reason vocabulary. A function-call output item means the turn
// stopped to call a tool.
func responsesStopReason(status string, output []responseOutputItem, _ responseIncompleteDetails) fwkrh.StopReason {
	status = strings.ToLower(strings.TrimSpace(status))
	switch status {
	case "failed", "cancelled", "incomplete":
		return fwkrh.StopReasonUnknown
	}
	for _, item := range output {
		if item.Type == "function_call" {
			return fwkrh.StopReasonToolCall
		}
	}
	if status == "" {
		return ""
	}
	return parsers.NormalizeStopReason(status)
}

// responsesEventStopReason maps a terminal Responses API stream event. Some
// providers omit response.status, so the event type is the fallback.
func responsesEventStopReason(eventType, status string, output []responseOutputItem, incomplete responseIncompleteDetails) fwkrh.StopReason {
	if eventType == "response.failed" {
		return fwkrh.StopReasonUnknown
	}
	if reason := responsesStopReason(status, output, incomplete); reason != "" {
		return reason
	}
	switch eventType {
	case "response.completed":
		return fwkrh.StopReasonTurnEnded
	case "response.incomplete":
		return fwkrh.StopReasonUnknown
	case "response.failed":
		return fwkrh.StopReasonUnknown
	default:
		return ""
	}
}

// completion returns the text of a completions response with exactly one choice
// that finished with "stop" or "length", or nil.
func (c *completionChoices) completion(bodyLen int) *fwkrh.Completion {
	if bodyLen > maxCompletionResponseBytes || c.Object != textCompletionObject || len(c.Choices) != 1 {
		return nil
	}
	choice := c.Choices[0]
	if choice.Text == nil || (choice.FinishReason != finishReasonStop && choice.FinishReason != finishReasonLength) {
		return nil
	}
	return &fwkrh.Completion{Text: *choice.Text, FinishReason: choice.FinishReason}
}

func (p *OpenAIParser) parseStreamResponse(chunk []byte) (*fwkrh.ParsedResponse, error) {
	// One pass collects the event count, usage, and stop reason. Usage keeps the
	// last event because continuous-usage streams report cumulative totals, and
	// the stop reason is the last terminal signal observed.
	resp := &fwkrh.ParsedResponse{}
	var usage *request.Usage
	stopReason := fwkrh.StopReason("")
	sawToolCall := false
	for line := range bytes.SplitSeq(chunk, []byte("\n")) {
		content, ok := bytes.CutPrefix(line, []byte(reqcommon.SSEDataPrefix))
		if !ok || isStreamTerminator(content) {
			continue
		}
		resp.StreamedEvents++
		var streamResponse struct {
			Type     string             `json:"type"`
			Choices  []responseChoice   `json:"choices"`
			Item     responseOutputItem `json:"item"`
			Usage    json.RawMessage    `json:"usage"`
			Response struct {
				Usage             json.RawMessage           `json:"usage"`
				Status            string                    `json:"status"`
				Output            []responseOutputItem      `json:"output"`
				IncompleteDetails responseIncompleteDetails `json:"incomplete_details"`
			} `json:"response"`
		}
		if err := json.Unmarshal(content, &streamResponse); err != nil {
			continue
		}
		if streamResponse.Item.Type == "function_call" {
			sawToolCall = true
			stopReason = fwkrh.StopReasonToolCall
		}
		if candidate := stopReasonFromChoices(streamResponse.Choices); candidate != "" {
			stopReason = candidate
		}
		if isResponsesTerminalEvent(streamResponse.Type) {
			candidate := responsesEventStopReason(streamResponse.Type, streamResponse.Response.Status, streamResponse.Response.Output, streamResponse.Response.IncompleteDetails)
			if candidate == fwkrh.StopReasonTurnEnded && sawToolCall && streamResponse.Type == "response.completed" {
				candidate = fwkrh.StopReasonToolCall
			}
			if candidate != "" {
				stopReason = candidate
			}
		}
		// Standard Chat Completions and vLLM report usage inline.
		if len(streamResponse.Usage) > 0 {
			if strings.HasPrefix(streamResponse.Type, "speech.audio.") {
				if streamResponse.Type == "speech.audio.done" {
					usage = extractRawUsage(streamResponse.Usage)
				}
			} else {
				var parsed *request.Usage
				if err := json.Unmarshal(streamResponse.Usage, &parsed); err == nil && parsed != nil {
					usage = parsed
				}
			}
		}
		// Responses API reports final usage inside the terminal response object.
		if len(streamResponse.Response.Usage) > 0 && isResponsesTerminalEvent(streamResponse.Type) {
			if parsed := extractRawUsage(streamResponse.Response.Usage); parsed != nil {
				usage = parsed
			}
		}
	}
	resp.Usage = usage
	resp.StopReason = stopReason
	return resp, nil
}

// isStreamTerminator reports whether an SSE data payload is the [DONE] terminator, tolerating a
// trailing \r left by CRLF line splitting.
func isStreamTerminator(content []byte) bool {
	return bytes.Equal(bytes.TrimSuffix(content, []byte("\r")), []byte(reqcommon.SSEDoneMarker))
}

func responseMediaType(headers map[string]string) string {
	value, ok := headerValue(headers, reqcommon.HeaderContentType)
	if !ok {
		return ""
	}
	mediaType, _, _ := strings.Cut(value, ";")
	return strings.ToLower(strings.TrimSpace(mediaType))
}

func headerValue(headers map[string]string, name string) (string, bool) {
	for key, value := range headers {
		if strings.EqualFold(key, name) {
			return value, true
		}
	}
	return "", false
}

func extractUsageHeaders(headers map[string]string) *request.Usage {
	usage := &request.Usage{}
	found := false

	for header, target := range map[string]*int{
		vllmOmniInputTokensHeader:  &usage.PromptTokens,
		vllmOmniOutputTokensHeader: &usage.CompletionTokens,
		vllmOmniTotalTokensHeader:  &usage.TotalTokens,
	} {
		value, ok := headerValue(headers, header)
		if !ok {
			continue
		}
		parsed, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil || parsed < 0 {
			continue
		}
		*target = parsed
		found = true
	}

	if !found {
		return nil
	}
	return usage
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
	if request.MatchPathSuffix(path, "/images/edits") {
		return imagesEditsAPI
	}
	if request.MatchPathSuffix(path, "/"+videosSyncAPI) {
		return videosSyncAPI
	}
	if request.MatchPathSuffix(path, "/"+videosAPI) {
		return videosAPI
	}
	if request.MatchPathSuffix(path, "/audio/generate") {
		return audioGenerateAPI
	}
	if request.MatchPathSuffix(path, "/audio/speech") {
		return audioSpeechAPI
	}
	if request.MatchPathSuffix(path, "/"+audioTranscriptionsAPI) {
		return audioTranscriptionsAPI
	}
	if request.MatchPathSuffix(path, "/"+audioTranslationsAPI) {
		return audioTranslationsAPI
	}
	if request.MatchPathSuffix(path, request.DecisionsPathSuffix) {
		return request.DecisionsPathSuffix
	}

	// Default to completions API for backward compatibility with existing clients and integration tests
	return completionsAPI
}

// extractRequestBody extracts the InferenceRequestBody from the given raw body
// for the already-resolved API type.
func extractRequestBody(apiType string, rawBody []byte) (*request.InferenceRequestBody, error) {
	switch apiType {
	case conversationsAPI:
		validationErr := errors.New("invalid conversations request: must have items field")
		var conversations request.ConversationsRequest
		if err := json.Unmarshal(rawBody, &conversations); err != nil {
			return nil, requestBodyDecodeError(err, "conversations", validationErr)
		}
		if len(conversations.Items) == 0 {
			return nil, validationErr
		}
		return &request.InferenceRequestBody{Conversations: &conversations}, nil

	case responsesAPI:
		validationErr := errors.New("invalid responses request: must have input field")
		var responses request.ResponsesRequest
		if err := json.Unmarshal(rawBody, &responses); err != nil {
			return nil, requestBodyDecodeError(err, "responses", validationErr)
		}
		if responses.Input == nil {
			return nil, validationErr
		}
		return &request.InferenceRequestBody{Responses: &responses}, nil

	case audioSpeechAPI:
		var speechRequest struct {
			Input            *string         `json:"input"`
			Instructions     string          `json:"instructions,omitempty"`
			RefText          string          `json:"ref_text,omitempty"`
			CacheSalt        string          `json:"cache_salt,omitempty"`
			Voice            json.RawMessage `json:"voice,omitempty"`
			Speaker          json.RawMessage `json:"speaker,omitempty"`
			Language         string          `json:"language,omitempty"`
			TaskType         string          `json:"task_type,omitempty"`
			RefAudio         json.RawMessage `json:"ref_audio,omitempty"`
			RefAudio2        json.RawMessage `json:"ref_audio_2,omitempty"`
			References       json.RawMessage `json:"references,omitempty"`
			RefAudioCodes    json.RawMessage `json:"ref_audio_codes,omitempty"`
			ExtraParams      json.RawMessage `json:"extra_params,omitempty"`
			SpeakerEmbedding json.RawMessage `json:"speaker_embedding,omitempty"`
			XVectorOnlyMode  *bool           `json:"x_vector_only_mode,omitempty"`
		}
		if err := json.Unmarshal(rawBody, &speechRequest); err != nil {
			return nil, requestBodyDecodeError(err, "text to speech", errors.New("invalid text to speech request: input, instructions, ref_text, and cache_salt must be strings and x_vector_only_mode a boolean"))
		}
		if speechRequest.Input == nil {
			return nil, errors.New("invalid text to speech request: must have string input field")
		}
		// The engine treats an explicit JSON null and an absent field as the
		// same None conditioning; keep the raw fields that way too.
		refAudio := rawFieldOrNull(speechRequest.RefAudio)
		refAudio2 := rawFieldOrNull(speechRequest.RefAudio2)
		references := rawFieldOrNull(speechRequest.References)
		refAudioCodes := rawFieldOrNull(speechRequest.RefAudioCodes)
		extraParams := rawFieldOrNull(speechRequest.ExtraParams)
		speakerEmbedding := rawFieldOrNull(speechRequest.SpeakerEmbedding)
		// The engine resolves "voice" and "speaker" by key presence, so
		// "speaker" applies only when "voice" is absent; an explicit null
		// "voice" selects the default voice. Both are a string or an object
		// carrying an id, normalized to one string.
		rawVoice := speechRequest.Voice
		if len(rawVoice) == 0 {
			rawVoice = speechRequest.Speaker
		}
		voice, err := voiceField(rawVoice)
		if err != nil {
			return nil, err
		}
		return &request.InferenceRequestBody{
			TextToSpeech: &request.TextToSpeechRequest{
				Input:            *speechRequest.Input,
				Instructions:     speechRequest.Instructions,
				RefText:          speechRequest.RefText,
				CacheSalt:        speechRequest.CacheSalt,
				Voice:            voice,
				Language:         speechRequest.Language,
				TaskType:         speechRequest.TaskType,
				RefAudio:         refAudio,
				RefAudio2:        refAudio2,
				References:       references,
				RefAudioCodes:    refAudioCodes,
				ExtraParams:      extraParams,
				SpeakerEmbedding: speakerEmbedding,
				XVectorOnlyMode:  speechRequest.XVectorOnlyMode,
			},
		}, nil

	case audioGenerateAPI:
		var generateRequest struct {
			Input          *string `json:"input"`
			NegativePrompt string  `json:"negative_prompt,omitempty"`
		}
		if err := json.Unmarshal(rawBody, &generateRequest); err != nil {
			return nil, requestBodyDecodeError(err, "audio generate", errors.New("invalid audio generate request: input and negative_prompt must be strings"))
		}
		if generateRequest.Input == nil {
			return nil, errors.New("invalid audio generate request: must have string input field")
		}
		return &request.InferenceRequestBody{
			AudioGenerate: &request.AudioGenerateRequest{Input: *generateRequest.Input, NegativePrompt: generateRequest.NegativePrompt},
		}, nil

	case chatCompletionsAPI:
		validationErr := errors.New("invalid chat completions request: must have valid messages field")
		var chatCompletions request.ChatCompletionsRequest
		if err := json.Unmarshal(rawBody, &chatCompletions); err != nil {
			return nil, requestBodyDecodeError(err, "chat completions", validationErr)
		}
		if err := validateChatCompletionsMessages(chatCompletions.Messages); err != nil {
			return nil, validationErr
		}
		return &request.InferenceRequestBody{ChatCompletions: &chatCompletions}, nil

	case completionsAPI:
		validationErr := errors.New("invalid completions request: must have prompt field")
		var completions struct {
			request.CompletionsRequest
			// vLLM accepts prompt_embeds in place of prompt.
			PromptEmbeds jsonPresent `json:"prompt_embeds"`
		}
		if err := json.Unmarshal(rawBody, &completions); err != nil {
			return nil, requestBodyDecodeError(err, "completions", validationErr)
		}
		if completions.Prompt.IsEmpty() && !bool(completions.PromptEmbeds) {
			return nil, validationErr
		}
		return &request.InferenceRequestBody{Completions: &completions.CompletionsRequest}, nil

	case embeddingsAPI:
		validationErr := errors.New("invalid embeddings request: must have input field")
		var embeddings request.EmbeddingsRequest
		if err := json.Unmarshal(rawBody, &embeddings); err != nil {
			return nil, requestBodyDecodeError(err, "embeddings", validationErr)
		}
		if embeddings.Input.IsEmpty() {
			return nil, validationErr
		}
		return &request.InferenceRequestBody{Embeddings: &embeddings}, nil

	case imagesGenerationsAPI:
		validationErr := errors.New("invalid images generations request: must have prompt field")
		var images request.ImagesGenerationsRequest
		if err := json.Unmarshal(rawBody, &images); err != nil {
			return nil, requestBodyDecodeError(err, "images generations", validationErr)
		}
		if images.Prompt == "" {
			return nil, validationErr
		}
		return &request.InferenceRequestBody{Images: &images}, nil
	default:
		return nil, errors.New("unsupported API endpoint")
	}
}

// parseDecisionsRequest reads the routing envelope of a /v1/decisions request.
func parseDecisionsRequest(body []byte) (*fwkrh.ParseResult, error) {
	bodyMap, err := parserutil.UnmarshalEnvelope(body, reqcommon.FieldInput)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling request body: %w", err)
	}
	questions, ok := bodyMap["questions"].(json.RawMessage)
	if !ok || len(questions) == 0 || questions[0] != '[' {
		return nil, errors.New("invalid decisions request: questions must be a JSON array")
	}
	result := &request.InferenceRequestBody{
		Payload: request.PayloadMap(bodyMap),
		RawBody: body,
		// The engine renders one prompt per question, which the router cannot reproduce.
		RenderRequest: true,
	}
	if model, ok := bodyMap["model"].(string); ok {
		result.Model = model
	}
	return &fwkrh.ParseResult{Body: result, JSONValidated: true, SkipResponseProcessing: false}, nil
}

func parseMultipartFields(body []byte, headers map[string]string, kind string, visit func(string, []byte) error) error {
	return parseMultipartForm(body, headers, kind, visit, nil, nil)
}

// parseMultipartForm walks a multipart/form-data body, handing scalar fields
// to visit and file parts whose name wantFile accepts to visitFile. File parts
// no callback accepts are drained without capture.
func parseMultipartForm(body []byte, headers map[string]string, kind string, visit func(string, []byte) error,
	wantFile func(name string) bool, visitFile func(name string, media request.UploadedMedia) error) error {
	contentTypeValue, _ := headerValue(headers, reqcommon.HeaderContentType)
	mediaType, params, err := mime.ParseMediaType(contentTypeValue)
	if err != nil || mediaType != "multipart/form-data" {
		return fmt.Errorf("%s request must have a multipart/form-data content-type", kind)
	}
	boundary := params["boundary"]
	if boundary == "" {
		return fmt.Errorf("%s request: content-type is missing the multipart boundary", kind)
	}
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var field bytes.Buffer
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("error reading %s multipart body: %w", kind, err)
		}
		if part.FileName() != "" {
			if visitFile == nil || wantFile == nil || !wantFile(part.FormName()) {
				if _, err := io.Copy(io.Discard, part); err != nil {
					return fmt.Errorf("error reading %s file part %q: %w", kind, part.FormName(), err)
				}
				continue
			}
			media, err := readUploadedMedia(part)
			if err != nil {
				return fmt.Errorf("error reading %s file part %q: %w", kind, part.FormName(), err)
			}
			if err := visitFile(part.FormName(), media); err != nil {
				return err
			}
			continue
		}
		field.Reset()
		if _, err := field.ReadFrom(part); err != nil {
			return fmt.Errorf("error reading %s form field %q: %w", kind, part.FormName(), err)
		}
		if err := visit(part.FormName(), field.Bytes()); err != nil {
			return err
		}
	}
}

// readUploadedMedia streams a file part into its routing-relevant properties
// without retaining the body: size, content type, a bounded head for container
// sniffing, and a hash of the full stream.
func readUploadedMedia(part *multipart.Part) (request.UploadedMedia, error) {
	media := request.UploadedMedia{ContentType: part.Header.Get(reqcommon.HeaderContentType)}
	digest := xxhash.New()
	head := make([]byte, 0, request.UploadedMediaHeadBytes)
	buf := make([]byte, 32<<10)
	for {
		n, err := part.Read(buf)
		if n > 0 {
			media.SizeBytes += int64(n)
			if _, err := digest.Write(buf[:n]); err != nil {
				return request.UploadedMedia{}, err
			}
			if len(head) < request.UploadedMediaHeadBytes {
				head = append(head, buf[:min(n, request.UploadedMediaHeadBytes-len(head))]...)
			}
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return request.UploadedMedia{}, err
		}
	}
	media.Head = head
	media.Hash = digest.Sum64()
	return media, nil
}

// parseImagesEditsRequest parses a multipart/form-data /v1/images/edits request.
// Each uploaded image part, under the "image" or "image[]" form name, is an
// image-to-image conditioning input, so it is captured as UploadedMedia for
// input-cost estimation; the original body is forwarded unchanged.
func parseImagesEditsRequest(body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	images := &request.ImagesGenerationsRequest{}
	extractedBody := &request.InferenceRequestBody{
		Images:  images,
		Payload: request.RawPayload(body),
	}
	err := parseMultipartForm(body, headers, "images edits", func(name string, value []byte) error {
		switch name {
		case reqcommon.FieldModel:
			extractedBody.Model = string(value)
		case "prompt":
			images.Prompt = string(value)
		case "size":
			images.Size = string(value)
		case "negative_prompt":
			images.NegativePrompt = string(value)
		case "n":
			n, err := strconv.ParseInt(string(value), 10, 64)
			if err != nil {
				return fmt.Errorf("invalid images edits n field: %w", err)
			}
			images.N = &n
		case formFieldNumInferenceSteps:
			steps, err := strconv.ParseInt(string(value), 10, 64)
			if err != nil {
				return fmt.Errorf("invalid images edits num_inference_steps field: %w", err)
			}
			images.NumInferenceSteps = &steps
		case "stream":
			stream, err := parseFormBool(value)
			if err != nil {
				return fmt.Errorf("invalid images edits stream field: %w", err)
			}
			extractedBody.Stream = stream
		}
		return nil
	}, func(name string) bool {
		return name == "image" || name == "image[]"
	}, func(_ string, media request.UploadedMedia) error {
		images.InputImages = append(images.InputImages, media)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if images.Prompt == "" {
		return nil, errors.New("invalid images edits request: must have prompt field")
	}
	return &fwkrh.ParseResult{Body: extractedBody, SkipResponseProcessing: false}, nil
}

// parseVideosRequest parses a multipart/form-data /v1/videos or /v1/videos/sync
// request. Scheduler-relevant scalar fields are read while the original body is
// forwarded unchanged.
func parseVideosRequest(body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	videos := &request.VideoGenerationRequest{NumOutputsPerPrompt: ptr.To[int64](1)}
	extractedBody := &request.InferenceRequestBody{
		Videos:  videos,
		Payload: request.RawPayload(body),
	}
	err := parseMultipartFields(body, headers, "videos", func(name string, value []byte) error {
		switch name {
		case reqcommon.FieldModel:
			extractedBody.Model = string(value)
		case "prompt":
			videos.Prompt = string(value)
		case "negative_prompt":
			videos.NegativePrompt = string(value)
		case "size":
			videos.Size = string(value)
		case "seconds":
			videos.Seconds = string(value)
		case "width", "height", "num_frames", formFieldNumInferenceSteps, "num_outputs_per_prompt", "seed":
			number, err := strconv.ParseInt(string(value), 10, 64)
			if err != nil {
				return fmt.Errorf("invalid videos %s field: %w", name, err)
			}
			switch name {
			case "width":
				videos.Width = &number
			case "height":
				videos.Height = &number
			case "num_frames":
				videos.NumFrames = &number
			case formFieldNumInferenceSteps:
				videos.NumInferenceSteps = &number
			case "num_outputs_per_prompt":
				videos.NumOutputsPerPrompt = &number
			case "seed":
				videos.Seed = &number
			}
		case "fps":
			fps, err := strconv.ParseFloat(string(value), 64)
			if err != nil {
				return fmt.Errorf("invalid videos fps field: %w", err)
			}
			if math.IsNaN(fps) || math.IsInf(fps, 0) {
				return fmt.Errorf("invalid videos fps field: %q", value)
			}
			videos.FPS = &fps
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if videos.Prompt == "" {
		return nil, errors.New("invalid videos request: must have prompt field")
	}
	return &fwkrh.ParseResult{Body: extractedBody, SkipResponseProcessing: false}, nil
}

// jsonPresent records whether a field holds a non-null value without copying it.
type jsonPresent bool

func (p *jsonPresent) UnmarshalJSON(data []byte) error {
	*p = jsonPresent(!bytes.Equal(data, []byte("null")))
	return nil
}

// parseTranscriptionsRequest parses a multipart/form-data /v1/audio/transcriptions
// or /v1/audio/translations request. The audio file part drives the engine's
// cost, so it is captured as UploadedMedia for input-cost estimation; the
// engine is authoritative for the required file field, and the original body is
// forwarded unchanged.
func parseTranscriptionsRequest(apiType string, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	transcriptions := &request.TranscriptionRequest{}
	extractedBody := &request.InferenceRequestBody{
		Transcriptions: transcriptions,
		Payload:        request.RawPayload(body),
	}
	err := parseMultipartForm(body, headers, apiType, func(name string, value []byte) error {
		switch name {
		case reqcommon.FieldModel:
			extractedBody.Model = string(value)
		case "prompt":
			transcriptions.Prompt = string(value)
		case "stream":
			stream, err := parseFormBool(value)
			if err != nil {
				return fmt.Errorf("invalid %s stream field: %w", apiType, err)
			}
			extractedBody.Stream = stream
		}
		return nil
	}, func(name string) bool {
		return name == "file"
	}, func(_ string, media request.UploadedMedia) error {
		transcriptions.Audio = media
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &fwkrh.ParseResult{Body: extractedBody, SkipResponseProcessing: false}, nil
}

// requestBodyDecodeError reports a JSON type error in an api request by the
// field path and the JSON value that failed to decode. Other decode errors
// return validationErr.
func requestBodyDecodeError(err error, api string, validationErr error) error {
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return err
	}
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) && typeErr.Field != "" {
		return &fwkrh.RequestFieldError{
			Field: typeErr.Field,
			Err: fmt.Errorf("invalid %s request: %s: cannot unmarshal %s into %s",
				api, typeErr.Field, typeErr.Value, jsonTypeName(typeErr.Type)),
		}
	}
	return validationErr
}

// jsonTypeName names t as a JSON type, so client errors do not carry Go type
// names. Scalars keep their Go kind, which states the range a number exceeded.
func jsonTypeName(t reflect.Type) string {
	if t == nil {
		return "value"
	}
	switch t.Kind() {
	case reflect.Slice, reflect.Array:
		return "array"
	case reflect.Map, reflect.Struct:
		return "object"
	default:
		return t.Kind().String()
	}
}

func validateChatCompletionsMessages(messages []request.Message) error {
	if len(messages) == 0 {
		return errors.New("chat-completions request must have at least one message")
	}
	return nil
}

// rawFieldOrNull returns nil for a raw JSON field holding a literal null, so
// it salts and fingerprints the same as an absent field.
func rawFieldOrNull(raw json.RawMessage) json.RawMessage {
	if string(raw) == "null" {
		return nil
	}
	return raw
}

// voiceField returns the voice a voice or speaker field carries: the string
// itself, or the object form's id, the two shapes the engine normalizes to one
// voice name. An object's id may be null, which the engine treats as absent.
// Empty input is an absent field.
func voiceField(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return "", errors.New("invalid text to speech request: voice must be a string or an object with an id")
	}
	id, ok := obj["id"]
	if !ok {
		return "", errors.New("invalid text to speech request: a voice object must have an id")
	}
	if string(id) == "null" {
		return "", nil
	}
	if err := json.Unmarshal(id, &s); err != nil {
		return "", errors.New("invalid text to speech request: a voice object's id must be a string")
	}
	return s, nil
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

// extractResponse reads usage and completion text in one decode. A type
// mismatch outside "usage" leaves Completion nil and does not fail usage.
func extractResponse(responseBytes []byte) (*fwkrh.ParsedResponse, error) {
	var responseBody struct {
		Usage             map[string]any            `json:"usage"`
		Status            string                    `json:"status"`
		Output            []responseOutputItem      `json:"output"`
		IncompleteDetails responseIncompleteDetails `json:"incomplete_details"`
		completionChoices
	}
	var completion *fwkrh.Completion
	err := json.Unmarshal(responseBytes, &responseBody)
	var typeErr *json.UnmarshalTypeError
	switch {
	case err == nil:
		completion = responseBody.completion(len(responseBytes))
	case !errors.As(err, &typeErr) || typeErr.Field == reqcommon.FieldUsage:
		return nil, err
	}
	reason := fwkrh.StopReason("")
	if len(responseBody.Output) > 0 || responseBody.Status != "" {
		reason = responsesStopReason(responseBody.Status, responseBody.Output, responseBody.IncompleteDetails)
	} else {
		reason = stopReasonFromChoices(responseBody.Choices)
	}
	return &fwkrh.ParsedResponse{Usage: usageFromFields(responseBody.Usage), Completion: completion, StopReason: reason}, nil
}

func extractUsage(responseBytes []byte) (*request.Usage, error) {
	var responseBody struct {
		Usage map[string]any `json:"usage"`
	}
	err := json.Unmarshal(responseBytes, &responseBody)
	if err != nil {
		return nil, err
	}
	return usageFromFields(responseBody.Usage), nil
}

// usageFromFields converts a decoded "usage" object, or returns nil when absent.
func usageFromFields(fields map[string]any) *request.Usage {
	if fields == nil {
		return nil
	}

	usage := request.Usage{}

	// Chat/Completions APIs use prompt_tokens. Responses/Conversations APIs use input_tokens.
	for _, inputTokens := range []string{reqcommon.FieldPromptTokens, reqcommon.FieldInputTokens} {
		if v, ok := fields[inputTokens]; ok && v != nil {
			usage.PromptTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use completion_tokens. Responses/Conversations APIs use output_tokens.
	for _, outputTokens := range []string{reqcommon.FieldCompletionTokens, reqcommon.FieldOutputTokens} {
		if v, ok := fields[outputTokens]; ok && v != nil {
			usage.CompletionTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use prompt_tokens_details. Responses/Conversations APIs use input_tokens_details.
	for _, details := range []string{reqcommon.FieldPromptTokensDetails, reqcommon.FieldInputTokensDetails} {
		if detailsMap, ok := fields[details].(map[string]any); ok {
			if cachedTokens, ok := detailsMap[reqcommon.FieldCachedTokens]; ok {
				usage.PromptTokenDetails = &request.PromptTokenDetails{
					CachedTokens: toInt(cachedTokens),
				}
			}
		}
	}

	// total_tokens field name is consistent across all API types.
	if v, ok := fields[reqcommon.FieldTotalTokens]; ok && v != nil {
		usage.TotalTokens = toInt(v)
	}

	return &usage
}

// isResponsesTerminalEvent reports whether a Responses API stream event ends
// the stream and carries the final response. The in-progress snapshots carry a
// usage field too, with no final counts.
func isResponsesTerminalEvent(eventType string) bool {
	switch eventType {
	case "response.completed", "response.incomplete", "response.failed":
		return true
	default:
		return false
	}
}

func extractRawUsage(raw json.RawMessage) *request.Usage {
	jsonBytes, _ := json.Marshal(map[string]any{"usage": raw})
	usage, err := extractUsage(jsonBytes)
	if err != nil {
		return nil
	}
	return usage
}

// parseFormBool parses a boolean form field with the values the model servers'
// form parsing (pydantic) accepts, in any letter case.
func parseFormBool(value []byte) (bool, error) {
	switch strings.ToLower(string(value)) {
	case "1", "on", "t", "true", "y", "yes":
		return true, nil
	case "0", "off", "f", "false", "n", "no":
		return false, nil
	}
	return false, fmt.Errorf("invalid boolean %q", value)
}
