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

package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
	"time"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

const (
	// finishReasonLength is the finish reason when max_tokens was reached.
	finishReasonLength = "length"

	requestFieldN = "n"
)

// dispatchDecode routes a fully-prepared decode request to either chunked
// decode or the regular decoder proxy. Chunked decode is only used for
// chat completions requests when s.config.DecodeChunkSize > 0.
// body is the already-parsed JSON map; callers that hold it
// should use this instead of calling s.decoderProxy directly.
// raw supplies regular-decode bytes without rereading the request body.
func (s *Server) dispatchDecode(w http.ResponseWriter, r *http.Request, body map[string]any, raw []byte, toolFields *toolCallingRequest) {
	if s.config.DecodeChunkSize > 0 && r.URL.Path == reqcommon.PathChatCompletions {
		s.runChunkedDecodeFromMap(w, r, body, raw, toolFields)
		return
	}
	toolFields.compareValidatedDispatchedForSpan(raw, trace.SpanFromContext(r.Context()))
	s.decoderProxy.ServeHTTP(w, r)
}

// runChunkedDecode reads and parses the body, then delegates to
// runChunkedDecodeFromMap.
func (s *Server) runChunkedDecode(w http.ResponseWriter, r *http.Request) {
	original, body, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	toolFields, _ := captureInferenceToolCallingRequest(r, reqcommon.APITypeChatCompletions, original)
	defer toolFields.report(r.Context())
	s.runChunkedDecodeFromMap(w, cloneRequestWithBody(r.Context(), r, original), body, original, toolFields)
}

// runChunkedDecodeFromMap executes chunked decode given an already-parsed body map.
// Non-streaming: accumulated chunks are reassembled into a single JSON response.
// Streaming: each chunk is re-emitted as an SSE event; [DONE] closes the stream.
// The caller owns aggregate reporting across prefill and decode chunks.
func (s *Server) runChunkedDecodeFromMap(w http.ResponseWriter, r *http.Request, body map[string]any, raw []byte, toolFields *toolCallingRequest) {
	s.logger.V(logging.DEBUG).Info("running chunked decode", "chunkSize", s.config.DecodeChunkSize)

	ctx, span := tracing.Tracer(tracerScope).Start(r.Context(), "chunked_decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()
	// Chunk outcomes exclude prefill results held by the request aggregate.
	var chunkFields *toolCallingRequest
	if toolFields != nil && span.IsRecording() {
		chunkFields = &toolCallingRequest{original: toolFields.original}
		defer func() {
			if len(chunkFields.statuses) == 0 {
				return
			}
			attrs := chunkFields.original.SpanAttributes(chunkFields.statuses)
			span.SetAttributes(attrs...)
			trace.SpanFromContext(r.Context()).SetAttributes(attrs...)
		}()
	}
	compareChunk := func(raw []byte) {
		statuses, err := toolFields.compareValidatedDispatched(raw)
		if chunkFields != nil && err == nil {
			chunkFields.mergeStatuses(statuses)
		}
	}
	regularDecode := func() {
		compareChunk(raw)
		s.decoderProxy.ServeHTTP(w, r.WithContext(ctx))
	}

	streamingEnabled, _ := body[reqcommon.FieldStream].(bool)
	originalMaxTokens := resolveMaxTokens(body)

	span.SetAttributes(
		semconv.LLMDPDProxyChunkedDecodeChunkSize(s.config.DecodeChunkSize),
		semconv.LLMDPDProxyChunkedDecodeStreaming(streamingEnabled),
	)

	// The decoder echoes the final assistant message, which under chunking is the text of all earlier chunks.
	if echo, _ := body[reqcommon.FieldEcho].(bool); echo {
		s.logger.V(logging.DEBUG).Info("chunked decode: echo enabled, using regular decode")
		regularDecode()
		return
	}

	// Each chunk continues the first choice's text for every choice, and the budget counts the tokens of all choices.
	if n, ok := toInt(body[requestFieldN]); ok && n > 1 {
		s.logger.V(logging.DEBUG).Info("chunked decode: several choices requested, using regular decode", "n", n)
		regularDecode()
		return
	}

	// A tool call cannot be reassembled across chunk boundaries: vLLM parses it
	// from a single request's output, and a call split by the chunk budget comes
	// back as plain content. Bypass chunking when tools may be called. tools and
	// tool_choice stay raw (not in inspectedRequestFields), so inspect their bytes
	// without re-encoding them.
	if requestMayCallTools(body) {
		s.logger.V(logging.DEBUG).Info("chunked decode: tools requested, using regular decode")
		regularDecode()
		return
	}

	// If the token budget fits within a single chunk, skip chunking entirely.
	if originalMaxTokens > 0 && originalMaxTokens <= s.config.DecodeChunkSize {
		s.logger.V(logging.DEBUG).Info("chunked decode: token budget <= chunk size, using regular decode",
			"maxTokens", originalMaxTokens, "chunkSize", s.config.DecodeChunkSize)
		regularDecode()
		return
	}

	messages, err := requestMessages(body)
	if err != nil {
		regularDecode()
		return
	}
	finalMessage := map[string]json.RawMessage{reqcommon.FieldRole: json.RawMessage(`"` + reqcommon.RoleAssistant + `"`)}
	prefix := ""
	if continueFinal, _ := body[reqcommon.FieldContinueFinalMessage].(bool); continueFinal {
		var role string
		finalMessage = nil
		// Only an assistant message can carry the accumulated text; any other final message is sent as is.
		if len(messages) == 0 || json.Unmarshal(messages[len(messages)-1], &finalMessage) != nil ||
			json.Unmarshal(finalMessage[reqcommon.FieldRole], &role) != nil || role != reqcommon.RoleAssistant {
			regularDecode()
			return
		}
		content := finalMessage[reqcommon.FieldContent]
		// Content parts arrays and null have no string to extend.
		if len(content) == 0 || content[0] != '"' || json.Unmarshal(content, &prefix) != nil {
			regularDecode()
			return
		}
		messages = messages[:len(messages)-1]
	}

	if streamingEnabled {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Transfer-Encoding", "chunked")
	}

	var (
		totalTokens          int
		chunkIndex           int
		lastResponse         map[string]any
		originalPromptTokens int
		textAccum            strings.Builder
	)

	decodeStart := time.Now()

	for {
		if ctx.Err() != nil {
			if streamingEnabled && chunkIndex > 0 {
				fmt.Fprintf(w, "%s\n\n", reqcommon.SSEDone)
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			return
		}

		remaining := remainingTokens(originalMaxTokens, totalTokens)
		if remaining == 0 {
			s.logger.V(logging.DEBUG).Info("chunked decode: token budget exhausted", "totalTokens", totalTokens)
			break
		}

		chunkBudget := s.config.DecodeChunkSize
		if remaining > 0 && remaining < chunkBudget {
			chunkBudget = remaining
		}

		chunkReq := maps.Clone(body)
		chunkReq[reqcommon.FieldMaxTokens] = chunkBudget
		chunkReq[reqcommon.FieldMaxCompletionTokens] = chunkBudget
		chunkReq[reqcommon.FieldStream] = false
		delete(chunkReq, reqcommon.FieldStreamOptions)

		// From the second chunk onward: remove KV transfer params and instruct
		// to continue the last assistant message rather than start a new one.
		if chunkIndex > 0 {
			delete(chunkReq, reqcommon.FieldKVTransferParams)
			chunkReq[reqcommon.FieldContinueFinalMessage] = true
			chunkReq[reqcommon.FieldAddGenerationPrompt] = false
			content, _ := json.Marshal(prefix + textAccum.String())
			finalMessage[reqcommon.FieldContent] = content
			message, err := json.Marshal(finalMessage)
			if err != nil {
				if writeErr := errorInternalServerError(err, w); writeErr != nil {
					s.logger.Error(writeErr, "failed to send error response to client")
				}
				return
			}
			chunkReq[reqcommon.FieldMessages] = append(slices.Clone(messages), message)
		}

		chunkBody, err := json.Marshal(chunkReq)
		if err != nil {
			if err := errorJSONInvalid(err, w); err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
			return
		}

		s.logger.V(logging.DEBUG).Info("chunked decode: dispatching chunk",
			"chunk", chunkIndex, "chunkBudget", chunkBudget, "totalTokensSoFar", totalTokens)

		bw := &bufferedResponseWriter{}
		compareChunk(chunkBody)
		s.decoderProxy.ServeHTTP(bw, cloneRequestWithBody(ctx, r, chunkBody))

		if isHTTPError(bw.statusCode) {
			s.logger.Error(fmt.Errorf("chunk %d failed with status %d", chunkIndex, bw.statusCode),
				"chunked decode chunk error", "statusCode", bw.statusCode)
			span.SetStatus(codes.Error, "chunk decode failed")
			maps.Copy(w.Header(), bw.headers)
			w.WriteHeader(bw.statusCode)
			if streamingEnabled && chunkIndex > 0 {
				// The status went out with the first chunk, so the error is sent the
				// way vLLM reports a failure in a stream: as a data event before [DONE].
				fmt.Fprintf(w, "%s%s\n\n%s\n\n", reqcommon.SSEDataPrefix, bw.bodyBytes(), reqcommon.SSEDone)
				return
			}
			WriteAll(w, bw.bodyBytes())
			return
		}

		var chunkResponse map[string]any
		if err := json.Unmarshal(bw.bodyBytes(), &chunkResponse); err != nil {
			s.logger.Error(jsonErrorForLog(err), "failed to unmarshal chunk response", "chunk", chunkIndex)
			if err := errorInternalServerError(err, w); err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
			return
		}

		lastResponse = chunkResponse
		chunkTokens := countTokensInResponse(chunkResponse)
		totalTokens += chunkTokens
		if chunkIndex == 0 {
			originalPromptTokens = extractPromptTokens(chunkResponse)
		}
		chunkIndex++

		s.logger.V(logging.DEBUG).Info("chunked decode: chunk complete", "chunkTokens", chunkTokens, "totalTokens", totalTokens)

		finishReason := extractFinishReason(chunkResponse)
		chunkText := extractChoiceText(firstChoice(chunkResponse))
		// Accumulated text forms the assistant message for the next chunk.
		textAccum.WriteString(chunkText)

		terminal := finishReason != "" && finishReason != finishReasonLength
		// Guard against infinite loop: if the chunk produced no tokens and no
		// text there is nothing to continue from.
		empty := chunkTokens == 0 && chunkText == ""

		if streamingEnabled {
			last := terminal || empty || remainingTokens(originalMaxTokens, totalTokens) == 0
			if err := emitSSEChunk(w, chunkResponse, last); err != nil {
				s.logger.Error(err, "failed to write SSE chunk to client")
				return
			}
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}

		if terminal {
			s.logger.V(logging.DEBUG).Info("chunked decode: terminal finish reason, stopping",
				"finishReason", finishReason, "chunks", chunkIndex)
			break
		}

		if empty {
			s.logger.Info("chunked decode: empty chunk with no tokens, stopping to avoid infinite loop",
				"chunk", chunkIndex)
			break
		}

		s.logger.V(logging.TRACE).Info("chunked decode: appending chunk text to request", "chunkTextBytes", len(chunkText))
	}

	span.SetAttributes(
		semconv.LLMDPDProxyChunkedDecodeChunks(chunkIndex),
		semconv.LLMDPDProxyChunkedDecodeTotalTokens(totalTokens),
		semconv.LLMDPDProxyChunkedDecodeDurationMs(float64(time.Since(decodeStart).Milliseconds())),
	)

	// Corrected cumulative usage: prompt_tokens from first chunk, completion_tokens summed.
	cumulativeUsage := map[string]any{
		reqcommon.FieldPromptTokens:     originalPromptTokens,
		reqcommon.FieldCompletionTokens: totalTokens,
		reqcommon.FieldTotalTokens:      originalPromptTokens + totalTokens,
	}

	if streamingEnabled {
		// Emit corrected cumulative usage as a final event before [DONE].
		// Individual chunk events have usage stripped by emitSSEChunk.
		if lastResponse != nil {
			usageEvent := map[string]any{
				reqcommon.FieldUsage:   cumulativeUsage,
				reqcommon.FieldChoices: []any{},
			}
			if data, err := json.Marshal(usageEvent); err == nil {
				fmt.Fprintf(w, "%s%s\n\n", reqcommon.SSEDataPrefix, data)
			}
		}
		fmt.Fprintf(w, "%s\n\n", reqcommon.SSEDone)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		return
	}

	// Non-streaming: reassemble the full response.
	// overwrite choices with the text accumulated across all chunks.
	if lastResponse == nil {
		if err := errorInternalServerError(errors.New("no chunks produced"), w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	lastResponse[reqcommon.FieldUsage] = cumulativeUsage

	if choice := firstChoice(lastResponse); choice != nil {
		choice = maps.Clone(choice)
		fullText := textAccum.String()
		if msg, ok := choice[reqcommon.FieldMessage].(map[string]any); ok {
			msg = maps.Clone(msg)
			msg[reqcommon.FieldContent] = fullText
			choice[reqcommon.FieldMessage] = msg
		}
		lastResponse[reqcommon.FieldChoices] = []any{choice}
	}

	respBody, err := json.Marshal(lastResponse)
	if err != nil {
		if err := errorInternalServerError(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	WriteAll(w, respBody)
}

// resolveMaxTokens returns the effective max-tokens limit from the request map.
// Prefers max_completion_tokens (OpenAI v1) over max_tokens (legacy).
// Returns -1 when neither field is set (no explicit limit).
func resolveMaxTokens(req map[string]any) int {
	for _, field := range []string{reqcommon.FieldMaxCompletionTokens, reqcommon.FieldMaxTokens} {
		if v, ok := req[field]; ok {
			if n, ok := toInt(v); ok && n > 0 {
				return n
			}
		}
	}
	return -1
}

// remainingTokens returns how many more tokens may be generated.
// Returns -1 for no cap, 0 when the budget is exhausted.
func remainingTokens(budget, used int) int {
	if budget < 0 {
		return -1
	}
	if used >= budget {
		return 0
	}
	return budget - used
}

// countTokensInResponse returns completion_tokens from the usage field, or 0.
func countTokensInResponse(response map[string]any) int {
	if usage, ok := response[reqcommon.FieldUsage].(map[string]any); ok {
		if n, ok := toInt(usage[reqcommon.FieldCompletionTokens]); ok {
			return n
		}
	}
	return 0
}

// extractPromptTokens returns prompt_tokens from the usage field, or 0.
func extractPromptTokens(response map[string]any) int {
	if usage, ok := response[reqcommon.FieldUsage].(map[string]any); ok {
		if n, ok := toInt(usage[reqcommon.FieldPromptTokens]); ok {
			return n
		}
	}
	return 0
}

// extractFinishReason returns finish_reason or "".
func extractFinishReason(response map[string]any) string {
	choices, ok := response[reqcommon.FieldChoices].([]any)
	if !ok || len(choices) == 0 {
		return ""
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return ""
	}
	reason, _ := choice[reqcommon.FieldFinishReason].(string)
	return reason
}

// emitSSEChunk writes one SSE data event to w, converting a buffered
// non-streaming response into a streaming delta event (chat or legacy).
// Unless last is set, finish_reason is null: the decoder stopped that chunk at
// its chunk budget, and the next chunk continues the response.
func emitSSEChunk(w http.ResponseWriter, chunkResponse map[string]any, last bool) error {
	streamChunk := maps.Clone(chunkResponse)

	if choices, _ := chunkResponse[reqcommon.FieldChoices].([]any); len(choices) > 0 {
		streamChoices := make([]any, 0, len(choices))
		for _, c := range choices {
			choice, ok := c.(map[string]any)
			if !ok {
				continue
			}
			text := extractChoiceText(choice)
			streamChoice := map[string]any{
				reqcommon.FieldIndex:        choice[reqcommon.FieldIndex],
				reqcommon.FieldFinishReason: nil,
			}
			if last {
				streamChoice[reqcommon.FieldFinishReason] = choice[reqcommon.FieldFinishReason]
			}
			streamChoice[reqcommon.FieldDelta] = map[string]any{reqcommon.FieldContent: text, reqcommon.FieldRole: reqcommon.RoleAssistant}
			streamChoices = append(streamChoices, streamChoice)
		}
		streamChunk[reqcommon.FieldChoices] = streamChoices
	}

	delete(streamChunk, reqcommon.FieldUsage)

	data, err := json.Marshal(streamChunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "%s%s\n\n", reqcommon.SSEDataPrefix, data)
	return err
}

// firstChoice returns choices[0] from a response map, or nil.
func firstChoice(response map[string]any) map[string]any {
	choices, ok := response[reqcommon.FieldChoices].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	choice, _ := choices[0].(map[string]any)
	return choice
}

// extractChoiceText returns the generated text from a choice's message.content.
func extractChoiceText(choice map[string]any) string {
	if msg, ok := choice[reqcommon.FieldMessage].(map[string]any); ok {
		if content, ok := msg[reqcommon.FieldContent].(string); ok {
			return content
		}
	}
	return ""
}

// toInt converts a JSON number value (float64, int, or json.Number) to int.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	}
	return 0, false
}

// requestMayCallTools reports whether the request declares a non-empty tools list
// and does not disable tool calls via tool_choice: "none". tools and tool_choice
// are not in inspectedRequestFields, so they arrive as json.RawMessage; this reads
// their bytes without decoding and re-encoding them.
func requestMayCallTools(body map[string]any) bool {
	raw, ok := body[reqcommon.FieldTools].(json.RawMessage)
	if !ok {
		return false
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(raw, &tools); err != nil || len(tools) == 0 {
		return false
	}
	if choice, ok := body[reqcommon.FieldToolChoice].(json.RawMessage); ok {
		var value string
		if json.Unmarshal(choice, &value) == nil && value == reqcommon.ToolChoiceNone {
			return false
		}
	}
	return true
}
