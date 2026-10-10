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
	"strings"
	"time"

	"github.com/go-logr/logr"
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

	responseFieldMessage = "message"
	responseFieldIndex   = "index"
	responseFieldDelta   = "delta"

	roleAssistant = "assistant"
)

// dispatchDecode routes a fully-prepared decode request to either chunked
// decode or the regular decoder proxy. Chunked decode is only used for
// chat completions requests when s.config.DecodeChunkSize > 0.
// body is the already-parsed JSON map; callers that hold it
// should use this instead of calling s.decoderProxy directly.
func (s *Server) dispatchDecode(w http.ResponseWriter, r *http.Request, body map[string]any) {
	if s.config.DecodeChunkSize > 0 && r.URL.Path == reqcommon.PathChatCompletions {
		s.runChunkedDecodeFromMap(w, r, body)
		return
	}
	s.decoderProxy.ServeHTTP(w, r)
}

// runChunkedDecode reads and parses the body, then delegates to
// runChunkedDecodeFromMap.
func (s *Server) runChunkedDecode(w http.ResponseWriter, r *http.Request) {
	original, body, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	s.runChunkedDecodeFromMap(w, cloneRequestWithBody(r.Context(), r, original), body)
}

// runChunkedDecodeFromMap executes chunked decode given an already-parsed body map.
// Non-streaming: accumulated chunks are reassembled into a single JSON response.
// Streaming: each chunk is re-emitted as an SSE event; [DONE] closes the stream.
func (s *Server) runChunkedDecodeFromMap(w http.ResponseWriter, r *http.Request, body map[string]any) {
	s.logger.V(logging.DEBUG).Info("running chunked decode", "chunkSize", s.config.DecodeChunkSize)

	ctx, span := tracing.Tracer(tracerScope).Start(r.Context(), "chunked_decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	streamingEnabled, _ := body[reqcommon.FieldStream].(bool)
	originalMaxTokens := resolveMaxTokens(body)

	span.SetAttributes(
		semconv.LLMDPDProxyChunkedDecodeChunkSize(s.config.DecodeChunkSize),
		semconv.LLMDPDProxyChunkedDecodeStreaming(streamingEnabled),
	)

	// The decoder echoes the final assistant message, which under chunking is the text of all earlier chunks.
	if echo, _ := body[reqcommon.FieldEcho].(bool); echo {
		s.logger.V(logging.DEBUG).Info("chunked decode: echo enabled, using regular decode")
		s.decoderProxy.ServeHTTP(w, r)
		return
	}

	// If the token budget fits within a single chunk, skip chunking entirely.
	if originalMaxTokens > 0 && originalMaxTokens <= s.config.DecodeChunkSize {
		s.logger.V(logging.DEBUG).Info("chunked decode: token budget <= chunk size, using regular decode",
			"maxTokens", originalMaxTokens, "chunkSize", s.config.DecodeChunkSize)
		s.decoderProxy.ServeHTTP(w, r)
		return
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
		s.decoderProxy.ServeHTTP(bw, cloneRequestWithBody(ctx, r, chunkBody))

		if isHTTPError(bw.statusCode) {
			s.logger.Error(fmt.Errorf("chunk %d failed with status %d", chunkIndex, bw.statusCode),
				"chunked decode chunk error", "statusCode", bw.statusCode)
			span.SetStatus(codes.Error, "chunk decode failed")
			maps.Copy(w.Header(), bw.headers)
			w.WriteHeader(bw.statusCode)
			WriteAll(w, bw.bodyBytes())
			return
		}

		var chunkResponse map[string]any
		if err := json.Unmarshal(bw.bodyBytes(), &chunkResponse); err != nil {
			s.logger.Error(err, "failed to unmarshal chunk response", "chunk", chunkIndex)
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

		if streamingEnabled {
			if err := emitSSEChunk(w, chunkResponse); err != nil {
				s.logger.Error(err, "failed to write SSE chunk to client")
				return
			}
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		} else {
			textAccum.WriteString(chunkText)
		}

		if finishReason != "" && finishReason != finishReasonLength {
			s.logger.V(logging.DEBUG).Info("chunked decode: terminal finish reason, stopping",
				"finishReason", finishReason, "chunks", chunkIndex)
			break
		}

		// Guard against infinite loop: if the chunk produced no tokens and no
		// text there is nothing to continue from.
		if chunkTokens == 0 && chunkText == "" {
			s.logger.Info("chunked decode: empty chunk with no tokens, stopping to avoid infinite loop",
				"chunk", chunkIndex)
			break
		}

		// Append the generated text to the request so the next chunk continues
		// from where this one left off.
		s.logger.V(logging.TRACE).Info("chunked decode: appending chunk text to request", "chunkText", chunkText)
		appendChunkToRequest(s.logger, body, chunkText)
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
				reqcommon.FieldUsage: cumulativeUsage,
				responseFieldChoices: []any{},
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

	if choices, ok := lastResponse[responseFieldChoices].([]any); ok && len(choices) > 0 {
		choice := maps.Clone(choices[0].(map[string]any))
		fullText := textAccum.String()
		if msg, ok := choice[responseFieldMessage].(map[string]any); ok {
			msg = maps.Clone(msg)
			msg[reqcommon.FieldContent] = fullText
			choice[responseFieldMessage] = msg
		}
		lastResponse[responseFieldChoices] = []any{choice}
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
	choices, ok := response[responseFieldChoices].([]any)
	if !ok || len(choices) == 0 {
		return ""
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return ""
	}
	reason, _ := choice[responseFieldFinishReason].(string)
	return reason
}

// emitSSEChunk writes one SSE data event to w, converting a buffered
// non-streaming response into a streaming delta event (chat or legacy).
func emitSSEChunk(w http.ResponseWriter, chunkResponse map[string]any) error {
	streamChunk := maps.Clone(chunkResponse)

	if choices, _ := chunkResponse[responseFieldChoices].([]any); len(choices) > 0 {
		streamChoices := make([]any, 0, len(choices))
		for _, c := range choices {
			choice, ok := c.(map[string]any)
			if !ok {
				continue
			}
			text := extractChoiceText(choice)
			streamChoice := map[string]any{
				responseFieldIndex:        choice[responseFieldIndex],
				responseFieldFinishReason: choice[responseFieldFinishReason],
			}
			streamChoice[responseFieldDelta] = map[string]any{reqcommon.FieldContent: text, reqcommon.FieldRole: roleAssistant}
			streamChoices = append(streamChoices, streamChoice)
		}
		streamChunk[responseFieldChoices] = streamChoices
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
	choices, ok := response[responseFieldChoices].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	choice, _ := choices[0].(map[string]any)
	return choice
}

// extractChoiceText returns the generated text from a choice's message.content.
func extractChoiceText(choice map[string]any) string {
	if msg, ok := choice[responseFieldMessage].(map[string]any); ok {
		if content, ok := msg[reqcommon.FieldContent].(string); ok {
			return content
		}
	}
	return ""
}

// appendChunkToRequest appends the generated text from a chunk to the request
// so the next chunk continues from where this one left off. The messages the
// client sent are appended to as raw bytes, keeping their key order intact
// across chunks.
func appendChunkToRequest(logger logr.Logger, req map[string]any, text string) {
	if text == "" {
		return
	}
	messages, err := requestMessages(req)
	if err != nil {
		logger.V(logging.DEBUG).Info("chunked decode: cannot read request messages", "error", err)
	}
	chunk, err := json.Marshal(map[string]any{
		reqcommon.FieldRole:    roleAssistant,
		reqcommon.FieldContent: text,
	})
	if err != nil {
		logger.V(logging.DEBUG).Info("chunked decode: cannot encode chunk text", "error", err)
		return
	}
	req[reqcommon.FieldMessages] = append(messages, chunk)
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
