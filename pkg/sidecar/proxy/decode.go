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
	"io"
	"maps"
	"net/http"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/telemetry"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const (
	// finishReasonLength is the finish reason when max_tokens was reached.
	finishReasonLength = "length"

	sseDataPrefix = "data: "
	sseDone       = "data: [DONE]"

	responseFieldUsage            = "usage"
	responseFieldCompletionTokens = "completion_tokens"
	responseFieldPromptTokens     = "prompt_tokens"
	responseFieldTotalTokens      = "total_tokens"
	responseFieldMessage          = "message"
	responseFieldIndex            = "index"
	responseFieldDelta            = "delta"

	requestFieldMessages = "messages"
	requestFieldRole     = "role"
	requestFieldContent  = "content"

	roleAssistant = "assistant"
)

// dispatchDecode routes a fully-prepared decode request to either chunked
// decode or the regular decoder proxy. Chunked decode is only used for
// chat completions requests when s.config.DecodeChunkSize > 0.
// completionRequest is the already-parsed JSON map; callers that hold it
// should use this instead of calling s.decoderProxy directly.
func (s *Server) dispatchDecode(w http.ResponseWriter, r *http.Request, completionRequest map[string]any) {
	if s.config.DecodeChunkSize > 0 && r.URL.Path == ChatCompletionsPath {
		s.runChunkedDecodeFromMap(w, r, completionRequest)
		return
	}
	s.decoderProxy.ServeHTTP(w, r)
}

// runChunkedDecode reads and parses the body, then delegates to
// runChunkedDecodeFromMap.
func (s *Server) runChunkedDecode(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close() //nolint:errcheck
	original, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(err.Error())) //nolint:errcheck
		return
	}

	var completionRequest map[string]any
	if err := json.Unmarshal(original, &completionRequest); err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	s.runChunkedDecodeFromMap(w, cloneRequestWithBody(r, original), completionRequest)
}

// runChunkedDecodeFromMap executes chunked decode given an already-parsed completionRequest map.
// Non-streaming: accumulated chunks are reassembled into a single JSON response.
// Streaming: each chunk is re-emitted as an SSE event; [DONE] closes the stream.
func (s *Server) runChunkedDecodeFromMap(w http.ResponseWriter, r *http.Request, completionRequest map[string]any) {
	s.logger.V(4).Info("running chunked decode", "chunkSize", s.config.DecodeChunkSize)

	ctx, span := telemetry.Tracer().Start(r.Context(), "llm_d.pd_proxy.chunked_decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	streamingEnabled, _ := completionRequest[requestFieldStream].(bool)
	originalMaxTokens := resolveMaxTokens(completionRequest)

	span.SetAttributes(
		attribute.Int("llm_d.pd_proxy.chunked_decode.chunk_size", s.config.DecodeChunkSize),
		attribute.Bool("llm_d.pd_proxy.chunked_decode.streaming", streamingEnabled),
	)

	// If the token budget fits within a single chunk, skip chunking entirely.
	if originalMaxTokens > 0 && originalMaxTokens <= s.config.DecodeChunkSize {
		s.logger.V(4).Info("chunked decode: token budget <= chunk size, using regular decode",
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
				fmt.Fprintf(w, "%s\n\n", sseDone) //nolint:errcheck
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			return
		}

		remaining := remainingTokens(originalMaxTokens, totalTokens)
		if remaining == 0 {
			s.logger.V(4).Info("chunked decode: token budget exhausted", "totalTokens", totalTokens)
			break
		}

		chunkBudget := s.config.DecodeChunkSize
		if remaining > 0 && remaining < chunkBudget {
			chunkBudget = remaining
		}

		chunkReq := maps.Clone(completionRequest)
		chunkReq[requestFieldMaxTokens] = chunkBudget
		chunkReq[requestFieldMaxCompletionTokens] = chunkBudget
		chunkReq[requestFieldStream] = false
		delete(chunkReq, requestFieldStreamOptions)

		// From the second chunk onward: remove KV transfer params and instruct
		// to continue the last assistant message rather than start a new one.
		if chunkIndex > 0 {
			delete(chunkReq, requestFieldKVTransferParams)
			chunkReq[requestFieldContinueFinalMessage] = true
			chunkReq[requestFieldAddGenerationPrompt] = false
		}

		chunkBody, err := json.Marshal(chunkReq)
		if err != nil {
			if err := errorJSONInvalid(err, w); err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
			return
		}

		s.logger.V(4).Info("chunked decode: dispatching chunk",
			"chunk", chunkIndex, "chunkBudget", chunkBudget, "totalTokensSoFar", totalTokens)

		bw := &bufferedResponseWriter{}
		s.decoderProxy.ServeHTTP(bw, cloneRequestWithBody(r.WithContext(ctx), chunkBody))

		if isHTTPError(bw.statusCode) {
			s.logger.Error(fmt.Errorf("chunk %d failed with status %d", chunkIndex, bw.statusCode),
				"chunked decode chunk error", "statusCode", bw.statusCode)
			span.SetStatus(codes.Error, "chunk decode failed")
			maps.Copy(w.Header(), bw.headers)
			w.WriteHeader(bw.statusCode)
			w.Write(bw.bodyBytes()) //nolint:errcheck
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

		s.logger.V(4).Info("chunked decode: chunk complete", "chunkTokens", chunkTokens, "totalTokens", totalTokens)

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
			s.logger.V(4).Info("chunked decode: terminal finish reason, stopping",
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
		s.logger.V(5).Info("chunked decode: appending chunk text to request", "chunkText", chunkText)
		appendChunkToRequest(completionRequest, chunkText)
	}

	span.SetAttributes(
		attribute.Int("llm_d.pd_proxy.chunked_decode.chunks", chunkIndex),
		attribute.Int("llm_d.pd_proxy.chunked_decode.total_tokens", totalTokens),
		attribute.Float64("llm_d.pd_proxy.chunked_decode.duration_ms", float64(time.Since(decodeStart).Milliseconds())),
	)

	// Corrected cumulative usage: prompt_tokens from first chunk, completion_tokens summed.
	cumulativeUsage := map[string]any{
		responseFieldPromptTokens:     originalPromptTokens,
		responseFieldCompletionTokens: totalTokens,
		responseFieldTotalTokens:      originalPromptTokens + totalTokens,
	}

	if streamingEnabled {
		// Emit corrected cumulative usage as a final event before [DONE]
		// when multiple chunks were produced (individual chunk usage is stripped).
		if chunkIndex > 1 && lastResponse != nil {
			usageEvent := map[string]any{
				responseFieldUsage:   cumulativeUsage,
				responseFieldChoices: []any{},
			}
			if data, err := json.Marshal(usageEvent); err == nil {
				fmt.Fprintf(w, "%s%s\n\n", sseDataPrefix, data) //nolint:errcheck
			}
		}
		fmt.Fprintf(w, "%s\n\n", sseDone) //nolint:errcheck
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

	lastResponse[responseFieldUsage] = cumulativeUsage

	if choices, ok := lastResponse[responseFieldChoices].([]any); ok && len(choices) > 0 {
		choice := maps.Clone(choices[0].(map[string]any))
		fullText := textAccum.String()
		if msg, ok := choice[responseFieldMessage].(map[string]any); ok {
			msg = maps.Clone(msg)
			msg[requestFieldContent] = fullText
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
	w.Write(respBody) //nolint:errcheck
}

// resolveMaxTokens returns the effective max-tokens limit from the request map.
// Prefers max_completion_tokens (OpenAI v1) over max_tokens (legacy).
// Returns -1 when neither field is set (no explicit limit).
func resolveMaxTokens(req map[string]any) int {
	for _, field := range []string{requestFieldMaxCompletionTokens, requestFieldMaxTokens} {
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
	if usage, ok := response[responseFieldUsage].(map[string]any); ok {
		if n, ok := toInt(usage[responseFieldCompletionTokens]); ok {
			return n
		}
	}
	return 0
}

// extractPromptTokens returns prompt_tokens from the usage field, or 0.
func extractPromptTokens(response map[string]any) int {
	if usage, ok := response[responseFieldUsage].(map[string]any); ok {
		if n, ok := toInt(usage[responseFieldPromptTokens]); ok {
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
			streamChoice[responseFieldDelta] = map[string]any{requestFieldContent: text, requestFieldRole: roleAssistant}
			streamChoices = append(streamChoices, streamChoice)
		}
		streamChunk[responseFieldChoices] = streamChoices
	}

	// Omit usage on intermediate chunks (finish_reason == length or absent).
	if fr := extractFinishReason(chunkResponse); fr == "" || fr == finishReasonLength {
		delete(streamChunk, responseFieldUsage)
	}

	data, err := json.Marshal(streamChunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "%s%s\n\n", sseDataPrefix, data)
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
		if content, ok := msg[requestFieldContent].(string); ok {
			return content
		}
	}
	return ""
}

// appendChunkToRequest appends the generated text from a chunk to the request
// so the next chunk continues from where this one left off.
func appendChunkToRequest(req map[string]any, text string) {
	if text == "" {
		return
	}
	messages, _ := req[requestFieldMessages].([]any)
	req[requestFieldMessages] = append(messages, map[string]any{
		requestFieldRole:    roleAssistant,
		requestFieldContent: text,
	})
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
