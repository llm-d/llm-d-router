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

package toolcalling

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"go.opentelemetry.io/otel/attribute"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

const maxResponseSSEEventBytes = 1 << 20

// ResponseSummary contains only bounded presence information. It deliberately
// does not retain response bodies, tool names, or tool arguments.
type ResponseSummary struct {
	ToolCallingRequested         bool
	UpstreamToolCallPresent      bool
	ForwardedToolCallPresent     bool
	UpstreamDetectionIncomplete  bool
	ForwardedDetectionIncomplete bool
}

// SpanAttributes returns response presence attributes only for requests that
// contained tool-calling fields. This keeps non-tool-calling spans unchanged.
func (summary ResponseSummary) SpanAttributes() []attribute.KeyValue {
	if !summary.ToolCallingRequested {
		return nil
	}

	return []attribute.KeyValue{
		semconv.LLMDToolCallingResponseUpstreamPresentKey.Bool(summary.UpstreamToolCallPresent),
		semconv.LLMDToolCallingResponseForwardedPresentKey.Bool(summary.ForwardedToolCallPresent),
		semconv.LLMDToolCallingResponseUpstreamDetectionIncompleteKey.Bool(summary.UpstreamDetectionIncomplete),
		semconv.LLMDToolCallingResponseForwardedDetectionIncompleteKey.Bool(summary.ForwardedDetectionIncomplete),
	}
}

// ResponseDetector observes one response body. It buffers JSON until completion.
// For SSE it buffers only the current line and event data, clearing completed events.
type ResponseDetector struct {
	surface             reqcommon.APIType
	eventStream         bool
	toolCallPresent     bool
	detectionIncomplete bool
	discardingLine      bool
	discardingEvent     bool
	afterCR             bool
	bomChecked          bool
	line                []byte
	eventData           []byte
	finished            bool
}

// NewResponseDetector creates a detector for a supported response API surface.
func NewResponseDetector(surface reqcommon.APIType, eventStream bool) (*ResponseDetector, error) {
	if surface != reqcommon.APITypeChatCompletions && surface != reqcommon.APITypeMessages {
		return nil, fmt.Errorf("unsupported response API type: %s", surface)
	}
	return &ResponseDetector{surface: surface, eventStream: eventStream}, nil
}

// Observe adds a body chunk and returns whether a tool call has been observed.
// JSON responses are parsed once at endOfStream; SSE responses are parsed one
// completed event at a time. Unfinished SSE data events are marked incomplete.
func (detector *ResponseDetector) Observe(chunk []byte, endOfStream bool) bool {
	if detector == nil {
		return false
	}
	if detector.finished {
		return detector.toolCallPresent
	}
	if detector.toolCallPresent {
		if endOfStream {
			detector.Close()
		}
		return true
	}

	switch {
	case detector.eventStream:
		detector.observeSSE(chunk)
		if endOfStream && !detector.toolCallPresent {
			if len(detector.line) > 0 {
				detector.processSSELine()
			}
			detector.detectionIncomplete = detector.detectionIncomplete || len(detector.eventData) > 0
		}
	case !endOfStream:
		detector.eventData = append(detector.eventData, chunk...)
	case len(detector.eventData) == 0:
		detector.observeJSON(chunk)
	default:
		detector.eventData = append(detector.eventData, chunk...)
		detector.observeJSON(detector.eventData)
	}

	if endOfStream {
		detector.Close()
	}
	return detector.toolCallPresent
}

// DetectionIncomplete reports whether a payload could not be fully inspected.
func (detector *ResponseDetector) DetectionIncomplete() bool {
	return detector != nil && detector.detectionIncomplete
}

// Close clears buffered response data and prevents further observation.
// Process calls it when the response lifecycle ends, including cancellation
// or errors before the stream reaches end-of-stream.
func (detector *ResponseDetector) Close() {
	if detector == nil {
		return
	}
	detector.clearLine()
	detector.clearEventData()
	detector.discardingLine = false
	detector.discardingEvent = false
	detector.afterCR = false
	detector.finished = true
}

func (detector *ResponseDetector) observeSSE(chunk []byte) {
	const utf8BOM = "\xef\xbb\xbf"
	for len(chunk) > 0 {
		// Treat CRLF as one boundary even when it spans two body chunks.
		if detector.afterCR {
			detector.afterCR = false
			if chunk[0] == '\n' {
				chunk = chunk[1:]
				continue
			}
		}
		// Append line segments in bulk to avoid per-byte work on large events.
		end := bytes.IndexAny(chunk, "\r\n")
		line := chunk
		if end >= 0 {
			line = chunk[:end]
		}
		if detector.discardingEvent {
			detector.discardingLine = detector.discardingLine || len(line) > 0
		} else {
			if !detector.bomChecked {
				// Check the BOM prefix separately because its bytes may span chunks.
				prefix := min(len(utf8BOM)-len(detector.line), len(line))
				detector.line = append(detector.line, line[:prefix]...)
				line = line[prefix:]
				if len(detector.line) == len(utf8BOM) {
					detector.bomChecked = true
					if bytes.Equal(detector.line, []byte(utf8BOM)) {
						detector.clearLine()
					}
				}
			}
			if len(detector.line)+len(detector.eventData)+len(line) > maxResponseSSEEventBytes {
				detector.discardSSEEvent()
				detector.discardingLine = true
			} else {
				detector.line = append(detector.line, line...)
			}
		}
		if end < 0 {
			return
		}
		b := chunk[end]
		chunk = chunk[end+1:]
		detector.bomChecked = true
		detector.afterCR = b == '\r'
		detector.processSSELine()
		if detector.toolCallPresent {
			return
		}
	}
}

func (detector *ResponseDetector) processSSELine() {
	if detector.discardingEvent {
		// Discard the entire event so its remaining lines cannot report presence.
		if !detector.discardingLine {
			detector.discardingEvent = false
		}
		detector.discardingLine = false
		return
	}
	if len(detector.line) == 0 {
		detector.processSSEEvent()
		return
	}
	payload, isData := bytes.CutPrefix(detector.line, []byte("data:"))
	if isData || bytes.Equal(detector.line, []byte("data")) {
		if !isData {
			payload = nil
		}
		payload = bytes.TrimPrefix(payload, []byte{' '})
		if len(detector.eventData)+len(payload)+1 > maxResponseSSEEventBytes {
			detector.discardSSEEvent()
		} else {
			detector.eventData = append(detector.eventData, payload...)
			detector.eventData = append(detector.eventData, '\n')
		}
	}
	detector.clearLine()
}

func (detector *ResponseDetector) processSSEEvent() {
	if len(detector.eventData) > 0 {
		payload := detector.eventData[:len(detector.eventData)-1]
		if len(bytes.TrimSpace(payload)) > 0 && !bytes.Equal(payload, []byte("[DONE]")) {
			detector.observeJSON(payload)
		}
	}
	detector.clearEventData()
}

func (detector *ResponseDetector) discardSSEEvent() {
	detector.clearLine()
	detector.clearEventData()
	detector.line = nil
	detector.eventData = nil
	detector.discardingEvent = true
	detector.detectionIncomplete = true
}

func (detector *ResponseDetector) clearEventData() {
	clear(detector.eventData)
	detector.eventData = detector.eventData[:0]
}

func (detector *ResponseDetector) clearLine() {
	clear(detector.line)
	detector.line = detector.line[:0]
}

type responseJSONContext uint8

const (
	responseJSONIgnore responseJSONContext = iota
	responseJSONRoot
	responseJSONChoice
	responseJSONMessage
	responseJSONContentBlock
)

func (detector *ResponseDetector) observeJSON(body []byte) {
	if !json.Valid(body) {
		detector.detectionIncomplete = true
		return
	}
	hasToolMarker := false
	switch detector.surface {
	case reqcommon.APITypeChatCompletions:
		hasToolMarker = bytes.Contains(body, []byte("tool_calls")) || bytes.Contains(body, []byte("function_call"))
	case reqcommon.APITypeMessages:
		hasToolMarker = bytes.Contains(body, []byte("tool_use"))
	}
	// Only a literal marker or Unicode escape can require a token walk.
	if !hasToolMarker && !bytes.Contains(body, []byte(`\u`)) {
		return
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	found, err := scanResponseJSONValue(decoder, responseJSONRoot, detector.surface)
	if err != nil {
		detector.detectionIncomplete = true
		return
	}
	if found {
		// The whole payload was validated above, so the walk can stop here.
		detector.toolCallPresent = true
		return
	}
	if _, err := decoder.Token(); err != io.EOF {
		detector.detectionIncomplete = true
		return
	}
	detector.toolCallPresent = found
}

func scanResponseJSONValue(decoder *json.Decoder, context responseJSONContext, surface reqcommon.APIType) (bool, error) {
	token, err := decoder.Token()
	if err != nil {
		return false, err
	}

	delim, isDelim := token.(json.Delim)
	if !isDelim {
		return false, nil
	}
	switch delim {
	case '{':
		return scanResponseJSONObject(decoder, context, surface)
	case '[':
		itemContext := responseArrayItemContext(context)
		for decoder.More() {
			itemFound, err := scanResponseJSONValue(decoder, itemContext, surface)
			if err != nil {
				return false, err
			}
			if itemFound {
				return true, nil
			}
		}
		_, err := decoder.Token()
		return false, err
	default:
		return false, nil
	}
}

func scanResponseJSONObject(decoder *json.Decoder, context responseJSONContext, surface reqcommon.APIType) (bool, error) {
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return false, err
		}
		key, ok := token.(string)
		if !ok {
			return false, nil
		}

		if context == responseJSONMessage && key == "tool_calls" && surface == reqcommon.APITypeChatCompletions {
			present, err := scanNonEmptyJSONArray(decoder)
			if err != nil {
				return false, err
			}
			if present {
				return true, nil
			}
			continue
		}
		if context == responseJSONMessage && key == "function_call" && surface == reqcommon.APITypeChatCompletions {
			present, err := scanNonEmptyJSONObject(decoder)
			if err != nil {
				return false, err
			}
			if present {
				return true, nil
			}
			continue
		}
		if context == responseJSONContentBlock && key == "type" {
			value, err := decoder.Token()
			if err != nil {
				return false, err
			}
			if surface == reqcommon.APITypeMessages && value == "tool_use" {
				return true, nil
			}
			if err := skipResponseJSONComposite(decoder, value); err != nil {
				return false, err
			}
			continue
		}

		childContext := responseObjectChildContext(context, key, surface)
		childFound, err := scanResponseJSONValue(decoder, childContext, surface)
		if err != nil {
			return false, err
		}
		if childFound {
			return true, nil
		}
	}
	_, err := decoder.Token()
	return false, err
}

func responseArrayItemContext(context responseJSONContext) responseJSONContext {
	switch context {
	case responseJSONChoice, responseJSONContentBlock:
		return context
	default:
		return responseJSONIgnore
	}
}

func responseObjectChildContext(context responseJSONContext, key string, surface reqcommon.APIType) responseJSONContext {
	switch context {
	case responseJSONRoot:
		switch key {
		case "choices":
			return responseJSONChoice
		case "message":
			if surface == reqcommon.APITypeMessages {
				return responseJSONMessage
			}
		case "content", "content_block":
			return responseJSONContentBlock
		}
	case responseJSONChoice:
		if key == "message" || key == "delta" {
			return responseJSONMessage
		}
	case responseJSONMessage:
		if key == "content" && surface == reqcommon.APITypeMessages {
			return responseJSONContentBlock
		}
	case responseJSONContentBlock:
		if key == "content_block" {
			return responseJSONContentBlock
		}
	}
	return responseJSONIgnore
}

func scanNonEmptyJSONArray(decoder *json.Decoder) (bool, error) {
	token, err := decoder.Token()
	if err != nil {
		return false, err
	}
	if token != json.Delim('[') {
		return false, skipResponseJSONComposite(decoder, token)
	}
	if decoder.More() {
		return true, nil
	}
	_, err = decoder.Token()
	return false, err
}

func scanNonEmptyJSONObject(decoder *json.Decoder) (bool, error) {
	token, err := decoder.Token()
	if err != nil {
		return false, err
	}
	if token != json.Delim('{') {
		return false, skipResponseJSONComposite(decoder, token)
	}
	if decoder.More() {
		return true, nil
	}
	_, err = decoder.Token()
	return false, err
}

func skipResponseJSONComposite(decoder *json.Decoder, token json.Token) error {
	delim, ok := token.(json.Delim)
	if !ok || (delim != '{' && delim != '[') {
		return nil
	}
	depth := 1
	for depth > 0 {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		if delim, ok := token.(json.Delim); ok {
			switch delim {
			case '{', '[':
				depth++
			case '}', ']':
				depth--
			}
		}
	}
	return nil
}
