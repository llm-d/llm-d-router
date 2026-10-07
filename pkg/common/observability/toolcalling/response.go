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

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

const (
	ResponseAttributeUpstreamToolCallPresent      = "llm_d.tool_calling.response.upstream_present"
	ResponseAttributeForwardedToolCallPresent     = "llm_d.tool_calling.response.forwarded_present"
	ResponseAttributeUpstreamDetectionIncomplete  = "llm_d.tool_calling.response.upstream_detection_incomplete"
	ResponseAttributeForwardedDetectionIncomplete = "llm_d.tool_calling.response.forwarded_detection_incomplete"
	maxResponseSSEEventBytes                      = 1 << 20
)

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
		attribute.Bool(ResponseAttributeUpstreamToolCallPresent, summary.UpstreamToolCallPresent),
		attribute.Bool(ResponseAttributeForwardedToolCallPresent, summary.ForwardedToolCallPresent),
		attribute.Bool(ResponseAttributeUpstreamDetectionIncomplete, summary.UpstreamDetectionIncomplete),
		attribute.Bool(ResponseAttributeForwardedDetectionIncomplete, summary.ForwardedDetectionIncomplete),
	}
}

// ResponseDetector observes one response body. For SSE responses it accepts
// arbitrarily chunked input and buffers only the current line and event data.
// Completed events are parsed and cleared.
type ResponseDetector struct {
	surface             reqcommon.APIType
	eventStream         bool
	toolCallPresent     bool
	detectionIncomplete bool
	discardingLine      bool
	discardingEvent     bool
	afterCR             bool
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
// Non-streaming responses are parsed once at endOfStream; streaming responses
// are parsed one completed SSE event at a time. A final unterminated payload
// is also inspected at endOfStream.
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

	if detector.eventStream {
		detector.observeSSE(chunk)
		if endOfStream && !detector.toolCallPresent && !detector.discardingEvent {
			detector.processSSELine()
			if !detector.toolCallPresent && !detector.discardingEvent {
				detector.processSSEEvent()
			}
		}
	} else if endOfStream {
		detector.observeJSON(chunk)
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

// Close clears buffered SSE data and prevents further observation.
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
	for _, b := range chunk {
		// Treat CRLF as one boundary even when it spans two body chunks.
		if detector.afterCR {
			detector.afterCR = false
			if b == '\n' {
				continue
			}
		}
		if b == '\n' || b == '\r' {
			detector.afterCR = b == '\r'
			detector.processSSELine()
			if detector.toolCallPresent {
				return
			}
			continue
		}
		if detector.discardingEvent {
			detector.discardingLine = true
			continue
		}
		if len(detector.line)+len(detector.eventData) == maxResponseSSEEventBytes {
			detector.discardSSEEvent()
			detector.discardingLine = true
			continue
		}
		detector.line = append(detector.line, b)
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
	for i := range detector.line {
		detector.line[i] = 0
	}
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

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	found, err := scanResponseJSONValue(decoder, responseJSONRoot, detector.surface)
	if err != nil {
		detector.detectionIncomplete = true
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
		found := false
		itemContext := responseArrayItemContext(context)
		for decoder.More() {
			itemFound, err := scanResponseJSONValue(decoder, itemContext, surface)
			if err != nil {
				return false, err
			}
			found = found || itemFound
		}
		_, err := decoder.Token()
		return found, err
	default:
		return false, nil
	}
}

func scanResponseJSONObject(decoder *json.Decoder, context responseJSONContext, surface reqcommon.APIType) (bool, error) {
	found := false
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
			found = found || present
			continue
		}
		if context == responseJSONMessage && key == "function_call" && surface == reqcommon.APITypeChatCompletions {
			present, err := scanNonEmptyJSONObject(decoder)
			if err != nil {
				return false, err
			}
			found = found || present
			continue
		}
		if context == responseJSONContentBlock && key == "type" {
			value, err := decoder.Token()
			if err != nil {
				return false, err
			}
			if surface == reqcommon.APITypeMessages && value == "tool_use" {
				found = true
			}
			continue
		}

		childContext := responseObjectChildContext(context, key)
		childFound, err := scanResponseJSONValue(decoder, childContext, surface)
		if err != nil {
			return false, err
		}
		found = found || childFound
	}
	_, err := decoder.Token()
	return found, err
}

func responseArrayItemContext(context responseJSONContext) responseJSONContext {
	switch context {
	case responseJSONChoice, responseJSONContentBlock:
		return context
	default:
		return responseJSONIgnore
	}
}

func responseObjectChildContext(context responseJSONContext, key string) responseJSONContext {
	switch context {
	case responseJSONRoot:
		switch key {
		case "choices":
			return responseJSONChoice
		case "content", "content_block":
			return responseJSONContentBlock
		}
	case responseJSONChoice:
		if key == "message" || key == "delta" {
			return responseJSONMessage
		}
	case responseJSONMessage:
		if key == "content" {
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
		return false, nil
	}
	nonEmpty := decoder.More()
	for decoder.More() {
		if _, err := scanResponseJSONValue(decoder, responseJSONIgnore, reqcommon.APITypeChatCompletions); err != nil {
			return false, err
		}
	}
	_, err = decoder.Token()
	return nonEmpty, err
}

func scanNonEmptyJSONObject(decoder *json.Decoder) (bool, error) {
	token, err := decoder.Token()
	if err != nil {
		return false, err
	}
	if token != json.Delim('{') {
		return false, nil
	}
	nonEmpty := decoder.More()
	for decoder.More() {
		if _, err := decoder.Token(); err != nil { // object key
			return false, err
		}
		if _, err := scanResponseJSONValue(decoder, responseJSONIgnore, reqcommon.APITypeChatCompletions); err != nil {
			return false, err
		}
	}
	_, err = decoder.Token()
	return nonEmpty, err
}
