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
	ResponseAttributeUpstreamToolCallPresent  = "llm_d.tool_calling.response.upstream_present"
	ResponseAttributeForwardedToolCallPresent = "llm_d.tool_calling.response.forwarded_present"
)

// ResponseSummary contains only bounded presence information. It deliberately
// does not retain response bodies, tool names, or tool arguments.
type ResponseSummary struct {
	ToolCallingRequested     bool
	UpstreamToolCallPresent  bool
	ForwardedToolCallPresent bool
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
	}
}

// ResponseDetector observes one response body. For SSE responses it accepts
// arbitrarily chunked input and keeps only the incomplete current line between
// calls; completed lines are parsed immediately and cleared.
type ResponseDetector struct {
	surface         reqcommon.APIType
	eventStream     bool
	toolCallPresent bool
	line            []byte
	finished        bool
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
// are parsed one completed SSE data line at a time.
func (detector *ResponseDetector) Observe(chunk []byte, endOfStream bool) bool {
	if detector == nil || detector.finished {
		return detector != nil && detector.toolCallPresent
	}

	if detector.eventStream {
		detector.observeSSE(chunk)
		if endOfStream && !detector.toolCallPresent {
			detector.processSSELine()
		}
	} else if endOfStream {
		detector.toolCallPresent = detectToolCallJSON(detector.surface, chunk)
	}

	if endOfStream {
		detector.clearLine()
		detector.finished = true
	}
	return detector.toolCallPresent
}

func (detector *ResponseDetector) observeSSE(chunk []byte) {
	for _, b := range chunk {
		if b == '\n' {
			detector.processSSELine()
			if detector.toolCallPresent {
				detector.clearLine()
				return
			}
			continue
		}
		detector.line = append(detector.line, b)
	}
}

func (detector *ResponseDetector) processSSELine() {
	line := bytes.TrimSuffix(detector.line, []byte{'\r'})
	if bytes.HasPrefix(line, []byte("data:")) {
		payload := bytes.TrimPrefix(line, []byte("data:"))
		payload = bytes.TrimPrefix(payload, []byte{' '})
		if !bytes.Equal(payload, []byte("[DONE]")) {
			detector.toolCallPresent = detectToolCallJSON(detector.surface, payload)
		}
	}
	detector.clearLine()
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

func detectToolCallJSON(surface reqcommon.APIType, body []byte) bool {
	if !json.Valid(body) {
		return false
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	found, err := scanResponseJSONValue(decoder, responseJSONRoot, surface)
	if err != nil {
		return false
	}
	if _, err := decoder.Token(); err != io.EOF {
		return false
	}
	return found
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
