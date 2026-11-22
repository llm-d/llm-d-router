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
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"

	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// Only Read owns observation state. Delegated Close can unblock an in-flight
// transport read without racing with the extractor or changing cleanup.
type toolCallingPassthroughBody struct {
	io.ReadCloser
	stream *toolCallingJSONStream
	length int64
	read   int64
	span   trace.Span
}

func observeToolCallingPassthrough(r *http.Request, api reqcommon.APIType) {
	if !toolCallingTelemetryEnabled(r.Context()) || api != reqcommon.APITypeChatCompletions && api != reqcommon.APITypeMessages || r.Body == nil || r.Body == http.NoBody {
		return
	}
	span, _ := r.Context().Value(toolCallingRequestSpanKey).(trace.Span)
	r.Body = &toolCallingPassthroughBody{
		ReadCloser: r.Body, stream: newToolCallingJSONStream(api), length: r.ContentLength,
		span: span,
	}
}

func (b *toolCallingPassthroughBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if b.stream == nil {
		return n, err
	}
	b.read += int64(n)
	if err != nil && err != io.EOF || b.length > 0 && b.read > b.length {
		b.stream = nil
		return n, err
	}
	b.stream.feed(p[:n])
	if b.stream.invalid {
		b.stream = nil
		return n, err
	}
	// Known-length HTTP transports need not request EOF after the final byte.
	if err != io.EOF && (b.length <= 0 || b.read != b.length) {
		return n, err
	}
	stream := b.stream
	b.stream = nil
	if b.length > 0 && b.read != b.length || len(stream.fields) == 0 {
		return n, err
	}
	raw, ok := stream.finish()
	if !ok {
		return n, err
	}
	snapshot, captureErr := toolcalling.CaptureValidatedRequestJSON(stream.api, raw)
	if captureErr != nil {
		return n, err
	}
	// Passthrough uses the inbound bytes unchanged; one snapshot suffices.
	statuses, _ := toolcalling.CompareRequests(snapshot, snapshot)
	metrics.RecordToolCallingFieldStatuses(statuses)
	if b.span != nil && b.span.IsRecording() {
		b.span.SetAttributes(snapshot.SpanAttributes(statuses)...)
	}
	return n, err
}

type toolJSONState uint8

const (
	toolJSONRoot toolJSONState = iota
	toolJSONDone
	toolJSONObjectFirstKey
	toolJSONObjectKey
	toolJSONObjectColon
	toolJSONObjectValue
	toolJSONObjectComma
	toolJSONArrayFirstValue
	toolJSONArrayValue
	toolJSONArrayComma
)

type toolJSONNumberState uint8

const (
	toolJSONNumberMinus toolJSONNumberState = iota
	toolJSONNumberZero
	toolJSONNumberInteger
	toolJSONNumberFractionStart
	toolJSONNumberFraction
	toolJSONNumberExponentSign
	toolJSONNumberExponentStart
	toolJSONNumberExponent
)

// toolCallingJSONStream validates an object incrementally and retains only
// supported top-level fields. Discarded prompt strings and numbers require no
// value-sized buffers. Nesting uses the same 10,000-level limit as encoding/json.
type toolCallingJSONStream struct {
	api       reqcommon.APIType
	stack     []toolJSONState
	invalid   bool
	mode      byte
	escaped   bool
	unicode   int
	number    toolJSONNumberState
	literal   string
	read      int
	key       []byte
	keyTooBig bool
	field     toolcalling.Field
	capture   []byte
	fields    map[string]json.RawMessage
}

func newToolCallingJSONStream(api reqcommon.APIType) *toolCallingJSONStream {
	return &toolCallingJSONStream{
		api: api, stack: []toolJSONState{toolJSONRoot},
		invalid: api != reqcommon.APITypeChatCompletions && api != reqcommon.APITypeMessages,
	}
}

func (s *toolCallingJSONStream) fail() {
	s.invalid = true
	s.capture, s.key, s.stack, s.fields = nil, nil, nil, nil
}

func (s *toolCallingJSONStream) feed(data []byte) {
	for len(data) > 0 {
		if s.invalid {
			return
		}
		// Ordinary string bytes preserve parser state and can be copied together.
		if s.mode == 's' && (s.key == nil || s.keyTooBig) {
			if !s.escaped && s.unicode == 0 {
				i := 0
				if len(data) >= 2048 {
					i = indexToolJSONStringSpecial(data)
					if i < 0 {
						i = len(data)
					}
				} else {
					for i < len(data) && data[i] >= 0x20 && data[i] != '"' && data[i] != '\\' {
						i++
					}
				}
				if s.capture != nil {
					s.capture = append(s.capture, data[:i]...)
				}
				data = data[i:]
				if len(data) == 0 {
					return
				}
			}
			if s.capture != nil {
				s.capture = append(s.capture, data[0])
			}
			s.consumeString(data[0])
		} else {
			s.consume(data[0])
		}
		data = data[1:]
	}
}

// Scan long strings eight bytes at a time; candidates are checked bytewise
// because the zero-byte test may also match a neighboring byte after a borrow.
func indexToolJSONStringSpecial(data []byte) int {
	const controlMask uint64 = 0xe0e0e0e0e0e0e0e0
	const quotes uint64 = 0x2222222222222222
	const slashes uint64 = 0x5c5c5c5c5c5c5c5c
	containsZeroByte := func(x uint64) bool {
		return (x-0x0101010101010101)&^x&0x8080808080808080 != 0
	}
	i := 0
	for ; i+8 <= len(data); i += 8 {
		word := binary.LittleEndian.Uint64(data[i:])
		if containsZeroByte(word&controlMask) || containsZeroByte(word^quotes) || containsZeroByte(word^slashes) {
			for j, c := range data[i : i+8] {
				if c < 0x20 || c == '"' || c == '\\' {
					return i + j
				}
			}
		}
	}
	for j, c := range data[i:] {
		if c < 0x20 || c == '"' || c == '\\' {
			return i + j
		}
	}
	return -1
}

func (s *toolCallingJSONStream) consume(b byte) {
	// A number's delimiter belongs to the enclosing object or array.
	if s.mode == 'n' && (b == ',' || b == ']' || b == '}' || isToolJSONSpace(b)) {
		if !s.finishNumber() {
			s.fail()
			return
		}
	}
	if s.capture != nil {
		s.capture = append(s.capture, b)
	}
	switch s.mode {
	case 's':
		if s.key != nil && !s.keyTooBig {
			// Even fully escaped spellings of the longest supported key fit.
			if len(s.key) < 6*len(toolcalling.FieldParallelToolCalls)+2 {
				s.key = append(s.key, b)
			} else {
				s.keyTooBig = true
			}
		}
		s.consumeString(b)
		return
	case 'n':
		s.consumeNumber(b)
		return
	case 'l':
		if b != s.literal[s.read] {
			s.fail()
			return
		}
		s.read++
		if s.read == len(s.literal) {
			s.mode = 0
			s.finishValue()
		}
		return
	}
	if isToolJSONSpace(b) {
		return
	}
	state := s.stack[len(s.stack)-1]
	switch state {
	case toolJSONObjectFirstKey, toolJSONObjectKey:
		switch {
		case b == '}' && state == toolJSONObjectFirstKey:
			s.closeContainer()
		case b == '"':
			s.mode = 's'
			if len(s.stack) == 2 {
				s.key = []byte{'"'}
				s.keyTooBig = false
			}
		default:
			s.fail()
		}
	case toolJSONObjectColon:
		if b != ':' {
			s.fail()
		} else {
			s.stack[len(s.stack)-1] = toolJSONObjectValue
		}
	case toolJSONObjectComma, toolJSONArrayComma:
		switch {
		case state == toolJSONObjectComma && b == '}' || state == toolJSONArrayComma && b == ']':
			s.closeContainer()
		case b == ',':
			if state == toolJSONObjectComma {
				s.stack[len(s.stack)-1] = toolJSONObjectKey
			} else {
				s.stack[len(s.stack)-1] = toolJSONArrayValue
			}
		default:
			s.fail()
		}
	case toolJSONArrayFirstValue:
		if b == ']' {
			s.closeContainer()
			return
		}
		s.beginValue(b)
	case toolJSONRoot, toolJSONObjectValue, toolJSONArrayValue:
		s.beginValue(b)
	default:
		s.fail()
	}
}

func (s *toolCallingJSONStream) beginValue(b byte) {
	if len(s.stack) == 1 && b != '{' {
		s.fail()
		return
	}
	if len(s.stack) == 2 && s.field != "" && s.stack[1] == toolJSONObjectValue {
		s.capture = []byte{b}
	}
	switch {
	case b == '{' || b == '[':
		if len(s.stack) > 10000 {
			s.fail()
			return
		}
		state := toolJSONObjectFirstKey
		if b == '[' {
			state = toolJSONArrayFirstValue
		}
		s.stack = append(s.stack, state)
	case b == '"':
		s.mode = 's'
	case b == 't' || b == 'f' || b == 'n':
		s.mode, s.read = 'l', 1
		s.literal = "null"
		switch b {
		case 't':
			s.literal = "true"
		case 'f':
			s.literal = "false"
		}
	case b == '-' || b >= '0' && b <= '9':
		s.mode = 'n'
		s.number = toolJSONNumberInteger
		switch b {
		case '-':
			s.number = toolJSONNumberMinus
		case '0':
			s.number = toolJSONNumberZero
		}
	default:
		s.fail()
	}
}

func (s *toolCallingJSONStream) consumeString(b byte) {
	switch {
	case s.unicode > 0:
		if (b < '0' || b > '9') && (b < 'a' || b > 'f') && (b < 'A' || b > 'F') {
			s.fail()
			return
		}
		s.unicode--
	case s.escaped:
		s.escaped = false
		switch b {
		case 'u':
			s.unicode = 4
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		default:
			s.fail()
		}
	case b == '\\':
		s.escaped = true
	case b < 0x20:
		s.fail()
	case b == '"':
		s.mode = 0
		state := s.stack[len(s.stack)-1]
		if state == toolJSONObjectFirstKey || state == toolJSONObjectKey {
			if len(s.stack) == 2 {
				s.field = ""
				if !s.keyTooBig {
					var key string
					if json.Unmarshal(s.key, &key) == nil {
						field := toolcalling.Field(key)
						if field == toolcalling.FieldTools || field == toolcalling.FieldToolChoice ||
							s.api == reqcommon.APITypeChatCompletions && (field == toolcalling.FieldParallelToolCalls || field == toolcalling.FieldResponseFormat) {
							s.field = field
						}
					}
				}
				s.key = nil
			}
			s.stack[len(s.stack)-1] = toolJSONObjectColon
		} else {
			s.finishValue()
		}
	}
}

func (s *toolCallingJSONStream) consumeNumber(b byte) {
	digit := b >= '0' && b <= '9'
	switch {
	case s.number == toolJSONNumberMinus && b == '0':
		s.number = toolJSONNumberZero
	case s.number == toolJSONNumberMinus && b >= '1' && b <= '9':
		s.number = toolJSONNumberInteger
	case s.number == toolJSONNumberInteger && digit:
	case (s.number == toolJSONNumberInteger || s.number == toolJSONNumberZero) && b == '.':
		s.number = toolJSONNumberFractionStart
	case s.number == toolJSONNumberFractionStart && digit:
		s.number = toolJSONNumberFraction
	case s.number == toolJSONNumberFraction && digit:
	case (s.number == toolJSONNumberInteger || s.number == toolJSONNumberZero || s.number == toolJSONNumberFraction) && (b == 'e' || b == 'E'):
		s.number = toolJSONNumberExponentSign
	case s.number == toolJSONNumberExponentSign && (b == '+' || b == '-'):
		s.number = toolJSONNumberExponentStart
	case (s.number == toolJSONNumberExponentSign || s.number == toolJSONNumberExponentStart) && digit:
		s.number = toolJSONNumberExponent
	case s.number == toolJSONNumberExponent && digit:
	default:
		s.fail()
	}
}

func (s *toolCallingJSONStream) finishNumber() bool {
	switch s.number {
	case toolJSONNumberZero, toolJSONNumberInteger, toolJSONNumberFraction, toolJSONNumberExponent:
		s.mode = 0
		s.finishValue()
		return true
	default:
		return false
	}
}

func (s *toolCallingJSONStream) closeContainer() {
	s.stack = s.stack[:len(s.stack)-1]
	s.finishValue()
}

func (s *toolCallingJSONStream) finishValue() {
	if len(s.stack) == 2 && s.capture != nil {
		if s.fields == nil {
			s.fields = make(map[string]json.RawMessage)
		}
		s.fields[string(s.field)] = s.capture
		s.capture, s.field = nil, ""
	}
	state := s.stack[len(s.stack)-1]
	switch state {
	case toolJSONRoot:
		s.stack[0] = toolJSONDone
	case toolJSONObjectValue:
		s.stack[len(s.stack)-1] = toolJSONObjectComma
	case toolJSONArrayFirstValue, toolJSONArrayValue:
		s.stack[len(s.stack)-1] = toolJSONArrayComma
	default:
		s.fail()
	}
}

func (s *toolCallingJSONStream) finish() ([]byte, bool) {
	if s.invalid || s.mode != 0 || len(s.stack) != 1 || s.stack[0] != toolJSONDone {
		return nil, false
	}
	if s.fields == nil {
		return []byte("{}"), true
	}
	raw, err := json.Marshal(s.fields)
	return raw, err == nil
}

func isToolJSONSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\r' || b == '\n'
}
