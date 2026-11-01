/*
Copyright 2026 The Kubernetes Authors.

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

package parserutil

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"unicode/utf8"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

var ErrTrailingData = errors.New("unexpected trailing data after JSON value")

// ParseRenderRequest reads only the envelope required for model routing.
func ParseRenderRequest(data []byte) (*fwkrh.ParseResult, error) {
	payload, err := UnmarshalEnvelope(data, "prompt", "system")
	if err != nil {
		return nil, err
	}
	model, _ := payload["model"].(string)
	return &fwkrh.ParseResult{
		Body:                   &fwkrequest.InferenceRequestBody{Payload: fwkrequest.PayloadMap(payload), RawBody: data, Model: model, RenderRequest: true},
		SkipResponseProcessing: true,
	}, nil
}

// UnmarshalEnvelope keeps objects, arrays, and rawFields opaque so routing
// metadata edits cannot round-trip content through Go values.
func UnmarshalEnvelope(data []byte, rawFields ...string) (map[string]any, error) {
	fields, ok := splitObject(data)
	if !ok {
		if err := json.Unmarshal(data, &fields); err != nil {
			if fallbackErr := Unmarshal(data, &fields); fallbackErr != nil {
				return nil, fallbackErr
			}
		}
	}
	result := make(map[string]any, len(fields))
	for key, raw := range fields {
		if slices.Contains(rawFields, key) || raw[0] == '{' || raw[0] == '[' {
			result[key] = raw
			continue
		}
		var value any
		decode := Unmarshal
		if raw[0] == '"' {
			decode = json.Unmarshal
		}
		if err := decode(raw, &value); err != nil {
			return nil, err
		}
		result[key] = value
	}
	return result, nil
}

// ObjectField returns payload[key] as a JSON object. UnmarshalEnvelope keeps
// object fields as json.RawMessage, so such a value is decoded with Unmarshal;
// a caller that edits the result must store it back under key.
func ObjectField(payload map[string]any, key string) (map[string]any, bool) {
	switch value := payload[key].(type) {
	case map[string]any:
		return value, true
	case json.RawMessage:
		var decoded map[string]any
		if err := Unmarshal(value, &decoded); err == nil && decoded != nil {
			return decoded, true
		}
	}
	return nil, false
}

// splitObject returns the members of a valid JSON object as slices of data, so the
// envelope does not copy message content. Each slice has its capacity capped at its
// length, so an append cannot write into data. It reports false for invalid JSON and
// for any top-level value that is not an object; json.Unmarshal handles those.
func splitObject(data []byte) (map[string]json.RawMessage, bool) {
	if !json.Valid(data) {
		return nil, false
	}
	i := skipSpace(data, 0)
	if data[i] != '{' {
		return nil, false
	}
	fields := make(map[string]json.RawMessage)
	i = skipSpace(data, i+1)
	if data[i] == '}' {
		return fields, true
	}
	for {
		keyEnd := stringEnd(data, i)
		key, ok := objectKey(data[i:keyEnd])
		if !ok {
			return nil, false
		}
		i = skipSpace(data, skipSpace(data, keyEnd)+1) // past ':'
		valueEnd := jsonValueEnd(data, i)
		fields[key] = json.RawMessage(data[i:valueEnd:valueEnd])
		i = skipSpace(data, valueEnd)
		if data[i] == '}' {
			return fields, true
		}
		i = skipSpace(data, i+1) // past ','
	}
}

// objectKey decodes a quoted key. A key that is plain ASCII without escapes is
// used as is; any other key goes through json.Unmarshal, which unescapes it and
// replaces invalid UTF-8.
func objectKey(quoted []byte) (string, bool) {
	plain := true
	for _, c := range quoted {
		if c == '\\' || c >= utf8.RuneSelf {
			plain = false
			break
		}
	}
	if plain {
		return string(quoted[1 : len(quoted)-1]), true
	}
	var key string
	if err := json.Unmarshal(quoted, &key); err != nil {
		return "", false
	}
	return key, true
}

// The scanners below assume data is valid JSON and i is at the start of a token.

func skipSpace(data []byte, i int) int {
	for i < len(data) {
		switch data[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// stringEnd returns the index after the closing quote of the string at i.
func stringEnd(data []byte, i int) int {
	for j := i + 1; ; j++ {
		switch data[j] {
		case '\\':
			j++
		case '"':
			return j + 1
		}
	}
}

// jsonValueEnd returns the index after the value at i.
func jsonValueEnd(data []byte, i int) int {
	switch data[i] {
	case '"':
		return stringEnd(data, i)
	case '{', '[':
		depth := 0
		for j := i; ; j++ {
			switch data[j] {
			case '"':
				j = stringEnd(data, j) - 1
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					return j + 1
				}
			}
		}
	default:
		j := i
		for j < len(data) {
			switch data[j] {
			case ',', '}', ']', ' ', '\t', '\n', '\r':
				return j
			}
			j++
		}
		return j
	}
}

// Unmarshal decodes one JSON value, preserves numbers as json.Number, and rejects trailing data.
func Unmarshal(data []byte, v any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(v); err != nil {
		return err
	}
	switch err := decoder.Decode(&struct{}{}); {
	case errors.Is(err, io.EOF):
		return nil
	case err == nil:
		return ErrTrailingData
	default:
		return fmt.Errorf("%w: %v", ErrTrailingData, err)
	}
}

// UnmarshalMapWithRawField decodes a JSON object while preserving one field's
// original JSON representation. All other numbers are preserved as json.Number.
func UnmarshalMapWithRawField(data []byte, rawField string) (map[string]any, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		if fallbackErr := Unmarshal(data, &fields); fallbackErr != nil {
			return nil, fallbackErr
		}
	}

	result := make(map[string]any, len(fields))
	for key, raw := range fields {
		if key == rawField {
			result[key] = raw
			continue
		}

		var value any
		if err := Unmarshal(raw, &value); err != nil {
			return nil, err
		}
		result[key] = value
	}
	return result, nil
}
