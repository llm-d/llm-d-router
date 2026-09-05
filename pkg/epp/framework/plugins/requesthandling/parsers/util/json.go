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
		Body:                   &fwkrh.InferenceRequestBody{Payload: fwkrh.PayloadMap(payload), RawBody: data, Model: model, RenderRequest: true},
		SkipResponseProcessing: true,
	}, nil
}

// UnmarshalEnvelope keeps objects, arrays, and rawFields opaque so routing
// metadata edits cannot round-trip content through Go values.
func UnmarshalEnvelope(data []byte, rawFields ...string) (map[string]any, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		if fallbackErr := Unmarshal(data, &fields); fallbackErr != nil {
			return nil, fallbackErr
		}
		return nil, err
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
		var fallback map[string]json.RawMessage
		if fallbackErr := Unmarshal(data, &fallback); fallbackErr != nil {
			return nil, fallbackErr
		}
		return nil, err
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
