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

package protocol

import (
	"bytes"
	"encoding/json/jsontext"
	"fmt"
	"strconv"

	"google.golang.org/protobuf/reflect/protoreflect"
)

// streamJSON adapts protojson output without building a second object tree.
func streamJSON(data []byte, descriptor protoreflect.MessageDescriptor) ([]byte, error) {
	// Input comes from protojson; the encoder checks names after flattening.
	decoder := jsontext.NewDecoder(bytes.NewReader(data), jsontext.AllowDuplicateNames(true))
	var output bytes.Buffer
	output.Grow(len(data))
	encoder := jsontext.NewEncoder(&output, jsontext.EscapeForHTML(true))
	if err := streamMessage(decoder, encoder, descriptor, false); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(output.Bytes(), []byte("\n")), nil
}

func streamMessage(decoder *jsontext.Decoder, encoder *jsontext.Encoder, descriptor protoreflect.MessageDescriptor, flatten bool) error {
	if descriptor.ParentFile().Package() == "google.protobuf" {
		return streamValue(decoder, encoder)
	}
	begin, err := decoder.ReadToken()
	if err != nil {
		return err
	}
	if !flatten {
		if err := encoder.WriteToken(begin); err != nil {
			return err
		}
	}
	plan := planFor(descriptor)
	for decoder.PeekKind() != '}' {
		name, err := decoder.ReadToken()
		if err != nil {
			return err
		}
		field := plan.byName[name.String()]
		if plan.features != nil && field != nil && field.Name() == "features" {
			if err := streamMessage(decoder, encoder, field.Message(), true); err != nil {
				return err
			}
			continue
		}
		if err := encoder.WriteToken(name); err != nil {
			return err
		}
		if err := streamField(decoder, encoder, field); err != nil {
			return err
		}
	}
	end, err := decoder.ReadToken()
	if err != nil {
		return err
	}
	if !flatten {
		return encoder.WriteToken(end)
	}
	return nil
}

func streamField(decoder *jsontext.Decoder, encoder *jsontext.Encoder, field protoreflect.FieldDescriptor) error {
	if field == nil {
		return streamValue(decoder, encoder)
	}
	if field.IsMap() || field.IsList() {
		end := jsontext.Kind(']')
		if field.IsMap() {
			end = '}'
		}
		begin, err := decoder.ReadToken()
		if err != nil {
			return err
		}
		if err := encoder.WriteToken(begin); err != nil {
			return err
		}
		for decoder.PeekKind() != end {
			valueField := field
			if field.IsMap() {
				name, err := decoder.ReadToken()
				if err != nil {
					return err
				}
				if err := encoder.WriteToken(name); err != nil {
					return err
				}
				valueField = field.MapValue()
			}
			if err := streamScalar(decoder, encoder, valueField); err != nil {
				return err
			}
		}
		close, err := decoder.ReadToken()
		if err != nil {
			return err
		}
		return encoder.WriteToken(close)
	}
	return streamScalar(decoder, encoder, field)
}

func streamScalar(decoder *jsontext.Decoder, encoder *jsontext.Encoder, field protoreflect.FieldDescriptor) error {
	if field.Kind() == protoreflect.MessageKind {
		return streamMessage(decoder, encoder, field.Message(), false)
	}
	switch field.Kind() {
	case protoreflect.DoubleKind, protoreflect.FloatKind:
		if decoder.PeekKind() == '"' {
			return fmt.Errorf("field %s: non-finite floating-point value", field.FullName())
		}
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind, protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		text, err := decoder.ReadToken()
		if err != nil {
			return err
		}
		if field.Kind() == protoreflect.Uint64Kind || field.Kind() == protoreflect.Fixed64Kind {
			value, err := strconv.ParseUint(text.String(), 10, 64)
			if err != nil {
				return err
			}
			return encoder.WriteToken(jsontext.Uint(value))
		}
		value, err := strconv.ParseInt(text.String(), 10, 64)
		if err != nil {
			return err
		}
		return encoder.WriteToken(jsontext.Int(value))
	}
	value, err := decoder.ReadToken()
	if err != nil {
		return err
	}
	return encoder.WriteToken(value)
}

func streamValue(decoder *jsontext.Decoder, encoder *jsontext.Encoder) error {
	value, err := decoder.ReadValue()
	if err != nil {
		return err
	}
	return encoder.WriteValue(value)
}
