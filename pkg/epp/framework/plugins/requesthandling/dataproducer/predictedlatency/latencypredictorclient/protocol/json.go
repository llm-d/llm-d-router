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

// Package protocol defines the latency predictor's JSON/HTTP contract.
package protocol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// Marshal preserves flat snake_case fields and numeric integers on HTTP.
// Optional fields are emitted only when present; application validation is separate.
func Marshal(message proto.Message) ([]byte, error) {
	if message == nil || !message.ProtoReflect().IsValid() {
		return nil, errors.New("cannot marshal a nil latency protocol message")
	}
	data, err := (protojson.MarshalOptions{UseProtoNames: true, EmitDefaultValues: true}).Marshal(message)
	if err != nil {
		return nil, err
	}
	descriptor := message.ProtoReflect().Descriptor()
	plan := planFor(descriptor)
	if plan.features != nil || plan.flatBatch {
		return streamJSON(data, descriptor)
	}
	return transformMessage(data, descriptor, true)
}

// Unmarshal ignores unknown HTTP fields for mixed-version rollout.
// Parsing succeeds before the destination is replaced.
func Unmarshal(data []byte, message proto.Message) error {
	if message == nil || !message.ProtoReflect().IsValid() {
		return errors.New("cannot unmarshal into a nil latency protocol message")
	}
	data, err := transformMessage(data, message.ProtoReflect().Descriptor(), false)
	if err != nil {
		return err
	}
	parsed := message.ProtoReflect().New().Interface()
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(data, parsed); err != nil {
		return err
	}
	proto.Reset(message)
	proto.Merge(message, parsed)
	return nil
}

type jsonField struct {
	descriptor protoreflect.FieldDescriptor
	keys       []string
}

type jsonPlan struct {
	fields    []jsonField
	features  []jsonField
	byName    map[string]protoreflect.FieldDescriptor
	flatBatch bool
}

var jsonPlans sync.Map

var flatMessages = sync.OnceValue(func() map[protoreflect.FullName]bool {
	return map[protoreflect.FullName]bool{
		(&PredictionRequest{}).ProtoReflect().Descriptor().FullName(): true,
		(&TrainingEntry{}).ProtoReflect().Descriptor().FullName():     true,
	}
})

func fieldsFor(descriptor protoreflect.MessageDescriptor) []jsonField {
	fields := descriptor.Fields()
	result := make([]jsonField, fields.Len())
	for index := range fields.Len() {
		field := fields.Get(index)
		keys := []string{string(field.Name())}
		if keys[0] != field.JSONName() {
			keys = append(keys, field.JSONName())
		}
		result[index] = jsonField{descriptor: field, keys: keys}
	}
	return result
}

func planFor(descriptor protoreflect.MessageDescriptor) *jsonPlan {
	if cached, ok := jsonPlans.Load(descriptor); ok {
		return cached.(*jsonPlan)
	}
	plan := &jsonPlan{fields: fieldsFor(descriptor), byName: make(map[string]protoreflect.FieldDescriptor)}
	for _, field := range plan.fields {
		plan.byName[string(field.descriptor.Name())] = field.descriptor
		if field.descriptor.IsList() && field.descriptor.Kind() == protoreflect.MessageKind && flatMessages()[field.descriptor.Message().FullName()] {
			plan.flatBatch = true
		}
	}
	if flatMessages()[descriptor.FullName()] {
		plan.features = fieldsFor(descriptor.Fields().ByName("features").Message())
	}
	actual, _ := jsonPlans.LoadOrStore(descriptor, plan)
	return actual.(*jsonPlan)
}

func transformMessage(data []byte, descriptor protoreflect.MessageDescriptor, encode bool) ([]byte, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	if len(bytes.TrimSpace(data[decoder.InputOffset():])) != 0 {
		return nil, errors.New("trailing JSON data")
	}
	converted, err := transformObject(value, descriptor, encode)
	if err != nil {
		return nil, err
	}
	return json.Marshal(converted)
}

func transformObject(value any, descriptor protoreflect.MessageDescriptor, encode bool) (any, error) {
	// Well-known types have special JSON mappings, including array-valued trees.
	if descriptor.ParentFile().Package() == "google.protobuf" {
		return value, nil
	}
	object, ok := value.(map[string]any)
	if !ok || object == nil {
		return nil, errors.New("expected a latency protocol JSON object")
	}
	plan := planFor(descriptor)
	if plan.features != nil && !encode {
		delete(object, "features")
		values := make(map[string]any)
		for _, field := range plan.features {
			for _, key := range field.keys {
				if item, ok := object[key]; ok {
					values[key] = item
					delete(object, key)
				}
			}
		}
		if len(values) > 0 {
			object["features"] = values
		}
	}
	for _, field := range plan.fields {
		for _, key := range field.keys {
			item, ok := object[key]
			if !ok || item == nil {
				continue
			}
			converted, err := transformField(item, field.descriptor, encode)
			if err != nil {
				return nil, fmt.Errorf("field %s: %w", field.descriptor.FullName(), err)
			}
			object[key] = converted
		}
	}
	if plan.features != nil && encode {
		if values, ok := object["features"].(map[string]any); ok {
			delete(object, "features")
			for key, item := range values {
				object[key] = item
			}
		}
	}
	return object, nil
}

func transformField(value any, field protoreflect.FieldDescriptor, encode bool) (any, error) {
	if field.IsMap() {
		values, ok := value.(map[string]any)
		if !ok {
			return nil, errors.New("expected a JSON map")
		}
		for key, item := range values {
			converted, err := transformValue(item, field.MapValue(), encode)
			if err != nil {
				return nil, err
			}
			values[key] = converted
		}
		return values, nil
	}
	if field.IsList() {
		values, ok := value.([]any)
		if !ok {
			return nil, errors.New("expected a JSON array")
		}
		for index, item := range values {
			converted, err := transformValue(item, field, encode)
			if err != nil {
				return nil, err
			}
			values[index] = converted
		}
		return values, nil
	}
	return transformValue(value, field, encode)
}

func transformValue(value any, field protoreflect.FieldDescriptor, encode bool) (any, error) {
	if field.Kind() == protoreflect.MessageKind {
		return transformObject(value, field.Message(), encode)
	}
	if encode {
		switch field.Kind() {
		case protoreflect.DoubleKind, protoreflect.FloatKind:
			if _, ok := value.(string); ok {
				return nil, errors.New("non-finite floating-point value")
			}
		case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind, protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
			text, ok := value.(string)
			if !ok {
				return nil, errors.New("expected a ProtoJSON integer string")
			}
			// Keep the decimal digits supplied by protojson without a float conversion.
			return json.Number(text), nil
		}
	}
	return value, nil
}
