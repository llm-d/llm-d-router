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

package latencypredictorclient

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
)

type featureBinding struct {
	index     int
	field     protoreflect.FieldDescriptor
	omitEmpty bool
}

var featureBindingCache sync.Map

func featureBindings(t reflect.Type) []featureBinding {
	if cached, ok := featureBindingCache.Load(t); ok {
		return cached.([]featureBinding)
	}
	fields := (&protocol.Features{}).ProtoReflect().Descriptor().Fields()
	bindings := make([]featureBinding, 0, t.NumField())
	for index := range t.NumField() {
		tag := t.Field(index).Tag.Get("json")
		name, _, _ := strings.Cut(tag, ",")
		if field := fields.ByName(protoreflect.Name(name)); field != nil {
			bindings = append(bindings, featureBinding{index: index, field: field, omitEmpty: strings.Contains(tag, ",omitempty")})
		}
	}
	actual, _ := featureBindingCache.LoadOrStore(t, bindings)
	return actual.([]featureBinding)
}

func facadeFeatures(value any, additional *protocol.Features) *protocol.Features {
	features := &protocol.Features{}
	if additional != nil {
		features = proto.Clone(additional).(*protocol.Features)
	}
	destination := features.ProtoReflect()
	source := reflect.ValueOf(value)
	for _, binding := range featureBindings(source.Type()) {
		value := source.Field(binding.index)
		if binding.omitEmpty && value.IsZero() {
			destination.Clear(binding.field)
			continue
		}
		switch value.Kind() {
		case reflect.Float64:
			destination.Set(binding.field, protoreflect.ValueOfFloat64(value.Float()))
		case reflect.Int, reflect.Int64:
			destination.Set(binding.field, protoreflect.ValueOfInt64(value.Int()))
		case reflect.String:
			destination.Set(binding.field, protoreflect.ValueOfString(value.String()))
		}
	}
	return features
}

func projectFeatures(features *protocol.Features, destination any) {
	value := reflect.ValueOf(destination).Elem()
	source := features.ProtoReflect()
	for _, binding := range featureBindings(value.Type()) {
		field := value.Field(binding.index)
		switch field.Kind() {
		case reflect.Float64:
			field.SetFloat(source.Get(binding.field).Float())
		case reflect.Int, reflect.Int64:
			field.SetInt(source.Get(binding.field).Int())
		case reflect.String:
			field.SetString(source.Get(binding.field).String())
		}
	}
}

// PredictionFromFeatures preserves the legacy field accessors for existing callers.
func PredictionFromFeatures(features *protocol.Features) PredictionRequest {
	if features == nil {
		features = &protocol.Features{}
	}
	request := PredictionRequest{Features: proto.Clone(features).(*protocol.Features)}
	projectFeatures(request.Features, &request)
	return request
}

// TrainingFromFeatures combines a feature snapshot with the observed labels.
func TrainingFromFeatures(features *protocol.Features, entry TrainingEntry) TrainingEntry {
	if features == nil {
		features = &protocol.Features{}
	}
	entry.Features = proto.Clone(features).(*protocol.Features)
	projectFeatures(entry.Features, &entry)
	return entry
}

func (r PredictionRequest) protocolMessage() *protocol.PredictionRequest {
	return &protocol.PredictionRequest{Features: facadeFeatures(r, r.Features)}
}

// MarshalJSON uses the generated schema while keeping the flat HTTP contract.
func (r PredictionRequest) MarshalJSON() ([]byte, error) {
	return protocol.Marshal(r.protocolMessage())
}

// MarshalJSON encodes a batch through one protobuf message.
func (r BulkPredictionRequest) MarshalJSON() ([]byte, error) {
	requests := make([]*protocol.PredictionRequest, len(r.Requests))
	for index := range r.Requests {
		requests[index] = r.Requests[index].protocolMessage()
	}
	return protocol.Marshal(&protocol.BulkPredictionRequest{Requests: requests})
}

// MarshalJSON keeps the original timestamp offset used in training sample hashes.
func (entry TrainingEntry) MarshalJSON() ([]byte, error) {
	message := &protocol.TrainingEntry{
		Features:     facadeFeatures(entry, entry.Features),
		ActualTtftMs: proto.Float64(entry.ActualTTFT), ActualTpotMs: proto.Float64(entry.ActualTPOT),
		PredictedTtftMs: entry.PredictedTTFT, PredictedTpotMs: entry.PredictedTPOT,
		Timestamp: timestamppb.New(entry.Timestamp),
	}
	data, err := protocol.Marshal(message)
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		return nil, err
	}
	object["timestamp"], err = entry.Timestamp.MarshalJSON()
	if err != nil {
		return nil, err
	}
	return json.Marshal(object)
}

func validateFeatures(features *protocol.Features) error {
	message := features.ProtoReflect()
	fields := message.Descriptor().Fields()
	for index := range fields.Len() {
		field := fields.Get(index)
		rules := protocol.RulesFor(field)
		if !rules.GetGoValidate() {
			continue
		}
		number, err := featureNumber(message, field)
		if err != nil {
			return err
		}
		if rules.Minimum != nil && number < rules.GetMinimum() {
			return fmt.Errorf("%s must be at least %v, got %v", field.Name(), rules.GetMinimum(), number)
		}
		if rules.Maximum != nil && number > rules.GetMaximum() {
			return fmt.Errorf("%s must be at most %v, got %v", field.Name(), rules.GetMaximum(), number)
		}
	}
	if features.GetEncoderMatchedSize() > features.GetEncoderInputSize() {
		return fmt.Errorf("encoder_matched_size must not exceed encoder_input_size (%d), got %d", features.GetEncoderInputSize(), features.GetEncoderMatchedSize())
	}
	return nil
}

func ridgeFields(target string) []protoreflect.FieldDescriptor {
	fields := (&protocol.Features{}).ProtoReflect().Descriptor().Fields()
	result := make([]protoreflect.FieldDescriptor, 0, fields.Len())
	order := func(field protoreflect.FieldDescriptor) int32 {
		rules := protocol.RulesFor(field)
		if target == "ttft" {
			return rules.GetRidgeTtftOrder()
		}
		return rules.GetRidgeTpotOrder()
	}
	for index := range fields.Len() {
		if field := fields.Get(index); order(field) > 0 {
			result = append(result, field)
		}
	}
	slices.SortFunc(result, func(a, b protoreflect.FieldDescriptor) int { return int(order(a) - order(b)) })
	return result
}

var ridgeTTFTFields = ridgeFields("ttft")
var ridgeTPOTFields = ridgeFields("tpot")

// PredictionRequestOption adds features without extending positional arguments.
type PredictionRequestOption func(*PredictionRequest) error

// WithFeatures accepts the fields generated from the shared schema.
func WithFeatures(features *protocol.Features) PredictionRequestOption {
	return func(request *PredictionRequest) error {
		if features == nil {
			return nil
		}
		merged := facadeFeatures(*request, request.Features)
		proto.Merge(merged, features)
		*request = PredictionFromFeatures(merged)
		return nil
	}
}

// WithFeature sets a field declared in the schema and validates its JSON value.
func WithFeature(name string, value any) PredictionRequestOption {
	return func(request *PredictionRequest) error {
		field := (&protocol.Features{}).ProtoReflect().Descriptor().Fields().ByName(protoreflect.Name(name))
		if field == nil {
			return fmt.Errorf("unknown prediction feature %q", name)
		}
		data, err := json.Marshal(map[string]any{name: value})
		if err != nil {
			return err
		}
		message := &protocol.Features{}
		if err := protocol.Unmarshal(data, message); err != nil {
			return err
		}
		return WithFeatures(message)(request)
	}
}

func ridgePrediction(intercept float64, coefficients map[string]float64, features *protocol.Features, fields []protoreflect.FieldDescriptor) (float64, error) {
	message := features.ProtoReflect()
	for _, field := range fields {
		number, err := featureNumber(message, field)
		if err != nil {
			return 0, err
		}
		intercept += coefficients[string(field.Name())] * number
	}
	return intercept, nil
}

func featureNumber(message protoreflect.Message, field protoreflect.FieldDescriptor) (float64, error) {
	value := message.Get(field)
	if !message.Has(field) {
		var err error
		value, err = protocol.DefaultFor(field)
		if err != nil {
			return 0, err
		}
	}
	switch field.Kind() {
	case protoreflect.DoubleKind, protoreflect.FloatKind:
		return value.Float(), nil
	case protoreflect.Int32Kind, protoreflect.Int64Kind, protoreflect.Sint32Kind, protoreflect.Sint64Kind, protoreflect.Sfixed32Kind, protoreflect.Sfixed64Kind:
		return float64(value.Int()), nil
	case protoreflect.Uint32Kind, protoreflect.Uint64Kind, protoreflect.Fixed32Kind, protoreflect.Fixed64Kind:
		return float64(value.Uint()), nil
	case protoreflect.BoolKind:
		if value.Bool() {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("feature %s must be numeric", field.Name())
	}
}
