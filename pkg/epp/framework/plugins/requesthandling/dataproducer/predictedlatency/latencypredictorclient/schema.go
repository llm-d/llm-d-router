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
	"reflect"
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
