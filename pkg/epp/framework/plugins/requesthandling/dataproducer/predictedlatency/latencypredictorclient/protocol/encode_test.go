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
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func requireGenericEncoding(t *testing.T, message proto.Message) []byte {
	t.Helper()
	data, err := (protojson.MarshalOptions{UseProtoNames: true, EmitDefaultValues: true}).Marshal(message)
	require.NoError(t, err)
	want, oldErr := transformMessage(data, message.ProtoReflect().Descriptor(), true)
	got, err := Marshal(message)
	if oldErr != nil {
		require.Error(t, err)
		return nil
	}
	require.NoError(t, err)
	require.True(t, json.Valid(got))
	require.JSONEq(t, string(want), string(got))
	return got
}

func TestStreamingRequestsMatchGenericAdapter(t *testing.T) {
	data, err := os.ReadFile("testdata/wire.json")
	require.NoError(t, err)
	var fixtures map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &fixtures))
	for _, name := range []string{"prediction_zero", "prediction_extended", "prediction_legacy"} {
		t.Run(name, func(t *testing.T) {
			message := &PredictionRequest{}
			require.NoError(t, Unmarshal(fixtures[name], message))
			requireGenericEncoding(t, message)
			for _, count := range []int{0, 1, 100} {
				batch := &BulkPredictionRequest{}
				for range count {
					batch.Requests = append(batch.Requests, message)
				}
				requireGenericEncoding(t, batch)
			}
		})
	}
	for _, name := range []string{"training_ttft", "training_tpot", "training_legacy"} {
		t.Run(name, func(t *testing.T) {
			message := &TrainingEntry{}
			require.NoError(t, Unmarshal(fixtures[name], message))
			requireGenericEncoding(t, message)
			requireGenericEncoding(t, &BulkTrainingRequest{Entries: []*TrainingEntry{message, message}})
		})
	}
	requireGenericEncoding(t, &TrainingEntry{Features: &Features{PodType: proto.String("prefill\"\n<>&\u2028雪")}, Timestamp: timestamppb.New(time.Unix(0, 123))})
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		requireGenericEncoding(t, &PredictionRequest{Features: &Features{KvCachePercentage: proto.Float64(value)}})
		requireGenericEncoding(t, &TrainingEntry{ActualTtftMs: proto.Float64(value)})
	}
}

func TestStreamingFeaturePreservesNegativeZero(t *testing.T) {
	data := requireGenericEncoding(t, &PredictionRequest{Features: &Features{KvCachePercentage: proto.Float64(math.Copysign(0, -1))}})
	var values map[string]float64
	require.NoError(t, json.Unmarshal(data, &values))
	require.True(t, math.Signbit(values["kv_cache_percentage"]))
}

func TestStreamingDiscoversAdditionalSchemaFields(t *testing.T) {
	cases := []struct {
		kind  descriptorpb.FieldDescriptorProto_Type
		value protoreflect.Value
		want  string
	}{
		{descriptorpb.FieldDescriptorProto_TYPE_INT64, protoreflect.ValueOfInt64(math.MinInt64), strconv.FormatInt(math.MinInt64, 10)},
		{descriptorpb.FieldDescriptorProto_TYPE_UINT64, protoreflect.ValueOfUint64(math.MaxUint64), strconv.FormatUint(math.MaxUint64, 10)},
		{descriptorpb.FieldDescriptorProto_TYPE_BOOL, protoreflect.ValueOfBool(false), "false"},
		{descriptorpb.FieldDescriptorProto_TYPE_DOUBLE, protoreflect.ValueOfFloat64(0), "0"},
	}
	for _, tc := range cases {
		t.Run(tc.kind.String(), func(t *testing.T) {
			file := protodesc.ToFileDescriptorProto(File_common_protocol_latency_proto)
			var features *descriptorpb.DescriptorProto
			for _, message := range file.MessageType {
				if message.GetName() == "Features" {
					features = message
					break
				}
			}
			require.NotNil(t, features)
			features.OneofDecl = append(features.OneofDecl, &descriptorpb.OneofDescriptorProto{Name: proto.String("_extension_size")})
			features.Field = append(features.Field, &descriptorpb.FieldDescriptorProto{
				Name: proto.String("extension_size"), Number: proto.Int32(99), Type: tc.kind.Enum(),
				Label: descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(), Proto3Optional: proto.Bool(true),
				OneofIndex: proto.Int32(int32(len(features.OneofDecl) - 1)),
			})
			descriptor, err := protodesc.NewFile(file, protoregistry.GlobalFiles)
			require.NoError(t, err)
			feature := dynamicpb.NewMessage(descriptor.Messages().ByName("Features"))
			feature.Set(feature.Descriptor().Fields().ByName("extension_size"), tc.value)
			request := dynamicpb.NewMessage(descriptor.Messages().ByName("PredictionRequest"))
			request.Set(request.Descriptor().Fields().ByName("features"), protoreflect.ValueOfMessage(feature))
			data := requireGenericEncoding(t, request)
			var values map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(data, &values))
			require.Equal(t, tc.want, string(values["extension_size"]))
		})
	}
}
