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

package protoscan

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/types/known/structpb"
)

// google.protobuf.Struct field numbers, resolved from the descriptors.
var (
	structFields = FieldNum(&structpb.Struct{}, "fields")
	entryKey     = mapEntryFieldNum(&structpb.Struct{}, "fields", "key")
	entryValue   = mapEntryFieldNum(&structpb.Struct{}, "fields", "value")
	valueNull    = FieldNum(&structpb.Value{}, "null_value")
	valueNumber  = FieldNum(&structpb.Value{}, "number_value")
	valueString  = FieldNum(&structpb.Value{}, "string_value")
	valueBool    = FieldNum(&structpb.Value{}, "bool_value")
	valueStruct  = FieldNum(&structpb.Value{}, "struct_value")
	valueList    = FieldNum(&structpb.Value{}, "list_value")
	listValues   = FieldNum(&structpb.ListValue{}, "values")
)

// maxValueDepth bounds Struct nesting so a hostile payload cannot exhaust
// the stack. It counts one level per nested map or list and is far below
// protowire.DefaultRecursionLimit (10000), because transfer parameters are
// flat maps and a deep payload has no legitimate use here. The encoder and
// the decoder apply the same count, so a map that one accepts round-trips
// through the other.
const maxValueDepth = 100

var errStructDepth = errors.New("google.protobuf.Struct nesting too deep")

// SizeStruct returns the encoded length of m as a google.protobuf.Struct.
// Supported value kinds: nil, bool, string, every Go integer and float type,
// json.Number, []any, []string, []int, []map[string]any, and map[string]any.
// A non-finite float or any other kind is an error. SizeStruct and
// AppendStruct agree byte for byte, so callers size one allocation first.
func SizeStruct(m map[string]any) (int, error) {
	return sizeStruct(m, 0)
}

func sizeStruct(m map[string]any, depth int) (int, error) {
	if depth > maxValueDepth {
		return 0, errStructDepth
	}
	total := 0
	for k, v := range m {
		valSize, err := sizeValue(v, depth+1)
		if err != nil {
			return 0, fmt.Errorf("field %q: %w", k, err)
		}
		total += protowire.SizeTag(structFields) + protowire.SizeBytes(entrySize(k, valSize))
	}
	return total, nil
}

// entrySize is the size of one map entry. Go protobuf always writes the key
// field, even for an empty key, so the encoder does the same.
func entrySize(k string, valSize int) int {
	return protowire.SizeTag(entryKey) + protowire.SizeBytes(len(k)) +
		protowire.SizeTag(entryValue) + protowire.SizeBytes(valSize)
}

// AppendStruct appends m encoded as a google.protobuf.Struct to dst. The
// caller must have validated m with SizeStruct first: AppendStruct skips
// values that SizeStruct rejects.
func AppendStruct(dst []byte, m map[string]any) []byte {
	for k, v := range m {
		valSize, err := sizeValue(v, 0)
		if err != nil {
			continue
		}
		dst = protowire.AppendTag(dst, structFields, protowire.BytesType)
		dst = AppendLen(dst, entrySize(k, valSize))
		dst = protowire.AppendTag(dst, entryKey, protowire.BytesType)
		dst = protowire.AppendString(dst, k)
		dst = protowire.AppendTag(dst, entryValue, protowire.BytesType)
		dst = AppendLen(dst, valSize)
		dst = appendValue(dst, v)
	}
	return dst
}

func sizeValue(v any, depth int) (int, error) {
	switch x := v.(type) {
	case nil:
		return protowire.SizeTag(valueNull) + protowire.SizeVarint(0), nil
	case bool:
		return protowire.SizeTag(valueBool) + protowire.SizeVarint(protowire.EncodeBool(x)), nil
	case string:
		return protowire.SizeTag(valueString) + protowire.SizeBytes(len(x)), nil
	case map[string]any:
		size, err := sizeStruct(x, depth)
		if err != nil {
			return 0, err
		}
		return protowire.SizeTag(valueStruct) + protowire.SizeBytes(size), nil
	case []any:
		total := 0
		for _, item := range x {
			size, err := sizeValue(item, depth+1)
			if err != nil {
				return 0, err
			}
			total += listElemSize(size)
		}
		return listSize(total), nil
	case []string:
		total := 0
		for _, s := range x {
			total += listElemSize(stringValueSize(s))
		}
		return listSize(total), nil
	case []int:
		return listSize(len(x) * listElemSize(numberValueSize)), nil
	case []map[string]any:
		total := 0
		for _, m := range x {
			size, err := sizeStruct(m, depth+1)
			if err != nil {
				return 0, err
			}
			total += listElemSize(structValueSize(size))
		}
		return listSize(total), nil
	default:
		if _, err := toFiniteFloat64(v); err != nil {
			return 0, err
		}
		return numberValueSize, nil
	}
}

// numberValueSize is the encoded size of a Value holding number_value.
var numberValueSize = protowire.SizeTag(valueNumber) + protowire.SizeFixed64()

func stringValueSize(s string) int {
	return protowire.SizeTag(valueString) + protowire.SizeBytes(len(s))
}

func structValueSize(bodySize int) int {
	return protowire.SizeTag(valueStruct) + protowire.SizeBytes(bodySize)
}

// listElemSize is the size of one ListValue.values entry holding a Value of
// valueSize bytes.
func listElemSize(valueSize int) int {
	return protowire.SizeTag(listValues) + protowire.SizeBytes(valueSize)
}

// listSize is the size of a Value holding list_value with a body of
// bodySize bytes.
func listSize(bodySize int) int {
	return protowire.SizeTag(valueList) + protowire.SizeBytes(bodySize)
}

func appendValue(dst []byte, v any) []byte {
	switch x := v.(type) {
	case nil:
		dst = protowire.AppendTag(dst, valueNull, protowire.VarintType)
		return protowire.AppendVarint(dst, 0)
	case bool:
		dst = protowire.AppendTag(dst, valueBool, protowire.VarintType)
		return protowire.AppendVarint(dst, protowire.EncodeBool(x))
	case string:
		dst = protowire.AppendTag(dst, valueString, protowire.BytesType)
		return protowire.AppendString(dst, x)
	case map[string]any:
		size, _ := sizeStruct(x, 0)
		dst = protowire.AppendTag(dst, valueStruct, protowire.BytesType)
		dst = AppendLen(dst, size)
		return AppendStruct(dst, x)
	case []any:
		total := 0
		for _, item := range x {
			size, _ := sizeValue(item, 0)
			total += listElemSize(size)
		}
		dst = appendListHeader(dst, total)
		for _, item := range x {
			size, _ := sizeValue(item, 0)
			dst = appendListElemHeader(dst, size)
			dst = appendValue(dst, item)
		}
		return dst
	case []string:
		total := 0
		for _, s := range x {
			total += listElemSize(stringValueSize(s))
		}
		dst = appendListHeader(dst, total)
		for _, s := range x {
			dst = appendListElemHeader(dst, stringValueSize(s))
			dst = protowire.AppendTag(dst, valueString, protowire.BytesType)
			dst = protowire.AppendString(dst, s)
		}
		return dst
	case []int:
		dst = appendListHeader(dst, len(x)*listElemSize(numberValueSize))
		for _, n := range x {
			dst = appendListElemHeader(dst, numberValueSize)
			dst = appendNumber(dst, float64(n))
		}
		return dst
	case []map[string]any:
		total := 0
		for _, m := range x {
			size, _ := sizeStruct(m, 0)
			total += listElemSize(structValueSize(size))
		}
		dst = appendListHeader(dst, total)
		for _, m := range x {
			size, _ := sizeStruct(m, 0)
			dst = appendListElemHeader(dst, structValueSize(size))
			dst = protowire.AppendTag(dst, valueStruct, protowire.BytesType)
			dst = AppendLen(dst, size)
			dst = AppendStruct(dst, m)
		}
		return dst
	default:
		f, _ := toFiniteFloat64(v)
		return appendNumber(dst, f)
	}
}

func appendNumber(dst []byte, f float64) []byte {
	dst = protowire.AppendTag(dst, valueNumber, protowire.Fixed64Type)
	return protowire.AppendFixed64(dst, math.Float64bits(f))
}

func appendListHeader(dst []byte, bodySize int) []byte {
	dst = protowire.AppendTag(dst, valueList, protowire.BytesType)
	return AppendLen(dst, bodySize)
}

func appendListElemHeader(dst []byte, valueSize int) []byte {
	dst = protowire.AppendTag(dst, listValues, protowire.BytesType)
	return AppendLen(dst, valueSize)
}

func toFiniteFloat64(v any) (float64, error) {
	var f float64
	switch x := v.(type) {
	case int:
		f = float64(x)
	case int8:
		f = float64(x)
	case int16:
		f = float64(x)
	case int32:
		f = float64(x)
	case int64:
		f = float64(x)
	case uint:
		f = float64(x)
	case uint8:
		f = float64(x)
	case uint16:
		f = float64(x)
	case uint32:
		f = float64(x)
	case uint64:
		f = float64(x)
	case float32:
		f = float64(x)
	case float64:
		f = x
	case json.Number:
		var err error
		f, err = x.Float64()
		if err != nil {
			return 0, err
		}
	default:
		return 0, fmt.Errorf("unsupported Struct value type %T", v)
	}
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, errors.New("non-finite number in Struct")
	}
	return f, nil
}

// DecodeStruct decodes google.protobuf.Struct wire bytes into a Go map with
// the same value kinds as structpb.Struct.AsMap: nil, bool, string, float64,
// []any, and map[string]any. Concatenated Struct bodies merge like
// proto.Unmarshal: a later entry for the same key replaces the earlier one.
func DecodeStruct(raw []byte) (map[string]any, error) {
	out := make(map[string]any)
	if err := decodeStructInto(out, raw, 0); err != nil {
		return nil, err
	}
	return out, nil
}

func decodeStructInto(out map[string]any, raw []byte, depth int) error {
	if depth > maxValueDepth {
		return fmt.Errorf("%w: %w", ErrMalformed, errStructDepth)
	}
	return ForEachField(raw, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		if num != structFields {
			return nil
		}
		if wt != protowire.BytesType {
			return ErrMalformed
		}
		var key string
		var val any
		err := ForEachField(value, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
			switch num {
			case entryKey:
				if wt != protowire.BytesType {
					return ErrMalformed
				}
				key = string(value)
			case entryValue:
				if wt != protowire.BytesType {
					return ErrMalformed
				}
				// A second value field in one entry merges into the
				// first, as proto.Unmarshal merges repeated message
				// fields.
				v, err := decodeValueInto(val, value, depth+1)
				if err != nil {
					return err
				}
				val = v
			}
			return nil
		})
		if err != nil {
			return err
		}
		out[key] = val
		return nil
	})
}

// decodeValueInto decodes a Value starting from prev, so a struct_value or
// list_value merges into an earlier one of the same kind and any other kind
// replaces it.
func decodeValueInto(prev any, raw []byte, depth int) (any, error) {
	if depth > maxValueDepth {
		return nil, fmt.Errorf("%w: %w", ErrMalformed, errStructDepth)
	}
	out := prev
	err := ForEachField(raw, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		switch num {
		case valueNull:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			out = nil
		case valueNumber:
			if wt != protowire.Fixed64Type {
				return ErrMalformed
			}
			bits, _ := protowire.ConsumeFixed64(value)
			out = math.Float64frombits(bits)
		case valueString:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			out = string(value)
		case valueBool:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			v, _ := protowire.ConsumeVarint(value)
			out = v != 0
		case valueStruct:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			m, ok := out.(map[string]any)
			if !ok {
				m = make(map[string]any)
			}
			if err := decodeStructInto(m, value, depth); err != nil {
				return err
			}
			out = m
		case valueList:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			list, ok := out.([]any)
			if !ok {
				list = []any{}
			}
			err := ForEachField(value, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
				if num != listValues {
					return nil
				}
				if wt != protowire.BytesType {
					return ErrMalformed
				}
				v, err := decodeValueInto(nil, value, depth+1)
				if err != nil {
					return err
				}
				list = append(list, v)
				return nil
			})
			if err != nil {
				return err
			}
			out = list
		}
		return nil
	})
	return out, err
}
