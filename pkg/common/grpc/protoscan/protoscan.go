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

// Package protoscan holds engine-agnostic protobuf wire primitives. The
// engine-specific packages (vllmwire, sglangwire) build their scanners and
// rewriters on top of it so that one field iterator, one packed-varint
// reader, and one google.protobuf.Struct codec serve every engine.
//
// The scanners and the Struct codec do not validate UTF-8 in string fields.
// A string that a scan accepts can still fail proto.Unmarshal on the engine
// side, which then reports the error to the client.
package protoscan

import (
	"errors"
	"fmt"
	"math"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
)

// ErrMalformed reports bytes that are not valid protobuf wire data.
var ErrMalformed = errors.New("malformed protobuf wire data")

// FieldNum resolves a field number from the generated descriptor of m. It
// panics on an unknown name so a renumbered or renamed field fails at init.
func FieldNum(m proto.Message, name protoreflect.Name) protowire.Number {
	fd := m.ProtoReflect().Descriptor().Fields().ByName(name)
	if fd == nil {
		panic("protoscan: unknown field " + string(name))
	}
	return fd.Number()
}

// mapEntryFieldNum resolves the key or value field of the map entry message
// behind mapField on m. entryField is "key" or "value".
func mapEntryFieldNum(m proto.Message, mapField, entryField protoreflect.Name) protowire.Number {
	fd := m.ProtoReflect().Descriptor().Fields().ByName(mapField)
	if fd == nil || !fd.IsMap() {
		panic("protoscan: unknown map field " + string(mapField))
	}
	efd := fd.Message().Fields().ByName(entryField)
	if efd == nil {
		panic("protoscan: unknown map entry field " + string(entryField))
	}
	return efd.Number()
}

// ForEachField visits every top-level field of b in order. value holds the
// raw value bytes. For BytesType the length prefix is already consumed, so
// value is the string, bytes, or submessage content. field holds the full
// tag plus value bytes. An error from fn stops the walk and is returned.
func ForEachField(b []byte, fn func(num protowire.Number, wt protowire.Type, value, field []byte) error) error {
	for len(b) > 0 {
		num, wt, n := protowire.ConsumeTag(b)
		if n < 0 {
			return ErrMalformed
		}
		m := protowire.ConsumeFieldValue(num, wt, b[n:])
		if m < 0 {
			return ErrMalformed
		}
		value := b[n : n+m]
		if wt == protowire.BytesType {
			value, _ = protowire.ConsumeBytes(value)
		}
		if err := fn(num, wt, value, b[:n+m]); err != nil {
			return err
		}
		b = b[n+m:]
	}
	return nil
}

// Validate reports ErrMalformed when b is not a well-formed sequence of
// top-level fields. It does not descend into submessages.
func Validate(b []byte) error {
	return ForEachField(b, func(protowire.Number, protowire.Type, []byte, []byte) error { return nil })
}

// Uint32RangeError reports a token value above math.MaxUint32. Index is the
// position the value would take in the destination slice. It matches
// ErrMalformed under errors.Is.
type Uint32RangeError struct {
	Index int
	Value uint64
}

func (e *Uint32RangeError) Error() string {
	return fmt.Sprintf("value[%d] out of uint32 range: %d", e.Index, int64(e.Value)) //#nosec G115 -- reports the int64 the client sent
}

// Unwrap returns ErrMalformed.
func (e *Uint32RangeError) Unwrap() error { return ErrMalformed }

// AppendPackedUint32 appends the uint32 values of one field occurrence to
// dst. wt is the occurrence's wire type: BytesType holds a packed run of
// varints, VarintType holds a single value. Any other wire type is
// ErrMalformed. A value above math.MaxUint32 is a *Uint32RangeError. One
// policy for every engine: reject, never truncate.
func AppendPackedUint32(dst []uint32, wt protowire.Type, value []byte) ([]uint32, error) {
	switch wt {
	case protowire.BytesType:
		for len(value) > 0 {
			v, n := protowire.ConsumeVarint(value)
			if n < 0 {
				return dst, ErrMalformed
			}
			if v > math.MaxUint32 {
				return dst, &Uint32RangeError{Index: len(dst), Value: v}
			}
			dst = append(dst, uint32(v))
			value = value[n:]
		}
		return dst, nil
	case protowire.VarintType:
		v, n := protowire.ConsumeVarint(value)
		if n < 0 || n != len(value) {
			return dst, ErrMalformed
		}
		if v > math.MaxUint32 {
			return dst, &Uint32RangeError{Index: len(dst), Value: v}
		}
		return append(dst, uint32(v)), nil
	default:
		return dst, ErrMalformed
	}
}

// CountVarints returns the number of varints in a packed field body. It
// counts bytes with the continuation bit clear and returns 0 when the body
// ends inside a varint.
func CountVarints(packed []byte) int {
	if len(packed) > 0 && packed[len(packed)-1]&0x80 != 0 {
		return 0
	}
	n := 0
	for _, c := range packed {
		if c&0x80 == 0 {
			n++
		}
	}
	return n
}

// SizeWithout returns the length of src after dropping every top-level field
// whose number is in drop. It lets a caller size one allocation before
// AppendWithout.
func SizeWithout(src []byte, drop ...protowire.Number) (int, error) {
	size := 0
	err := ForEachField(src, func(num protowire.Number, _ protowire.Type, _, field []byte) error {
		if !contains(drop, num) {
			size += len(field)
		}
		return nil
	})
	return size, err
}

// AppendWithout appends src to dst, dropping every top-level field whose
// number is in drop. Fields keep their original order and encoding.
func AppendWithout(dst, src []byte, drop ...protowire.Number) ([]byte, error) {
	err := ForEachField(src, func(num protowire.Number, _ protowire.Type, _, field []byte) error {
		if !contains(drop, num) {
			dst = append(dst, field...)
		}
		return nil
	})
	return dst, err
}

func contains(nums []protowire.Number, num protowire.Number) bool {
	for _, n := range nums {
		if n == num {
			return true
		}
	}
	return false
}

// AppendLen appends a length prefix. n must be a non-negative size.
func AppendLen(dst []byte, n int) []byte {
	return protowire.AppendVarint(dst, uint64(n)) //#nosec G115 -- n is a non-negative encoded size
}

// ReplaceVarintField returns frame with every top-level field num removed
// and num set to value. frame is a full gRPC frame, as is the result. When
// omitZero is true, a zero value writes no field, which matches proto3
// implicit presence. Set omitZero to false for an optional field, where an
// explicit zero differs from an absent field. When nothing changes, the
// result is frame itself. Otherwise the result takes one allocation.
func ReplaceVarintField(frame []byte, num protowire.Number, value uint64, omitZero bool) ([]byte, error) {
	write := value != 0 || !omitZero
	fieldLen := 0
	if write {
		fieldLen = protowire.SizeTag(num) + protowire.SizeVarint(value)
	}
	out, unchanged, err := startReplace(frame, num, fieldLen)
	if err != nil || unchanged {
		return out, err
	}
	if write {
		out = protowire.AppendTag(out, num, protowire.VarintType)
		out = protowire.AppendVarint(out, value)
	}
	return grpcframe.FinishFrame(out)
}

// ReplaceBytesField is ReplaceVarintField for a string, bytes, or message
// field. For a message field, value is the serialized message. An empty value
// removes the field.
func ReplaceBytesField[V string | []byte](frame []byte, num protowire.Number, value V) ([]byte, error) {
	fieldLen := 0
	if len(value) != 0 {
		fieldLen = protowire.SizeTag(num) + protowire.SizeBytes(len(value))
	}
	out, unchanged, err := startReplace(frame, num, fieldLen)
	if err != nil || unchanged {
		return out, err
	}
	if len(value) != 0 {
		out = protowire.AppendTag(out, num, protowire.BytesType)
		out = AppendLen(out, len(value))
		out = append(out, value...)
	}
	return grpcframe.FinishFrame(out)
}

// startReplace copies the payload of frame without field num into a new
// frame buffer with room for fieldLen more bytes. When fieldLen is 0 and the
// payload has no field num, it returns frame and unchanged set to true.
func startReplace(frame []byte, num protowire.Number, fieldLen int) (out []byte, unchanged bool, err error) {
	payload, err := grpcframe.ParsePayload(frame)
	if err != nil {
		return nil, false, err
	}
	if grpcframe.HeaderLen+len(payload) != len(frame) {
		return nil, false, grpcframe.ErrMalformed
	}
	keptLen, err := SizeWithout(payload, num)
	if err != nil {
		return nil, false, err
	}
	if fieldLen == 0 && keptLen == len(payload) {
		return frame, true, nil
	}
	if keptLen > grpcframe.MaxPayloadLen-fieldLen {
		return nil, false, grpcframe.ErrTooLarge
	}
	out = make([]byte, grpcframe.HeaderLen, grpcframe.HeaderLen+keptLen+fieldLen)
	out, err = AppendWithout(out, payload, num)
	return out, false, err
}
