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
	"errors"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
)

func TestFieldNum(t *testing.T) {
	require.Equal(t, protowire.Number(1), FieldNum(&structpb.Struct{}, "fields"))
	require.Equal(t, protowire.Number(1), mapEntryFieldNum(&structpb.Struct{}, "fields", "key"))
	require.Equal(t, protowire.Number(2), mapEntryFieldNum(&structpb.Struct{}, "fields", "value"))
	require.Panics(t, func() { FieldNum(&structpb.Struct{}, "nope") })
	require.Panics(t, func() { mapEntryFieldNum(&structpb.Struct{}, "nope", "key") })
	require.Panics(t, func() { mapEntryFieldNum(&structpb.Value{}, "string_value", "key") })
	require.Panics(t, func() { mapEntryFieldNum(&structpb.Struct{}, "fields", "nope") })
}

func TestForEachField(t *testing.T) {
	var b []byte
	b = protowire.AppendTag(b, 1, protowire.VarintType)
	b = protowire.AppendVarint(b, 300)
	b = protowire.AppendTag(b, 2, protowire.BytesType)
	b = protowire.AppendString(b, "hello")
	b = protowire.AppendTag(b, 3, protowire.Fixed32Type)
	b = protowire.AppendFixed32(b, 7)

	type visit struct {
		num   protowire.Number
		wt    protowire.Type
		value []byte
		field []byte
	}
	var got []visit
	require.NoError(t, ForEachField(b, func(num protowire.Number, wt protowire.Type, value, field []byte) error {
		got = append(got, visit{num, wt, value, field})
		return nil
	}))
	require.Len(t, got, 3)
	require.Equal(t, protowire.Number(1), got[0].num)
	require.Equal(t, protowire.VarintType, got[0].wt)
	v, _ := protowire.ConsumeVarint(got[0].value)
	require.Equal(t, uint64(300), v)
	require.Equal(t, b[:3], got[0].field)
	require.Equal(t, "hello", string(got[1].value), "BytesType value excludes the length prefix")
	require.Equal(t, protowire.AppendString(protowire.AppendTag(nil, 2, protowire.BytesType), "hello"), got[1].field)
	require.Equal(t, protowire.Fixed32Type, got[2].wt)
	require.Len(t, got[2].value, 4)

	// Error propagation stops the walk.
	stop := errors.New("stop")
	calls := 0
	err := ForEachField(b, func(protowire.Number, protowire.Type, []byte, []byte) error {
		calls++
		return stop
	})
	require.ErrorIs(t, err, stop)
	require.Equal(t, 1, calls)

	// Malformed inputs.
	for name, bad := range map[string][]byte{
		"truncated tag":   {0x80},
		"truncated bytes": {0x12, 0x05, 'a'},
		"bad wire type":   {0x0f},
		"zero field":      {0x00},
	} {
		require.ErrorIs(t, Validate(bad), ErrMalformed, name)
	}
	require.NoError(t, Validate(nil))
	require.Zero(t, testing.AllocsPerRun(100, func() { _ = Validate(b) }))
}

func TestAppendPackedUint32(t *testing.T) {
	packed := protowire.AppendVarint(protowire.AppendVarint(protowire.AppendVarint(nil, 1), 300), math.MaxUint32)
	got, err := AppendPackedUint32(nil, protowire.BytesType, packed)
	require.NoError(t, err)
	require.Equal(t, []uint32{1, 300, math.MaxUint32}, got)

	got, err = AppendPackedUint32(got, protowire.VarintType, protowire.AppendVarint(nil, 5))
	require.NoError(t, err)
	require.Equal(t, []uint32{1, 300, math.MaxUint32, 5}, got)

	// Overflow policy: reject, never truncate.
	_, err = AppendPackedUint32(nil, protowire.BytesType, protowire.AppendVarint(nil, math.MaxUint32+1))
	require.ErrorIs(t, err, ErrMalformed)
	_, err = AppendPackedUint32(nil, protowire.VarintType, protowire.AppendVarint(nil, math.MaxUint32+1))
	require.ErrorIs(t, err, ErrMalformed)
	// The range error carries the destination index and the value.
	_, err = AppendPackedUint32([]uint32{7}, protowire.BytesType, protowire.AppendVarint(protowire.AppendVarint(nil, 1), math.MaxUint64))
	rangeErr, ok := errors.AsType[*Uint32RangeError](err)
	require.True(t, ok)
	require.Equal(t, 2, rangeErr.Index)
	require.Equal(t, uint64(math.MaxUint64), rangeErr.Value)
	require.EqualError(t, err, "value[2] out of uint32 range: -1")
	// Truncated varint inside a packed run.
	_, err = AppendPackedUint32(nil, protowire.BytesType, []byte{0x80})
	require.ErrorIs(t, err, ErrMalformed)
	// Trailing bytes after a single varint.
	_, err = AppendPackedUint32(nil, protowire.VarintType, []byte{0x01, 0x01})
	require.ErrorIs(t, err, ErrMalformed)
	// Wrong wire type.
	_, err = AppendPackedUint32(nil, protowire.Fixed32Type, []byte{0, 0, 0, 0})
	require.ErrorIs(t, err, ErrMalformed)
	// Empty packed body appends nothing.
	got, err = AppendPackedUint32(nil, protowire.BytesType, nil)
	require.NoError(t, err)
	require.Empty(t, got)
}

func TestCountVarints(t *testing.T) {
	require.Equal(t, 0, CountVarints(nil))
	require.Equal(t, 3, CountVarints(protowire.AppendVarint(protowire.AppendVarint(protowire.AppendVarint(nil, 1), 300), 1<<40)))
	require.Equal(t, 0, CountVarints([]byte{0x01, 0x80}), "body ending inside a varint")
}

func TestAppendWithout(t *testing.T) {
	var src []byte
	src = protowire.AppendTag(src, 1, protowire.VarintType)
	src = protowire.AppendVarint(src, 1)
	src = protowire.AppendTag(src, 2, protowire.BytesType)
	src = protowire.AppendString(src, "drop me")
	src = protowire.AppendTag(src, 3, protowire.VarintType)
	src = protowire.AppendVarint(src, 3)
	src = protowire.AppendTag(src, 2, protowire.BytesType)
	src = protowire.AppendString(src, "drop me too")

	want := protowire.AppendVarint(protowire.AppendTag(nil, 1, protowire.VarintType), 1)
	want = protowire.AppendVarint(protowire.AppendTag(want, 3, protowire.VarintType), 3)

	size, err := SizeWithout(src, 2)
	require.NoError(t, err)
	require.Equal(t, len(want), size)
	got, err := AppendWithout([]byte{0xff}, src, 2)
	require.NoError(t, err)
	require.Equal(t, append([]byte{0xff}, want...), got)

	// Dropping nothing copies verbatim. Dropping everything leaves dst.
	got, err = AppendWithout(nil, src)
	require.NoError(t, err)
	require.Equal(t, src, got)
	got, err = AppendWithout(nil, src, 1, 2, 3)
	require.NoError(t, err)
	require.Empty(t, got)

	_, err = AppendWithout(nil, []byte{0x80}, 2)
	require.ErrorIs(t, err, ErrMalformed)
	_, err = SizeWithout([]byte{0x80}, 2)
	require.ErrorIs(t, err, ErrMalformed)
}

func TestReplaceVarintField(t *testing.T) {
	const num = protowire.Number(3)
	other := protowire.AppendString(protowire.AppendTag(nil, 1, protowire.BytesType), "keep")
	withField := protowire.AppendVarint(protowire.AppendTag(append([]byte{}, other...), num, protowire.VarintType), 9)
	// A repeated occurrence: every copy must go.
	withField = protowire.AppendVarint(protowire.AppendTag(withField, num, protowire.VarintType), 11)
	frameWith, err := grpcframe.FramePayload(withField)
	require.NoError(t, err)
	frameWithout, err := grpcframe.FramePayload(other)
	require.NoError(t, err)

	want := func(value uint64, write bool) []byte {
		b := append([]byte{}, other...)
		if write {
			b = protowire.AppendVarint(protowire.AppendTag(b, num, protowire.VarintType), value)
		}
		framed, ferr := grpcframe.FramePayload(b)
		require.NoError(t, ferr)
		return framed
	}

	tests := []struct {
		name     string
		frame    []byte
		value    uint64
		omitZero bool
		want     []byte
		same     bool
	}{
		{name: "replace", frame: frameWith, value: 5, omitZero: true, want: want(5, true)},
		{name: "add", frame: frameWithout, value: 5, omitZero: true, want: want(5, true)},
		{name: "omit zero drops field", frame: frameWith, value: 0, omitZero: true, want: want(0, false)},
		{name: "omit zero absent is no-op", frame: frameWithout, value: 0, omitZero: true, want: frameWithout, same: true},
		{name: "explicit zero", frame: frameWithout, value: 0, omitZero: false, want: want(0, true)},
		{name: "negative int64", frame: frameWith, value: math.MaxUint64, omitZero: false, want: want(math.MaxUint64, true)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ReplaceVarintField(tc.frame, num, tc.value, tc.omitZero)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
			require.Equal(t, tc.same, &got[0] == &tc.frame[0])
			if !tc.same {
				require.Equal(t, len(got), cap(got), "one exact-size allocation")
			}
		})
	}
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = ReplaceVarintField(frameWithout, num, 0, true) }))
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = ReplaceVarintField(frameWith, num, 5, true) }))

	malformed, err := grpcframe.FramePayload([]byte{0xff})
	require.NoError(t, err)
	_, err = ReplaceVarintField(malformed, num, 1, true)
	require.ErrorIs(t, err, ErrMalformed)
	_, err = ReplaceVarintField([]byte{0, 0}, num, 1, true)
	require.Error(t, err)
	// Bytes after the frame are rejected on both the no-op and the rewrite
	// path, so a caller cannot lose them.
	trailing := append(append([]byte{}, frameWithout...), 'x')
	_, err = ReplaceVarintField(trailing, num, 0, true)
	require.ErrorIs(t, err, grpcframe.ErrMalformed)
	_, err = ReplaceVarintField(trailing, num, 7, true)
	require.ErrorIs(t, err, grpcframe.ErrMalformed)
}

func TestReplaceBytesField(t *testing.T) {
	const num = protowire.Number(2)
	other := protowire.AppendVarint(protowire.AppendTag(nil, 1, protowire.VarintType), 4)
	withField := protowire.AppendString(protowire.AppendTag(append([]byte{}, other...), num, protowire.BytesType), "old")
	frameWith, err := grpcframe.FramePayload(withField)
	require.NoError(t, err)
	frameWithout, err := grpcframe.FramePayload(other)
	require.NoError(t, err)

	got, err := ReplaceBytesField(frameWith, num, "new")
	require.NoError(t, err)
	wantFrame, err := grpcframe.FramePayload(protowire.AppendString(protowire.AppendTag(append([]byte{}, other...), num, protowire.BytesType), "new"))
	require.NoError(t, err)
	require.Equal(t, wantFrame, got)
	require.Equal(t, len(got), cap(got))

	got, err = ReplaceBytesField(frameWith, num, "")
	require.NoError(t, err)
	require.Equal(t, frameWithout, got)

	got, err = ReplaceBytesField(frameWithout, num, "")
	require.NoError(t, err)
	require.Same(t, &frameWithout[0], &got[0])

	// A []byte value, such as a serialized message, writes the same field
	// without a string copy.
	got, err = ReplaceBytesField(frameWith, num, []byte("new"))
	require.NoError(t, err)
	require.Equal(t, wantFrame, got)
	value := []byte("new")
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = ReplaceBytesField(frameWith, num, value) }))

	got, err = ReplaceBytesField(frameWith, num, []byte(nil))
	require.NoError(t, err)
	require.Equal(t, frameWithout, got)
}
