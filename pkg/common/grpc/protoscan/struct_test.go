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
	"math"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

func marshalStruct(t *testing.T, m map[string]any) []byte {
	t.Helper()
	st, err := structpb.NewStruct(m)
	require.NoError(t, err)
	b, err := proto.Marshal(st)
	require.NoError(t, err)
	return b
}

func encodeStruct(t *testing.T, m map[string]any) []byte {
	t.Helper()
	size, err := SizeStruct(m)
	require.NoError(t, err)
	out := AppendStruct(make([]byte, 0, size), m)
	require.Len(t, out, size, "SizeStruct and AppendStruct must agree")
	return out
}

func nixlMap() map[string]any {
	return map[string]any{
		"do_remote_decode":  false,
		"do_remote_prefill": true,
		"remote_engine_id":  "eng-1",
		"remote_block_ids":  []int{10, 11, 12},
		"remote_host":       "10.0.0.8",
		"remote_port":       json.Number("5557"),
	}
}

// TestStructEncodeMatchesStructpb checks byte equality against structpb for
// single-key maps, where proto.Marshal's map ordering cannot differ.
func TestStructEncodeMatchesStructpb(t *testing.T) {
	cases := map[string]struct {
		in   any
		want any // structpb-compatible value when in is not one
	}{
		"nil":            {in: nil},
		"bool":           {in: true},
		"string":         {in: "hello"},
		"empty string":   {in: ""},
		"float64":        {in: 1.5},
		"float32":        {in: float32(0.25)},
		"int":            {in: 42, want: 42},
		"int32":          {in: int32(-7)},
		"int64":          {in: int64(1 << 40), want: float64(1 << 40)},
		"uint32":         {in: uint32(9)},
		"uint64":         {in: uint64(9), want: float64(9)},
		"json.Number":    {in: json.Number("12.5"), want: 12.5},
		"[]any":          {in: []any{1, "a", nil, true, []any{2.0}}},
		"[]string":       {in: []string{"a", "b"}, want: []any{"a", "b"}},
		"[]int":          {in: []int{1, 2}, want: []any{1, 2}},
		"[]map":          {in: []map[string]any{{"k": "v"}}, want: []any{map[string]any{"k": "v"}}},
		"map":            {in: map[string]any{"nested": map[string]any{"deep": 1}}},
		"empty map":      {in: map[string]any{}},
		"empty list":     {in: []any{}},
		"zero number":    {in: 0},
		"negative zero":  {in: math.Copysign(0, -1)},
		"max safe float": {in: math.MaxFloat64},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			want := tc.in
			if tc.want != nil {
				want = tc.want
			}
			wantBytes := marshalStruct(t, map[string]any{"k": want})
			got := encodeStruct(t, map[string]any{"k": tc.in})
			require.Equal(t, wantBytes, got)

			decoded, err := DecodeStruct(got)
			require.NoError(t, err)
			var st structpb.Struct
			require.NoError(t, proto.Unmarshal(got, &st))
			require.Equal(t, st.AsMap(), decoded)
		})
	}

	// Empty key: Go protobuf still writes the key field.
	require.Equal(t, marshalStruct(t, map[string]any{"": 1}), encodeStruct(t, map[string]any{"": 1}))
}

func TestStructEncodeMultiKeyMatchesSemantically(t *testing.T) {
	m := nixlMap()
	got := encodeStruct(t, m)
	var st structpb.Struct
	require.NoError(t, proto.Unmarshal(got, &st))
	want, err := structpb.NewStruct(map[string]any{
		"do_remote_decode":  false,
		"do_remote_prefill": true,
		"remote_engine_id":  "eng-1",
		"remote_block_ids":  []any{10, 11, 12},
		"remote_host":       "10.0.0.8",
		"remote_port":       5557,
	})
	require.NoError(t, err)
	require.True(t, proto.Equal(want, &st))

	decoded, err := DecodeStruct(got)
	require.NoError(t, err)
	require.Equal(t, want.AsMap(), decoded)
}

func TestStructEncodeRejectsUnsupportedValues(t *testing.T) {
	for name, m := range map[string]map[string]any{
		"NaN":              {"x": math.NaN()},
		"+Inf":             {"x": math.Inf(1)},
		"-Inf float32":     {"x": float32(math.Inf(-1))},
		"bad json.Number":  {"x": json.Number("abc")},
		"struct":           {"x": struct{}{}},
		"[]float64":        {"x": []float64{1}},
		"nested bad":       {"x": map[string]any{"y": math.NaN()}},
		"list bad":         {"x": []any{math.NaN()}},
		"list of maps bad": {"x": []map[string]any{{"y": math.NaN()}}},
	} {
		_, err := SizeStruct(m)
		require.Error(t, err, name)
	}

	// Nesting deeper than the limit is an error, not a stack overflow.
	deep := map[string]any{}
	cur := deep
	for i := 0; i < maxValueDepth+5; i++ {
		next := map[string]any{}
		cur["n"] = next
		cur = next
	}
	_, err := SizeStruct(deep)
	require.ErrorIs(t, err, errStructDepth)
}

func TestAppendStructAllocations(t *testing.T) {
	m := nixlMap()
	size, err := SizeStruct(m)
	require.NoError(t, err)
	dst := make([]byte, 0, size)
	require.Zero(t, testing.AllocsPerRun(100, func() {
		dst = AppendStruct(dst[:0], m)
	}))
	require.Zero(t, testing.AllocsPerRun(100, func() {
		_, _ = SizeStruct(m)
	}))
}

func TestDecodeStruct(t *testing.T) {
	m, err := DecodeStruct(nil)
	require.NoError(t, err)
	require.Empty(t, m)
	require.NotNil(t, m)

	// Concatenated Struct bodies merge with later entries winning.
	a := marshalStruct(t, map[string]any{"k": 1, "only_a": true})
	b := marshalStruct(t, map[string]any{"k": "two"})
	merged, err := DecodeStruct(append(append([]byte{}, a...), b...))
	require.NoError(t, err)
	require.Equal(t, map[string]any{"k": "two", "only_a": true}, merged)
	var st structpb.Struct
	require.NoError(t, proto.Unmarshal(append(append([]byte{}, a...), b...), &st))
	require.Equal(t, st.AsMap(), merged)

	// A Value with no kind set decodes as nil, as AsMap does.
	emptyValue := protowire.AppendBytes(protowire.AppendTag(nil, entryValue, protowire.BytesType), nil)
	emptyValue = protowire.AppendString(protowire.AppendTag(emptyValue, entryKey, protowire.BytesType), "k")
	raw := protowire.AppendBytes(protowire.AppendTag(nil, structFields, protowire.BytesType), emptyValue)
	got, err := DecodeStruct(raw)
	require.NoError(t, err)
	require.Equal(t, map[string]any{"k": nil}, got)

	// Malformed inputs.
	for name, bad := range map[string][]byte{
		"truncated":          {0x0a, 0x05, 0x01},
		"fields wrong type":  protowire.AppendVarint(protowire.AppendTag(nil, structFields, protowire.VarintType), 1),
		"value wrong type":   protowire.AppendBytes(protowire.AppendTag(nil, structFields, protowire.BytesType), protowire.AppendVarint(protowire.AppendTag(nil, entryValue, protowire.VarintType), 1)),
		"number wrong type":  protowire.AppendBytes(protowire.AppendTag(nil, structFields, protowire.BytesType), protowire.AppendBytes(protowire.AppendTag(nil, entryValue, protowire.BytesType), protowire.AppendVarint(protowire.AppendTag(nil, valueNumber, protowire.VarintType), 1))),
		"truncated in value": protowire.AppendBytes(protowire.AppendTag(nil, structFields, protowire.BytesType), protowire.AppendBytes(protowire.AppendTag(nil, entryValue, protowire.BytesType), []byte{0x80})),
	} {
		_, err := DecodeStruct(bad)
		require.ErrorIs(t, err, ErrMalformed, name)
	}

	// A list element with a wrong wire type is malformed.
	badList := protowire.AppendVarint(protowire.AppendTag(nil, listValues, protowire.VarintType), 1)
	badList = protowire.AppendBytes(protowire.AppendTag(nil, valueList, protowire.BytesType), badList)
	badList = protowire.AppendBytes(protowire.AppendTag(nil, entryValue, protowire.BytesType), badList)
	_, err = DecodeStruct(protowire.AppendBytes(protowire.AppendTag(nil, structFields, protowire.BytesType), badList))
	require.ErrorIs(t, err, ErrMalformed)
}

// A map entry whose value field repeats merges like proto.Unmarshal: two
// struct_value fields merge their keys, two list_value fields concatenate,
// and a different kind replaces what came before.
func TestDecodeStructRepeatedEntryValue(t *testing.T) {
	valueField := func(v any) []byte {
		pv, err := structpb.NewValue(v)
		require.NoError(t, err)
		raw, err := proto.Marshal(pv)
		require.NoError(t, err)
		return protowire.AppendBytes(protowire.AppendTag(nil, entryValue, protowire.BytesType), raw)
	}
	entry := func(key string, values ...[]byte) []byte {
		e := protowire.AppendString(protowire.AppendTag(nil, entryKey, protowire.BytesType), key)
		for _, v := range values {
			e = append(e, v...)
		}
		return protowire.AppendBytes(protowire.AppendTag(nil, structFields, protowire.BytesType), e)
	}
	for name, raw := range map[string][]byte{
		"struct merge":     entry("k", valueField(map[string]any{"a": 1.0}), valueField(map[string]any{"b": 2.0})),
		"list concat":      entry("k", valueField([]any{1.0}), valueField([]any{"x"})),
		"kind switch":      entry("k", valueField(map[string]any{"a": 1.0}), valueField("s")),
		"scalar to struct": entry("k", valueField("s"), valueField(map[string]any{"a": 1.0})),
	} {
		t.Run(name, func(t *testing.T) {
			var st structpb.Struct
			require.NoError(t, proto.Unmarshal(raw, &st))
			got, err := DecodeStruct(raw)
			require.NoError(t, err)
			require.Equal(t, st.AsMap(), got)
		})
	}
}

// The encoder and the decoder count nesting the same way, so a map that
// SizeStruct accepts decodes, and one level deeper fails on both sides.
func TestStructDepthBound(t *testing.T) {
	nested := func(levels int) map[string]any {
		m := map[string]any{}
		cur := m
		for range levels {
			next := map[string]any{}
			cur["x"] = next
			cur = next
		}
		return m
	}
	atBound := nested(maxValueDepth)
	raw := encodeStruct(t, atBound)
	var st structpb.Struct
	require.NoError(t, proto.Unmarshal(raw, &st))
	got, err := DecodeStruct(raw)
	require.NoError(t, err)
	require.Equal(t, st.AsMap(), got)

	_, err = SizeStruct(nested(maxValueDepth + 1))
	require.ErrorIs(t, err, errStructDepth)
	deep := []byte{}
	for range maxValueDepth + 2 {
		inner := protowire.AppendBytes(protowire.AppendTag(nil, valueStruct, protowire.BytesType), deep)
		entry := protowire.AppendBytes(protowire.AppendTag(nil, entryValue, protowire.BytesType), inner)
		deep = protowire.AppendBytes(protowire.AppendTag(nil, structFields, protowire.BytesType), entry)
	}
	_, err = DecodeStruct(deep)
	require.ErrorIs(t, err, ErrMalformed)
}

func BenchmarkAppendStructNIXL(b *testing.B) {
	m := nixlMap()
	size, err := SizeStruct(m)
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]byte, 0, size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst = AppendStruct(dst[:0], m)
	}
}

func BenchmarkDecodeStructNIXL(b *testing.B) {
	m := nixlMap()
	size, err := SizeStruct(m)
	if err != nil {
		b.Fatal(err)
	}
	raw := AppendStruct(make([]byte, 0, size), m)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeStruct(raw); err != nil {
			b.Fatal(err)
		}
	}
}
