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

package prefix

import (
	"testing"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func encodeAll(s *PrefixHashSet) [][]byte {
	var out [][]byte
	for gen := s.SequenceBase(); gen < s.SequenceTip(); gen++ {
		elem := s.AppendElement(nil, gen)
		if len(elem) != s.ElementSize(gen) {
			panic("ElementSize disagrees with AppendElement")
		}
		out = append(out, elem)
	}
	return out
}

func TestPrefixHashSetSequenceRoundTrip(t *testing.T) {
	src := &PrefixHashSet{Base: 7, Prompts: [][]uint64{{1, 2}, {1 << 63}, {}}}
	require.Equal(t, uint64(7), src.SequenceBase())
	require.Equal(t, uint64(10), src.SequenceTip())

	got, err := (&PrefixHashSet{}).NewSequence(7, encodeAll(src))
	require.NoError(t, err)
	require.Equal(t, uint64(7), got.SequenceBase())
	require.Equal(t, [][]uint64{{1, 2}, {1 << 63}, {}}, got.(*PrefixHashSet).Prompts)

	_, err = (&PrefixHashSet{}).NewSequence(0, [][]byte{make([]byte, 7)})
	require.Error(t, err)
}

func TestPrefixHashSetAdvance(t *testing.T) {
	prev := &PrefixHashSet{Base: 10, Prompts: [][]uint64{{1}, {2}, {3}}}
	elem := (&PrefixHashSet{Prompts: [][]uint64{{4, 5}}}).AppendElement(nil, 0)

	for _, tc := range []struct {
		name     string
		base     uint64
		elems    [][]byte
		wantBase uint64
		want     [][]uint64
	}{
		{name: "append", base: 10, elems: [][]byte{elem}, wantBase: 10, want: [][]uint64{{1}, {2}, {3}, {4, 5}}},
		{name: "evict and append", base: 12, elems: [][]byte{elem}, wantBase: 12, want: [][]uint64{{3}, {4, 5}}},
		{name: "evict only", base: 11, wantBase: 11, want: [][]uint64{{2}, {3}}},
		{name: "base past tip", base: 20, elems: [][]byte{elem}, wantBase: 20, want: [][]uint64{{4, 5}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := prev.Advance(tc.base, tc.elems)
			require.NoError(t, err)
			require.Equal(t, tc.wantBase, got.SequenceBase())
			require.Equal(t, tc.want, got.(*PrefixHashSet).Prompts)
		})
	}
	require.Equal(t, [][]uint64{{1}, {2}, {3}}, prev.Prompts, "Advance does not modify the receiver")
}

func TestPrefixHashSetRetractionRoundTrip(t *testing.T) {
	src := &PrefixHashSet{Base: 4, Prompts: [][]uint64{{1}, {2}, {3}}}
	require.True(t, src.Retract([]uint64{2}))
	require.Equal(t, uint64(8), src.SequenceTip())
	got, err := (&PrefixHashSet{}).NewSequence(4, encodeAll(src))
	require.NoError(t, err)
	require.Equal(t, [][]uint64{{1}, {}, {3}, nil}, got.(*PrefixHashSet).Prompts)
	require.Equal(t, 9, got.(*PrefixHashSet).ElementSize(7))

	prev := &PrefixHashSet{Base: 4, Prompts: [][]uint64{{1}, {2}, {3}}}
	advanced, err := prev.Advance(4, [][]byte{src.AppendElement(nil, 7)})
	require.NoError(t, err)
	require.Equal(t, [][]uint64{{1}, {}, {3}, nil}, advanced.(*PrefixHashSet).Prompts)
	require.Equal(t, [][]uint64{{1}, {2}, {3}}, prev.Prompts)
}

func TestPrefixHashSetOversizedPromptEncodesAsRetract(t *testing.T) {
	huge := make([]uint64, fwkdl.MaxSequencedElementBytes/8+1)
	src := &PrefixHashSet{Prompts: [][]uint64{{1}, huge, {2}}}
	require.Equal(t, 9, src.ElementSize(1))
	require.Equal(t, huge, src.Prompts[1], "the local prompt is kept")
	got, err := (&PrefixHashSet{}).NewSequence(0, encodeAll(src))
	require.NoError(t, err)
	require.Equal(t, [][]uint64{{1}, nil, {2}}, got.(*PrefixHashSet).Prompts)
}

func TestPrefixHashSetCloneKeepsBase(t *testing.T) {
	src := &PrefixHashSet{Base: 3, Prompts: [][]uint64{{1}}}
	require.Equal(t, uint64(3), src.Clone().(*PrefixHashSet).Base)
}
