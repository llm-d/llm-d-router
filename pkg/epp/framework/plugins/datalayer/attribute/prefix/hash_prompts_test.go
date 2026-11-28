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
	"encoding/binary"
	"math/rand/v2"
	"testing"

	"github.com/cespare/xxhash/v2"
	"github.com/stretchr/testify/require"
)

// hashPromptsRef is one xxhash.Write per uint64 of the length-prefixed encoding.
func hashPromptsRef(prompts [][]uint64) uint64 {
	h := xxhash.New()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(prompts)))
	_, _ = h.Write(buf[:])
	for _, p := range prompts {
		binary.LittleEndian.PutUint64(buf[:], uint64(len(p)))
		_, _ = h.Write(buf[:])
		for _, v := range p {
			binary.LittleEndian.PutUint64(buf[:], v)
			_, _ = h.Write(buf[:])
		}
	}
	return h.Sum64()
}

func TestHashPromptsMatchesSingleWrites(t *testing.T) {
	long := make([]uint64, 600)
	for i := range long {
		long[i] = uint64(i)
	}
	cases := make([][][]uint64, 0, 106)
	cases = append(cases,
		nil,
		[][]uint64{},
		[][]uint64{{}},
		[][]uint64{{1}},
		[][]uint64{{1, 2}, {}, {3}},
		[][]uint64{long},
	)
	r := rand.New(rand.NewPCG(1, 2))
	for range 100 {
		n := r.IntN(6)
		ps := make([][]uint64, n)
		for i := range ps {
			ps[i] = make([]uint64, r.IntN(1200))
			for j := range ps[i] {
				ps[i][j] = r.Uint64()
			}
		}
		cases = append(cases, ps)
	}
	for i, c := range cases {
		require.Equal(t, hashPromptsRef(c), hashPrompts(c), "case %d", i)
	}
}

func BenchmarkHashPrompts(b *testing.B) {
	// One endpoint of the Produce churn fixture: four prompts fill the default
	// 31250-hash window, and one peer prompt has the same length.
	const prompts, hashesPerPrompt = 5, 31250 / 4
	window := make([][]uint64, prompts)
	for i := range window {
		h := make([]uint64, hashesPerPrompt)
		for k := range h {
			h[k] = uint64(i+1)<<16 | uint64(k+1)
		}
		window[i] = h
	}
	b.ReportAllocs()
	for b.Loop() {
		hashPrompts(window)
	}
}
