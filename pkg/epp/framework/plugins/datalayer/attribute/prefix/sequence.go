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
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

var _ fwkdl.SequencedValue = (*PrefixHashSet)(nil)

// SequenceBase is the generation of the oldest prompt.
func (s *PrefixHashSet) SequenceBase() uint64 { return s.Base }

// SequenceTip is one past the generation of the newest prompt.
func (s *PrefixHashSet) SequenceTip() uint64 { return s.Base + uint64(len(s.Prompts)) }

// ElementSize is the encoded size of the prompt at gen: 8 bytes per hash.
func (s *PrefixHashSet) ElementSize(gen uint64) int { return 8 * len(s.Prompts[gen-s.Base]) }

// AppendElement appends the prompt at gen as little-endian uint64 hashes.
func (s *PrefixHashSet) AppendElement(dst []byte, gen uint64) []byte {
	for _, h := range s.Prompts[gen-s.Base] {
		dst = binary.LittleEndian.AppendUint64(dst, h)
	}
	return dst
}

// Advance shares the retained prompt slices with s; prompts are not modified
// after they are published.
func (s *PrefixHashSet) Advance(base uint64, elems [][]byte) (fwkdl.SequencedValue, error) {
	start := max(base, s.Base)
	var kept [][]uint64
	if start < s.SequenceTip() {
		kept = s.Prompts[start-s.Base:]
	} else {
		start = base
	}
	prompts := make([][]uint64, len(kept), len(kept)+len(elems))
	copy(prompts, kept)
	prompts, err := appendDecoded(prompts, elems)
	if err != nil {
		return nil, err
	}
	return &PrefixHashSet{Prompts: prompts, Base: start}, nil
}

func (s *PrefixHashSet) NewSequence(base uint64, elems [][]byte) (fwkdl.SequencedValue, error) {
	prompts, err := appendDecoded(make([][]uint64, 0, len(elems)), elems)
	if err != nil {
		return nil, err
	}
	return &PrefixHashSet{Prompts: prompts, Base: base}, nil
}

func appendDecoded(prompts [][]uint64, elems [][]byte) ([][]uint64, error) {
	for _, elem := range elems {
		if len(elem)%8 != 0 {
			return nil, fmt.Errorf("prefix: encoded prompt length %d is not a multiple of 8", len(elem))
		}
		prompt := make([]uint64, len(elem)/8)
		for i := range prompt {
			prompt[i] = binary.LittleEndian.Uint64(elem[8*i:])
		}
		prompts = append(prompts, prompt)
	}
	return prompts, nil
}
