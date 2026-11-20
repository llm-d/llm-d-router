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
	"slices"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

var _ fwkdl.SequencedValue = (*PrefixHashSet)(nil)

// A retraction element is 0x00 followed by the generation it withdraws.
// Prompt elements stay a multiple of 8, so the two encodings do not overlap.
const retractMarker byte = 0

type dropSlot struct {
	on  bool
	gen uint64
}

// SequenceBase is the generation of the oldest element.
func (s *PrefixHashSet) SequenceBase() uint64 { return s.Base }

// SequenceTip is one past the generation of the newest element.
func (s *PrefixHashSet) SequenceTip() uint64 { return s.Base + uint64(len(s.Prompts)) }

// ElementSize is the encoded size of the element at gen.
func (s *PrefixHashSet) ElementSize(gen uint64) int {
	if _, ok := s.wireRetract(gen); ok {
		return 9
	}
	return 8 * len(s.Prompts[gen-s.Base])
}

// AppendElement appends the encoding of the element at gen to dst.
func (s *PrefixHashSet) AppendElement(dst []byte, gen uint64) []byte {
	if target, ok := s.wireRetract(gen); ok {
		dst = append(dst, retractMarker)
		return binary.LittleEndian.AppendUint64(dst, target)
	}
	for _, h := range s.Prompts[gen-s.Base] {
		dst = binary.LittleEndian.AppendUint64(dst, h)
	}
	return dst
}

// wireRetract reports the generation this element withdraws. A prompt larger
// than MaxSequencedElementBytes retracts its own generation.
func (s *PrefixHashSet) wireRetract(gen uint64) (uint64, bool) {
	if target, ok := s.dropAt(gen - s.Base); ok {
		return target, true
	}
	i := int(gen - s.Base) //#nosec G115 -- negative or oversized values fail the bounds check below
	if i >= 0 && i < len(s.Prompts) && 8*len(s.Prompts[i]) > fwkdl.MaxSequencedElementBytes {
		return gen, true
	}
	return 0, false
}

func (s *PrefixHashSet) dropAt(index uint64) (uint64, bool) {
	i := int(index) //#nosec G115 -- negative or oversized values fail the bounds check below
	if s == nil || i < 0 || i >= len(s.drops) || !s.drops[i].on {
		return 0, false
	}
	return s.drops[i].gen, true
}

func (s *PrefixHashSet) AppendContent(prompt []uint64) {
	s.Prompts = append(s.Prompts, prompt)
	if s.drops != nil {
		s.drops = append(s.drops, dropSlot{})
	}
	s.digestCached = false
}

func (s *PrefixHashSet) AppendRetract(gen uint64) {
	if s.drops == nil {
		s.drops = make([]dropSlot, len(s.Prompts))
	}
	s.Prompts = append(s.Prompts, nil)
	s.drops = append(s.drops, dropSlot{on: true, gen: gen})
	s.digestCached = false
}

// Retract empties the first matching prompt and appends a retraction of its
// generation. Later elements keep their generations.
func (s *PrefixHashSet) Retract(prompt []uint64) bool {
	for i, p := range s.Prompts {
		if _, drop := s.dropAt(uint64(i)); drop || len(p) == 0 {
			continue
		}
		if !slices.Equal(p, prompt) {
			continue
		}
		s.Prompts[i] = []uint64{}
		s.AppendRetract(s.Base + uint64(i))
		return true
	}
	return false
}

// Trim drops the oldest n elements and advances Base by the same amount.
func (s *PrefixHashSet) Trim(n int) {
	if n <= 0 {
		return
	}
	if n > len(s.Prompts) {
		n = len(s.Prompts)
	}
	s.Base += uint64(n)
	s.Prompts = append([][]uint64(nil), s.Prompts[n:]...)
	if s.drops != nil {
		s.drops = append([]dropSlot(nil), s.drops[n:]...)
	}
	s.digestCached = false
}

// Advance shares the retained prompt slices with s; prompts are not modified
// after they are published.
func (s *PrefixHashSet) Advance(base uint64, elems [][]byte) (fwkdl.SequencedValue, error) {
	start := max(base, s.Base)
	out := &PrefixHashSet{Base: start}
	if start < s.SequenceTip() {
		out = s.cloneFrom(start)
	}
	if err := out.appendEncoded(elems); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *PrefixHashSet) NewSequence(base uint64, elems [][]byte) (fwkdl.SequencedValue, error) {
	out := &PrefixHashSet{Base: base}
	if err := out.appendEncoded(elems); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *PrefixHashSet) cloneFrom(start uint64) *PrefixHashSet {
	i := int(start - s.Base) //#nosec G115 -- Advance only calls this with start below SequenceTip, so i indexes Prompts
	out := &PrefixHashSet{Base: start, Prompts: make([][]uint64, len(s.Prompts)-i)}
	copy(out.Prompts, s.Prompts[i:])
	if len(s.drops) > 0 {
		out.drops = append([]dropSlot(nil), s.drops[i:]...)
	}
	return out
}

func (s *PrefixHashSet) appendEncoded(elems [][]byte) error {
	for _, elem := range elems {
		if len(elem) == 9 && elem[0] == retractMarker {
			gen := binary.LittleEndian.Uint64(elem[1:])
			s.clear(gen)
			s.AppendRetract(gen)
			continue
		}
		if len(elem)%8 != 0 {
			return fmt.Errorf("prefix: encoded prompt length %d is not a multiple of 8", len(elem))
		}
		prompt := make([]uint64, len(elem)/8)
		for i := range prompt {
			prompt[i] = binary.LittleEndian.Uint64(elem[8*i:])
		}
		s.AppendContent(prompt)
	}
	return nil
}

// clear withdraws the prompt at gen when it is still in this set.
func (s *PrefixHashSet) clear(gen uint64) {
	if gen < s.Base || gen >= s.Base+uint64(len(s.Prompts)) {
		return
	}
	i := gen - s.Base
	if _, drop := s.dropAt(i); drop {
		return
	}
	s.Prompts[i] = []uint64{}
	s.digestCached = false
}
