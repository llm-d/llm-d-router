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

package outlensketch

import (
	"strings"
	"testing"
)

func testVocab() *wordPiece {
	return newWordPiece(map[string]int32{
		"[UNK]": 0,
		"play":  1,
		"##ing": 2,
		"foo":   3,
		"!":     4,
		"cafe":  5,
		"a":     6,
	})
}

func TestWordPieceGreedySubword(t *testing.T) {
	// "playing" -> "play" + "##ing".
	if got := testVocab().encode("playing"); !equalIDs(got, []int32{1, 2}) {
		t.Errorf("encode(playing) = %v, want [1 2]", got)
	}
}

func TestWordPiecePunctuationIsolated(t *testing.T) {
	// BertPreTokenizer splits "foo!" into "foo" and "!".
	if got := testVocab().encode("foo!"); !equalIDs(got, []int32{3, 4}) {
		t.Errorf("encode(foo!) = %v, want [3 4]", got)
	}
}

func TestWordPieceLowercases(t *testing.T) {
	if got := testVocab().encode("FOO"); !equalIDs(got, []int32{3}) {
		t.Errorf("encode(FOO) = %v, want [3] (lowercased)", got)
	}
}

func TestWordPieceStripsAccents(t *testing.T) {
	// "café" normalizes to "cafe".
	if got := testVocab().encode("café"); !equalIDs(got, []int32{5}) {
		t.Errorf("encode(café) = %v, want [5] (accent stripped)", got)
	}
}

func TestWordPieceUnknownWord(t *testing.T) {
	if got := testVocab().encode("zebra"); !equalIDs(got, []int32{0}) {
		t.Errorf("encode(zebra) = %v, want [0] ([UNK])", got)
	}
}

func TestWordPieceTooLongWord(t *testing.T) {
	long := strings.Repeat("a", maxInputCharsPerWord+1)
	if got := testVocab().encode(long); !equalIDs(got, []int32{0}) {
		t.Errorf("encode(very long word) = %v, want [0] ([UNK])", got)
	}
}

func TestWordPieceWhitespaceSplit(t *testing.T) {
	if got := testVocab().encode("  foo\tplaying\n"); !equalIDs(got, []int32{3, 1, 2}) {
		t.Errorf("encode(whitespace) = %v, want [3 1 2]", got)
	}
}

func TestWordPieceEmpty(t *testing.T) {
	if got := testVocab().encode("   "); len(got) != 0 {
		t.Errorf("encode(blank) = %v, want empty", got)
	}
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
