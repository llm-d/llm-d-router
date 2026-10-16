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
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// WordPiece tokenizer matching the BAAI/bge-base-en-v1.5 tokenizer that
// potion-base-8M distils from: a BertNormalizer (clean_text, handle_chinese_chars,
// lowercase, strip_accents) followed by a BertPreTokenizer and a WordPiece model.
// Reproduced in pure Go so prompt featurization needs no tokenizer runtime on the
// hot path. The algorithm is standard and fixed; the vocabulary is loaded from the
// model's own vocab.txt (see embed.go).
const (
	unkToken             = "[UNK]"
	subwordPrefix        = "##"
	maxInputCharsPerWord = 100
)

// wordPiece holds the token-to-id vocabulary and the resolved [UNK] id.
type wordPiece struct {
	vocab map[string]int32
	unkID int32 // -1 when the vocabulary carries no [UNK] token
}

func newWordPiece(vocab map[string]int32) *wordPiece {
	unk := int32(-1)
	if id, ok := vocab[unkToken]; ok {
		unk = id
	}
	return &wordPiece{vocab: vocab, unkID: unk}
}

// encode normalizes, pre-tokenizes, and WordPiece-tokenizes text into token ids
// (add_special_tokens=false: no [CLS]/[SEP]).
func (wp *wordPiece) encode(text string) []int32 {
	words := bertPreTokenize(bertNormalize(text))
	ids := make([]int32, 0, len(words))
	for _, w := range words {
		ids = append(ids, wp.tokenizeWord(w)...)
	}
	return ids
}

// tokenizeWord applies greedy longest-match-first WordPiece to a single
// pre-token. A word longer than maxInputCharsPerWord, or one with any piece not
// in the vocabulary, maps to a single [UNK].
func (wp *wordPiece) tokenizeWord(word string) []int32 {
	runes := []rune(word)
	if len(runes) == 0 {
		return nil
	}
	if len(runes) > maxInputCharsPerWord {
		return []int32{wp.unkID}
	}
	out := make([]int32, 0, 4)
	for start := 0; start < len(runes); {
		end := len(runes)
		cur := int32(-1)
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = subwordPrefix + sub
			}
			if id, ok := wp.vocab[sub]; ok {
				cur = id
				break
			}
			end--
		}
		if cur < 0 { // no vocabulary piece covers runes[start:]; the whole word is unknown
			return []int32{wp.unkID}
		}
		out = append(out, cur)
		start = end
	}
	return out
}

// bertNormalize applies the BertNormalizer stages in order: clean_text,
// handle_chinese_chars, strip_accents (NFD + drop nonspacing marks), lowercase.
// strip_accents is unset in the model config, which resolves to the lowercase
// value (true).
func bertNormalize(s string) string {
	var cleaned strings.Builder
	cleaned.Grow(len(s))
	for _, r := range s {
		if r == 0 || r == 0xFFFD || isControl(r) {
			continue
		}
		if isWhitespace(r) {
			cleaned.WriteByte(' ')
			continue
		}
		cleaned.WriteRune(r)
	}

	var chinese strings.Builder
	chinese.Grow(cleaned.Len())
	for _, r := range cleaned.String() {
		if isCJK(r) {
			chinese.WriteByte(' ')
			chinese.WriteRune(r)
			chinese.WriteByte(' ')
		} else {
			chinese.WriteRune(r)
		}
	}

	var stripped strings.Builder
	stripped.Grow(chinese.Len())
	for _, r := range norm.NFD.String(chinese.String()) {
		if unicode.Is(unicode.Mn, r) { // nonspacing combining mark
			continue
		}
		stripped.WriteRune(r)
	}

	return strings.ToLower(stripped.String())
}

// bertPreTokenize splits on whitespace and isolates each punctuation rune as its
// own token, matching the BertPreTokenizer.
func bertPreTokenize(s string) []string {
	var out []string
	for _, word := range strings.Fields(s) {
		var cur strings.Builder
		for _, r := range word {
			if isPunctuation(r) {
				if cur.Len() > 0 {
					out = append(out, cur.String())
					cur.Reset()
				}
				out = append(out, string(r))
				continue
			}
			cur.WriteRune(r)
		}
		if cur.Len() > 0 {
			out = append(out, cur.String())
		}
	}
	return out
}

// isControl matches BERT's _is_control: tab/newline/carriage-return are treated as
// whitespace (not control), everything else in a Unicode "Other" category is control.
func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.In(r, unicode.C)
}

// isWhitespace matches BERT's _is_whitespace: the ASCII whitespace set plus the
// Unicode space-separator category.
func isWhitespace(r rune) bool {
	if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

// isPunctuation matches BERT's _is_punctuation: the ASCII punctuation ranges plus
// the Unicode punctuation category.
func isPunctuation(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.In(r, unicode.P)
}

// isCJK matches BERT's _is_chinese_char CJK code-point ranges.
func isCJK(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF,
		r >= 0x3400 && r <= 0x4DBF,
		r >= 0x20000 && r <= 0x2A6DF,
		r >= 0x2A700 && r <= 0x2B73F,
		r >= 0x2B740 && r <= 0x2B81F,
		r >= 0x2B820 && r <= 0x2CEAF,
		r >= 0xF900 && r <= 0xFAFF,
		r >= 0x2F800 && r <= 0x2FA1F:
		return true
	}
	return false
}
