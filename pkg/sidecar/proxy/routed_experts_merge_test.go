/*
Copyright 2025 The llm-d Authors.

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

package proxy

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"testing"
)

// makeNpyU8 builds a tiny C-order .npy v1.0 blob for a (n, l, k) uint8 array
// where every element of row r equals base+r, so rows are distinguishable.
func makeNpyU8(n, l, k int, base byte) []byte {
	header := fmt.Sprintf("{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d, %d), }", n, l, k)
	h := []byte(header)
	total := 10 + len(h) + 1
	pad := (64 - (total % 64)) % 64
	for i := 0; i < pad; i++ {
		h = append(h, ' ')
	}
	h = append(h, '\n')

	buf := []byte("\x93NUMPY")
	buf = append(buf, 1, 0)
	lenField := make([]byte, 2)
	binary.LittleEndian.PutUint16(lenField, uint16(len(h)))
	buf = append(buf, lenField...)
	buf = append(buf, h...)
	for r := 0; r < n; r++ {
		for i := 0; i < l*k; i++ {
			buf = append(buf, base+byte(r))
		}
	}
	return buf
}

func b64(v []byte) string { return base64.StdEncoding.EncodeToString(v) }

func TestParseNpyHeader(t *testing.T) {
	h := parseNpyHeader(makeNpyU8(3, 2, 4, 0))
	if h.descr != "|u1" {
		t.Fatalf("descr wrong: %+v", h)
	}
	if len(h.shape) != 3 || h.shape[0] != 3 || h.shape[1] != 2 || h.shape[2] != 4 {
		t.Fatalf("shape wrong: %v", h.shape)
	}
	if h.itemsize() != 1 {
		t.Fatalf("itemsize wrong: %d", h.itemsize())
	}
	if h.rowBytes() != 8 {
		t.Fatalf("rowBytes wrong: %d", h.rowBytes())
	}
}

func TestSpliceReplacesPrefixOnly(t *testing.T) {
	l, k := 2, 3
	prefill := makeNpyU8(2, l, k, 100)
	decode := makeNpyU8(5, l, k, 0)
	out, err := spliceRoutedExpertsNpy(b64(prefill), b64(decode))
	if err != nil {
		t.Fatalf("splice: %v", err)
	}
	merged, _ := base64.StdEncoding.DecodeString(out)
	h := parseNpyHeader(merged)
	if h.shape[0] != 5 {
		t.Fatalf("merged keeps decode rows, got %d", h.shape[0])
	}
	rb := l * k
	data := merged[h.dataOffset:]
	// rows 0,1 from prefill (100,101); rows 2,3,4 from decode (2,3,4).
	for r, want := range []byte{100, 101, 2, 3, 4} {
		if got := data[r*rb]; got != want {
			t.Fatalf("row %d: got %d want %d", r, got, want)
		}
	}
}

func TestSpliceEqualLengthsUsesAllPrefill(t *testing.T) {
	out, err := spliceRoutedExpertsNpy(b64(makeNpyU8(4, 1, 1, 50)), b64(makeNpyU8(4, 1, 1, 0)))
	if err != nil {
		t.Fatalf("splice: %v", err)
	}
	merged, _ := base64.StdEncoding.DecodeString(out)
	h := parseNpyHeader(merged)
	data := merged[h.dataOffset:]
	for i, want := range []byte{50, 51, 52, 53} {
		if data[i] != want {
			t.Fatalf("byte %d: got %d want %d", i, data[i], want)
		}
	}
}

func TestMergeRoutedExpertsJSON(t *testing.T) {
	l, k := 2, 2
	prefillBody, _ := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"index": 0, "routed_experts": b64(makeNpyU8(2, l, k, 100))}},
	})
	decodeBody, _ := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"index": 0, "routed_experts": b64(makeNpyU8(5, l, k, 0))}},
	})

	var prefillResp map[string]any
	_ = json.Unmarshal(prefillBody, &prefillResp)
	prefillByIdx := extractRoutedExperts(prefillResp)
	if len(prefillByIdx) != 1 {
		t.Fatalf("extract: got %d", len(prefillByIdx))
	}

	out, merged := mergeRoutedExpertsJSON(prefillByIdx, decodeBody)
	if !merged {
		t.Fatal("expected merge")
	}
	var resp map[string]any
	_ = json.Unmarshal(out, &resp)
	re := resp["choices"].([]any)[0].(map[string]any)["routed_experts"].(string)
	mergedBytes, _ := base64.StdEncoding.DecodeString(re)
	h := parseNpyHeader(mergedBytes)
	if h.shape[0] != 5 {
		t.Fatalf("merged shape rows: got %d", h.shape[0])
	}
	if mergedBytes[h.dataOffset] != 100 {
		t.Fatalf("first row not from prefill: got %d", mergedBytes[h.dataOffset])
	}
}

func TestMergeNoOpWhenAbsent(t *testing.T) {
	decodeBody, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"index": 0}}})
	if _, merged := mergeRoutedExpertsJSON(map[int]string{0: "x"}, decodeBody); merged {
		t.Fatal("expected no merge when decode choice has no routed_experts")
	}
	if _, merged := mergeRoutedExpertsJSON(nil, decodeBody); merged {
		t.Fatal("expected no merge with empty prefill map")
	}
}
