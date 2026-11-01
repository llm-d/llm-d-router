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
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeTestModel writes a synthetic model directory (vocab.txt + model.safetensors)
// so the embedder can be exercised without the 30 MB production asset.
func writeTestModel(t *testing.T, tokens []string, rows [][]float32) string {
	t.Helper()
	dir := t.TempDir()

	var vocab string
	for _, tok := range tokens {
		vocab += tok + "\n"
	}
	if err := os.WriteFile(filepath.Join(dir, vocabFile), []byte(vocab), 0o600); err != nil {
		t.Fatal(err)
	}

	dim := len(rows[0])
	data := make([]byte, 0, len(rows)*dim*4)
	for _, row := range rows {
		if len(row) != dim {
			t.Fatalf("ragged table row: got %d, want %d", len(row), dim)
		}
		for _, v := range row {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			data = append(data, b[:]...)
		}
	}
	header, err := json.Marshal(map[string]safetensorsHeader{
		tableTensorName: {DType: "F32", Shape: []int{len(rows), dim}, DataOffsets: []int{0, len(data)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var buf []byte
	var lenBytes [8]byte
	binary.LittleEndian.PutUint64(lenBytes[:], uint64(len(header)))
	buf = append(buf, lenBytes[:]...)
	buf = append(buf, header...)
	buf = append(buf, data...)
	if err := os.WriteFile(filepath.Join(dir, tableFile), buf, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestEmbedMeanPoolAndNormalize(t *testing.T) {
	dir := writeTestModel(t,
		[]string{"[UNK]", "aa", "bb"},
		[][]float32{{9, 9}, {1, 0}, {0, 1}},
	)
	e, err := newEmbedder(dir)
	if err != nil {
		t.Fatalf("newEmbedder: %v", err)
	}

	// mean([1,0],[0,1]) = [0.5,0.5] -> L2 -> [0.707,0.707].
	got := e.encode("aa bb")
	want := float32(1 / math.Sqrt2)
	if !approxF32(got[0], want) || !approxF32(got[1], want) {
		t.Errorf("encode(aa bb) = %v, want [%.3f %.3f]", got, want, want)
	}

	// Unknown tokens are dropped; the [UNK] row (9,9) must not leak in.
	got = e.encode("aa zz")
	if !approxF32(got[0], 1) || !approxF32(got[1], 0) {
		t.Errorf("encode(aa zz) = %v, want [1 0] (unk dropped)", got)
	}

	// All-unknown prompt yields the zero vector.
	if got := e.encode("zz yy"); got[0] != 0 || got[1] != 0 {
		t.Errorf("encode(all-unknown) = %v, want zero vector", got)
	}
}

func TestEmbedLoaderErrors(t *testing.T) {
	if _, err := newEmbedder(t.TempDir()); err == nil {
		t.Error("expected error loading from an empty directory")
	}

	// Table rows must match the vocabulary size.
	dir := writeTestModel(t, []string{"[UNK]", "aa"}, [][]float32{{1, 2}, {3, 4}, {5, 6}})
	if _, err := newEmbedder(dir); err == nil {
		t.Error("expected error when table rows do not match vocab size")
	}
}

func approxF32(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-5
}
