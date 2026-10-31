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
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// Static-embedding encoder for the model2vec potion-base-8M distillation: tokenize
// with WordPiece, look up each token's row in a precomputed table, mean-pool, and
// L2-normalize. PCA and Zipf weighting are baked into the table at distillation
// time, so encoding is a table read and an average -- no neural forward pass. The
// table is loaded once at startup from a mounted model directory and is read-only
// on the hot path.
const (
	// vocabFile and tableFile are the model directory entries the encoder reads.
	vocabFile = "vocab.txt"
	tableFile = "model.safetensors"
	// tableTensorName is the safetensors tensor holding the embedding matrix.
	tableTensorName = "embeddings"
	// maxEncodeTokens caps the tokens averaged per prompt, matching model2vec's
	// default encode max_length. At the W1 window budget the cap is never reached.
	maxEncodeTokens = 512
	// l2Epsilon guards the L2 normalization against a zero-norm vector, matching
	// model2vec's +1e-32.
	l2Epsilon = 1e-32
)

// embedder encodes windowed prompt text into a dim-dimensional unit vector.
type embedder struct {
	wp    *wordPiece
	table []float32 // row-major vocabN x dim
	dim   int
}

// newEmbedder loads the vocabulary and embedding table from a mounted model
// directory (the unpacked potion-base-8M model: vocab.txt + model.safetensors).
func newEmbedder(modelDir string) (*embedder, error) {
	vocab, err := loadVocab(filepath.Join(modelDir, vocabFile))
	if err != nil {
		return nil, err
	}
	table, rows, dim, err := loadTable(filepath.Join(modelDir, tableFile))
	if err != nil {
		return nil, err
	}
	if rows != len(vocab) {
		return nil, fmt.Errorf("embedding table rows (%d) do not match vocabulary size (%d)", rows, len(vocab))
	}
	return &embedder{wp: newWordPiece(vocab), table: table, dim: dim}, nil
}

// encode returns the mean-pooled, L2-normalized embedding of text. Unknown tokens
// are dropped before pooling; an all-unknown or empty prompt yields the zero vector.
func (e *embedder) encode(text string) []float32 {
	ids := e.wp.encode(text)
	sum := make([]float64, e.dim)
	n := 0
	for _, id := range ids {
		if id == e.wp.unkID || id < 0 || int(id) >= len(e.table)/e.dim {
			continue
		}
		if n >= maxEncodeTokens {
			break
		}
		row := e.table[int(id)*e.dim : int(id)*e.dim+e.dim]
		for j, v := range row {
			sum[j] += float64(v)
		}
		n++
	}
	out := make([]float32, e.dim)
	if n == 0 {
		return out
	}
	var norm float64
	for j := range sum {
		mean := sum[j] / float64(n)
		sum[j] = mean
		norm += mean * mean
	}
	norm = math.Sqrt(norm) + l2Epsilon
	for j := range out {
		out[j] = float32(sum[j] / norm)
	}
	return out
}

// loadVocab reads vocab.txt, assigning id n to the token on line n (0-indexed),
// matching the model's tokenizer vocabulary ordering.
func loadVocab(path string) (map[string]int32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open vocab %q: %w", path, err)
	}
	defer f.Close()

	vocab := make(map[string]int32)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var id int32
	for sc.Scan() {
		// vocab.txt tokens never contain surrounding whitespace; the file is one
		// token per line, so the raw line text (minus the line ending Scan strips)
		// is the token. Blank lines would be real tokens, so they are not skipped.
		vocab[sc.Text()] = id
		id++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read vocab %q: %w", path, err)
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("vocab %q is empty", path)
	}
	return vocab, nil
}

// safetensorsHeader is the per-tensor metadata in a safetensors file header.
type safetensorsHeader struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

// loadTable reads the embedding matrix from a safetensors file. The format is an
// 8-byte little-endian header length, a JSON header mapping tensor names to
// {dtype, shape, data_offsets}, then the raw tensor bytes. Only the float32
// embeddings tensor is read.
func loadTable(path string) (table []float32, rows, dim int, err error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("open embedding table %q: %w", path, err)
	}
	if len(raw) < 8 {
		return nil, 0, 0, fmt.Errorf("embedding table %q too short", path)
	}
	hdrLen := binary.LittleEndian.Uint64(raw[:8])
	if uint64(len(raw)) < 8+hdrLen {
		return nil, 0, 0, fmt.Errorf("embedding table %q header length %d exceeds file size", path, hdrLen)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+hdrLen], &header); err != nil {
		return nil, 0, 0, fmt.Errorf("parse embedding table header: %w", err)
	}
	rawTensor, ok := header[tableTensorName]
	if !ok {
		return nil, 0, 0, fmt.Errorf("embedding table %q has no %q tensor", path, tableTensorName)
	}
	var meta safetensorsHeader
	if err := json.Unmarshal(rawTensor, &meta); err != nil {
		return nil, 0, 0, fmt.Errorf("parse %q tensor metadata: %w", tableTensorName, err)
	}
	if meta.DType != "F32" {
		return nil, 0, 0, fmt.Errorf("embedding table dtype %q is not F32", meta.DType)
	}
	if len(meta.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("embedding table shape %v is not 2-dimensional", meta.Shape)
	}
	rows, dim = meta.Shape[0], meta.Shape[1]
	if len(meta.DataOffsets) != 2 {
		return nil, 0, 0, fmt.Errorf("embedding table data_offsets %v malformed", meta.DataOffsets)
	}
	dataStart := 8 + int(hdrLen)
	begin, end := dataStart+meta.DataOffsets[0], dataStart+meta.DataOffsets[1]
	if begin < 0 || end > len(raw) || begin > end {
		return nil, 0, 0, fmt.Errorf("embedding table tensor bounds [%d,%d] outside file", begin, end)
	}
	tensorBytes := raw[begin:end]
	if len(tensorBytes) != rows*dim*4 {
		return nil, 0, 0, fmt.Errorf("embedding table has %d bytes, expected %d for shape %v", len(tensorBytes), rows*dim*4, meta.Shape)
	}
	table = make([]float32, rows*dim)
	for i := range table {
		table[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensorBytes[i*4:]))
	}
	return table, rows, dim, nil
}
