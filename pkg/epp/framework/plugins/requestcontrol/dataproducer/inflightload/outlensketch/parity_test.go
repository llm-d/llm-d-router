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
	"encoding/json"
	"math"
	"os"
	"testing"
)

// Parity against the reference model2vec potion-base-8M implementation. Skipped
// unless both OSL_MODEL_DIR (the unpacked model directory) and OSL_PARITY_FIXTURE
// (a JSON dump of reference token ids and embeddings) are set, so it never runs in
// CI without the model asset. Generate the fixture with the study's model2vec.
type parityFixture struct {
	Dim   int `json:"dim"`
	Cases []struct {
		Text      string    `json:"text"`
		RawIDs    []int32   `json:"raw_ids"`
		Embedding []float64 `json:"embedding"`
	} `json:"cases"`
}

func loadParity(t *testing.T) (*embedder, parityFixture) {
	t.Helper()
	modelDir := os.Getenv("OSL_MODEL_DIR")
	fixturePath := os.Getenv("OSL_PARITY_FIXTURE")
	if modelDir == "" || fixturePath == "" {
		t.Skip("set OSL_MODEL_DIR and OSL_PARITY_FIXTURE to run model2vec parity")
	}
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx parityFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	emb, err := newEmbedder(modelDir)
	if err != nil {
		t.Fatalf("load embedder: %v", err)
	}
	if emb.dim != fx.Dim {
		t.Fatalf("dim mismatch: go %d, fixture %d", emb.dim, fx.Dim)
	}
	return emb, fx
}

func TestWordPieceParity(t *testing.T) {
	emb, fx := loadParity(t)
	for _, c := range fx.Cases {
		got := emb.wp.encode(c.Text)
		if len(got) != len(c.RawIDs) {
			t.Errorf("token count for %q: got %d, want %d\n got=%v\nwant=%v", c.Text, len(got), len(c.RawIDs), got, c.RawIDs)
			continue
		}
		for i := range got {
			if got[i] != c.RawIDs[i] {
				t.Errorf("token %d for %q: got %d, want %d", i, c.Text, got[i], c.RawIDs[i])
				break
			}
		}
	}
}

func TestEmbeddingParity(t *testing.T) {
	emb, fx := loadParity(t)
	for _, c := range fx.Cases {
		got := emb.encode(c.Text)
		if len(got) != len(c.Embedding) {
			t.Fatalf("embedding dim for %q: got %d, want %d", c.Text, len(got), len(c.Embedding))
		}
		var dot, nGot, nWant float64
		for i := range got {
			g, w := float64(got[i]), c.Embedding[i]
			dot += g * w
			nGot += g * g
			nWant += w * w
		}
		if nGot == 0 && nWant == 0 { // both zero vectors (empty prompt)
			continue
		}
		cos := dot / (math.Sqrt(nGot)*math.Sqrt(nWant) + 1e-32)
		if cos < 0.9999 {
			t.Errorf("embedding cosine for %q = %.6f, want >= 0.9999", c.Text, cos)
		}
	}
}
