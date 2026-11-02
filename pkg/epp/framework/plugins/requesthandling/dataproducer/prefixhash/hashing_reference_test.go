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

package prefixhash

import (
	"context"
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"
	"unsafe"

	"github.com/cespare/xxhash/v2"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// referenceSeed is the (model, salt) seed: xxhash over each string
// length-prefixed as an 8-byte little-endian integer. The length prefix removes
// boundaries that would otherwise let distinct pairs share a seed.
func referenceSeed(model, salt string) uint64 {
	var seed []byte
	for _, s := range []string{model, salt} {
		var lenBuf [8]byte
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(s)))
		seed = append(seed, lenBuf[:]...)
		seed = append(seed, s...)
	}
	return xxhash.Sum64(seed)
}

// referenceBlockHashes is an independent statement of the block hash chain:
// the seed is xxhash over the length-prefixed model and salt; block i hashes
// its token bytes, and its chain hash is xxhash(LE(block hash) || LE(chain hash
// of block i-1) || mmExtra), where mmExtra covers the multimodal features
// overlapping the block in offset order. Every prompt starts from the seed,
// the final block may be partial, and empty prompts produce no entry.
func referenceBlockHashes(model, salt string, prompts []fwkrequest.PromptTokens, blockSize, maxBlocks int) ([][]BlockHash, []int) {
	var hashes [][]BlockHash
	var tokens []int
	seed := referenceSeed(model, salt)
	for _, p := range prompts {
		prompt := p.TokenIDs
		if len(prompt) == 0 || blockSize <= 0 {
			continue
		}
		var chain []BlockHash
		prev := seed
		for start := 0; start < len(prompt) && len(chain) < maxBlocks; start += blockSize {
			block := prompt[start:min(start+blockSize, len(prompt))]
			content := xxhash.Sum64(unsafe.Slice((*byte)(unsafe.Pointer(&block[0])), len(block)*4)) //#nosec G103 -- test reference over a non-empty slice
			input := make([]byte, 0, 16)
			var buf [16]byte
			binary.LittleEndian.PutUint64(buf[:8], content)
			binary.LittleEndian.PutUint64(buf[8:], prev)
			input = append(input, buf[:]...)
			input = append(input, referenceMMExtra(p.MultiModalFeatures, start, start+len(block))...)
			prev = xxhash.Sum64(input)
			chain = append(chain, BlockHash(prev))
		}
		hashes = append(hashes, chain)
		tokens = append(tokens, len(prompt))
	}
	if len(hashes) == 0 {
		return nil, nil
	}
	return hashes, tokens
}

// referenceMMExtra is the independent statement of the media bytes mixed into
// a block's hash: for each feature overlapping [start, end), in offset order,
// its length-prefixed hash followed by its entry offset from the block start.
// The end is the block's clamped extent, so a span past a partial final block
// is ignored.
func referenceMMExtra(features []fwkrequest.MultiModalFeature, start, end int) []byte {
	sorted := append([]fwkrequest.MultiModalFeature(nil), features...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Offset < sorted[j].Offset })
	var out []byte
	var lenBuf [8]byte
	for _, f := range sorted {
		if f.Length <= 0 || f.Hash == "" {
			continue
		}
		if f.Offset >= end || f.Offset+f.Length <= start {
			continue
		}
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(f.Hash)))
		out = append(out, lenBuf[:]...)
		out = append(out, f.Hash...)
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(max(f.Offset, start)-start))
		out = append(out, lenBuf[:]...)
	}
	return out
}

// TestGetBlockHashesMatchesReference compares GetBlockHashesWithPromptTokens
// with the reference chain over random prompts, block sizes, caps, and salts.
func TestGetBlockHashesMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	for iter := range 2000 {
		prompts := make([]fwkrequest.PromptTokens, rng.IntN(4))
		for i := range prompts {
			tokens := make([]uint32, rng.IntN(70))
			for j := range tokens {
				tokens[j] = rng.Uint32()
			}
			prompts[i] = fwkrequest.PromptTokens{TokenIDs: tokens}
			// Features: in-range spans, spans past the prompt end, zero
			// lengths, and duplicate offsets, so the reference covers the
			// mixing, the clamped block end, and the position ordering.
			for range rng.IntN(4) {
				hash := fmt.Sprintf("hash-%d", rng.IntN(5))
				if rng.IntN(4) == 0 {
					hash = "" // a hashless feature must encode as featureless
				}
				prompts[i].MultiModalFeatures = append(prompts[i].MultiModalFeatures, fwkrequest.MultiModalFeature{
					Modality: "image",
					Hash:     hash,
					Offset:   rng.IntN(len(tokens)+10) - 2,
					Length:   rng.IntN(9) - 1,
				})
			}
		}
		blockSize := rng.IntN(18) - 1
		maxBlocks := 1 + rng.IntN(20)
		salt := ""
		if rng.IntN(2) == 0 {
			salt = fmt.Sprintf("salt-%d", rng.IntN(3))
		}
		model := fmt.Sprintf("model-%d", rng.IntN(3))

		req := &fwksched.InferenceRequest{TargetModel: model, Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{CacheSalt: salt, Prompts: prompts}}}

		gotHashes, gotTokens := GetBlockHashesWithPromptTokens(context.Background(), req, blockSize, maxBlocks)
		wantHashes, wantTokens := referenceBlockHashes(model, salt, prompts, blockSize, maxBlocks)
		assert.Equal(t, wantHashes, gotHashes, "iter %d: block size %d, cap %d", iter, blockSize, maxBlocks)
		assert.Equal(t, wantTokens, gotTokens, "iter %d", iter)
	}
}

func TestGetBlockHashesMultiModalSpansMatchReference(t *testing.T) {
	for _, tt := range []struct {
		name     string
		features []fwkrequest.MultiModalFeature
	}{
		{
			name: "disjoint spans across blocks and gaps",
			features: []fwkrequest.MultiModalFeature{
				{Hash: "last", Offset: 18, Length: 4},
				{Hash: "first", Offset: 2, Length: 4},
				{Hash: "middle", Offset: 10, Length: 2},
			},
		},
		{
			name: "long span outlives nested spans",
			features: []fwkrequest.MultiModalFeature{
				{Hash: "long", Offset: 0, Length: 24},
				{Hash: "short", Offset: 2, Length: 2},
				{Hash: "middle", Offset: 6, Length: 4},
				{Hash: "later", Offset: 18, Length: 2},
			},
		},
		{
			name: "equal offsets retain input order",
			features: []fwkrequest.MultiModalFeature{
				{Hash: "first", Offset: 2, Length: 2},
				{Hash: "second", Offset: 2, Length: 12},
				{Hash: "third", Offset: 2, Length: 6},
			},
		},
		{
			name: "invalid spans and partial final block",
			features: []fwkrequest.MultiModalFeature{
				{Hash: "past-end", Offset: 23, Length: 2},
				{Hash: "negative-start", Offset: -2, Length: 5},
				{Hash: "empty", Offset: 1, Length: 0},
				{Hash: "negative-length", Offset: 5, Length: -1},
				{Hash: "partial", Offset: 21, Length: 8},
				{Hash: "before-start", Offset: -4, Length: 2},
			},
		},
		{
			name: "single spanning feature",
			features: []fwkrequest.MultiModalFeature{
				{Hash: "single", Offset: 2, Length: 12},
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			original := slices.Clone(tt.features)
			prompts := []fwkrequest.PromptTokens{{TokenIDs: make([]uint32, 23), MultiModalFeatures: tt.features}}
			req := &fwksched.InferenceRequest{TargetModel: "m", Body: &fwkrequest.InferenceRequestBody{
				TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: prompts}}}
			for _, maxBlocks := range []int{3, 6} {
				wantHashes, wantTokens := referenceBlockHashes("m", "", prompts, 4, maxBlocks)
				gotHashes, gotTokens := GetBlockHashesWithPromptTokens(context.Background(), req, 4, maxBlocks)
				assert.Equal(t, wantHashes, gotHashes, "cap %d", maxBlocks)
				assert.Equal(t, wantTokens, gotTokens, "cap %d", maxBlocks)
				assert.Equal(t, original, tt.features, "hashing must not mutate the request's features")
			}
		})
	}
}

func BenchmarkGetBlockHashesMultiModal(b *testing.B) {
	for _, tt := range []struct {
		name         string
		blocks       int
		featureCount int
		longSpan     bool
	}{
		{name: "single", blocks: 2048, featureCount: 1},
		{name: "small", blocks: 8, featureCount: 4},
		{name: "sparse", blocks: 2048, featureCount: 256},
		{name: "nested", blocks: 2048, featureCount: 256, longSpan: true},
	} {
		b.Run(tt.name, func(b *testing.B) {
			ctx := log.IntoContext(b.Context(), logr.Discard())
			tokens := make([]uint32, tt.blocks*64)
			for i := range tokens {
				tokens[i] = uint32(i)
			}
			features := make([]fwkrequest.MultiModalFeature, tt.featureCount)
			for i := range features {
				features[i] = fwkrequest.MultiModalFeature{
					Modality: "image",
					Hash:     fmt.Sprintf("%064d", i),
					Offset:   i * (len(tokens) / tt.featureCount),
					Length:   96,
				}
			}
			if tt.longSpan {
				features[0].Length = len(tokens)
			}
			req := &fwksched.InferenceRequest{TargetModel: "bench-model", Body: &fwkrequest.InferenceRequestBody{
				TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{
					TokenIDs: tokens, MultiModalFeatures: features,
				}}}}}
			b.ReportAllocs()
			for b.Loop() {
				GetBlockHashesWithPromptTokens(ctx, req, 64, 2048)
			}
		})
	}
}

// BenchmarkGetBlockHashes hashes one 32,768-token prompt in 64-token blocks.
func BenchmarkGetBlockHashes(b *testing.B) {
	ctx := log.IntoContext(b.Context(), logr.Discard())
	tokens := make([]uint32, 512*64)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	req := &fwksched.InferenceRequest{TargetModel: "bench-model", Body: &fwkrequest.InferenceRequestBody{
		TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens}}}}}
	b.ReportAllocs()
	for b.Loop() {
		GetBlockHashesWithPromptTokens(ctx, req, 64, 2048)
	}
}
