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
	"testing"
	"unsafe"

	"github.com/cespare/xxhash/v2"
	"github.com/stretchr/testify/assert"

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
// of block i-1)). A block overlapping multimodal items replaces its token hash
// with xxhash(LE(token hash) || the length-prefixed hash of each overlapping
// item, in feature order). Every prompt starts from the seed, the final block
// may be partial, and empty prompts produce no entry.
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
			var mm []byte
			for _, f := range p.MultiModalFeatures {
				if f.Offset < start+len(block) && start < f.Offset+f.Length {
					mm = binary.LittleEndian.AppendUint64(mm, uint64(len(f.Hash)))
					mm = append(mm, f.Hash...)
				}
			}
			if mm != nil {
				content = xxhash.Sum64(append(binary.LittleEndian.AppendUint64(nil, content), mm...))
			}
			var buf [16]byte
			binary.LittleEndian.PutUint64(buf[:8], content)
			binary.LittleEndian.PutUint64(buf[8:], prev)
			prev = xxhash.Sum64(buf[:])
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
			prompts[i].TokenIDs = tokens
			// Up to three items, in offset order as token producers emit them.
			offset := 0
			for range rng.IntN(4) {
				if offset >= len(tokens) {
					break
				}
				offset += rng.IntN(len(tokens) - offset)
				length := 1 + rng.IntN(len(tokens)-offset)
				prompts[i].MultiModalFeatures = append(prompts[i].MultiModalFeatures, fwkrequest.MultiModalFeature{
					Hash: fmt.Sprintf("item-%d", rng.IntN(1000)), Offset: offset, Length: length,
				})
				offset += length
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
			TokenizedRequest: &fwkrequest.TokenizedRequest{CacheSalt: salt}}}
		req.Body.TokenizedRequest.Prompts = prompts

		gotHashes, gotTokens := GetBlockHashesWithPromptTokens(context.Background(), req, blockSize, maxBlocks)
		wantHashes, wantTokens := referenceBlockHashes(model, salt, prompts, blockSize, maxBlocks)
		assert.Equal(t, wantHashes, gotHashes, "iter %d: block size %d, cap %d", iter, blockSize, maxBlocks)
		assert.Equal(t, wantTokens, gotTokens, "iter %d", iter)
	}
}

// BenchmarkGetBlockHashes hashes one 32,768-token prompt in 64-token blocks.
func BenchmarkGetBlockHashes(b *testing.B) {
	tokens := make([]uint32, 512*64)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	req := &fwksched.InferenceRequest{TargetModel: "bench-model", Body: &fwkrequest.InferenceRequestBody{
		TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens}}}}}
	b.ReportAllocs()
	for b.Loop() {
		GetBlockHashesWithPromptTokens(context.Background(), req, 64, 2048)
	}
}
