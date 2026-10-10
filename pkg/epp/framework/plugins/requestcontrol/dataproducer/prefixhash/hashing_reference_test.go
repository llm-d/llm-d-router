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

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// referenceBlockHashes is an independent statement of the block hash chain:
// the seed is xxhash(model || salt); block i hashes its token bytes, and its
// chain hash is xxhash(LE(block hash) || LE(chain hash of block i-1)). Every
// prompt starts from the seed, the final block may be partial, and empty
// prompts produce no entry.
func referenceBlockHashes(model, salt string, prompts [][]uint32, blockSize, maxBlocks int) ([][]BlockHash, []int) {
	var hashes [][]BlockHash
	var tokens []int
	seed := xxhash.Sum64String(model + salt)
	for _, prompt := range prompts {
		if len(prompt) == 0 || blockSize <= 0 {
			continue
		}
		var chain []BlockHash
		prev := seed
		for start := 0; start < len(prompt) && len(chain) < maxBlocks; start += blockSize {
			block := prompt[start:min(start+blockSize, len(prompt))]
			content := xxhash.Sum64(unsafe.Slice((*byte)(unsafe.Pointer(&block[0])), len(block)*4)) //#nosec G103 -- test reference over a non-empty slice
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
		prompts := make([][]uint32, rng.IntN(4))
		for i := range prompts {
			prompts[i] = make([]uint32, rng.IntN(70))
			for j := range prompts[i] {
				prompts[i][j] = rng.Uint32()
			}
		}
		blockSize := rng.IntN(18) - 1
		maxBlocks := 1 + rng.IntN(20)
		salt := ""
		if rng.IntN(2) == 0 {
			salt = fmt.Sprintf("salt-%d", rng.IntN(3))
		}
		model := fmt.Sprintf("model-%d", rng.IntN(3))

		req := &fwksched.InferenceRequest{TargetModel: model, Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{CacheSalt: salt}}}
		for _, p := range prompts {
			req.Body.TokenizedRequest.Prompts = append(req.Body.TokenizedRequest.Prompts, fwkrh.PromptTokens{TokenIDs: p})
		}

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
	req := &fwksched.InferenceRequest{TargetModel: "bench-model", Body: &fwkrh.InferenceRequestBody{
		TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: tokens}}}}}
	b.ReportAllocs()
	for b.Loop() {
		GetBlockHashesWithPromptTokens(context.Background(), req, 64, 2048)
	}
}
