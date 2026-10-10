/*
Copyright 2026 The Kubernetes Authors.
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

// Package prefixhash computes stable prefix block hashes for a tokenized
// prompt. It is shared by the prefix-aware data producers so they derive
// identical block hashes for the same request.
package prefixhash

import (
	"context"
	"encoding/binary"
	"unsafe"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// BlockHash is a hash of a block of request data.
type BlockHash uint64

// HashBlock wraps a block of token IDs used for calculating prefix hashes.
type HashBlock struct {
	// Tokens are the token IDs covered by this block.
	Tokens []uint32
}

// Hash computes a stable unique identifier for the HashBlock content.
func (b HashBlock) Hash() uint64 {
	if len(b.Tokens) > 0 {
		// Reinterprets the uint32 slice as bytes to hash without copying. Safe
		// because the length check above guarantees a valid backing array, and
		// the byte length (len(Tokens) * 4) matches uint32's size exactly, so
		// the resulting slice stays within the array's bounds.
		byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&b.Tokens[0])), len(b.Tokens)*4) //#nosec G103 -- see comment above
		return xxhash.Sum64(byteSlice)
	}

	return 0
}

// GetBlockHashes divides the tokenized prompt into blocks and calculates a
// prefix cache hash for each block. Each prompt in Prompts is hashed
// independently so cross-prompt block adjacency is avoided. The first block
// hash of every prompt includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
// It requires the token producer to populate TokenizedPromptDataKey.
func GetBlockHashes(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) [][]BlockHash {
	hashes, _ := GetBlockHashesWithPromptTokens(ctx, request, blockSizeTokens, maxPrefixBlocks)
	return hashes
}

// GetBlockHashesWithPromptTokens hashes as GetBlockHashes does and additionally
// returns the token count of the prompt behind each entry, positionally aligned
// with the hashes. A prompt's final block may be partial, so converting a
// matched block count back to tokens overshoots unless it is bounded by the
// length of the prompt that produced the blocks.
func GetBlockHashesWithPromptTokens(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) ([][]BlockHash, []int) {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil {
		loggerDebug.Info("Request is nil, skipping hashing")
		return nil, nil
	}

	tp, ok := scheduling.ReadRequestAttribute[*scheduling.TokenizedRequest](request, tokenproducer.TokenizedPromptDataKey)
	if !ok || tp.TokenCount() == 0 {
		loggerDebug.Info("TokenizedRequest is empty, skipping hashing")
		return nil, nil
	}

	var result [][]BlockHash
	var promptTokens []int
	for _, p := range tp.Prompts {
		hashes := computeBlockHashes(request.TargetModel, tp.CacheSalt, p.TokenIDs, blockSizeTokens, maxPrefixBlocks)
		if len(hashes) > 0 {
			result = append(result, hashes)
			promptTokens = append(promptTokens, len(p.TokenIDs))
		}
	}
	if len(result) == 0 {
		loggerDebug.Info("No kv cache block found")
		return nil, nil
	}
	return result, promptTokens
}

// computeBlockHashes hashes a prompt in blocks of blockSizeTokens tokens, the
// last possibly partial, up to maxPrefixBlocks blocks. Block i's hash is xxhash
// over its content hash and block i-1's hash, both little-endian; block 0
// chains from a seed hashed over the target model and cache salt.
func computeBlockHashes(targetModel, cacheSalt string, tokens []uint32, blockSizeTokens, maxPrefixBlocks int) []BlockHash {
	if len(tokens) == 0 || blockSizeTokens <= 0 || maxPrefixBlocks <= 0 {
		return nil
	}
	count := min((len(tokens)-1)/blockSizeTokens+1, maxPrefixBlocks)
	blockHashes := make([]BlockHash, 0, count)

	// Different models should have different hashes even with the same body.
	var seed xxhash.Digest
	seed.Reset()
	_, _ = seed.WriteString(targetModel)
	_, _ = seed.WriteString(cacheSalt)
	prevBlockHash := BlockHash(seed.Sum64())

	var buf [16]byte
	for i := range count {
		block := HashBlock{Tokens: tokens[i*blockSizeTokens : min((i+1)*blockSizeTokens, len(tokens))]}
		PutBlockHash(buf[:8], BlockHash(block.Hash()))
		PutBlockHash(buf[8:], prevBlockHash)
		prevBlockHash = BlockHash(xxhash.Sum64(buf[:]))
		blockHashes = append(blockHashes, prevBlockHash)
	}

	return blockHashes
}

// PutBlockHash writes h into the first 8 bytes of buf in little-endian order.
// buf must have length at least 8. It lets callers serialize block hashes into
// a reused buffer without allocating per hash.
func PutBlockHash(buf []byte, h BlockHash) {
	binary.LittleEndian.PutUint64(buf, uint64(h))
}
