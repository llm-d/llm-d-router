/*
Copyright 2026 The Kubernetes Authors.

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
	"iter"
	"unsafe"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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
		byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&b.Tokens[0])), len(b.Tokens)*4)
		return xxhash.Sum64(byteSlice)
	}

	return 0
}

// GetBlockHashes divides the tokenized prompt into blocks and calculates a
// prefix cache hash for each block. Each prompt in Prompts is hashed
// independently so cross-prompt block adjacency is avoided. The first block
// hash of every prompt includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
// It requires request.Body.TokenizedRequest to be populated by a token-producer backend.
func GetBlockHashes(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) [][]BlockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}

	tp := request.Body.TokenizedRequest
	if tp == nil || tp.TokenCount() == 0 {
		loggerDebug.Info("TokenizedRequest is empty, skipping hashing")
		return nil
	}

	result := make([][]BlockHash, 0, len(tp.Prompts))
	for _, p := range tp.Prompts {
		hashes := hashPrompt(p.TokenIDs, blockSizeTokens, maxPrefixBlocks, request.TargetModel, tp.CacheSalt)
		if len(hashes) > 0 {
			result = append(result, hashes)
		}
	}
	if len(result) == 0 {
		loggerDebug.Info("No kv cache block found")
		return nil
	}
	return result
}

func hashPrompt(tokens []uint32, blockSizeTokens, maxPrefixBlocks int, targetModel, cacheSalt string) []BlockHash {
	if len(tokens) == 0 || blockSizeTokens <= 0 || maxPrefixBlocks <= 0 {
		return nil
	}

	numBlocks := (len(tokens) + blockSizeTokens - 1) / blockSizeTokens
	if numBlocks > maxPrefixBlocks {
		numBlocks = maxPrefixBlocks
	}

	// Different models should have different hashes even with the same body.
	var prevBlockHash BlockHash
	if cacheSalt == "" {
		prevBlockHash = BlockHash(xxhash.Sum64String(targetModel))
	} else {
		var h xxhash.Digest
		h.Reset()
		_, _ = h.WriteString(targetModel)
		_, _ = h.WriteString(cacheSalt)
		prevBlockHash = BlockHash(h.Sum64())
	}

	blockHashes := make([]BlockHash, 0, numBlocks)
	var buf [16]byte
	for i := range numBlocks {
		start := i * blockSizeTokens
		end := min(start+blockSizeTokens, len(tokens))
		blockID := (HashBlock{Tokens: tokens[start:end]}).Hash()
		PutBlockHash(buf[:8], BlockHash(blockID))
		PutBlockHash(buf[8:16], prevBlockHash)
		prevBlockHash = BlockHash(xxhash.Sum64(buf[:16]))
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

func getKVCacheBlocksFromTokens(ids []uint32, blockSizeTokens int) iter.Seq[HashBlock] {
	return func(yield func(HashBlock) bool) {
		if len(ids) == 0 || blockSizeTokens <= 0 {
			return
		}
		for i := 0; i < len(ids); i += blockSizeTokens {
			end := i + blockSizeTokens
			if end > len(ids) {
				end = len(ids)
			}
			if !yield(HashBlock{Tokens: ids[i:end]}) {
				return
			}
		}
	}
}
