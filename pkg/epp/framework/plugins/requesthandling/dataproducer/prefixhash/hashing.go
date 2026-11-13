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
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
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
// It requires request.Body.TokenizedRequest to be populated by a token-producer backend.
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
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil, nil
	}

	tp := request.Body.TokenizedRequest
	if tp == nil || tp.TokenCount() == 0 {
		loggerDebug.Info("TokenizedRequest is empty, skipping hashing")
		return nil, nil
	}

	var result [][]BlockHash
	var promptTokens []int
	for _, p := range tp.Prompts {
		hashes := computeBlockHashes(request, p.TokenIDs, p.MultiModalFeatures, blockSizeTokens, maxPrefixBlocks)
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
// chains from a seed hashed over the target model and cache salt. A block that
// overlaps a multimodal placeholder range also covers that item's hash, because
// placeholder token IDs are the same for every item of a given size.
func computeBlockHashes(request *scheduling.InferenceRequest, tokens []uint32, mmFeatures []fwkrequest.MultiModalFeature, blockSizeTokens, maxPrefixBlocks int) []BlockHash {
	if len(tokens) == 0 || blockSizeTokens <= 0 || maxPrefixBlocks <= 0 {
		return nil
	}
	count := min((len(tokens)-1)/blockSizeTokens+1, maxPrefixBlocks)
	blockHashes := make([]BlockHash, 0, count)

	// Different models should have different hashes even with the same body.
	// Length-prefixing each field keeps distinct model and salt pairs from
	// encoding to the same bytes, so model "ab" with salt "c" and model "a"
	// with salt "bc" no longer share a seed.
	var seed xxhash.Digest
	seed.Reset()
	var seedLen [8]byte
	writeLengthPrefixedString(&seed, seedLen[:], request.TargetModel)
	writeLengthPrefixedString(&seed, seedLen[:], request.Body.TokenizedRequest.CacheSalt)
	prevBlockHash := BlockHash(seed.Sum64())

	var buf [16]byte
	for i := range count {
		start, end := i*blockSizeTokens, min((i+1)*blockSizeTokens, len(tokens))
		block := HashBlock{Tokens: tokens[start:end]}
		content := block.Hash()
		if len(mmFeatures) > 0 {
			content = withMMHashes(content, mmFeatures, start, end)
		}
		PutBlockHash(buf[:8], BlockHash(content))
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

// writeLengthPrefixedString writes s to d preceded by its length as an 8-byte
// little-endian integer. Length-prefixing makes a sequence of strings
// unambiguous: no two sequences of distinct strings share an encoding. lenBuf
// must have length at least 8 and is overwritten.
func writeLengthPrefixedString(d *xxhash.Digest, lenBuf []byte, s string) {
	binary.LittleEndian.PutUint64(lenBuf, uint64(len(s)))
	_, _ = d.Write(lenBuf)
	_, _ = d.WriteString(s)
}

// withMMHashes returns tokenHash unchanged when no item in mm overlaps the
// block [start, end), so text-only blocks keep their hashes. Otherwise it
// hashes tokenHash followed by each overlapping item's hash, length-prefixed,
// in mm order.
func withMMHashes(tokenHash uint64, mm []fwkrequest.MultiModalFeature, start, end int) uint64 {
	var d xxhash.Digest
	var buf [8]byte
	mixed := false
	for _, f := range mm {
		if f.Offset >= end || f.Offset+f.Length <= start {
			continue
		}
		if !mixed {
			d.Reset()
			binary.LittleEndian.PutUint64(buf[:], tokenHash)
			_, _ = d.Write(buf[:])
			mixed = true
		}
		writeLengthPrefixedString(&d, buf[:], f.Hash)
	}
	if !mixed {
		return tokenHash
	}
	return d.Sum64()
}
