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

package kvblock

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/fnv"

	"github.com/cespare/xxhash/v2"
	"github.com/fxamacker/cbor/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/collections"
)

// defaultBlockSize is the default number of tokens per block.
// 16 is the default value used by vLLM.
const defaultBlockSize = 16

// Accepted TokenProcessorConfig.HashAlgorithm values.
const (
	// HashAlgorithmCBORFNV hashes each block with FNV-64a over the canonical
	// CBOR encoding of [parent, tokens, extra]. This is the default.
	HashAlgorithmCBORFNV = "cbor-fnv"
	// HashAlgorithmXXH64 hashes each block with a single XXH64 digest over
	// the parent hash (8 bytes little-endian), the raw token-ID bytes, and
	// each extra-feature string prefixed with its 8-byte little-endian length.
	HashAlgorithmXXH64 = "xxh64"
)

// TokenProcessorConfig holds the configuration for the token processor.
type TokenProcessorConfig struct {
	// BlockSize is deprecated. Use BlockSizeTokens instead.
	//
	// Deprecated: Use BlockSizeTokens instead.
	BlockSize int `json:"blockSize,omitempty"`
	// BlockSizeTokens is the number of tokens per block.
	// A value of zero is treated as "not set" and resolved to the default (16) by NewChunkedTokenDatabase.
	BlockSizeTokens int `json:"blockSizeTokens"`
	// HashSeed is used to prefix initial hash chunks, similarly to vLLM's NONE_HASH.
	// This should be aligned with vLLM's `PYTHONHASHSEED` environment variable.
	// The system's deployer is responsible for aligning the vLLM deployments
	// with the same seed value.
	HashSeed string `json:"hashSeed"`
	// HashAlgorithm selects the block hashing chain: HashAlgorithmCBORFNV
	// (the default when empty) or HashAlgorithmXXH64. Block keys are
	// internal to the index and need not match vLLM's hashes, but they must
	// be consistent across the index's whole lifetime: like HashSeed, the
	// algorithm must not change while a persisted or shared index (e.g. the
	// Redis backend) holds keys, or while engines still hold blocks whose
	// stored engine-to-request mappings were built under the old keys —
	// parent-chain resolution would then mix algorithms and produce request-key
	// lookups that can never match. Change it only together with an index
	// flush and engine cache reset, and roll all replicas sharing an index
	// to the same value.
	HashAlgorithm string `json:"hashAlgorithm"`
	initHash      uint64 // cache once
}

// DefaultTokenProcessorConfig returns the default configuration for the token processor.
func DefaultTokenProcessorConfig() *TokenProcessorConfig {
	return &TokenProcessorConfig{
		BlockSizeTokens: defaultBlockSize,
		HashSeed:        "",
	}
}

// TokenProcessor defines the interface for converting tokens to
// KVBlockKeys.
type TokenProcessor interface {
	// TokensToKVBlockKeys converts tokens into kv_block.Keys.
	// It accepts an optional parentKey to continue a hash chain.
	// extraFeatures provides per-block multimodal data that taints the hash;
	// nil means text-only (no taint). When non-nil, its length must match the
	// number of token chunks.
	// It returns a slice of generated Keys.
	TokensToKVBlockKeys(
		parentKey BlockHash, tokens []uint32, modelName string,
		extraFeatures []*BlockExtraFeatures,
	) ([]BlockHash, error)

	// BlockSize returns the number of tokens per block used by this processor.
	BlockSize() int
}

// chunkedTokenDatabase is a concrete implementation of TokenDatabase.
// It mimics the chunkedTokenDatabase in the Python code.
type chunkedTokenDatabase struct {
	TokenProcessorConfig
	encoder cbor.EncMode // cached CBOR encoder for interoperable encoding
}

var _ TokenProcessor = &chunkedTokenDatabase{}

// NewChunkedTokenDatabase creates a new instance with the given config and metadata.
func NewChunkedTokenDatabase(config *TokenProcessorConfig) (TokenProcessor, error) {
	var cfg TokenProcessorConfig
	if config == nil {
		cfg = *DefaultTokenProcessorConfig()
	} else {
		cfg = *config // local copy — caller's struct is never mutated
	}

	// Apply defaults for omitted fields so partial configs (e.g. only hashSeed set) work correctly.
	if cfg.BlockSizeTokens == 0 && cfg.BlockSize == 0 {
		cfg.BlockSizeTokens = defaultBlockSize
	}

	// Handle backward compatibility: if only deprecated BlockSize is set, promote it.
	if cfg.BlockSizeTokens == 0 && cfg.BlockSize > 0 {
		cfg.BlockSizeTokens = cfg.BlockSize
	}

	if cfg.BlockSizeTokens <= 0 {
		// Report the actual invalid value the caller set, not the zero from the other field.
		invalidBlockSize := cfg.BlockSizeTokens
		if cfg.BlockSizeTokens == 0 && cfg.BlockSize != 0 {
			invalidBlockSize = cfg.BlockSize
		}
		return nil, fmt.Errorf("blockSizeTokens must be greater than 0, got %d", invalidBlockSize)
	}

	switch cfg.HashAlgorithm {
	case "", HashAlgorithmCBORFNV, HashAlgorithmXXH64:
	default:
		return nil, fmt.Errorf("unsupported hashAlgorithm %q: accepted values are %q and %q",
			cfg.HashAlgorithm, HashAlgorithmCBORFNV, HashAlgorithmXXH64)
	}

	if cfg.initHash == 0 {
		if cfg.HashAlgorithm == HashAlgorithmXXH64 {
			cfg.initHash = xxhash.Sum64String(cfg.HashSeed)
		} else {
			h := fnv.New64a()
			_, _ = h.Write([]byte(cfg.HashSeed))
			cfg.initHash = h.Sum64()
		}
	}

	encoder, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		return nil, fmt.Errorf("failed to create CBOR encoder: %w", err)
	}

	return &chunkedTokenDatabase{
		TokenProcessorConfig: cfg,
		encoder:              encoder,
	}, nil
}

// getInitHash returns the initial hash for the given model name.
func (db *chunkedTokenDatabase) getInitHash(modelName string, digest *xxhash.Digest) uint64 {
	if digest != nil {
		// The model name occupies the extra-string position, mirroring the
		// CBOR chain where it is passed as the extra of the seed hash.
		return hashXXH64(digest, db.initHash, nil, []MMHash{{Hash: modelName}})
	}
	return db.hash(db.initHash, nil, modelName)
}

// hash computes the uint64 FNV-64a hash of the given parent, tokens,
// and extra keys.
//
// The hash is computed using FNV-64a over the CBOR canonical encoding of
// [parent, tokens, extra], ensuring deterministic results across runs and
// compatibility with vLLM's prefix caching algorithm.
//
// The extra parameter enables cache differentiation for LoRA adapters and
// multi-modal content. Supported types: nil, int, string, map[string]interface{}.
// Must be CBOR-serializable.
func (db *chunkedTokenDatabase) hash(parent uint64, tokens []uint32, extra interface{}) uint64 {
	payload := []interface{}{parent, tokens, extra}

	b, err := db.encoder.Marshal(payload)
	if err != nil {
		log.FromContext(context.Background()).Error(err, "failed to marshal payload to CBOR")
		return 0
	}

	h := fnv.New64a()
	_, _ = h.Write(b)
	return h.Sum64()
}

// hashXXH64 computes one XXH64 digest over the parent hash (8 bytes
// little-endian), the token IDs (4 bytes little-endian each, so keys are
// identical across architectures and shareable through a Redis-backed index),
// and each extra string prefixed with its 8-byte little-endian length. The
// length prefix prevents concatenation ambiguity between adjacent extra
// strings. The digest is reset before use so callers can reuse one across
// blocks.
func hashXXH64(h *xxhash.Digest, parent uint64, tokens []uint32, extras []MMHash) uint64 {
	h.Reset()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], parent)
	_, _ = h.Write(buf[:])
	var tb [256]byte
	for i := 0; i < len(tokens); {
		n := min(len(tokens)-i, len(tb)/4)
		for j := 0; j < n; j++ {
			binary.LittleEndian.PutUint32(tb[j*4:], tokens[i+j])
		}
		_, _ = h.Write(tb[:n*4])
		i += n
	}
	for _, mm := range extras {
		binary.LittleEndian.PutUint64(buf[:], uint64(len(mm.Hash)))
		_, _ = h.Write(buf[:])
		_, _ = h.WriteString(mm.Hash)
	}
	return h.Sum64()
}

// prefixHashes returns a slice of uint64 hashes.
// extraFeatures must be the same length as tokenChunks (callers guarantee this).
func (db *chunkedTokenDatabase) prefixHashes(
	parentHash uint64, tokenChunks [][]uint32, extraFeatures []*BlockExtraFeatures,
	digest *xxhash.Digest,
) []uint64 {
	prefix := parentHash
	hashes := make([]uint64, len(tokenChunks))
	for i, chunk := range tokenChunks {
		var extras []MMHash
		if extraFeatures[i] != nil {
			extras = extraFeatures[i].MMHashes
		}
		if digest != nil {
			prefix = hashXXH64(digest, prefix, chunk, extras)
		} else {
			prefix = db.hash(prefix, chunk, extras)
		}
		hashes[i] = prefix
	}
	return hashes
}

// BlockSize returns the number of tokens per block.
func (db *chunkedTokenDatabase) BlockSize() int {
	return db.BlockSizeTokens
}

// chunkTokens splits the input slice of tokens into chunks of size blockSize.
func (db *chunkedTokenDatabase) chunkTokens(tokens []uint32) [][]uint32 {
	bs := db.BlockSizeTokens
	var chunks [][]uint32
	for i := 0; i < len(tokens); i += bs {
		end := i + bs
		if end > len(tokens) {
			break // no partial blocks
		}

		chunks = append(chunks, tokens[i:end])
	}

	return chunks
}

// TokensToKVBlockKeys converts tokens into kv_block.Keys.
func (db *chunkedTokenDatabase) TokensToKVBlockKeys(
	parentKey BlockHash, tokens []uint32, modelName string,
	extraFeatures []*BlockExtraFeatures,
) ([]BlockHash, error) {
	chunks := db.chunkTokens(tokens)
	if len(chunks) == 0 {
		return nil, nil
	}

	var digest *xxhash.Digest // one per call
	if db.HashAlgorithm == HashAlgorithmXXH64 {
		digest = xxhash.New()
	}

	var currentParentHash uint64
	if parentKey != EmptyBlockHash {
		currentParentHash = uint64(parentKey)
	} else {
		currentParentHash = db.getInitHash(modelName, digest)
	}

	if extraFeatures == nil {
		extraFeatures = make([]*BlockExtraFeatures, len(chunks))
	} else if len(extraFeatures) != len(chunks) {
		return nil, fmt.Errorf("extraFeatures length %d does not match token chunk count %d (blockSizeTokens=%d, tokens=%d)",
			len(extraFeatures), len(chunks), db.BlockSizeTokens, len(tokens))
	}

	ph := db.prefixHashes(currentParentHash, chunks, extraFeatures, digest)

	return collections.SliceMap(ph, func(hashVal uint64) BlockHash {
		return BlockHash(hashVal)
	}), nil
}
