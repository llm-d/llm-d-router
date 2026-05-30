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

package approximateprefix

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

// getBlockHashes divides the tokenized prompt into blocks and calculates a
// prefix cache hash for each block. Each prompt in PerPromptTokens is hashed
// independently so cross-prompt block adjacency is avoided. The first block
// hash of every prompt includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
// It requires request.Body.TokenizedPrompt to be populated by a token-producer backend.
func getBlockHashes(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) [][]blockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}

	tp := request.Body.TokenizedPrompt
	if tp == nil || tp.TokenCount() == 0 {
		loggerDebug.Info("TokenizedPrompt is empty, skipping hashing")
		return nil
	}

	var result [][]blockHash
	for _, tokens := range tp.PerPromptTokens {
		seq := getKVCacheBlocksFromTokens(tokens, blockSizeTokens)
		hashes := computeBlockHashes(seq, request, maxPrefixBlocks)
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

// computeBlockHashes calculates the hash for content blocks.
func computeBlockHashes(seq iter.Seq[HashBlock], request *scheduling.InferenceRequest, maxPrefixBlocks int) []blockHash {
	var blockHashes []blockHash

	h := xxhash.New()
	// Different models should have different hashes even with the same body.
	_, _ = h.Write([]byte(request.TargetModel))
	if cacheSalt := request.Body.TokenizedPrompt.CacheSalt; cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := blockHash(h.Sum64())

	count := 0
	for block := range seq {
		if count >= maxPrefixBlocks {
			break
		}
		h.Reset()
		blockID := block.Hash()
		_, _ = h.Write(toBytes(blockHash(blockID)))
		_, _ = h.Write(toBytes(prevBlockHash))
		blockHashes = append(blockHashes, blockHash(h.Sum64()))

		prevBlockHash = blockHashes[len(blockHashes)-1]
		count++
	}

	return blockHashes
}

func toBytes(i blockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
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

func getKVCacheBlocksFromRawBytes(rawBytes []byte, blockSizeTokens int) iter.Seq[HashBlock] {
	return func(yield func(HashBlock) bool) {
		if len(rawBytes) == 0 {
			return
		}

		blockSizeBytes := blockSizeTokens * averageCharactersPerToken

		for i := 0; i < len(rawBytes); i += blockSizeBytes {
			blockEnd := i + blockSizeBytes
			if blockEnd > len(rawBytes) {
				blockEnd = len(rawBytes)
			}

			block := HashBlock{
				PseudoTokens: rawBytes[i:blockEnd],
			}
			if !yield(block) {
				return
			}
		}
	}
}

func getKVCacheBlocksFromChatCompletions(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, tokenEstimator TokenEstimator) iter.Seq[HashBlock] {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	messages := request.Body.ChatCompletions.Messages
	var allPseudoBytes []byte

	for _, msg := range messages {
		if msg.Role != "" {
			allPseudoBytes = append(allPseudoBytes, []byte(msg.Role)...)
		}
		if msg.Content.Raw != "" {
			allPseudoBytes = append(allPseudoBytes, []byte(msg.Content.Raw)...)
		} else if len(msg.Content.Structured) > 0 {
			for _, block := range msg.Content.Structured {
				switch block.Type {
				case "text":
					allPseudoBytes = append(allPseudoBytes, []byte(block.Text)...)
				case "image_url":
					// multimodal content can't be in the same pseudo token of text.
					allPseudoBytes = padToAlignment(allPseudoBytes, averageCharactersPerToken)
					url := block.ImageURL.URL
					numPlaceHolders := tokenEstimator.Estimate(fwkrh.ContentBlock{
						Type:     "image_url",
						ImageURL: fwkrh.ImageBlock{URL: url},
					})

					imgHashVal := xxhash.Sum64([]byte(url))
					imgHashBytes := make([]byte, 4)
					binary.LittleEndian.PutUint32(imgHashBytes, uint32(imgHashVal))
					for i := 0; i < numPlaceHolders; i++ {
						allPseudoBytes = append(allPseudoBytes, imgHashBytes...)
					}
				case "video_url":
					allPseudoBytes = padToAlignment(allPseudoBytes, averageCharactersPerToken)
					numPlaceHolders := tokenEstimator.Estimate(fwkrh.ContentBlock{
						Type:     "video_url",
						VideoURL: fwkrh.VideoBlock{URL: block.VideoURL.URL},
					})
					videoHashVal := xxhash.Sum64([]byte(block.VideoURL.URL))
					videoHashBytes := make([]byte, 4)
					binary.LittleEndian.PutUint32(videoHashBytes, uint32(videoHashVal))
					for i := 0; i < numPlaceHolders; i++ {
						allPseudoBytes = append(allPseudoBytes, videoHashBytes...)
					}
				case "input_audio", "audio_url":
					allPseudoBytes = padToAlignment(allPseudoBytes, averageCharactersPerToken)
					numPlaceHolders := tokenEstimator.Estimate(fwkrh.ContentBlock{
						Type:       "input_audio",
						InputAudio: fwkrh.AudioBlock{Data: block.InputAudio.Data, Format: block.InputAudio.Format},
					})
					audioHashVal := xxhash.Sum64([]byte(block.InputAudio.Data + block.InputAudio.Format))
					audioHashBytes := make([]byte, 4)
					binary.LittleEndian.PutUint32(audioHashBytes, uint32(audioHashVal))
					for i := 0; i < numPlaceHolders; i++ {
						allPseudoBytes = append(allPseudoBytes, audioHashBytes...)
					}
				default:
					loggerDebug.Info("Unsupported block type: " + block.Type)

				}
			}
		}
	}

	return getKVCacheBlocksFromRawBytes(allPseudoBytes, blockSizeTokens)
}

func padToAlignment(b []byte, alignment int) []byte {
	remainder := len(b) % alignment
	if remainder == 0 {
		return b
	}
	padding := alignment - remainder
	for i := 0; i < padding; i++ {
		b = append(b, 0)
	}
	return b
}
