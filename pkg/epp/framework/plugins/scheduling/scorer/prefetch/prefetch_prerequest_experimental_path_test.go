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

package prefetch

import (
	"encoding/hex"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBasePath(t *testing.T) {
	params := &KVFilePathBaseParams{
		RootDir:   "/var/kv",
		ModelName: "Qwen/Qwen3-8B",
		Digest:    "07d7b166f256",
	}
	// safeModelName replaces '/' with '_'; digest is appended after '_'.
	expected := filepath.Join("/var/kv", "Qwen_Qwen3-8B_07d7b166f256")
	assert.Equal(t, expected, params.basePath())
}

func TestDigestToFullPath_FormatsCorrectly(t *testing.T) {
	params := &KVFilePathBaseParams{
		RootDir:   "/var/kv",
		ModelName: "m",
		Digest:    "abcdef123456",
		GroupIdx:  0,
	}
	digest := []byte{0x12, 0x34, 0x56, 0x78, 0xaa, 0xbb, 0xcc, 0xdd}
	path := params.digestToFullPath(2, digest)
	expected := filepath.Join("/var/kv", "m_abcdef123456") + "_r2/123/45_g0/12345678aabbccdd.bin"
	assert.Equal(t, expected, path)
}

func TestDigestToFullPath_NonZeroGroup(t *testing.T) {
	params := &KVFilePathBaseParams{
		RootDir:   "/var/kv",
		ModelName: "m",
		Digest:    "abcdef123456",
		GroupIdx:  3,
	}
	digest := []byte{0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89}
	path := params.digestToFullPath(0, digest)
	expected := filepath.Join("/var/kv", "m_abcdef123456") + "_r0/abc/de_g3/abcdef0123456789.bin"
	assert.Equal(t, expected, path)
}

// digestN builds a 32-byte digest distinguished by its last byte.
func digestN(n byte) []byte {
	d := make([]byte, 32)
	d[31] = n
	return d
}

func TestDigestsToFilePaths_BatchedByBlocksPerFile(t *testing.T) {
	params := &KVFilePathBaseParams{
		RootDir:          "/var/kv",
		ModelName:        "m",
		Digest:           "abcdef123456",
		GpuBlocksPerFile: 4,
	}

	// fs-connector aggregates 4 vLLM blocks per file and names each file
	// after the last block. With 8 digests and GpuBlocksPerFile=4, only
	// digests[3] and digests[7] correspond to files on disk.
	digests := [][]byte{
		digestN(1), digestN(2), digestN(3), digestN(4),
		digestN(5), digestN(6), digestN(7), digestN(8),
	}
	paths := digestsToFilePaths(params, 0, digests)
	require.Len(t, paths, 2)
	assert.Contains(t, paths[0], hex.EncodeToString(digestN(4))+".bin")
	assert.Contains(t, paths[1], hex.EncodeToString(digestN(8))+".bin")
}

func TestDigestsToFilePaths_SingleBlockPerFile(t *testing.T) {
	params := &KVFilePathBaseParams{
		RootDir:          "/var/kv",
		ModelName:        "m",
		Digest:           "abcdef123456",
		GpuBlocksPerFile: 1,
	}
	digests := [][]byte{digestN(1), digestN(2), digestN(3)}
	paths := digestsToFilePaths(params, 0, digests)
	require.Len(t, paths, 3)
	assert.Contains(t, paths[0], hex.EncodeToString(digestN(1))+".bin")
	assert.Contains(t, paths[2], hex.EncodeToString(digestN(3))+".bin")
}

func TestDigestsToFilePaths_FewerDigestsThanBlocksPerFile(t *testing.T) {
	// vLLM hasn't aggregated enough blocks to write any file yet, so the
	// helper returns nil (no aggregated file exists for this request).
	params := &KVFilePathBaseParams{
		RootDir:          "/var/kv",
		ModelName:        "m",
		Digest:           "abcdef123456",
		GpuBlocksPerFile: 8,
	}
	digests := make([][]byte, 7) // 7 < 8
	for i := range digests {
		digests[i] = digestN(byte(i + 1))
	}
	paths := digestsToFilePaths(params, 0, digests)
	assert.Nil(t, paths)
}

func TestDigestsToFilePaths_NoDigests(t *testing.T) {
	params := &KVFilePathBaseParams{
		RootDir:          "/var/kv",
		ModelName:        "m",
		Digest:           "abcdef123456",
		GpuBlocksPerFile: 1,
	}
	assert.Nil(t, digestsToFilePaths(params, 0, nil))
}
