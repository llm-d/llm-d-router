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

package kvblock

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"
)

// hashCBORFNVBlock must equal the marshaled CBOR-FNV hash for nil tokens and
// for every CBOR head size of the parent, of each token, and of the token
// count.
func TestHashCBORFNVBlockMatchesMarshaledHash(t *testing.T) {
	processor, err := NewChunkedTokenDatabase(nil)
	require.NoError(t, err)
	db := processor.(*chunkedTokenDatabase)

	boundaries := []uint64{
		0, 23, 24, math.MaxUint8, math.MaxUint8 + 1, math.MaxUint16, math.MaxUint16 + 1,
		math.MaxUint32, math.MaxUint32 + 1, math.MaxUint64,
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, parent := range boundaries {
		require.Equal(t, db.hash(parent, nil, []MMHash(nil)), hashCBORFNVBlock(parent, nil), "parent=%d nil tokens", parent)
		for _, n := range []int{1, 16, 23, 24, 255, 256, 300, math.MaxUint16 + 1} {
			tokens := make([]uint32, n)
			for i := range tokens {
				b := boundaries[rng.IntN(len(boundaries))]
				tokens[i] = uint32(min(b, math.MaxUint32))
			}
			want := db.hash(parent, tokens, []MMHash(nil))
			require.Equal(t, want, hashCBORFNVBlock(parent, tokens), "parent=%d tokens=%d", parent, n)
		}
	}
}

// The default chain must produce the keys of a chain built only from the
// marshaled hash, across text-only blocks, blocks with empty extra features,
// and multimodal blocks.
func TestTokensToKVBlockKeysCBORFNVMatchesMarshaledChain(t *testing.T) {
	processor, err := NewChunkedTokenDatabase(&TokenProcessorConfig{BlockSizeTokens: 4})
	require.NoError(t, err)
	db := processor.(*chunkedTokenDatabase)

	tokens := make([]uint32, 4*6)
	for i := range tokens {
		tokens[i] = uint32(i * 1000)
	}
	extras := []*BlockExtraFeatures{
		nil,
		{},
		{MMHashes: []MMHash{}},
		{MMHashes: []MMHash{{Hash: "img-a"}}},
		nil,
		{MMHashes: []MMHash{{Hash: "img-a"}, {Hash: "img-b"}}},
	}

	got, err := db.TokensToKVBlockKeys(EmptyBlockHash, tokens, "model", extras)
	require.NoError(t, err)

	want := make([]BlockHash, len(extras))
	parent := db.hash(db.initHash, nil, "model")
	for i, ef := range extras {
		var mm []MMHash
		if ef != nil {
			mm = ef.MMHashes
		}
		parent = db.hash(parent, tokens[i*4:(i+1)*4], mm)
		want[i] = BlockHash(parent)
	}
	require.Equal(t, want, got)
}
