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
	"encoding/binary"

	"github.com/cespare/xxhash/v2"
)

// EngineScope is one physical KV cache that reports engine block hashes: an
// endpoint, its data-parallel rank, and its KV-cache group. A nil rank or
// group identifies an event that does not carry that dimension.
type EngineScope struct {
	Endpoint string
	// DataParallelRank is the engine's DP rank from the event batch. Ranks of
	// one endpoint hold separate caches.
	DataParallelRank *int
	// GroupIdx is the vLLM KV-cache group that reported the block. Groups of
	// one engine hold separate caches for the same block hash, as with hybrid
	// memory allocation.
	GroupIdx *int
}

// Keys projects engine block hashes into the key space of an Index dedicated
// to engine hashes, so the same hash reported by two scopes gets two entries.
// Insert with Add(nil, keys, entries) and evict with RequestKey. These keys
// must not share an index with token-derived keys.
func (s EngineScope) Keys(hashes []uint64) []BlockHash {
	// The fixed-size dimension suffix makes the endpoint prefix unambiguous.
	buf := make([]byte, len(s.Endpoint)+26)
	copy(buf, s.Endpoint)
	dims := buf[len(s.Endpoint):]
	if s.DataParallelRank != nil {
		dims[0] = 1
		binary.LittleEndian.PutUint64(dims[1:9], uint64(*s.DataParallelRank)) // #nosec G115 -- preserves the integer bit pattern
	}
	if s.GroupIdx != nil {
		dims[9] = 1
		binary.LittleEndian.PutUint64(dims[10:18], uint64(*s.GroupIdx)) // #nosec G115 -- preserves the integer bit pattern
	}
	keys := make([]BlockHash, len(hashes))
	for i, hash := range hashes {
		binary.LittleEndian.PutUint64(dims[18:], hash)
		keys[i] = BlockHash(xxhash.Sum64(buf))
	}
	return keys
}
