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

// EngineScope projects a logical cache namespace into a physical location.
// Nil rank or group identifies a legacy event without that dimension.
type EngineScope struct {
	CacheNamespace   string
	Endpoint         string
	DataParallelRank *int
	GroupIdx         *int
}

// Keys projects engine hashes into the key space of a dedicated Index.
// Insert with Add(nil, keys, entries) and evict with RequestKey. These keys
// must not share an index namespace with token-derived keys.
func (s EngineScope) Keys(hashes []uint64) []BlockHash {
	buf := make([]byte, 8+len(s.CacheNamespace)+len(s.Endpoint)+26)
	binary.LittleEndian.PutUint64(buf, uint64(len(s.CacheNamespace)))
	copy(buf[8:], s.CacheNamespace)
	copy(buf[8+len(s.CacheNamespace):], s.Endpoint)
	dims := buf[8+len(s.CacheNamespace)+len(s.Endpoint):]
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
