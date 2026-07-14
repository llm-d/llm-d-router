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

package engineadapter

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

const (
	// Common event type tags shared across engine adapters.
	eventTagBlockStored      = "BlockStored"
	eventTagBlockRemoved     = "BlockRemoved"
	eventTagAllBlocksCleared = "AllBlocksCleared"
)

// parseTopic extracts pod ID and model name from the topic format "kv@<pod-id>@<model-name>".
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func parseTopic(topic string) (string, string) {
	topicParts := strings.Split(topic, "@")
	if len(topicParts) == 3 {
		return topicParts[1], topicParts[2]
	}
	return topic, ""
}

// getHashAsUint64 converts engine hash formats (uint64, int64, or []byte) to uint64.
// This handles both legacy uint64 hashes and new []byte hashes by taking
// the last 8 bytes and interpreting them as a big-endian integer.
func getHashAsUint64(raw any) (uint64, error) {
	switch val := raw.(type) {
	case uint64:
		return val, nil
	case int64:
		// msgpack can decode small integers as int64
		//nolint:gosec // int64 to uint64 conversion is safe here
		return uint64(val), nil
	case []byte:
		if len(val) == 0 {
			return 0, fmt.Errorf("hash byte slice is empty")
		}
		if len(val) >= 8 {
			return binary.BigEndian.Uint64(val[len(val)-8:]), nil
		}
		padded := make([]byte, 8)
		copy(padded[8-len(val):], val)
		return binary.BigEndian.Uint64(padded), nil
	default:
		return 0, fmt.Errorf("unsupported hash type: %T", val)
	}
}

// decodeEvent decodes a single msgpack event, extracts the tag, and dispatches to the appropriate converter.
// Used by SGLang adapter. The vLLM adapter uses its own single-pass []any decoder.
func decodeEvent(
	rawEventBytes []byte,
	converters map[string]func([]byte) (kvevents.GenericEvent, error),
) (kvevents.GenericEvent, error) {
	var taggedUnion []any
	if err := msgpack.Unmarshal(rawEventBytes, &taggedUnion); err != nil {
		return nil, fmt.Errorf("failed to decode tagged union: %w", err)
	}

	if len(taggedUnion) < 1 {
		return nil, fmt.Errorf("malformed tagged union: no tag")
	}

	tag, ok := taggedUnion[0].(string)
	if !ok {
		return nil, fmt.Errorf("event tag is not a string: %T", taggedUnion[0])
	}

	converter, exists := converters[tag]
	if !exists {
		return nil, fmt.Errorf("unknown event tag: %s", tag)
	}

	return converter(rawEventBytes)
}

// convertBlockHashes converts raw hash values to uint64 slice.
func convertBlockHashes(rawHashes []any) ([]uint64, error) {
	blockHashes := make([]uint64, 0, len(rawHashes))
	for _, rawHash := range rawHashes {
		hash, err := getHashAsUint64(rawHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse block hash: %w", err)
		}
		blockHashes = append(blockHashes, hash)
	}
	return blockHashes, nil
}

// convertExtraKeys converts raw extra_keys to typed slice.
func convertExtraKeys(rawExtraKeys []any) ([][]any, error) {
	if rawExtraKeys == nil {
		return nil, nil
	}
	extraKeys := make([][]any, 0, len(rawExtraKeys))
	for i, rawKey := range rawExtraKeys {
		if rawKey == nil {
			extraKeys = append(extraKeys, nil)
		} else if keySlice, ok := rawKey.([]any); ok {
			extraKeys = append(extraKeys, keySlice)
		} else {
			return nil, fmt.Errorf("extra_keys[%d] has invalid type %T, expected []any or nil", i, rawKey)
		}
	}
	return extraKeys, nil
}
