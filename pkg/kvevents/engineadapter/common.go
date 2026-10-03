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

type msgpackEventBatch struct {
	_                struct{} `msgpack:",array"`
	TS               float64
	Events           []msgpack.RawMessage
	DataParallelRank *int `msgpack:",omitempty"`
}

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
		// #nosec G115 -- hash values are unsigned bit patterns; a msgpack encoder
		// may tag one as Int64 whenever its top bit is set, regardless of the number sign.
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

// decodeEvent decodes a single msgpack event, normalizes map-encoded events to
// positional fields via toFields, and dispatches to the matching converter.
// toFields is optional: pass nil for an adapter that never receives
// map-encoded events; a map payload with no toFields is a decode error.
// Shared by the vLLM and SGLang adapters, which differ only in their
// map-to-fields layout and converter set.
func decodeEvent(
	rawEventBytes []byte,
	toFields func(map[string]any) ([]any, error),
	converters map[string]func([]any) (kvevents.GenericEvent, error),
) (kvevents.GenericEvent, error) {
	var decoded any
	if err := msgpack.Unmarshal(rawEventBytes, &decoded); err != nil {
		return nil, fmt.Errorf("unmarshal event payload: %w", err)
	}

	var fields []any
	switch ev := decoded.(type) {
	case []any:
		fields = ev
	case map[string]any:
		if toFields == nil {
			return nil, fmt.Errorf("map-encoded event but no field mapper is configured")
		}
		var err error
		if fields, err = toFields(ev); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("event is neither an array nor a map: %T", decoded)
	}

	if len(fields) < 1 {
		return nil, fmt.Errorf("malformed tagged union: no tag")
	}

	tag, ok := fields[0].(string)
	if !ok {
		return nil, fmt.Errorf("event tag is not a string: %T", fields[0])
	}

	converter, exists := converters[tag]
	if !exists {
		return nil, fmt.Errorf("unknown event tag: %s", tag)
	}

	return converter(fields)
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
