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

	"github.com/llm-d/llm-d-router/pkg/common/clamp"
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

// DecodeMsgpack decodes [ts, events, data_parallel_rank] and skips trailing
// elements, such as the publisher_id a vLLM publisher that serves snapshots
// appends. Array-encoded struct decoding rejects any other element count.
func (b *msgpackEventBatch) DecodeMsgpack(dec *msgpack.Decoder) error {
	n, err := dec.DecodeArrayLen()
	if err != nil {
		return err
	}
	if n < 3 {
		return fmt.Errorf("event batch has %d elements, want at least 3", n)
	}
	if b.TS, err = dec.DecodeFloat64(); err != nil {
		return err
	}
	if err := dec.Decode(&b.Events); err != nil {
		return err
	}
	if err := dec.Decode(&b.DataParallelRank); err != nil {
		return err
	}
	for range n - 3 {
		if err := dec.Skip(); err != nil {
			return err
		}
	}
	return nil
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

// getHashAsUint64 converts MessagePack integer types or []byte to uint64.
// MessagePack may decode small Python int values into narrow Go integer types,
// so all supported integer widths are accepted.
// Signed integer hashes preserve their two's-complement bit pattern. Byte
// hashes use the last 8 bytes, interpreted as a big-endian integer.
func getHashAsUint64(raw any) (uint64, error) {
	switch val := raw.(type) {
	case uint64:
		return val, nil
	case uint32:
		return uint64(val), nil
	case uint16:
		return uint64(val), nil
	case uint8:
		return uint64(val), nil
	case int32:
		//nolint:gosec // preserve the two's-complement bit pattern of signed hashes
		return uint64(val), nil
	case int16:
		//nolint:gosec // preserve the two's-complement bit pattern of signed hashes
		return uint64(val), nil
	case int8:
		//nolint:gosec // preserve the two's-complement bit pattern of signed hashes
		return uint64(val), nil
	case int64:
		//nolint:gosec // preserve the two's-complement bit pattern of signed hashes
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

// The SGLang converters use positional fields for both array and map events.
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

func fieldAt(fields []any, i int) any {
	if i < len(fields) {
		return fields[i]
	}
	return nil
}

func toUint32Slice(raw any) ([]uint32, error) {
	arr, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("token_ids is not an array: %T", raw)
	}
	result := make([]uint32, len(arr))
	for i, v := range arr {
		n, err := toInt(v)
		if err != nil {
			return nil, fmt.Errorf("token_ids[%d]: %w", i, err)
		}
		result[i] = clamp.Uint32(n)
	}
	return result, nil
}

func toInt(raw any) (int, error) {
	switch v := raw.(type) {
	case int64:
		return int(v), nil
	case uint64:
		//#nosec -- token IDs and lora IDs fit in the platform int range
		return int(v), nil
	case int8:
		return int(v), nil
	case int16:
		return int(v), nil
	case int32:
		return int(v), nil
	case uint8:
		return int(v), nil
	case uint16:
		return int(v), nil
	case uint32:
		return int(v), nil
	default:
		return 0, fmt.Errorf("unsupported numeric type: %T", raw)
	}
}
