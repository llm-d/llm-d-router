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
	"io"
	"math"

	"github.com/vmihailenco/msgpack/v5"
	"github.com/vmihailenco/msgpack/v5/msgpcode"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// Keep initial collections small because their wire lengths are untrusted.
// Valid large collections grow as the decoder reads their elements.
const maxDecodePreallocate = 1024

// Limit recursive values in extra_keys to protect the goroutine stack.
const maxDecodeDepth = 64

type msgpackVLLMEventBatch struct {
	timestamp        float64
	events           []kvevents.GenericEvent
	dataParallelRank *int
}

func (b *msgpackVLLMEventBatch) DecodeMsgpack(dec *msgpack.Decoder) error {
	fieldCount, err := dec.DecodeArrayLen()
	if err != nil {
		return err
	}
	if fieldCount < 2 {
		return fmt.Errorf("vLLM event batch: need at least 2 fields, got %d", fieldCount)
	}

	b.timestamp, err = dec.DecodeFloat64()
	if err != nil {
		return fmt.Errorf("vLLM event batch timestamp: %w", err)
	}

	eventCount, err := dec.DecodeArrayLen()
	if err != nil {
		return fmt.Errorf("vLLM event batch events: %w", err)
	}
	if eventCount < 0 {
		b.events = nil
	} else {
		b.events = make([]kvevents.GenericEvent, 0, min(eventCount, maxDecodePreallocate))
		for range eventCount {
			event, err := decodeVLLMEventFromDecoder(dec)
			if err != nil {
				return fmt.Errorf("failed to decode vLLM event: %w", err)
			}
			b.events = append(b.events, event)
		}
	}

	if fieldCount >= 3 {
		b.dataParallelRank, err = decodeOptionalInt(dec)
		if err != nil {
			return fmt.Errorf("vLLM event batch data parallel rank: %w", err)
		}
	}
	for range fieldCount - 3 {
		if err := skipValue(dec); err != nil {
			return fmt.Errorf("vLLM event batch trailing field: %w", err)
		}
	}
	return nil
}

type msgpackVLLMEvent struct {
	event kvevents.GenericEvent
}

func (e *msgpackVLLMEvent) DecodeMsgpack(dec *msgpack.Decoder) error {
	var err error
	e.event, err = decodeVLLMEventFromDecoder(dec)
	return err
}

func decodeVLLMEvent(payload []byte) (kvevents.GenericEvent, error) {
	var decoded msgpackVLLMEvent
	if err := msgpack.Unmarshal(payload, &decoded); err != nil {
		return nil, err
	}
	return decoded.event, nil
}

func decodeVLLMEventFromDecoder(dec *msgpack.Decoder) (kvevents.GenericEvent, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return nil, err
	}

	switch {
	case msgpcode.IsFixedArray(code) || code == msgpcode.Array16 || code == msgpcode.Array32:
		return decodeArrayVLLMEvent(dec)
	case msgpcode.IsFixedMap(code) || code == msgpcode.Map16 || code == msgpcode.Map32:
		return decodeMapVLLMEvent(dec)
	default:
		return nil, fmt.Errorf("event is neither an array nor a map: MessagePack code %#x", code)
	}
}

func decodeArrayVLLMEvent(dec *msgpack.Decoder) (kvevents.GenericEvent, error) {
	fieldCount, err := dec.DecodeArrayLen()
	if err != nil {
		return nil, err
	}
	if fieldCount < 1 {
		return nil, fmt.Errorf("malformed tagged union: no tag")
	}

	tag, err := decodeEventTag(dec)
	if err != nil {
		return nil, err
	}

	switch tag {
	case eventTagBlockStored:
		return decodeArrayBlockStored(dec, fieldCount)
	case eventTagBlockRemoved:
		return decodeArrayBlockRemoved(dec, fieldCount)
	case eventTagAllBlocksCleared:
		if err := skipFields(dec, fieldCount-1); err != nil {
			return nil, err
		}
		return &kvevents.AllBlocksClearedEvent{}, nil
	default:
		return nil, fmt.Errorf("unknown vLLM event tag: %s", tag)
	}
}

func decodeArrayBlockStored(dec *msgpack.Decoder, fieldCount int) (kvevents.GenericEvent, error) {
	if fieldCount < 5 {
		return nil, fmt.Errorf("BlockStored: need at least 5 fields, got %d", fieldCount)
	}

	fields := vllmEventFields{}
	var err error
	if fields.blockHashes, err = decodeBlockHashes(dec); err != nil {
		return nil, err
	}
	if fields.parentHash, err = decodeNullableHash(dec); err != nil {
		return nil, fmt.Errorf("failed to parse parent hash: %w", err)
	}
	if fields.tokens, err = decodeTokenIDs(dec); err != nil {
		return nil, fmt.Errorf("BlockStored: %w", err)
	}
	if fields.blockSize, err = decodeInt(dec); err != nil {
		return nil, fmt.Errorf("BlockStored: block_size: %w", err)
	}

	optionalDecoders := []func(*msgpack.Decoder, *vllmEventFields) error{
		decodeLoraID,
		decodeMedium,
		decodeLoraName,
		decodeExtraKeysField,
		decodeGroupIdx,
		decodeKVCacheSpecKind,
		decodeSlidingWindow,
	}
	optionalCount := min(fieldCount-5, len(optionalDecoders))
	for i := range optionalCount {
		if err := optionalDecoders[i](dec, &fields); err != nil {
			return nil, fmt.Errorf("BlockStored: %w", err)
		}
	}
	if err := skipFields(dec, fieldCount-5-optionalCount); err != nil {
		return nil, err
	}

	return fields.blockStoredEvent(), nil
}

func decodeArrayBlockRemoved(dec *msgpack.Decoder, fieldCount int) (kvevents.GenericEvent, error) {
	if fieldCount < 2 {
		return nil, fmt.Errorf("BlockRemoved: need at least 2 fields, got %d", fieldCount)
	}

	fields := vllmEventFields{}
	var err error
	if fields.blockHashes, err = decodeBlockHashes(dec); err != nil {
		return nil, err
	}

	optionalDecoders := []func(*msgpack.Decoder, *vllmEventFields) error{
		decodeMedium,
		decodeGroupIdx,
	}
	optionalCount := min(fieldCount-2, len(optionalDecoders))
	for i := range optionalCount {
		if err := optionalDecoders[i](dec, &fields); err != nil {
			return nil, fmt.Errorf("BlockRemoved: %w", err)
		}
	}
	if err := skipFields(dec, fieldCount-2-optionalCount); err != nil {
		return nil, err
	}

	return fields.blockRemovedEvent(), nil
}

type vllmEventFields struct {
	tag           string
	hasTag        bool
	blockHashes   []uint64
	hasHashes     bool
	parentHash    uint64
	tokens        []uint32
	hasTokens     bool
	blockSize     int
	hasBlockSize  bool
	loraID        *int
	medium        string
	loraName      *string
	extraKeys     [][]any
	groupIdx      *int
	specKind      kvevents.KVCacheSpecKind
	slidingWindow *int
}

type deferredVLLMMapField struct {
	name  string
	value any
}

type deferredMapEntry struct {
	key   any
	value any
}

type deferredMap []deferredMapEntry

func (m deferredMap) EncodeMsgpack(enc *msgpack.Encoder) error {
	if err := enc.EncodeMapLen(len(m)); err != nil {
		return err
	}
	for _, entry := range m {
		if err := enc.Encode(entry.key); err != nil {
			return err
		}
		if err := enc.Encode(entry.value); err != nil {
			return err
		}
	}
	return nil
}

type deferredExtension struct {
	id   int8
	data []byte
}

func (e deferredExtension) EncodeMsgpack(enc *msgpack.Encoder) error {
	if err := enc.EncodeExtHeader(e.id, len(e.data)); err != nil {
		return err
	}
	written, err := enc.Writer().Write(e.data)
	if err == nil && written != len(e.data) {
		return io.ErrShortWrite
	}
	return err
}

type msgpackVLLMMapField struct {
	name   string
	tag    string
	fields *vllmEventFields
}

func (f *msgpackVLLMMapField) DecodeMsgpack(dec *msgpack.Decoder) error {
	return decodeVLLMMapField(dec, f.name, f.tag, f.fields)
}

func decodeMapVLLMEvent(dec *msgpack.Decoder) (kvevents.GenericEvent, error) {
	fieldCount, err := dec.DecodeMapLen()
	if err != nil {
		return nil, err
	}

	fields := vllmEventFields{}
	// Buffer fields before the tag because the tag defines their schema.
	var deferred []deferredVLLMMapField
	for range fieldCount {
		name, err := dec.DecodeString()
		if err != nil {
			return nil, fmt.Errorf("map-encoded event field name: %w", err)
		}

		switch {
		case name == "type":
			if fields.hasTag {
				return nil, fmt.Errorf("map-encoded event has more than one %q tag", "type")
			}
			fields.tag, err = decodeMapEventTag(dec)
			fields.hasTag = err == nil
			if err == nil && isKnownVLLMEventTag(fields.tag) {
				for _, field := range deferred {
					payload, marshalErr := msgpack.Marshal(field.value)
					if marshalErr != nil {
						return nil, fmt.Errorf("map-encoded event field %q: %w", field.name, marshalErr)
					}
					wrapped := msgpackVLLMMapField{
						name: field.name, tag: fields.tag, fields: &fields,
					}
					if err := msgpack.Unmarshal(payload, &wrapped); err != nil {
						return nil, fmt.Errorf("map-encoded event field %q: %w", field.name, err)
					}
				}
			}
			deferred = nil
		case !fields.hasTag:
			value, decodeErr := decodeDeferredAny(dec, 0)
			if decodeErr != nil {
				return nil, fmt.Errorf("map-encoded event field %q: %w", name, decodeErr)
			}
			deferred = append(deferred, deferredVLLMMapField{name: name, value: value})
		case isKnownVLLMEventTag(fields.tag):
			err = decodeVLLMMapField(dec, name, fields.tag, &fields)
		default:
			err = skipValue(dec)
		}
		if err != nil {
			return nil, fmt.Errorf("map-encoded event field %q: %w", name, err)
		}
	}

	if !fields.hasTag {
		return nil, fmt.Errorf("map-encoded event is missing the %q tag", "type")
	}

	switch fields.tag {
	case eventTagBlockStored:
		if err := fields.requireBlockStoredFields(); err != nil {
			return nil, err
		}
		return fields.blockStoredEvent(), nil
	case eventTagBlockRemoved:
		if !fields.hasHashes {
			return nil, fmt.Errorf("BlockRemoved: missing required field %q", "block_hashes")
		}
		return fields.blockRemovedEvent(), nil
	case eventTagAllBlocksCleared:
		return &kvevents.AllBlocksClearedEvent{}, nil
	default:
		return nil, fmt.Errorf("unknown vLLM event tag: %s", fields.tag)
	}
}

func decodeVLLMMapField(
	dec *msgpack.Decoder,
	name string,
	tag string,
	fields *vllmEventFields,
) error {
	if !isVLLMMapFieldForTag(name, tag) {
		return skipValue(dec)
	}

	var err error
	switch name {
	case "block_hashes":
		fields.blockHashes, err = decodeBlockHashes(dec)
		fields.hasHashes = err == nil
	case "parent_block_hash":
		fields.parentHash, err = decodeNullableHash(dec)
	case "token_ids":
		fields.tokens, err = decodeTokenIDs(dec)
		fields.hasTokens = err == nil
	case "block_size":
		fields.blockSize, err = decodeInt(dec)
		fields.hasBlockSize = err == nil
	case "lora_id":
		err = decodeLoraID(dec, fields)
	case "medium":
		err = decodeMedium(dec, fields)
	case "lora_name":
		err = decodeLoraName(dec, fields)
	case "extra_keys":
		err = decodeExtraKeysField(dec, fields)
	case "group_idx":
		err = decodeGroupIdx(dec, fields)
	case "kv_cache_spec_kind":
		err = decodeKVCacheSpecKind(dec, fields)
	case "kv_cache_spec_sliding_window":
		err = decodeSlidingWindow(dec, fields)
	default:
		err = skipValue(dec)
	}
	return err
}

func isVLLMMapFieldForTag(name, tag string) bool {
	switch tag {
	case eventTagBlockStored:
		switch name {
		case "block_hashes", "parent_block_hash", "token_ids", "block_size", "lora_id",
			"medium", "lora_name", "extra_keys", "group_idx", "kv_cache_spec_kind",
			"kv_cache_spec_sliding_window":
			return true
		}
	case eventTagBlockRemoved:
		return name == "block_hashes" || name == "medium" || name == "group_idx"
	}
	return false
}

func isKnownVLLMEventTag(tag string) bool {
	switch tag {
	case eventTagBlockStored, eventTagBlockRemoved, eventTagAllBlocksCleared:
		return true
	default:
		return false
	}
}

func (f *vllmEventFields) requireBlockStoredFields() error {
	for _, required := range []struct {
		name string
		has  bool
	}{
		{"block_hashes", f.hasHashes},
		{"token_ids", f.hasTokens},
		{"block_size", f.hasBlockSize},
	} {
		if !required.has {
			return fmt.Errorf("BlockStored: missing required field %q", required.name)
		}
	}
	return nil
}

func (f *vllmEventFields) blockStoredEvent() *kvevents.BlockStoredEvent {
	return &kvevents.BlockStoredEvent{
		BlockHashes:                  f.blockHashes,
		Tokens:                       f.tokens,
		ParentHash:                   f.parentHash,
		BlockSize:                    f.blockSize,
		DeviceTier:                   f.medium,
		LoraID:                       f.loraID,
		LoraName:                     f.loraName,
		ExtraKeys:                    f.extraKeys,
		GroupIdx:                     f.groupIdx,
		KVCacheSpecKind:              f.specKind,
		KVCacheSpecSlidingWindowSize: f.slidingWindow,
	}
}

func (f *vllmEventFields) blockRemovedEvent() *kvevents.BlockRemovedEvent {
	return &kvevents.BlockRemovedEvent{
		BlockHashes: f.blockHashes,
		DeviceTier:  f.medium,
		GroupIdx:    f.groupIdx,
	}
}

func decodeEventTag(dec *msgpack.Decoder) (string, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return "", err
	}
	if msgpcode.IsString(code) {
		return dec.DecodeString()
	}
	return "", fmt.Errorf("event tag is not a string: MessagePack code %#x", code)
}

func decodeMapEventTag(dec *msgpack.Decoder) (string, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return "", err
	}
	if msgpcode.IsString(code) {
		return dec.DecodeString()
	}
	return "", fmt.Errorf(
		"map-encoded event tag (%q) is not a string: MessagePack code %#x", "type", code)
}

func decodeBlockHashes(dec *msgpack.Decoder) ([]uint64, error) {
	count, err := dec.DecodeArrayLen()
	if err != nil {
		return nil, fmt.Errorf("block_hashes is not an array: %w", err)
	}
	if count < 0 {
		return nil, fmt.Errorf("block_hashes is not an array: <nil>")
	}

	hashes := make([]uint64, 0, min(count, maxDecodePreallocate))
	for range count {
		hash, err := decodeHash(dec)
		if err != nil {
			return nil, fmt.Errorf("failed to parse block hash: %w", err)
		}
		hashes = append(hashes, hash)
	}
	return hashes, nil
}

func decodeNullableHash(dec *msgpack.Decoder) (uint64, error) {
	isNil, err := decodeNil(dec)
	if err != nil || isNil {
		return 0, err
	}
	return decodeHash(dec)
}

func decodeHash(dec *msgpack.Decoder) (uint64, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return 0, err
	}

	if msgpcode.IsBin(code) {
		length, err := dec.DecodeBytesLen()
		if err != nil {
			return 0, err
		}
		if length == 0 {
			return 0, fmt.Errorf("hash byte slice is empty")
		}

		// Only the final eight bytes contribute to the router hash.
		var discard [64]byte
		for remaining := length - min(length, 8); remaining > 0; {
			chunk := min(remaining, len(discard))
			if err := dec.ReadFull(discard[:chunk]); err != nil {
				return 0, err
			}
			remaining -= chunk
		}

		var value [8]byte
		keptLength := min(length, len(value))
		if err := dec.ReadFull(value[len(value)-keptLength:]); err != nil {
			return 0, err
		}
		return binary.BigEndian.Uint64(value[:]), nil
	}
	if isIntegerCode(code) {
		return dec.DecodeUint64()
	}

	return 0, fmt.Errorf("unsupported hash type: MessagePack code %#x", code)
}

func decodeTokenIDs(dec *msgpack.Decoder) ([]uint32, error) {
	count, err := dec.DecodeArrayLen()
	if err != nil {
		return nil, fmt.Errorf("token_ids is not an array: %w", err)
	}
	if count < 0 {
		return nil, fmt.Errorf("token_ids is not an array: <nil>")
	}

	tokens := make([]uint32, 0, min(count, maxDecodePreallocate))
	for i := range count {
		code, err := dec.PeekCode()
		if err != nil {
			return nil, fmt.Errorf("token_ids[%d]: %w", i, err)
		}
		if !isIntegerCode(code) {
			return nil, fmt.Errorf("token_ids[%d]: unsupported numeric type: MessagePack code %#x", i, code)
		}
		value, err := dec.DecodeUint64()
		if err != nil {
			return nil, fmt.Errorf("token_ids[%d]: %w", i, err)
		}
		if value > math.MaxUint32 {
			return nil, fmt.Errorf("token_ids[%d]: value %d exceeds uint32", i, value)
		}
		tokens = append(tokens, uint32(value))
	}
	return tokens, nil
}

func decodeInt(dec *msgpack.Decoder) (int, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return 0, err
	}
	if !isIntegerCode(code) {
		return 0, fmt.Errorf("unsupported numeric type: MessagePack code %#x", code)
	}
	if code <= msgpcode.PosFixedNumHigh || code == msgpcode.Uint8 || code == msgpcode.Uint16 ||
		code == msgpcode.Uint32 || code == msgpcode.Uint64 {
		value, err := dec.DecodeUint64()
		if err != nil {
			return 0, err
		}
		if value > math.MaxInt {
			return 0, fmt.Errorf("integer %d is outside the platform int range", value)
		}
		return int(value), nil
	}
	value, err := dec.DecodeInt64()
	if err != nil {
		return 0, err
	}
	if value < math.MinInt || value > math.MaxInt {
		return 0, fmt.Errorf("integer %d is outside the platform int range", value)
	}
	return int(value), nil
}

func decodeLoraID(dec *msgpack.Decoder, fields *vllmEventFields) error {
	value, err := decodeOptionalInt(dec)
	if err != nil {
		return fmt.Errorf("lora_id: %w", err)
	}
	fields.loraID = value
	return nil
}

func decodeMedium(dec *msgpack.Decoder, fields *vllmEventFields) error {
	value, err := decodeOptionalString(dec)
	if err != nil {
		return fmt.Errorf("medium is not a string: %w", err)
	}
	if value != nil {
		fields.medium = *value
	}
	return nil
}

func decodeLoraName(dec *msgpack.Decoder, fields *vllmEventFields) error {
	value, err := decodeOptionalString(dec)
	if err != nil {
		return fmt.Errorf("lora_name is not a string: %w", err)
	}
	fields.loraName = value
	return nil
}

func decodeExtraKeysField(dec *msgpack.Decoder, fields *vllmEventFields) error {
	value, err := decodeExtraKeys(dec)
	if err != nil {
		return err
	}
	fields.extraKeys = value
	return nil
}

func decodeGroupIdx(dec *msgpack.Decoder, fields *vllmEventFields) error {
	value, err := decodeOptionalInt(dec)
	if err != nil {
		return fmt.Errorf("group_idx: %w", err)
	}
	if value != nil && *value < 0 {
		return fmt.Errorf("group_idx: negative value: %d", *value)
	}
	fields.groupIdx = value
	return nil
}

func decodeKVCacheSpecKind(dec *msgpack.Decoder, fields *vllmEventFields) error {
	value, err := decodeOptionalString(dec)
	if err != nil {
		return fmt.Errorf("kv_cache_spec_kind is not a string: %w", err)
	}
	if value != nil {
		fields.specKind = kvevents.KVCacheSpecKind(*value)
	}
	return nil
}

func decodeSlidingWindow(dec *msgpack.Decoder, fields *vllmEventFields) error {
	value, err := decodeOptionalInt(dec)
	if err != nil {
		return fmt.Errorf("kv_cache_spec_sliding_window: %w", err)
	}
	fields.slidingWindow = value
	return nil
}

func decodeOptionalInt(dec *msgpack.Decoder) (*int, error) {
	isNil, err := decodeNil(dec)
	if err != nil || isNil {
		return nil, err
	}
	value, err := decodeInt(dec)
	if err != nil {
		return nil, err
	}
	return &value, nil
}

func decodeOptionalString(dec *msgpack.Decoder) (*string, error) {
	isNil, err := decodeNil(dec)
	if err != nil || isNil {
		return nil, err
	}
	code, err := dec.PeekCode()
	if err != nil {
		return nil, err
	}
	if !msgpcode.IsString(code) {
		return nil, fmt.Errorf("unsupported string type: MessagePack code %#x", code)
	}
	value, err := dec.DecodeString()
	if err != nil {
		return nil, err
	}
	return &value, nil
}

func decodeExtraKeys(dec *msgpack.Decoder) ([][]any, error) {
	count, err := dec.DecodeArrayLen()
	if err != nil {
		return nil, fmt.Errorf("extra_keys is not an array: %w", err)
	}
	if count < 0 {
		return nil, nil
	}

	extraKeys := make([][]any, 0, min(count, maxDecodePreallocate))
	for i := range count {
		code, err := dec.PeekCode()
		if err != nil {
			return nil, err
		}
		if code == msgpcode.Nil {
			if err := dec.DecodeNil(); err != nil {
				return nil, err
			}
			extraKeys = append(extraKeys, nil)
			continue
		}
		if !msgpcode.IsFixedArray(code) && code != msgpcode.Array16 && code != msgpcode.Array32 {
			return nil, fmt.Errorf(
				"extra_keys[%d] has invalid type with MessagePack code %#x, expected []any or nil", i, code)
		}

		itemCount, err := dec.DecodeArrayLen()
		if err != nil {
			return nil, err
		}
		item := make([]any, 0, min(itemCount, maxDecodePreallocate))
		for range itemCount {
			value, err := decodeAny(dec, 0)
			if err != nil {
				return nil, err
			}
			item = append(item, value)
		}
		extraKeys = append(extraKeys, item)
	}
	return extraKeys, nil
}

func decodeAny(dec *msgpack.Decoder, depth int) (any, error) {
	if depth >= maxDecodeDepth {
		return nil, fmt.Errorf("MessagePack value exceeds the maximum nesting depth of %d", maxDecodeDepth)
	}

	code, err := dec.PeekCode()
	if err != nil {
		return nil, err
	}

	switch {
	case code == msgpcode.Nil:
		return nil, dec.DecodeNil()
	case code == msgpcode.False || code == msgpcode.True:
		return dec.DecodeBool()
	case msgpcode.IsFixedNum(code):
		return dec.DecodeInt8()
	case code == msgpcode.Uint8:
		return dec.DecodeUint8()
	case code == msgpcode.Uint16:
		return dec.DecodeUint16()
	case code == msgpcode.Uint32:
		return dec.DecodeUint32()
	case code == msgpcode.Uint64:
		return dec.DecodeUint64()
	case code == msgpcode.Int8:
		return dec.DecodeInt8()
	case code == msgpcode.Int16:
		return dec.DecodeInt16()
	case code == msgpcode.Int32:
		return dec.DecodeInt32()
	case code == msgpcode.Int64:
		return dec.DecodeInt64()
	case code == msgpcode.Float:
		return dec.DecodeFloat32()
	case code == msgpcode.Double:
		return dec.DecodeFloat64()
	case msgpcode.IsString(code):
		return dec.DecodeString()
	case msgpcode.IsBin(code):
		return dec.DecodeBytes()
	case msgpcode.IsFixedArray(code) || code == msgpcode.Array16 || code == msgpcode.Array32:
		return decodeAnyArray(dec, depth+1)
	case msgpcode.IsFixedMap(code) || code == msgpcode.Map16 || code == msgpcode.Map32:
		return decodeAnyMap(dec, depth+1)
	default:
		return nil, fmt.Errorf("unsupported MessagePack code %#x", code)
	}
}

func decodeDeferredAny(dec *msgpack.Decoder, depth int) (any, error) {
	if depth >= maxDecodeDepth {
		return nil, fmt.Errorf("MessagePack value exceeds the maximum nesting depth of %d", maxDecodeDepth)
	}

	code, err := dec.PeekCode()
	if err != nil {
		return nil, err
	}
	switch {
	case msgpcode.IsFixedArray(code) || code == msgpcode.Array16 || code == msgpcode.Array32:
		count, err := dec.DecodeArrayLen()
		if err != nil || count < 0 {
			return nil, err
		}
		values := make([]any, 0, min(count, maxDecodePreallocate))
		for range count {
			value, err := decodeDeferredAny(dec, depth+1)
			if err != nil {
				return nil, err
			}
			values = append(values, value)
		}
		return values, nil
	case msgpcode.IsFixedMap(code) || code == msgpcode.Map16 || code == msgpcode.Map32:
		count, err := dec.DecodeMapLen()
		if err != nil || count < 0 {
			return nil, err
		}
		values := make(deferredMap, 0, min(count, maxDecodePreallocate))
		for range count {
			key, err := decodeDeferredAny(dec, depth+1)
			if err != nil {
				return nil, err
			}
			value, err := decodeDeferredAny(dec, depth+1)
			if err != nil {
				return nil, err
			}
			values = append(values, deferredMapEntry{key: key, value: value})
		}
		return values, nil
	case msgpcode.IsExt(code):
		return decodeDeferredExtension(dec)
	default:
		return decodeAny(dec, depth)
	}
}

func decodeDeferredExtension(dec *msgpack.Decoder) (any, error) {
	id, length, err := dec.DecodeExtHeader()
	if err != nil {
		return nil, err
	}
	if length < 0 {
		return nil, fmt.Errorf("MessagePack extension has invalid length %d", length)
	}

	data := make([]byte, 0, min(length, maxDecodePreallocate))
	var chunkBuffer [maxDecodePreallocate]byte
	for remaining := length; remaining > 0; {
		chunkLength := min(remaining, len(chunkBuffer))
		if err := dec.ReadFull(chunkBuffer[:chunkLength]); err != nil {
			return nil, err
		}
		data = append(data, chunkBuffer[:chunkLength]...)
		remaining -= chunkLength
	}
	return deferredExtension{id: id, data: data}, nil
}

func decodeAnyArray(dec *msgpack.Decoder, depth int) ([]any, error) {
	count, err := dec.DecodeArrayLen()
	if err != nil || count < 0 {
		return nil, err
	}
	values := make([]any, 0, min(count, maxDecodePreallocate))
	for range count {
		value, err := decodeAny(dec, depth)
		if err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, nil
}

func decodeAnyMap(dec *msgpack.Decoder, depth int) (map[string]any, error) {
	count, err := dec.DecodeMapLen()
	if err != nil || count < 0 {
		return nil, err
	}
	values := make(map[string]any, min(count, maxDecodePreallocate))
	for range count {
		key, err := dec.DecodeString()
		if err != nil {
			return nil, err
		}
		value, err := decodeAny(dec, depth)
		if err != nil {
			return nil, err
		}
		values[key] = value
	}
	return values, nil
}

func decodeNil(dec *msgpack.Decoder) (bool, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return false, err
	}
	if code != msgpcode.Nil {
		return false, nil
	}
	return true, dec.DecodeNil()
}

func skipFields(dec *msgpack.Decoder, count int) error {
	for range count {
		if err := skipValue(dec); err != nil {
			return err
		}
	}
	return nil
}

func skipValue(dec *msgpack.Decoder) error {
	remaining := []int{1}
	for len(remaining) > 0 {
		level := len(remaining) - 1
		if remaining[level] == 0 {
			remaining = remaining[:level]
			continue
		}
		remaining[level]--

		code, err := dec.PeekCode()
		if err != nil {
			return err
		}
		childCount := 0
		switch {
		case msgpcode.IsFixedArray(code) || code == msgpcode.Array16 || code == msgpcode.Array32:
			childCount, err = dec.DecodeArrayLen()
		case msgpcode.IsFixedMap(code) || code == msgpcode.Map16 || code == msgpcode.Map32:
			childCount, err = dec.DecodeMapLen()
			if childCount > math.MaxInt/2 {
				return fmt.Errorf("MessagePack map has too many fields: %d", childCount)
			}
			childCount *= 2
		default:
			err = dec.Skip()
		}
		if err != nil {
			return err
		}
		if childCount > 0 {
			if len(remaining) >= maxDecodeDepth {
				return fmt.Errorf("MessagePack value exceeds the maximum nesting depth of %d", maxDecodeDepth)
			}
			remaining = append(remaining, childCount)
		}
	}
	return nil
}

func isIntegerCode(code byte) bool {
	return msgpcode.IsFixedNum(code) || code == msgpcode.Uint8 || code == msgpcode.Uint16 ||
		code == msgpcode.Uint32 || code == msgpcode.Uint64 || code == msgpcode.Int8 ||
		code == msgpcode.Int16 || code == msgpcode.Int32 || code == msgpcode.Int64
}
