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
	"bytes"
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

// Limit hash processing because only the final bytes contribute to the hash.
const maxDecodeHashBytes = 64

type msgpackTypedEventBatch struct {
	sglang           bool
	timestamp        float64
	events           []kvevents.GenericEvent
	dataParallelRank *int
}

func (b *msgpackTypedEventBatch) DecodeMsgpack(dec *msgpack.Decoder) error {
	fieldCount, err := dec.DecodeArrayLen()
	if err != nil {
		return err
	}
	minimumFields := 2
	engine := "vLLM"
	if b.sglang {
		minimumFields = 3
		engine = "SGLang"
	}
	if fieldCount < minimumFields {
		return fmt.Errorf("%s event batch: need at least %d fields, got %d", engine, minimumFields, fieldCount)
	}

	b.timestamp, err = dec.DecodeFloat64()
	if err != nil {
		return fmt.Errorf("%s event batch timestamp: %w", engine, err)
	}

	eventCount, err := dec.DecodeArrayLen()
	if err != nil {
		return fmt.Errorf("%s event batch events: %w", engine, err)
	}
	if eventCount < 0 {
		b.events = nil
	} else {
		b.events = make([]kvevents.GenericEvent, 0, min(eventCount, maxDecodePreallocate))
		for range eventCount {
			event, err := decodeTypedEventFromDecoder(dec, b.sglang)
			if err != nil {
				return fmt.Errorf("failed to decode %s event: %w", engine, err)
			}
			b.events = append(b.events, event)
		}
	}

	if fieldCount >= 3 {
		b.dataParallelRank, err = decodeOptionalInt(dec)
		if err != nil {
			return fmt.Errorf("%s event batch data parallel rank: %w", engine, err)
		}
	}
	for range fieldCount - 3 {
		if err := skipValue(dec); err != nil {
			return fmt.Errorf("%s event batch trailing field: %w", engine, err)
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
	return decodeTypedEventFromDecoder(dec, false)
}

func decodeTypedEventFromDecoder(dec *msgpack.Decoder, sglang bool) (kvevents.GenericEvent, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return nil, err
	}

	switch {
	case msgpcode.IsFixedArray(code) || code == msgpcode.Array16 || code == msgpcode.Array32:
		return decodeArrayTypedEvent(dec, sglang)
	case msgpcode.IsFixedMap(code) || code == msgpcode.Map16 || code == msgpcode.Map32:
		return decodeMapTypedEvent(dec, sglang)
	default:
		return nil, fmt.Errorf("event is neither an array nor a map: MessagePack code %#x", code)
	}
}

func decodeArrayTypedEvent(dec *msgpack.Decoder, sglang bool) (kvevents.GenericEvent, error) {
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
		return decodeArrayBlockStored(dec, fieldCount, sglang)
	case eventTagBlockRemoved:
		return decodeArrayBlockRemoved(dec, fieldCount, sglang)
	case eventTagAllBlocksCleared:
		if err := skipFields(dec, fieldCount-1); err != nil {
			return nil, err
		}
		return &kvevents.AllBlocksClearedEvent{}, nil
	default:
		if sglang {
			return nil, fmt.Errorf("unknown event tag: %s", tag)
		}
		return nil, fmt.Errorf("unknown vLLM event tag: %s", tag)
	}
}

type vllmFieldDecoder struct {
	name   string
	decode func(*msgpack.Decoder, *vllmEventFields) error
}

var blockStoredFieldDecoders = []vllmFieldDecoder{
	{"block_hashes", decodeHashesField},
	{"parent_block_hash", decodeParentHashField},
	{"token_ids", decodeTokensField},
	{"block_size", decodeBlockSizeField},
	{"lora_id", decodeLoraID},
	{"medium", decodeMedium},
	{"lora_name", decodeLoraName},
	{"extra_keys", decodeExtraKeysField},
	{"group_idx", decodeGroupIdx},
	{"kv_cache_spec_kind", decodeKVCacheSpecKind},
	{"kv_cache_spec_sliding_window", decodeSlidingWindow},
}

var blockRemovedFieldDecoders = []vllmFieldDecoder{
	blockStoredFieldDecoders[0],
	blockStoredFieldDecoders[5],
	blockStoredFieldDecoders[8],
}

func typedFieldDecoders(tag string, sglang bool) []vllmFieldDecoder {
	switch tag {
	case eventTagBlockStored:
		if sglang {
			return blockStoredFieldDecoders[:6]
		}
		return blockStoredFieldDecoders
	case eventTagBlockRemoved:
		if sglang {
			return blockRemovedFieldDecoders[:2]
		}
		return blockRemovedFieldDecoders
	default:
		return nil
	}
}

func decodeArrayFields(dec *msgpack.Decoder, fieldCount int, tag string, sglang bool) (*vllmEventFields, error) {
	fields := &vllmEventFields{}
	schema := typedFieldDecoders(tag, sglang)
	knownCount := min(fieldCount-1, len(schema))
	for _, field := range schema[:knownCount] {
		if err := field.decode(dec, fields); err != nil {
			return nil, fmt.Errorf("%s: %w", tag, err)
		}
	}
	if err := skipFields(dec, fieldCount-1-knownCount); err != nil {
		return nil, err
	}
	return fields, nil
}

func decodeArrayBlockStored(dec *msgpack.Decoder, fieldCount int, sglang bool) (kvevents.GenericEvent, error) {
	minimumFields := 5
	if sglang {
		minimumFields = 7
	}
	if fieldCount < minimumFields {
		return nil, fmt.Errorf("BlockStored: need at least %d fields, got %d", minimumFields, fieldCount)
	}
	fields, err := decodeArrayFields(dec, fieldCount, eventTagBlockStored, sglang)
	if err != nil {
		return nil, err
	}
	return fields.blockStoredEvent(), nil
}

func decodeArrayBlockRemoved(dec *msgpack.Decoder, fieldCount int, sglang bool) (kvevents.GenericEvent, error) {
	minimumFields := 2
	if sglang {
		minimumFields = 3
	}
	if fieldCount < minimumFields {
		return nil, fmt.Errorf("BlockRemoved: need at least %d fields, got %d", minimumFields, fieldCount)
	}
	fields, err := decodeArrayFields(dec, fieldCount, eventTagBlockRemoved, sglang)
	if err != nil {
		return nil, err
	}
	return fields.blockRemovedEvent(), nil
}

func decodeHashesField(dec *msgpack.Decoder, fields *vllmEventFields) error {
	var err error
	fields.blockHashes, err = decodeBlockHashes(dec)
	fields.hasHashes = err == nil
	return err
}

func decodeParentHashField(dec *msgpack.Decoder, fields *vllmEventFields) error {
	var err error
	fields.parentHash, err = decodeNullableHash(dec)
	if err != nil {
		return fmt.Errorf("failed to parse parent hash: %w", err)
	}
	return nil
}

func decodeTokensField(dec *msgpack.Decoder, fields *vllmEventFields) error {
	var err error
	fields.tokens, err = decodeTokenIDs(dec)
	fields.hasTokens = err == nil
	return err
}

func decodeBlockSizeField(dec *msgpack.Decoder, fields *vllmEventFields) error {
	var err error
	fields.blockSize, err = decodeInt(dec)
	fields.hasBlockSize = err == nil
	if err != nil {
		return fmt.Errorf("block_size: %w", err)
	}
	return nil
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

type rawVLLMMapField struct {
	name  string
	value []byte
}

// Record bytes during the bounded skip because DecodeRaw uses recursive Skip.
type recordingReader struct {
	reader  io.Reader
	scanner io.ByteScanner
	data    []byte
}

func (r *recordingReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	r.data = append(r.data, p[:n]...)
	return n, err
}

func (r *recordingReader) ReadByte() (byte, error) {
	value, err := r.scanner.ReadByte()
	if err == nil {
		r.data = append(r.data, value)
	}
	return value, err
}

func (r *recordingReader) UnreadByte() error {
	if err := r.scanner.UnreadByte(); err != nil {
		return err
	}
	r.data = r.data[:len(r.data)-1]
	return nil
}

func decodeBoundedRaw(dec *msgpack.Decoder, name string) ([]byte, error) {
	reader := dec.Buffered()
	scanner, ok := reader.(io.ByteScanner)
	if !ok {
		return nil, fmt.Errorf("MessagePack reader does not support byte scanning")
	}
	recording := &recordingReader{reader: reader, scanner: scanner}
	dec.ResetReader(recording)
	limit := maxDecodeDepth
	// Typed extra keys use the depth limit after the outer and item arrays.
	if name == "extra_keys" {
		limit += 2
	}
	err := skipValueWithDepth(dec, limit)
	dec.ResetReader(reader)
	return recording.data, err
}

func decodeMapTypedEvent(dec *msgpack.Decoder, sglang bool) (kvevents.GenericEvent, error) {
	fieldCount, err := dec.DecodeMapLen()
	if err != nil {
		return nil, err
	}

	fields := vllmEventFields{}
	// Buffer fields before the tag because the tag defines their schema.
	var deferred []rawVLLMMapField
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
					if err := decodeTypedMapField(msgpack.NewDecoder(bytes.NewReader(field.value)), field.name, fields.tag, &fields, sglang); err != nil {
						return nil, fmt.Errorf("map-encoded event field %q: %w", field.name, err)
					}
				}
			}
			deferred = nil
		case !fields.hasTag:
			knownField := false
			for _, field := range typedFieldDecoders(eventTagBlockStored, sglang) {
				if field.name == name {
					knownField = true
					break
				}
			}
			if !knownField {
				err = skipValue(dec)
				break
			}
			value, decodeErr := decodeBoundedRaw(dec, name)
			if decodeErr != nil {
				return nil, fmt.Errorf("map-encoded event field %q: %w", name, decodeErr)
			}
			deferred = append(deferred, rawVLLMMapField{name: name, value: value})
		case isKnownVLLMEventTag(fields.tag):
			err = decodeTypedMapField(dec, name, fields.tag, &fields, sglang)
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
		if sglang {
			return nil, fmt.Errorf("unknown event tag: %s", fields.tag)
		}
		return nil, fmt.Errorf("unknown vLLM event tag: %s", fields.tag)
	}
}

func decodeTypedMapField(
	dec *msgpack.Decoder,
	name string,
	tag string,
	fields *vllmEventFields,
	sglang bool,
) error {
	for _, field := range typedFieldDecoders(tag, sglang) {
		if field.name == name {
			return field.decode(dec, fields)
		}
	}
	return skipValue(dec)
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

		if length > maxDecodeHashBytes {
			return 0, fmt.Errorf("hash byte slice exceeds the maximum length of %d", maxDecodeHashBytes)
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
		var value uint64
		if code >= msgpcode.NegFixedNumLow || code == msgpcode.Int8 || code == msgpcode.Int16 || code == msgpcode.Int32 || code == msgpcode.Int64 {
			signed, decodeErr := dec.DecodeInt64()
			err = decodeErr
			if signed > 0 {
				value = uint64(signed)
			}
		} else {
			value, err = dec.DecodeUint64()
		}
		if err != nil {
			return nil, fmt.Errorf("token_ids[%d]: %w", i, err)
		}
		value = min(value, uint64(math.MaxUint32))
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

// Use an explicit stack because Decoder.Skip has no nesting limit.
func skipValue(dec *msgpack.Decoder) error {
	return skipValueWithDepth(dec, maxDecodeDepth)
}

func skipValueWithDepth(dec *msgpack.Decoder, limit int) error {
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
		case msgpcode.IsBin(code) || msgpcode.IsString(code):
			var length int
			length, err = dec.DecodeBytesLen()
			if err == nil {
				err = discardBytes(dec, length)
			}
		case msgpcode.IsExt(code):
			var length int
			_, length, err = dec.DecodeExtHeader()
			if err == nil {
				err = discardBytes(dec, length)
			}
		default:
			err = dec.Skip()
		}
		if err != nil {
			return err
		}
		if childCount > 0 {
			if len(remaining) >= limit {
				return fmt.Errorf("MessagePack value exceeds the maximum nesting depth of %d", maxDecodeDepth)
			}
			remaining = append(remaining, childCount)
		}
	}
	return nil
}

func discardBytes(dec *msgpack.Decoder, length int) error {
	var buffer [maxDecodePreallocate]byte
	for length > 0 {
		chunk := min(length, len(buffer))
		if err := dec.ReadFull(buffer[:chunk]); err != nil {
			return err
		}
		length -= chunk
	}
	return nil
}

func isIntegerCode(code byte) bool {
	return msgpcode.IsFixedNum(code) || code == msgpcode.Uint8 || code == msgpcode.Uint16 ||
		code == msgpcode.Uint32 || code == msgpcode.Uint64 || code == msgpcode.Int8 ||
		code == msgpcode.Int16 || code == msgpcode.Int32 || code == msgpcode.Int64
}
