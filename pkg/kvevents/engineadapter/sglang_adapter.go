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
	"fmt"

	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

const (
	// Minimum required fields, tag included (excluding trailing optional ones).
	// See: sglang/srt/disaggregation/kv_events.py (BlockStored, BlockRemoved classes).
	sglangBlockStoredMinFields  = 5 // tag + block_hashes + parent + tokens + block_size
	sglangBlockRemovedMinFields = 2 // tag + block_hashes
)

// Field-name order of map-encoded SGLang events, mirroring the converters'
// positional layouts below. "type" (the tag) is not listed here: it is read
// separately by name and placed at position 0 by hand, then these names fill
// the remaining positions in order.
var (
	sglangBlockStoredFieldOrder  = []string{"block_hashes", "parent_block_hash", "token_ids", "block_size", "lora_id", "medium"}
	sglangBlockRemovedFieldOrder = []string{"block_hashes", "medium"}
)

// SGLangAdapter implements the kvevents.EngineAdapter interface for SGLang engines.
//
// SGLang emits events either as positional msgpack arrays with trailing
// defaults omitted (msgspec array_like=True) or, since
// sgl-project/sglang#37482, as field-name maps tagged under "type". Both
// decode into the same positional []any layout, extracted with length guards
// instead of fixed structs, so the converters stay encoding-agnostic.
type SGLangAdapter struct {
	eventConverters map[string]func([]any) (kvevents.GenericEvent, error)
}

// NewSGLangAdapter creates a new SGLang adapter.
func NewSGLangAdapter() *SGLangAdapter {
	adapter := &SGLangAdapter{}

	adapter.eventConverters = map[string]func([]any) (kvevents.GenericEvent, error){
		eventTagBlockStored:      adapter.convertBlockStoredEvent,
		eventTagBlockRemoved:     adapter.convertBlockRemovedEvent,
		eventTagAllBlocksCleared: adapter.convertAllBlocksClearedEvent,
	}

	return adapter
}

// ShardingKey extracts the pod-id segment from a SGLang raw message topic.
// Expected topic format: "kv@<pod-id>@<model-name>" (same as vLLM).
func (s *SGLangAdapter) ShardingKey(msg *kvevents.RawMessage) string {
	podID, _ := parseTopic(msg.Topic)
	return podID
}

// ParseMessage parses a raw transport message into domain data.
// It extracts pod identity and model name from the topic,
// and decodes the msgpack payload into an EventBatch.
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (s *SGLangAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	podID, modelName := parseTopic(msg.Topic)

	var batch msgpackEventBatch
	if err := msgpack.Unmarshal(msg.Payload, &batch); err != nil {
		return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode SGLang event batch: %w", err)
	}

	genericEvents := make([]kvevents.GenericEvent, len(batch.Events))
	for i, rawEventBytes := range batch.Events {
		genericEvent, err := s.decodeSGLangEvent(rawEventBytes)
		if err != nil {
			return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode SGLang event: %w", err)
		}
		genericEvents[i] = genericEvent
	}

	eventBatch := kvevents.EventBatch{
		Timestamp:        batch.TS,
		Events:           genericEvents,
		DataParallelRank: batch.DataParallelRank,
	}

	return podID, modelName, eventBatch, nil
}

// decodeSGLangEvent decodes a single SGLang event from msgpack bytes and dispatches
// it to the matching converter. Map-encoded events are first normalized to the
// positional []any layout the converters consume; each converter enforces its
// own minimum-field count.
func (s *SGLangAdapter) decodeSGLangEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	var decoded any
	if err := msgpack.Unmarshal(rawEventBytes, &decoded); err != nil {
		return nil, fmt.Errorf("unmarshal event payload: %w", err)
	}

	var fields []any
	switch ev := decoded.(type) {
	case []any:
		fields = ev
	case map[string]any:
		var err error
		if fields, err = sglangMapEventToFields(ev); err != nil {
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

	converter, exists := s.eventConverters[tag]
	if !exists {
		return nil, fmt.Errorf("unknown SGLang event tag: %s", tag)
	}

	return converter(fields)
}

// sglangMapEventToFields normalizes a map-encoded SGLang event to positional []any.
// Absent fields become nil (same as an omitted trailing array field); unknown
// tags pass through so converter lookup reports them uniformly.
func sglangMapEventToFields(ev map[string]any) ([]any, error) {
	rawTag, exists := ev["type"]
	if !exists {
		return nil, fmt.Errorf("map-encoded event is missing the %q tag", "type")
	}
	tag, ok := rawTag.(string)
	if !ok {
		return nil, fmt.Errorf("map-encoded event tag (%q) is not a string: %T", "type", rawTag)
	}

	var order []string
	switch tag {
	case eventTagBlockStored:
		order = sglangBlockStoredFieldOrder
	case eventTagBlockRemoved:
		order = sglangBlockRemovedFieldOrder
	case eventTagAllBlocksCleared:
		// no payload fields
	default:
		return []any{tag}, nil
	}

	fields := make([]any, 0, len(order)+1)
	fields = append(fields, tag)
	for _, name := range order {
		fields = append(fields, ev[name])
	}
	return fields, nil
}

// convertBlockStoredEvent converts a decoded []any into a BlockStoredEvent.
// SGLang field positions (array_like=True, tag=True), also produced when
// normalizing the tagged-map form:
//
//	[0] tag                string            (consumed by decodeSGLangEvent)
//	[1] block_hashes       []hash
//	[2] parent_block_hash  hash|nil
//	[3] token_ids          []uint32
//	[4] block_size         int
//	[5] lora_id            int|nil    (optional, omit_defaults)
//	[6] medium             string|nil (optional, omit_defaults)
//
// cache_salt and session_id are map-only fields and have no positional representation.
// slot and are not carried into kvevents.BlockStoredEvent.
func (s *SGLangAdapter) convertBlockStoredEvent(fields []any) (kvevents.GenericEvent, error) {
	if len(fields) < sglangBlockStoredMinFields {
		return nil, fmt.Errorf("BlockStored event has too few fields: %d (minimum %d)", len(fields), sglangBlockStoredMinFields)
	}

	rawHashes, ok := fields[1].([]any)
	if !ok {
		return nil, fmt.Errorf("BlockStored: block_hashes is not an array: %T", fields[1])
	}
	blockHashes, err := convertBlockHashes(rawHashes)
	if err != nil {
		return nil, err
	}

	var parentHash uint64
	if fields[2] != nil {
		hash, err := getHashAsUint64(fields[2])
		if err != nil {
			return nil, fmt.Errorf("failed to parse parent hash: %w", err)
		}
		parentHash = hash
	}

	tokens, err := toUint32Slice(fields[3])
	if err != nil {
		return nil, fmt.Errorf("BlockStored: %w", err)
	}

	blockSize, err := toInt(fields[4])
	if err != nil {
		return nil, fmt.Errorf("BlockStored: block_size: %w", err)
	}

	var loraID *int
	if raw := fieldAt(fields, 5); raw != nil {
		id, err := toInt(raw)
		if err != nil {
			return nil, fmt.Errorf("BlockStored: lora_id: %w", err)
		}
		loraID = &id
	}

	var deviceTier string
	if raw := fieldAt(fields, 6); raw != nil {
		mediumStr, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("BlockStored: medium is not a string: %T", raw)
		}
		deviceTier = mediumStr
	}

	return &kvevents.BlockStoredEvent{
		BlockHashes: blockHashes,
		Tokens:      tokens,
		ParentHash:  parentHash,
		BlockSize:   blockSize,
		DeviceTier:  deviceTier,
		LoraID:      loraID,
	}, nil
}

// convertBlockRemovedEvent converts a decoded []any into a BlockRemovedEvent.
// SGLang field positions:
//
//	[0] tag           string
//	[1] block_hashes  []hash
//	[2] medium        string|nil (optional, omit_defaults)
func (s *SGLangAdapter) convertBlockRemovedEvent(fields []any) (kvevents.GenericEvent, error) {
	if len(fields) < sglangBlockRemovedMinFields {
		return nil, fmt.Errorf("BlockRemoved event has too few fields: %d (minimum %d)", len(fields), sglangBlockRemovedMinFields)
	}

	rawHashes, ok := fields[1].([]any)
	if !ok {
		return nil, fmt.Errorf("BlockRemoved: block_hashes is not an array: %T", fields[1])
	}
	blockHashes, err := convertBlockHashes(rawHashes)
	if err != nil {
		return nil, err
	}

	var deviceTier string
	if raw := fieldAt(fields, 2); raw != nil {
		mediumStr, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("BlockRemoved: medium is not a string: %T", raw)
		}
		deviceTier = mediumStr
	}

	return &kvevents.BlockRemovedEvent{
		BlockHashes: blockHashes,
		DeviceTier:  deviceTier,
	}, nil
}

// convertAllBlocksClearedEvent converts a decoded []any into an AllBlocksClearedEvent.
func (s *SGLangAdapter) convertAllBlocksClearedEvent(_ []any) (kvevents.GenericEvent, error) {
	return &kvevents.AllBlocksClearedEvent{}, nil
}
