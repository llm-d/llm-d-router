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
	"github.com/vmihailenco/msgpack/v5/msgpcode"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

var sglangWireSchema = eventWireSchema{
	name:                 "SGLang",
	minimumBatchFields:   3,
	minimumStoredFields:  7,
	minimumRemovedFields: 3,
	storedFields:         sglangBlockStoredFieldDecoders,
	removedFields:        sglangBlockRemovedFieldDecoders,
	storedEvent:          (*typedEventFields).sglangBlockStoredEvent,
	unknownTagPrefix:     "unknown event tag",
}

// SGLangAdapter parses SGLang KV events from arrays and tagged maps.
type SGLangAdapter struct{}

var sglangBlockStoredFieldDecoders = []eventFieldDecoder{
	{"block_hashes", decodeHashesField},
	{"parent_block_hash", decodeParentHashField},
	{"token_ids", decodeTokensField},
	{"block_size", decodeBlockSizeField},
	{"lora_id", decodeLoraID},
	{"medium", decodeMedium},
	{"cache_salt", decodeSGLangCacheSalt},
}

var sglangBlockRemovedFieldDecoders = []eventFieldDecoder{
	{"block_hashes", decodeHashesField},
	{"medium", decodeMedium},
}

// NewSGLangAdapter creates a SGLang adapter.
func NewSGLangAdapter() *SGLangAdapter {
	return &SGLangAdapter{}
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
func (s *SGLangAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	podID, modelName := parseTopic(msg.Topic)

	batch := msgpackTypedEventBatch{schema: &sglangWireSchema}
	if err := msgpack.Unmarshal(msg.Payload, &batch); err != nil {
		return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode SGLang event batch: %w", err)
	}
	eventBatch := kvevents.EventBatch{
		Timestamp:        batch.timestamp,
		Events:           batch.events,
		DataParallelRank: batch.dataParallelRank,
	}

	return podID, modelName, eventBatch, nil
}

func decodeSGLangCacheSalt(dec *msgpack.Decoder, fields *typedEventFields) error {
	code, err := dec.PeekCode()
	if err != nil {
		return err
	}
	if msgpcode.IsFixedMap(code) || code == msgpcode.Map16 || code == msgpcode.Map32 {
		return skipValue(dec)
	}
	value, err := decodeOptionalString(dec)
	if err != nil {
		return fmt.Errorf("cache_salt is not a string: %w", err)
	}
	if value != nil {
		fields.cacheSalt = *value
	}
	return nil
}

func (f *typedEventFields) sglangBlockStoredEvent() *kvevents.BlockStoredEvent {
	event := f.blockStoredEvent()
	// Child blocks inherit the salt through the parent's request key.
	if f.cacheSalt != "" && f.parentHash == 0 && len(f.blockHashes) > 0 {
		event.ExtraKeys = make([][]any, len(f.blockHashes))
		event.ExtraKeys[0] = []any{f.cacheSalt}
	}
	return event
}
