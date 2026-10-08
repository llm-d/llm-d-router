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

// VLLMAdapter parses vLLM KV-cache events.
type VLLMAdapter struct{}

// NewVLLMAdapter creates a vLLM adapter.
func NewVLLMAdapter() *VLLMAdapter {
	return &VLLMAdapter{}
}

// ShardingKey extracts the pod ID from a vLLM message topic.
func (v *VLLMAdapter) ShardingKey(msg *kvevents.RawMessage) string {
	podID, _ := parseTopic(msg.Topic)
	return podID
}

// ParseMessage decodes one vLLM event batch.
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (v *VLLMAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	podID, modelName := parseTopic(msg.Topic)

	var vllmBatch msgpackVLLMEventBatch
	if err := msgpack.Unmarshal(msg.Payload, &vllmBatch); err != nil {
		return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode vLLM event batch: %w", err)
	}

	return podID, modelName, kvevents.EventBatch{
		Timestamp:        vllmBatch.timestamp,
		Events:           vllmBatch.events,
		DataParallelRank: vllmBatch.dataParallelRank,
	}, nil
}

func (v *VLLMAdapter) decodeVLLMEvent(payload []byte) (kvevents.GenericEvent, error) {
	event, err := decodeVLLMEvent(payload)
	if err != nil {
		return nil, fmt.Errorf("unmarshal event payload: %w", err)
	}
	return event, nil
}
