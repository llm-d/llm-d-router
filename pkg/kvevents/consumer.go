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

package kvevents

import "context"

// EventSource identifies a decoded batch's serving endpoint and wire sequence.
type EventSource struct {
	Endpoint  string
	ModelName string
	Sequence  uint64
}

// EventConsumer receives decoded events in place of the pool's token-keyed
// index path. Calls are ordered per source endpoint and concurrent across
// endpoints. Implementations must not mutate batches. An error from
// ProcessEvents resets that source.
type EventConsumer interface {
	// ProcessEvents delivers one decoded batch, including wire
	// AllBlocksCleared events.
	ProcessEvents(ctx context.Context, source EventSource, batch EventBatch) error
	// Reset drops everything learned from the endpoint's stream. The pool
	// calls it on a sequence gap without replay, a failed replay, a
	// reconnect, subscriber attachment, and endpoint removal.
	Reset(ctx context.Context, endpoint string) error
}

// NewConsumerPool uses the existing subscriber queues and replay handling with
// a consumer that owns event processing. It performs no token hashing.
func NewConsumerPool(cfg *Config, adapter EngineAdapter, consumer EventConsumer) (*Pool, error) {
	p, err := NewPool(cfg, nil, nil, adapter)
	if err != nil {
		return nil, err
	}
	p.consumer = consumer
	return p, nil
}
