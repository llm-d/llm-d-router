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

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// EventSource identifies a decoded batch's serving endpoint and wire sequence.
type EventSource struct {
	Endpoint  string
	ModelName string
	Sequence  uint64
	// Retired is set on Reset when the endpoint is removed.
	Retired bool
	// Resync marks a reset or batch that rebuilds the endpoint for the
	// consumers that requested a resync. On a Resync reset, Sequence is the
	// last wire sequence the rebuild covers.
	Resync bool
}

// EventConsumer receives decoded events in place of the pool's token-keyed
// index path. Calls are ordered per source endpoint and concurrent across
// endpoints. Implementations must not mutate batches. An error from
// ProcessEvents resets that source.
type EventConsumer interface {
	// ProcessEvents delivers one decoded batch, including wire
	// AllBlocksCleared events.
	ProcessEvents(ctx context.Context, source EventSource, batch EventBatch) error
	// Reset invalidates observations from the endpoint's stream. The pool
	// calls it on a publisher restart, a failed replay, before a rebuild
	// from a snapshot or a full replay, on subscriber attachment, and on
	// endpoint removal. A dropped frame or an undecodable message without
	// replay or snapshot recovery does not reset.
	Reset(ctx context.Context, source EventSource) error
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

// NewIndexConsumer returns a Pool that indexes the decoded batches a
// subscribing pool hands it. It has no queues and no workers of its own.
func NewIndexConsumer(index kvblock.Index, tokenProcessor kvblock.TokenProcessor) *Pool {
	metrics.Register()
	return &Pool{
		index:          index,
		tokenProcessor: tokenProcessor,
		groupCatalog:   kvblock.NewGroupCatalog(),
		cacheFormats:   kvblock.NewCacheFormatCatalog(),
		dedup:          newEventDedupFilter(),
	}
}

// ProcessEvents indexes a decoded batch using this pool's token processor.
// The caller must serialize batches and resets for each endpoint.
func (p *Pool) ProcessEvents(ctx context.Context, source EventSource, batch EventBatch) error {
	p.processEventBatch(ctx, &batch, source.Endpoint, source.ModelName)
	return nil
}

// Reset clears indexed residency and, for retired endpoints, learned formats.
func (p *Pool) Reset(ctx context.Context, source EventSource) error {
	p.clearPod(ctx, source.Endpoint, source.Retired)
	return nil
}
