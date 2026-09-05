// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvblock

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

type tracedIndex struct {
	next Index
}

// NewTracedIndex wraps an Index and emits OpenTelemetry traces for index operations.
// This encapsulates all tracing logic for the kvblock.Index interface.
func NewTracedIndex(next Index) Index {
	return &tracedIndex{next: next}
}

func (t *tracedIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	tracer := tracing.Tracer(TracerScope)
	ctx, span := tracer.Start(ctx, "index_add",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.Int("llm_d.kv_cache.index.add.engine_key_count", len(engineKeys)),
		attribute.Int("llm_d.kv_cache.index.add.request_key_count", len(requestKeys)),
		attribute.Int("llm_d.kv_cache.index.add.pod_entry_count", len(entries)),
		attribute.Int("llm_d.kv_cache.index.add.device_tier_count", deviceTierCount(entries)),
	)

	err := t.next.Add(ctx, engineKeys, requestKeys, entries)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return err
	}

	return nil
}

func (t *tracedIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	tracer := tracing.Tracer(TracerScope)
	ctx, span := tracer.Start(ctx, "index_evict",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("llm_d.kv_cache.index.evict.key_type", keyTypeLabel(keyType)),
		attribute.Int("llm_d.kv_cache.index.evict.pod_entry_count", len(entries)),
		attribute.Int("llm_d.kv_cache.index.evict.device_tier_count", deviceTierCount(entries)),
	)

	err := t.next.Evict(ctx, key, keyType, entries)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return err
	}

	return nil
}

func (t *tracedIndex) Lookup(
	ctx context.Context,
	requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	tracer := tracing.Tracer(TracerScope)
	ctx, span := tracer.Start(ctx, "index_lookup",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.Int("llm_d.kv_cache.index.lookup.block_count", len(requestKeys)),
		attribute.Int("llm_d.kv_cache.lookup.pod_filter_count", podIdentifierSet.Len()),
	)

	result, err := t.next.Lookup(ctx, requestKeys, podIdentifierSet)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	// Calculate cache hit metrics
	blocksFound := 0
	for _, pods := range result {
		if len(pods) > 0 {
			blocksFound++
		}
	}
	cacheHit := blocksFound > 0

	span.SetAttributes(
		attribute.Bool("llm_d.kv_cache.lookup.cache_hit", cacheHit),
		attribute.Int("llm_d.kv_cache.lookup.blocks_found", blocksFound),
	)

	return result, nil
}

func (t *tracedIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	return t.next.GetRequestKey(ctx, engineKey)
}

func (t *tracedIndex) Clear(ctx context.Context, podIdentifier string) error {
	return t.next.Clear(ctx, podIdentifier)
}

func keyTypeLabel(keyType KeyType) string {
	switch keyType {
	case EngineKey:
		return "engine"
	case RequestKey:
		return "request"
	default:
		return "unknown"
	}
}

func deviceTierCount(entries []PodEntry) int {
	deviceTiers := make(map[string]struct{})
	for _, entry := range entries {
		if entry.DeviceTier == "" {
			continue
		}
		deviceTiers[entry.DeviceTier] = struct{}{}
	}
	return len(deviceTiers)
}
