/*
Copyright 2025 The llm-d Authors.

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

package kvcache

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

type tracedScorer struct {
	next KVBlockScorer
}

// NewTracedScorer wraps a KVBlockScorer and emits OpenTelemetry traces for Score operations.
// This encapsulates all tracing logic for the KVBlockScorer interface.
func NewTracedScorer(next KVBlockScorer) KVBlockScorer {
	return &tracedScorer{next: next}
}

func (t *tracedScorer) Strategy() KVScoringStrategy {
	return t.next.Strategy()
}

func (t *tracedScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	tracer := tracing.Tracer(TracerScope)
	_, span := tracer.Start(ctx, "compute_scores",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("llm_d.kv_cache.scorer.algorithm", string(t.next.Strategy())),
		attribute.Int("llm_d.kv_cache.scorer.key_count", len(keys)),
	)

	scores, err := t.next.Score(ctx, keys, keyToPods)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	// Calculate score distribution
	if len(scores) > 0 {
		maxScore := 0.0
		totalScore := 0.0
		for _, score := range scores {
			if score > maxScore {
				maxScore = score
			}
			totalScore += score
		}
		avgScore := totalScore / float64(len(scores))

		span.SetAttributes(
			attribute.Float64("llm_d.kv_cache.score.max", maxScore),
			attribute.Float64("llm_d.kv_cache.score.avg", avgScore),
			attribute.Int("llm_d.kv_cache.scorer.pods_scored", len(scores)),
		)
	}

	return scores, nil
}
