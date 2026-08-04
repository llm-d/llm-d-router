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

package kvevents

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

type tracedAdapter struct {
	next EngineAdapter
}

// NewTracedAdapter wraps an EngineAdapter and emits OpenTelemetry traces for
// message decoding. This encapsulates all tracing logic for the EngineAdapter
// interface.
func NewTracedAdapter(next EngineAdapter) EngineAdapter {
	return &tracedAdapter{next: next}
}

//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (t *tracedAdapter) ParseMessage(ctx context.Context, msg *RawMessage) (string, string, EventBatch, error) {
	tracer := tracing.Tracer(TracerScope)
	ctx, span := tracer.Start(ctx, "events_decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	podID, modelName, batch, err := t.next.ParseMessage(ctx, msg)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return podID, modelName, batch, err
	}

	span.SetAttributes(
		attribute.String("llm_d.kv_cache.events.pod_id", podID),
		attribute.String("gen_ai.request.model", modelName),
		attribute.Int("llm_d.kv_cache.events.event_count", len(batch.Events)),
	)

	return podID, modelName, batch, nil
}

func (t *tracedAdapter) ShardingKey(msg *RawMessage) string {
	return t.next.ShardingKey(msg)
}
