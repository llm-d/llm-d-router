/*
Copyright 2025 The Kubernetes Authors.

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

package scheduling

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	scorerSpanNamePrefix        = "llm_d.epp.scorer."
	scorerTypeAttribute         = "llm_d.scorer.type"
	scorerNameAttribute         = "llm_d.scorer.name"
	scorerWeightAttribute       = "llm_d.scorer.weight"
	scorerCandidateAttribute    = "llm_d.scorer.candidate_count"
	scorerEndpointsAttribute    = "llm_d.scorer.endpoints_scored"
	scorerMaxScoreAttribute     = "llm_d.scorer.score.max"
	scorerAverageScoreAttribute = "llm_d.scorer.score.avg"
)

func schedulerTracer() trace.Tracer {
	return otel.Tracer(schedulerInstrumentationName)
}

// TracedScorer decorates a scorer with request-local OpenTelemetry spans.
type TracedScorer struct {
	scorer fwksched.Scorer
	weight float64
}

// NewTracedScorer creates a tracing decorator for scorer.
func NewTracedScorer(scorer fwksched.Scorer, weight float64) *TracedScorer {
	return &TracedScorer{
		scorer: scorer,
		weight: weight,
	}
}

func (s *TracedScorer) TypedName() fwkplugin.TypedName {
	return s.scorer.TypedName()
}

func (s *TracedScorer) Category() fwksched.ScorerCategory {
	return s.scorer.Category()
}

func (s *TracedScorer) Score(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	typedName := s.TypedName()
	ctx, span := schedulerTracer().Start(ctx, scorerSpanNamePrefix+typedName.Type,
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(
			attribute.String(scorerTypeAttribute, typedName.Type),
			attribute.String(scorerNameAttribute, typedName.Name),
			attribute.Float64(scorerWeightAttribute, s.weight),
			attribute.Int(scorerCandidateAttribute, len(endpoints)),
		),
	)
	defer span.End()

	scores := s.scorer.Score(ctx, request, endpoints)

	attrs := []attribute.KeyValue{
		attribute.Int(scorerEndpointsAttribute, len(scores)),
	}
	if len(scores) > 0 {
		var maxScore, totalScore float64
		i := 0
		for _, score := range scores {
			if i == 0 || score > maxScore {
				maxScore = score
			}
			totalScore += score
			i++
		}
		attrs = append(attrs,
			attribute.Float64(scorerMaxScoreAttribute, maxScore),
			attribute.Float64(scorerAverageScoreAttribute, totalScore/float64(len(scores))),
		)
	}
	span.SetAttributes(attrs...)

	return scores
}
