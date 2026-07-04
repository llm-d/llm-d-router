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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	scorerSpanName              = "score_endpoints"
	scorerTypeAttribute         = "llm_d.epp.scorer.type"
	scorerNameAttribute         = "llm_d.epp.scorer.name"
	scorerWeightAttribute       = "llm_d.epp.scorer.weight"
	scorerCandidateAttribute    = "llm_d.epp.scorer.candidate_endpoints"
	scorerEndpointsAttribute    = "llm_d.epp.scorer.scored_endpoints"
	scorerMaxScoreAttribute     = "llm_d.epp.scorer.max_score"
	scorerAverageScoreAttribute = "llm_d.epp.scorer.average_score"
)

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
	attrs := []attribute.KeyValue{
		attribute.String(scorerTypeAttribute, typedName.Type),
		attribute.String(scorerNameAttribute, typedName.Name),
		attribute.Float64(scorerWeightAttribute, s.weight),
		attribute.Int(scorerCandidateAttribute, len(endpoints)),
	}
	if request != nil {
		if request.TargetModel != "" {
			attrs = append(attrs, attribute.String("gen_ai.request.model", request.TargetModel))
		}
		if request.RequestID != "" {
			attrs = append(attrs, attribute.String("gen_ai.request.id", request.RequestID))
		}
	}

	ctx, span := tracing.Tracer(TracerScope).Start(ctx, scorerSpanName,
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attrs...),
	)
	defer span.End()

	scores := s.scorer.Score(ctx, request, endpoints)

	attrs = []attribute.KeyValue{
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
