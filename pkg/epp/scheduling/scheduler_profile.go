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
	"fmt"
	"sort"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// internalSpanKind is hoisted to avoid allocating a span-start option on every
// scoring call.
var internalSpanKind = trace.WithSpanKind(trace.SpanKindInternal)

// NewSchedulerProfile creates a new SchedulerProfile object and returns its pointer.
func NewSchedulerProfile() *SchedulerProfile {
	return &SchedulerProfile{
		filters: []fwksched.Filter{},
		scorers: []*WeightedScorer{},
		// picker remains nil since profile doesn't support multiple pickers
	}
}

// SchedulerProfile provides a profile configuration for the scheduler which influence routing decisions.
type SchedulerProfile struct {
	filters []fwksched.Filter
	scorers []*WeightedScorer
	picker  fwksched.Picker
}

// WithFilters sets the given filter plugins as the Filter plugins.
// if the SchedulerProfile has Filter plugins, this call replaces the existing plugins with the given ones.
func (p *SchedulerProfile) WithFilters(filters ...fwksched.Filter) *SchedulerProfile {
	p.filters = filters
	return p
}

// WithScorers sets the given scorer plugins as the Scorer plugins.
// if the SchedulerProfile has Scorer plugins, this call replaces the existing plugins with the given ones.
func (p *SchedulerProfile) WithScorers(scorers ...*WeightedScorer) *SchedulerProfile {
	p.scorers = scorers
	return p
}

// WithPicker sets the given picker plugins as the Picker plugin.
// if the SchedulerProfile has Picker plugin, this call replaces the existing plugin with the given one.
func (p *SchedulerProfile) WithPicker(picker fwksched.Picker) *SchedulerProfile {
	p.picker = picker
	return p
}

// AddPlugins adds the given plugins to all scheduler plugins according to the interfaces each plugin implements.
// A plugin may implement more than one scheduler plugin interface.
// Special Case: In order to add a scorer, one must use the scorer.NewWeightedScorer function in order to provide a weight.
// if a scorer implements more than one interface, supplying a WeightedScorer is sufficient. The function will take the internal
// scorer object and register it to all interfaces it implements.
func (p *SchedulerProfile) AddPlugins(pluginObjects ...plugin.Plugin) error {
	for _, plugin := range pluginObjects {
		if weightedScorer, ok := plugin.(*WeightedScorer); ok {
			p.scorers = append(p.scorers, weightedScorer)
			plugin = weightedScorer.Scorer // if we got WeightedScorer, unwrap the plugin
		} else if scorer, ok := plugin.(fwksched.Scorer); ok { // if we got a Scorer instead of WeightedScorer that's an error.
			return fmt.Errorf("failed to register scorer '%s' without a weight. follow function documentation to register a scorer", scorer.TypedName())
		}
		if filter, ok := plugin.(fwksched.Filter); ok {
			p.filters = append(p.filters, filter)
		}
		if picker, ok := plugin.(fwksched.Picker); ok {
			if p.picker != nil {
				return fmt.Errorf("failed to set '%s' as picker, already have a registered picker plugin '%s'", picker.TypedName(), p.picker.TypedName())
			}
			p.picker = picker
		}
	}
	return nil
}

func (p *SchedulerProfile) String() string {
	filterNames := make([]string, len(p.filters))
	for i, filter := range p.filters {
		filterNames[i] = filter.TypedName().String()
	}
	scorerNames := make([]string, len(p.scorers))
	for i, scorer := range p.scorers {
		scorerNames[i] = fmt.Sprintf("%s: %f", scorer.TypedName(), scorer.Weight())
	}

	return fmt.Sprintf(
		"{Filters: [%s], Scorers: [%s], Picker: %s}",
		strings.Join(filterNames, ", "),
		strings.Join(scorerNames, ", "),
		p.picker.TypedName(),
	)
}

// Run runs a SchedulerProfile. It invokes all the SchedulerProfile plugins for the given request in this
// order - Filters, Scorers, Picker. After completing all, it returns the result.
func (p *SchedulerProfile) Run(ctx context.Context, request *fwksched.InferenceRequest, candidateEndpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	endpoints := p.runFilterPlugins(ctx, request, candidateEndpoints)
	if len(endpoints) == 0 {
		// Filters draining a non-empty candidate set means the pool is busy, not
		// broken: an empty pool is rejected in the director before scheduling
		// runs. Report it with the same status and drop-reason vocabulary as a
		// flow control capacity rejection.
		return nil, errcommon.Error{
			Code:    errcommon.ResourceExhausted,
			Msg:     "no endpoints available for the given request",
			Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonSaturated)},
		}
	}
	// if we got here, there is at least one endpoint to score
	weightedScorePerEndpoint := p.runScorerPlugins(ctx, request, endpoints)

	result := p.runPickerPlugin(ctx, request, weightedScorePerEndpoint)

	return result, nil
}

func (p *SchedulerProfile) runFilterPlugins(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	filteredEndpoints := endpoints
	debug := logger.V(logutil.DEBUG)
	debugEnabled := debug.Enabled()
	verbose := logger.V(logutil.VERBOSE)
	verboseEnabled := verbose.Enabled()
	if debugEnabled {
		debug.Info("Before running filter plugins", "endpoints", filteredEndpoints)
	}

	ctx, span := tracing.Tracer(TracerScope).Start(ctx, "filter_endpoints",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()
	tracingActive := span.IsRecording()
	if tracingActive {
		span.SetAttributes(attribute.Int("llm_d.epp.filter.candidate_endpoints", len(endpoints)))
		span.SetAttributes(requestSpanAttributes(request)...)
	}

	for _, filter := range p.filters {
		typedName := filter.TypedName()
		if verboseEnabled {
			verbose.Info("Running filter plugin", "plugin", typedName)
		}
		// The violations are dropped: Filter has no error return, so a rejected
		// write cannot fail the request here. Scope has already dropped the write,
		// logged it, and counted it under plugin_data_scope_violations_total, which
		// is where a misdeclared filter surfaces in production. Producers run under
		// executePluginsAsDAG, which does fail the request on one.
		scoped, _ := datalayer.Scope(logger, filterExtensionPoint, filter, filteredEndpoints)
		before := time.Now()
		filteredEndpoints = datalayer.Unscope(filter.Filter(ctx, request, scoped))
		metrics.RecordPluginProcessingLatency(filterExtensionPoint, typedName.Type, typedName.Name, time.Since(before))
		if debugEnabled {
			debug.Info("Completed running filter plugin successfully", "plugin", typedName, "endpoints", filteredEndpoints)
		}
		if len(filteredEndpoints) == 0 {
			if verboseEnabled {
				verbose.Info("Filter eliminated all endpoints", "plugin", typedName, "endpointsBefore", len(endpoints))
			}
			break
		}
	}
	if tracingActive {
		span.SetAttributes(attribute.Int("llm_d.epp.filter.filtered_endpoints", len(filteredEndpoints)))
	}
	if verboseEnabled {
		verbose.Info("Completed running filter plugins", "remainingEndpoints", len(filteredEndpoints))
	}

	return filteredEndpoints
}

func (p *SchedulerProfile) runScorerPlugins(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	logger := log.FromContext(ctx)
	// Cache the leveled loggers and their enabled state once. The per-endpoint
	// Info call in the scoring loop below evaluates and boxes its variadic args
	// even when the verbosity gate would suppress output; on a 100-endpoint,
	// 4-scorer fleet that line alone accounted for ~80% of total allocations per
	// Scheduler.Schedule call. Guarding by Enabled() preserves debugging behavior
	// while removing the allocation when the level is off (the production default).
	debug := logger.V(logutil.DEBUG)
	debugEnabled := debug.Enabled()
	verbose := logger.V(logutil.VERBOSE)
	verboseEnabled := verbose.Enabled()
	if debugEnabled {
		debug.Info("Before running scorer plugins", "endpoints", endpoints)
	}

	// Parent span over the whole scorer chain. Per-scorer child spans (and any
	// plugin-internal spans) nest under it. Attributes are request- and
	// chain-level only; no per-endpoint keys, to keep span cardinality bounded.
	// The tracer is resolved once and threaded into runScorer so the per-scorer
	// spans reuse it rather than rebuilding instrumentation options per scorer.
	tracer := tracing.Tracer(TracerScope)
	ctx, span := tracer.Start(ctx, "scoring", internalSpanKind)
	defer span.End()
	// On the default (tracing-disabled) path Start returns a non-recording span;
	// skip all attribute and child-span construction so the scoring hot path
	// stays allocation-free, matching the rest of this package.
	tracingActive := span.IsRecording()
	if tracingActive {
		span.SetAttributes(
			attribute.Int("llm_d.epp.scorer.count", len(p.scorers)),
			attribute.Int("llm_d.epp.scoring.candidate_endpoints", len(endpoints)),
		)
		span.SetAttributes(requestSpanAttributes(request)...)
	}

	weightedScorePerEndpoint := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		weightedScorePerEndpoint[endpoint] = float64(0) // initialize weighted score per endpoint with 0 value
	}
	// Iterate through each scorer in the chain and accumulate the weighted scores.
	for _, scorer := range p.scorers {
		typedName := scorer.TypedName()
		if verboseEnabled {
			verbose.Info("Running scorer plugin", "plugin", typedName)
		}
		scores := runScorer(ctx, tracer, tracingActive, scorer, request, endpoints)
		for endpoint, score := range scores { // weight is relative to the sum of weights
			if debugEnabled {
				debug.Info("Calculated score", "plugin", typedName, "endpoint", endpoint.GetMetadata().ID, "score", score)
			}
			weightedScorePerEndpoint[endpoint] += enforceScoreRange(score) * scorer.Weight()
		}
		if debugEnabled {
			debug.Info("Completed running scorer plugin successfully", "plugin", typedName)
		}
	}
	if verboseEnabled {
		verbose.Info("Completed running scorer plugins successfully")
	}

	return weightedScorePerEndpoint
}

// runScorer invokes a single weighted scorer and records its latency metric.
// When tracing is active it wraps the call in a scorer.<type> span
// annotated with the scorer's identity, weight, candidate count, and aggregate
// score signals; aggregates are derived from the returned score map only, with
// no per-endpoint attribute keys, to keep span cardinality bounded. When
// tracing is inactive no span or attribute work is performed.
func runScorer(ctx context.Context, tracer trace.Tracer, tracingActive bool, scorer *WeightedScorer, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	typedName := scorer.TypedName()

	// Scope against the wrapped plugin, not the WeightedScorer: the wrapper
	// embeds the Scorer interface, which does not carry Produces/Consumes, so
	// scoping the wrapper would hide every declaration the scorer makes.
	logger := log.FromContext(ctx)
	scoped, _ := datalayer.Scope(logger, scorerExtensionPoint, scorer.Scorer, endpoints) // violations dropped: Score has no error return, see runFilterPlugins

	if !tracingActive {
		before := time.Now()
		scores := datalayer.UnscopeScores(scorer.Score(ctx, request, scoped))
		metrics.RecordPluginProcessingLatency(scorerExtensionPoint, typedName.Type, typedName.Name, time.Since(before))
		return scores
	}

	ctx, span := tracer.Start(ctx, "scorer."+typedName.Type, internalSpanKind)
	defer span.End()
	span.SetAttributes(
		attribute.String("llm_d.epp.scorer.type", typedName.Type),
		attribute.String("llm_d.epp.scorer.name", typedName.Name),
		attribute.Float64("llm_d.epp.scorer.weight", scorer.Weight()),
		attribute.Int("llm_d.epp.scorer.candidate_endpoints", len(endpoints)),
	)

	before := time.Now()
	scores := datalayer.UnscopeScores(scorer.Score(ctx, request, scoped))
	metrics.RecordPluginProcessingLatency(scorerExtensionPoint, typedName.Type, typedName.Name, time.Since(before))

	if len(scores) > 0 {
		var maxScore, totalScore float64
		first := true
		for _, s := range scores {
			if first || s > maxScore {
				maxScore = s
			}
			first = false
			totalScore += s
		}
		span.SetAttributes(
			attribute.Float64("llm_d.epp.scorer.score.max", maxScore),
			attribute.Float64("llm_d.epp.scorer.score.avg", totalScore/float64(len(scores))),
			attribute.Int("llm_d.epp.scorer.endpoints_scored", len(scores)),
		)
	}

	return scores
}

func requestSpanAttributes(request *fwksched.InferenceRequest) []attribute.KeyValue {
	if request == nil {
		return nil
	}
	attributes := make([]attribute.KeyValue, 0, 2)
	if request.TargetModel != "" {
		attributes = append(attributes, attribute.String("gen_ai.request.model", request.TargetModel))
	}
	if request.RequestID != "" {
		attributes = append(attributes, attribute.String("gen_ai.request.id", request.RequestID))
	}
	return attributes
}

func (p *SchedulerProfile) runPickerPlugin(ctx context.Context, request *fwksched.InferenceRequest, weightedScorePerEndpoint map[fwksched.Endpoint]float64) *fwksched.ProfileRunResult {
	logger := log.FromContext(ctx)

	// Allocate the ScoredEndpoint values as a single contiguous backing array
	// and build the picker's pointer slice by indexing into it. Previously each
	// per-endpoint &ScoredEndpoint{...} was a separate heap allocation, which
	// at production fleet sizes (~100 pods) dominated per-request picker cost.
	// Pickers reorder the pointer slice (shuffle/sort) but do not realloc, so
	// pointer aliasing into the backing array is safe.
	n := len(weightedScorePerEndpoint)
	storage := make([]fwksched.ScoredEndpoint, n)
	scoredEndpoints := make([]*fwksched.ScoredEndpoint, n)
	i := 0
	for endpoint, score := range weightedScorePerEndpoint {
		storage[i] = fwksched.ScoredEndpoint{Endpoint: endpoint, Score: score}
		scoredEndpoints[i] = &storage[i]
		i++
	}
	typedName := p.picker.TypedName()
	debug := logger.V(logutil.DEBUG)
	debugEnabled := debug.Enabled()
	if verbose := logger.V(logutil.VERBOSE); verbose.Enabled() {
		verbose.Info("Running picker plugin", "plugin", typedName)
	}
	if debugEnabled {
		debug.Info("Candidate pods for picking", "endpoints-weighted-score", scoredEndpoints)
	}

	ctx, span := tracing.Tracer(TracerScope).Start(ctx, "pick_endpoints",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	if span.IsRecording() {
		span.SetAttributes(attribute.Int("llm_d.epp.picker.candidate_endpoints", len(scoredEndpoints)))
		// The picker almost always returns a single target, so its count carries
		// little signal. The score distribution across the strongest candidates is
		// what explains why an endpoint was chosen, so record the highest-scoring
		// few (names with their weighted scores). Captured before Pick because
		// pickers reorder scoredEndpoints in place.
		if names, scores := topScoredEndpoints(scoredEndpoints, maxTracedEndpointScores); len(names) > 0 {
			span.SetAttributes(
				attribute.StringSlice("llm_d.epp.picker.top_endpoints", names),
				attribute.Float64Slice("llm_d.epp.picker.top_scores", scores),
			)
		}
		span.SetAttributes(requestSpanAttributes(request)...)
	}

	before := time.Now()
	result := p.picker.Pick(ctx, scoredEndpoints)
	metrics.RecordPluginProcessingLatency(pickerExtensionPoint, typedName.Type, typedName.Name, time.Since(before))
	if debugEnabled {
		debug.Info("Completed running picker plugin successfully", "plugin", typedName, "result", result)
	}

	if result != nil {
		// Record the complete candidate set, which pickers narrow to their
		// selection. Pickers reorder and truncate the pointer slice, never the
		// backing array, so storage still holds every scored candidate.
		result.ScoredCandidates = storage
	}

	return result
}

// topScoredEndpoints returns the names and weighted scores of the highest
// scoring candidates, ordered by descending score with the endpoint name as a
// stable tiebreaker and capped at limit. The returned slices are index-aligned.
func topScoredEndpoints(scored []*fwksched.ScoredEndpoint, limit int) ([]string, []float64) {
	ranked := make([]*fwksched.ScoredEndpoint, len(scored))
	copy(ranked, scored)
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].Score != ranked[j].Score {
			return ranked[i].Score > ranked[j].Score
		}
		return ranked[i].GetMetadata().ID.String() <
			ranked[j].GetMetadata().ID.String()
	})
	if limit < len(ranked) {
		ranked = ranked[:limit]
	}
	names := make([]string, len(ranked))
	scores := make([]float64, len(ranked))
	for i, se := range ranked {
		names[i] = se.GetMetadata().ID.String()
		scores[i] = se.Score
	}
	return names, scores
}

func enforceScoreRange(score float64) float64 {
	if score < 0 {
		return 0
	}
	if score > 1 {
		return 1
	}
	return score
}
