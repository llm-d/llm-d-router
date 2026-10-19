/*
Copyright 2025 The Kubernetes Authors.
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

// Package prefixcacheaffinity provides a probabilistic filter that narrows
// candidates to "sticky" endpoints (those with high prefix cache scores).
// Can be instantiated multiple times with different thresholds (e.g., 0.99
// for global gate, 0.80 for within-tier gate).
package prefixcacheaffinity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	schedplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling"
)

const (
	PluginType = "prefix-cache-affinity-filter"
)

var (
	_ fwksched.Filter  = &Plugin{}
	_ fwkrh.PreRequest = &pairedPlugin{}
)

// TTFTSource selects the per-endpoint TTFT signal used by the load gate. The
// choice also determines which producer attribute the filter consumes.
type TTFTSource string

const (
	// TTFTSourceLatencyPredictor reads predicted TTFT from LatencyPredictionInfo
	// (produced by the predicted-latency-producer).
	TTFTSourceLatencyPredictor TTFTSource = "latencyPredictor"
	// TTFTSourcePrefillThroughput estimates TTFT from in-flight tokens and
	// PeakPrefillThroughput, reading InFlightLoad (produced by the
	// in-flight-load-producer).
	TTFTSourcePrefillThroughput TTFTSource = "prefillThroughput"
)

type Config struct {
	// AffinityThreshold is the prefix cache score threshold. Endpoints with
	// score >= this value are considered "sticky" (prompt is cached). Default: 0.80.
	AffinityThreshold float64 `json:"affinityThreshold,omitempty"`

	// ExplorationProbability is the probability of skipping the gate entirely,
	// keeping all endpoints for exploration. Range: [0, 1]. Default: 0.
	ExplorationProbability float64 `json:"explorationProbability,omitempty"`

	// MaxTTFTPenaltyMs is the max TTFT penalty (ms) before breaking stickiness.
	// If the best sticky endpoint's TTFT exceeds the best non-sticky endpoint's
	// TTFT by more than this value, all endpoints are kept. Set to 0 to always
	// stick. Default: 18000.
	MaxTTFTPenaltyMs float64 `json:"maxTTFTPenaltyMs,omitempty"`

	// TTFTSource selects where the load gate reads per-endpoint TTFT from.
	// TTFTSourcePrefillThroughput (default) estimates it from in-flight tokens and
	// PeakPrefillThroughput; TTFTSourceLatencyPredictor reads predicted TTFT from
	// the latency predictor.
	TTFTSource TTFTSource `json:"ttftSource,omitempty"`

	// PeakPrefillThroughput is the peak prefill throughput in tokens/sec, used to
	// estimate TTFT from in-flight tokens when TTFTSource is prefillThroughput:
	//   TTFT_ms = inFlightTokens / PeakPrefillThroughput * 1000
	// (tokens / (tokens/sec) * 1000 = ms). Default: 15928.
	PeakPrefillThroughput float64 `json:"peakPrefillThroughput,omitempty"`

	// UseUncachedTokens adds the incoming request's uncached token count
	// to in-flight tokens when estimating TTFT via the prefillThroughput source.
	UseUncachedTokens bool `json:"useUncachedTokens,omitempty"`

	PrefixMatchInfoProducerName       string `json:"prefixMatchInfoProducerName,omitempty"`
	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
	InFlightLoadProducerName          string `json:"inFlightLoadProducerName,omitempty"`

	// P2PSourceProducerName names the p2p-source-producer whose decision for
	// each request is paired with this filter's outcome in
	// llm_d_epp_prefix_cache_affinity_filter_p2p_decisions_total. Empty disables
	// the pairing.
	P2PSourceProducerName string `json:"p2pSourceProducerName,omitempty"`
}

var DefaultConfig = Config{
	AffinityThreshold:      0.80,
	ExplorationProbability: 0,
	MaxTTFTPenaltyMs:       18000,
	TTFTSource:             TTFTSourcePrefillThroughput,

	// Calibrated for Qwen 32B on 2x H100 80GB (TP=2), vLLM 0.19; see README.
	PeakPrefillThroughput: 15928,
	UseUncachedTokens:     false,
}

type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	prefixMatchDataKey           fwkplugin.DataKey
	latencyPredictionInfoDataKey fwkplugin.DataKey
	inFlightLoadDataKey          fwkplugin.DataKey
	outcomeDataKey               fwkplugin.DataKey
	p2pDecisionDataKey           fwkplugin.DataKey
	uncachedRequestDataKey       fwkplugin.DataKey
}

// pairedPlugin is the filter with p2pSourceProducerName set. Only it implements
// PreRequest, so an unpaired filter adds no per-request hook.
type pairedPlugin struct {
	*Plugin
}

// affinityOutcome is the outcome Filter stores on the request for PreRequest.
type affinityOutcome string

func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	p := &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
		outcomeDataKey:               fwkplugin.NewDataKey("affinity-outcome", PluginType).WithNonEmptyProducerName(name),
		p2pDecisionDataKey:           attrprefix.P2PSourceDecisionDataKey.WithNonEmptyProducerName(config.P2PSourceProducerName),
		uncachedRequestDataKey:       attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}
	if p.pairsP2P() {
		return &pairedPlugin{Plugin: p}, nil
	}
	return p, nil
}

func (c *Config) validate() error {
	if c.AffinityThreshold < 0 || c.AffinityThreshold > 1.0 {
		return fmt.Errorf("affinityThreshold must be in [0, 1], got %f", c.AffinityThreshold)
	}
	if c.ExplorationProbability < 0 || c.ExplorationProbability > 1.0 {
		return fmt.Errorf("explorationProbability must be in [0, 1], got %f", c.ExplorationProbability)
	}
	if c.MaxTTFTPenaltyMs < 0 {
		return fmt.Errorf("maxTTFTPenaltyMs must be >= 0, got %f", c.MaxTTFTPenaltyMs)
	}
	if c.PeakPrefillThroughput < 0 {
		return fmt.Errorf("peakPrefillThroughput must be >= 0, got %f", c.PeakPrefillThroughput)
	}
	switch c.TTFTSource {
	case TTFTSourceLatencyPredictor, TTFTSourcePrefillThroughput:
	default:
		return fmt.Errorf("ttftSource must be %q or %q, got %q", TTFTSourceLatencyPredictor, TTFTSourcePrefillThroughput, c.TTFTSource)
	}
	if !c.usesLatencyPredictor() && c.MaxTTFTPenaltyMs > 0 && c.PeakPrefillThroughput == 0 {
		return errors.New("peakPrefillThroughput must be > 0 when ttftSource is prefillThroughput")
	}
	return nil
}

// usesLatencyPredictor reports whether the load gate sources TTFT from the
// latency predictor. Throughput is the default; only an explicit
// latencyPredictor selects the predictor.
func (c *Config) usesLatencyPredictor() bool {
	return c.TTFTSource == TTFTSourceLatencyPredictor
}

func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)

	_, span := tracing.Tracer(schedplugins.TracerScope).Start(ctx, "filter_prefix_cache_affinity",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		semconv.LLMDEPPFilterCandidateEndpoints(len(endpoints)),
		semconv.LLMDEPPFilterAffinityThreshold(p.config.AffinityThreshold),
	)
	if request != nil {
		if request.TargetModel != "" {
			span.SetAttributes(semconv.GenAIRequestModel(request.TargetModel))
		}
		if clientRequestID := request.Headers[reqcommon.RequestIDHeaderKey]; clientRequestID != "" {
			span.SetAttributes(semconv.GenAIRequestID(clientRequestID))
		}
	}

	if len(endpoints) <= 1 || p.config.AffinityThreshold <= 0 {
		p.decide(request, span, outcomeNotApplicable)
		return endpoints
	}

	// Exploration: skip the gate with configured probability.
	if rand.Float64() < p.config.ExplorationProbability { //#nosec G404 -- exploration sampling, not security-sensitive
		logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: exploration skip, keeping all",
			"affinityThreshold", p.config.AffinityThreshold, "total", len(endpoints))
		p.decide(request, span, outcomeExploration)
		return endpoints
	}

	// Find sticky and non-sticky endpoints.
	var sticky, nonSticky []fwksched.Endpoint
	for _, ep := range endpoints {
		if p.prefixCacheScore(ep) >= p.config.AffinityThreshold {
			sticky = append(sticky, ep)
		} else {
			nonSticky = append(nonSticky, ep)
		}
	}

	span.SetAttributes(semconv.LLMDEPPFilterStickyEndpoints(len(sticky)))

	// No sticky endpoints found, keep all.
	if len(sticky) == 0 {
		logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: no sticky endpoints",
			"affinityThreshold", p.config.AffinityThreshold, "total", len(endpoints))
		p.decide(request, span, outcomeNoMatch)
		return endpoints
	}

	// TTFT load gate: break stickiness if sticky endpoints are too slow.
	if p.config.MaxTTFTPenaltyMs > 0 && len(nonSticky) > 0 {
		bestStickyTTFT, stickyOk := p.bestTTFT(sticky)
		bestNonStickyTTFT, nonStickyOk := p.bestTTFT(nonSticky)
		if !stickyOk || !nonStickyOk {
			logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: TTFT load gate skipped, missing signal",
				"stickyObserved", stickyOk, "nonStickyObserved", nonStickyOk)
			p.decide(request, span, outcomeMissingSignal)
			return endpoints
		}
		penalty := bestStickyTTFT - bestNonStickyTTFT
		span.SetAttributes(semconv.LLMDEPPFilterTTFTPenaltyMs(penalty))
		if penalty > p.config.MaxTTFTPenaltyMs {
			logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: TTFT load gate broken",
				"bestStickyTTFT", bestStickyTTFT, "bestNonStickyTTFT", bestNonStickyTTFT,
				"penalty", penalty, "maxPenalty", p.config.MaxTTFTPenaltyMs)
			p.decide(request, span, outcomeLoadOverride)
			return endpoints
		}
	}

	logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: narrowed to sticky",
		"affinityThreshold", p.config.AffinityThreshold, "sticky", len(sticky), "total", len(endpoints))
	p.decide(request, span, outcomeSticky)
	return sticky
}

// decide records the outcome. With pairing on, a repeat call for the same
// request (the instance runs in several scheduling profiles) stores
// outcomeMultiple.
func (p *Plugin) decide(request *fwksched.InferenceRequest, span trace.Span, outcome string) {
	recordDecision(p.typedName.Name, outcome)
	span.SetAttributes(semconv.LLMDEPPFilterDecision(outcome))
	if !p.pairsP2P() || request == nil {
		return
	}
	if _, decided := fwksched.ReadRequestAttribute[affinityOutcome](request, p.outcomeDataKey); decided {
		outcome = outcomeMultiple
	}
	request.PutAttribute(p.outcomeDataKey, affinityOutcome(outcome))
}

func (p *Plugin) pairsP2P() bool {
	return p.config.P2PSourceProducerName != ""
}

func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	required := map[fwkplugin.DataKey]any{
		p.prefixMatchDataKey: attrprefix.PrefixCacheMatchInfo{},
	}
	if p.config.MaxTTFTPenaltyMs > 0 {
		if p.config.usesLatencyPredictor() {
			required[p.latencyPredictionInfoDataKey] = attrlatency.LatencyPredictionInfo{}
		} else {
			required[p.inFlightLoadDataKey] = attrconcurrency.InFlightLoad{}
			if p.config.UseUncachedTokens {
				required[p.uncachedRequestDataKey] = attrconcurrency.UncachedRequestTokens{}
			}
		}
	}
	return fwkplugin.DataDependencies{Required: required}
}

// Consumes adds the p2p-source-producer's decision as optional: the filter
// routes the same without that producer, and the declared read orders the
// producer's PreRequest before this one.
func (p *pairedPlugin) Consumes() fwkplugin.DataDependencies {
	deps := p.Plugin.Consumes()
	deps.Optional = map[fwkplugin.DataKey]any{p.p2pDecisionDataKey: attrprefix.P2PSourceDecision{}}
	return deps
}

// Produces declares the outcome Filter stores on the request; PreRequest reads
// it back.
func (p *pairedPlugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.outcomeDataKey: affinityOutcome("")}
}

// PreRequest pairs the outcome Filter stored for the request with the
// p2p-source-producer's decision. A request this instance did not filter is
// not counted.
func (p *pairedPlugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if request == nil {
		return nil
	}
	outcome, ok := fwksched.ReadRequestAttribute[affinityOutcome](request, p.outcomeDataKey)
	if !ok {
		return nil
	}
	decision, ok := fwksched.ReadRequestAttribute[attrprefix.P2PSourceDecision](request, p.p2pDecisionDataKey)
	if !ok {
		decision = attrprefix.P2PSourceDecision{Outcome: p2pOutcomeUnreported}
	}
	recordP2PDecision(p.typedName.Name, decision.EndpointRole, string(outcome), decision.Outcome)
	return nil
}

func (p *Plugin) prefixCacheScore(ep fwksched.Endpoint) float64 {
	if raw, ok := ep.Get(p.prefixMatchDataKey); ok {
		info := raw.(*attrprefix.PrefixCacheMatchInfo)
		if info.TotalBlocks() > 0 {
			score := float64(info.MatchBlocks()) / float64(info.TotalBlocks())
			if !math.IsNaN(score) {
				return score
			}
		}
	}
	return 0
}

// bestTTFT returns the lowest per-endpoint TTFT (ms) across endpoints.
func (p *Plugin) bestTTFT(endpoints []fwksched.Endpoint) (float64, bool) {
	best := math.MaxFloat64
	found := false
	for _, ep := range endpoints {
		if ttft, ok := p.endpointTTFT(ep); ok && ttft < best {
			best = ttft
			found = true
		}
	}
	return best, found
}

// endpointTTFT returns the predicted TTFT (ms) for an endpoint, either from the
// latency predictor or estimated from in-flight tokens and peak prefill
// throughput. Endpoints missing the required attribute contribute no signal:
// the function returns false and the endpoint is excluded from the bestTTFT calculation.
func (p *Plugin) endpointTTFT(ep fwksched.Endpoint) (float64, bool) {
	if p.config.usesLatencyPredictor() {
		if raw, ok := ep.Get(p.latencyPredictionInfoDataKey); ok {
			info := raw.(*attrlatency.LatencyPredictionInfo)
			return info.TTFT(), true
		}
		return math.MaxFloat64, false
	}
	tokens, ok := p.inFlightTokens(ep)
	if !ok {
		return 0, false
	}
	if p.config.UseUncachedTokens {
		uncachedTokens, ok := p.uncachedRequestTokens(ep)
		if !ok {
			return 0, false
		}
		tokens += uncachedTokens
	}
	return float64(tokens) / p.config.PeakPrefillThroughput * 1000, true
}

// inFlightTokens returns an endpoint's in-flight token count. If the attribute is
// absent, it returns ok=false so the caller can treat the signal as missing.
func (p *Plugin) inFlightTokens(ep fwksched.Endpoint) (int64, bool) {
	if raw, ok := ep.Get(p.inFlightLoadDataKey); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return load.Tokens, true
		}
	}
	return 0, false
}

// uncachedRequestTokens returns the number of tokens from the current request
// that this endpoint would need to prefill from scratch. If the attribute is
// absent, it returns ok=false so the caller can treat the signal as missing.
func (p *Plugin) uncachedRequestTokens(ep fwksched.Endpoint) (int64, bool) {
	if raw, ok := ep.Get(p.uncachedRequestDataKey); ok {
		if uncached, ok := raw.(*attrconcurrency.UncachedRequestTokens); ok && uncached != nil {
			return uncached.Tokens, true
		}
	}
	return 0, false
}
