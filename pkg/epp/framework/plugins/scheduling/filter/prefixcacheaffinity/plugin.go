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
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	schedplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling"
)

const (
	PluginType = "prefix-cache-affinity-filter"
)

var _ fwksched.Filter = &Plugin{}

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

	// MaxInFlightRequestsDelta opens the gate when the least-loaded sticky
	// endpoint holds more than this many in-flight requests above the
	// least-loaded non-sticky endpoint and the request's estimated TTFT on the
	// best non-sticky endpoint exceeds its TTFT on the best sticky endpoint by
	// at most MaxTTFTPenaltyMs. Requires MaxTTFTPenaltyMs > 0. Default: 0
	// (disabled).
	MaxInFlightRequestsDelta int64 `json:"maxInFlightRequestsDelta,omitempty"`

	PrefixMatchInfoProducerName       string `json:"prefixMatchInfoProducerName,omitempty"`
	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
	InFlightLoadProducerName          string `json:"inFlightLoadProducerName,omitempty"`
}

var DefaultConfig = Config{
	AffinityThreshold:      0.80,
	ExplorationProbability: 0,
	MaxTTFTPenaltyMs:       18000,
	TTFTSource:             TTFTSourcePrefillThroughput,

	// Calibrated for Qwen 32B on 2x H100 80GB (TP=2), vLLM 0.19; see README.
	PeakPrefillThroughput: 15928,
}

type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	prefixMatchDataKey           fwkplugin.DataKey
	latencyPredictionInfoDataKey fwkplugin.DataKey
	inFlightLoadDataKey          fwkplugin.DataKey
	uncachedRequestTokensDataKey fwkplugin.DataKey
}

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
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
		uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}, nil
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
	if c.MaxInFlightRequestsDelta < 0 {
		return fmt.Errorf("maxInFlightRequestsDelta must be >= 0, got %d", c.MaxInFlightRequestsDelta)
	}
	if c.MaxInFlightRequestsDelta > 0 && c.MaxTTFTPenaltyMs == 0 {
		return errors.New("maxTTFTPenaltyMs must be > 0 when maxInFlightRequestsDelta is set")
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
		if request.RequestID != "" {
			span.SetAttributes(semconv.GenAIRequestID(request.RequestID))
		}
	}

	if len(endpoints) <= 1 || p.config.AffinityThreshold <= 0 {
		recordDecision(p.typedName.Name, outcomeNotApplicable)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeNotApplicable))
		return endpoints
	}

	// Exploration: skip the gate with configured probability.
	if rand.Float64() < p.config.ExplorationProbability {
		logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: exploration skip, keeping all",
			"affinityThreshold", p.config.AffinityThreshold, "total", len(endpoints))
		recordDecision(p.typedName.Name, outcomeExploration)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeExploration))
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
		recordDecision(p.typedName.Name, outcomeNoMatch)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeNoMatch))
		return endpoints
	}

	// TTFT load gate: break stickiness if sticky endpoints are too slow.
	if p.config.MaxTTFTPenaltyMs > 0 && len(nonSticky) > 0 {
		bestStickyTTFT, stickyOk := p.bestTTFT(sticky)
		bestNonStickyTTFT, nonStickyOk := p.bestTTFT(nonSticky)
		if !stickyOk || !nonStickyOk {
			logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: TTFT load gate skipped, missing signal",
				"stickyObserved", stickyOk, "nonStickyObserved", nonStickyOk)
			recordDecision(p.typedName.Name, outcomeMissingSignal)
			span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeMissingSignal))
			return endpoints
		}
		penalty := bestStickyTTFT - bestNonStickyTTFT
		span.SetAttributes(semconv.LLMDEPPFilterTTFTPenaltyMs(penalty))
		if penalty > p.config.MaxTTFTPenaltyMs {
			logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: TTFT load gate broken",
				"bestStickyTTFT", bestStickyTTFT, "bestNonStickyTTFT", bestNonStickyTTFT,
				"penalty", penalty, "maxPenalty", p.config.MaxTTFTPenaltyMs)
			recordDecision(p.typedName.Name, outcomeLoadOverride)
			span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeLoadOverride))
			return endpoints
		}
	}

	// In-flight request gate: break stickiness if sticky endpoints hold many
	// more in-flight requests and serving the request elsewhere costs at most
	// the penalty more than serving it on a sticky endpoint.
	if p.config.MaxInFlightRequestsDelta > 0 && len(nonSticky) > 0 {
		stickyRequests, stickyOk := p.leastInFlightRequests(sticky)
		nonStickyRequests, nonStickyOk := p.leastInFlightRequests(nonSticky)
		if stickyOk && nonStickyOk && stickyRequests-nonStickyRequests > p.config.MaxInFlightRequestsDelta {
			stickyTTFT, stickyOk := p.bestRequestTTFT(sticky)
			nonStickyTTFT, nonStickyOk := p.bestRequestTTFT(nonSticky)
			if stickyOk && nonStickyOk && nonStickyTTFT-stickyTTFT <= p.config.MaxTTFTPenaltyMs {
				logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: in-flight request gate broken",
					"stickyRequests", stickyRequests, "nonStickyRequests", nonStickyRequests,
					"maxDelta", p.config.MaxInFlightRequestsDelta,
					"stickyRequestTTFT", stickyTTFT, "nonStickyRequestTTFT", nonStickyTTFT,
					"maxPenalty", p.config.MaxTTFTPenaltyMs)
				recordDecision(p.typedName.Name, outcomeRequestLoadOverride)
				span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeRequestLoadOverride))
				return endpoints
			}
		}
	}

	logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: narrowed to sticky",
		"affinityThreshold", p.config.AffinityThreshold, "sticky", len(sticky), "total", len(endpoints))
	recordDecision(p.typedName.Name, outcomeSticky)
	span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeSticky))
	return sticky
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
		}
	}
	if p.config.MaxInFlightRequestsDelta > 0 {
		required[p.inFlightLoadDataKey] = attrconcurrency.InFlightLoad{}
		if !p.config.usesLatencyPredictor() {
			required[p.uncachedRequestTokensDataKey] = attrconcurrency.UncachedRequestTokens{}
		}
	}
	return fwkplugin.DataDependencies{Required: required}
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
	return float64(tokens) / p.config.PeakPrefillThroughput * 1000, true
}

// requestTTFT returns the estimated TTFT (ms) of the current request on an
// endpoint. The predictor path already accounts for the request; the
// throughput path adds the request's uncached tokens on the endpoint to the
// endpoint TTFT. It returns false when a required attribute is absent.
func (p *Plugin) requestTTFT(ep fwksched.Endpoint) (float64, bool) {
	base, ok := p.endpointTTFT(ep)
	if !ok {
		return 0, false
	}
	if p.config.usesLatencyPredictor() {
		return base, true
	}
	raw, ok := ep.Get(p.uncachedRequestTokensDataKey)
	if !ok {
		return 0, false
	}
	uncached, ok := raw.(*attrconcurrency.UncachedRequestTokens)
	if !ok || uncached == nil {
		return 0, false
	}
	return base + float64(uncached.Tokens)/p.config.PeakPrefillThroughput*1000, true
}

// bestRequestTTFT returns the lowest requestTTFT across endpoints, and false
// when no endpoint carries the required attributes.
func (p *Plugin) bestRequestTTFT(endpoints []fwksched.Endpoint) (float64, bool) {
	best := math.MaxFloat64
	found := false
	for _, ep := range endpoints {
		if ttft, ok := p.requestTTFT(ep); ok && ttft < best {
			best = ttft
			found = true
		}
	}
	return best, found
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

// leastInFlightRequests returns the lowest in-flight request count across
// endpoints, and false when no endpoint carries InFlightLoad.
func (p *Plugin) leastInFlightRequests(endpoints []fwksched.Endpoint) (int64, bool) {
	best := int64(math.MaxInt64)
	found := false
	for _, ep := range endpoints {
		if n, ok := p.inFlightRequests(ep); ok && n < best {
			best = n
			found = true
		}
	}
	return best, found
}

// inFlightRequests returns an endpoint's in-flight request count. If the
// attribute is absent, it returns ok=false so the caller can treat the signal
// as missing.
func (p *Plugin) inFlightRequests(ep fwksched.Endpoint) (int64, bool) {
	if raw, ok := ep.Get(p.inFlightLoadDataKey); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return load.Requests, true
		}
	}
	return 0, false
}
