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
	"maps"
	"math"
	"math/rand"
	"time"

	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
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
	_ fwksched.Filter             = &Plugin{}
	_ fwkrh.PreRequest            = &pairedPlugin{}
	_ fwkrh.ResponseBodyProcessor = &Plugin{}
)

// ttftEstimateStateKey is the PluginState key of a request's per-endpoint TTFT
// estimates, read back at the first response chunk.
const ttftEstimateStateKey = fwkplugin.StateKey("ttft-estimates")

// ttftEstimates holds the TTFT estimated for each candidate endpoint when the
// filter ran, keyed by endpoint ID, the time it ran and the request's target model.
type ttftEstimates struct {
	byEndpoint  map[string]time.Duration
	at          time.Time
	targetModel string
}

func (e *ttftEstimates) Clone() fwkplugin.StateData {
	return &ttftEstimates{byEndpoint: maps.Clone(e.byEndpoint), at: e.at, targetModel: e.targetModel}
}

// TTFTSource selects the per-endpoint TTFT signal used by the load gate. The
// choice also determines which producer attribute the filter consumes.
type TTFTSource string

// TTFTEstimate selects the method of estimation for the TTFT per-endpoint with
// the prefillThroughput source.
type TTFTEstimate string

const (
	// TTFTSourceLatencyPredictor reads predicted TTFT from LatencyPredictionInfo
	// (produced by the predicted-latency-producer).
	TTFTSourceLatencyPredictor TTFTSource = "latencyPredictor"
	// TTFTSourcePrefillThroughput estimates TTFT from in-flight tokens and
	// PeakPrefillThroughput, reading InFlightLoad (produced by the
	// in-flight-load-producer).
	TTFTSourcePrefillThroughput TTFTSource = "prefillThroughput"
)

const (
	// TTFTEstimateBackLog estimates TTFT only from the endpoint's current load (InFlightLoad).
	TTFTEstimateBackLog TTFTEstimate = "backlog"
	// TTFTEstimateFull estimates TTFT from the endpoint's current load (InFlightLoad) plus the
	// request impact (uncachedRequestTokens).
	TTFTEstimateFull TTFTEstimate = "full"
)

// minCompletionsPerSecond is the lowest completion rate assumed for an endpoint
// with queued requests, so that an endpoint that finished nothing in the
// inflight-load-producer's window still gets a finite queue wait.
const minCompletionsPerSecond = 1.0 / 30

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

	// TTFTEstimate selects how to compute the TTFT when the source is set to prefillThroughput.
	// TTFTEstimateBackLog (default) only accounts inFlightTokens and TTFTEstimateFull also handles
	// the request's uncached tokens.
	TTFTEstimate TTFTEstimate `json:"ttftEstimate,omitempty"`

	// MaxNumSeqs is the model server's limit on running requests (vLLM
	// max_num_seqs). With TTFTEstimateFull, a value > 0 adds the wait for a
	// running slot for the EPP's in-flight requests beyond it; 0 leaves it out.
	MaxNumSeqs int `json:"maxNumSeqs,omitempty"`

	// QueueWaitKVCacheThreshold is the KV cache usage, in [0, 1], at or above which
	// an endpoint's waiting requests wait for KV cache. With TTFTEstimateFull, a
	// value > 0 adds their wait to the estimate; 0 leaves it out. Default: 0.95.
	QueueWaitKVCacheThreshold float64 `json:"queueWaitKVCacheThreshold,omitempty"`

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
	PeakPrefillThroughput:     15928,
	TTFTEstimate:              TTFTEstimateBackLog,
	QueueWaitKVCacheThreshold: 0.95,
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
	// pluginState carries each request's TTFT estimates from Filter to the
	// first response chunk. Nil leaves the estimate metrics unrecorded.
	pluginState *fwkplugin.PluginState
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
	var pluginState *fwkplugin.PluginState
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
		ctx := handle.Context()
		if ctx == nil {
			ctx = context.Background()
		}
		pluginState = fwkplugin.NewPluginState(ctx)
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
		pluginState:                  pluginState,
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
	switch c.TTFTEstimate {
	case TTFTEstimateBackLog, TTFTEstimateFull:
	default:
		return fmt.Errorf("ttftEstimate must be %q or %q, got %q", TTFTEstimateBackLog, TTFTEstimateFull, c.TTFTEstimate)
	}
	if c.MaxNumSeqs < 0 {
		return fmt.Errorf("maxNumSeqs must be >= 0, got %d", c.MaxNumSeqs)
	}
	if c.QueueWaitKVCacheThreshold < 0 || c.QueueWaitKVCacheThreshold > 1.0 {
		return fmt.Errorf("queueWaitKVCacheThreshold must be in [0, 1], got %f", c.QueueWaitKVCacheThreshold)
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

func (c *Config) usesFullEstimate() bool {
	return c.TTFTEstimate == TTFTEstimateFull
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

	p.saveTTFTEstimates(ctx, request, endpoints)

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
			if p.config.usesFullEstimate() {
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
	load, ok := p.inFlightLoad(ep)
	if !ok {
		return 0, false
	}
	if !p.config.usesFullEstimate() {
		return float64(load.Tokens) / p.config.PeakPrefillThroughput * 1000, true
	}
	uncachedTokens, ok := p.uncachedRequestTokens(ep)
	if !ok {
		return 0, false
	}
	queued := p.queuedRequests(ep, load)
	tokens := float64(load.Tokens)
	if queued > 0 && load.PendingRequests > 0 {
		// The queued requests' prefill comes after their wait, which the queue
		// term already covers. Their tokens are taken as the average pending share.
		tokens *= float64(max(load.PendingRequests-queued, 0)) / float64(load.PendingRequests)
	}
	ttft := (tokens + float64(uncachedTokens)) / p.config.PeakPrefillThroughput * 1000
	return ttft + queueWaitMs(load, queued), true
}

// queueWaitMs returns the time (ms) for queued requests to be admitted at the
// endpoint's completion rate. The filter reads CompletionsPerSecond only here.
func queueWaitMs(load *attrconcurrency.InFlightLoad, queued int64) float64 {
	if queued <= 0 {
		return 0
	}
	rate := max(load.CompletionsPerSecond, minCompletionsPerSecond)
	return float64(queued) / rate * 1000
}

// queuedRequests returns the number of requests on the endpoint that wait to be
// admitted: the EPP's in-flight requests beyond MaxNumSeqs, or, when KV cache
// usage is at or above QueueWaitKVCacheThreshold, the model server's waiting
// queue if that is larger. Requests waiting for prefill token budget are not
// counted, since their tokens are part of the in-flight tokens.
func (p *Plugin) queuedRequests(ep fwksched.Endpoint, load *attrconcurrency.InFlightLoad) int64 {
	var queued int64
	if p.config.MaxNumSeqs > 0 {
		queued = max(load.Requests-int64(p.config.MaxNumSeqs), 0)
	}
	if p.config.QueueWaitKVCacheThreshold > 0 {
		if metrics := ep.GetMetrics(); metrics != nil && metrics.KVCacheUsagePercent >= p.config.QueueWaitKVCacheThreshold {
			queued = max(queued, int64(metrics.WaitingQueueSize))
		}
	}
	return queued
}

// saveTTFTEstimates stores the throughput-based TTFT estimate of every candidate
// endpoint for the request, so that ResponseBody can record the served
// endpoint's estimate next to the measured TTFT. The estimates are kept while
// ctx is alive, so a request still waiting for its first chunk after the
// PluginState staleness threshold is recorded too.
func (p *Plugin) saveTTFTEstimates(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) {
	if p.pluginState == nil || request == nil || request.RequestID == "" || p.config.usesLatencyPredictor() ||
		p.config.PeakPrefillThroughput <= 0 {
		return
	}
	byEndpoint := make(map[string]time.Duration, len(endpoints))
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			continue
		}
		if ttft, ok := p.endpointTTFT(ep); ok {
			byEndpoint[ep.GetMetadata().ID.String()] = time.Duration(ttft * float64(time.Millisecond))
		}
	}
	if len(byEndpoint) > 0 {
		p.pluginState.Write(request.RequestID, ttftEstimateStateKey,
			&ttftEstimates{byEndpoint: byEndpoint, at: time.Now(), targetModel: request.TargetModel})
		p.pluginState.BindLiveness(ctx, request.RequestID)
	}
}

// ResponseBody records, at the first response chunk, the TTFT the filter
// estimated for the endpoint that served the request and the TTFT measured
// since the filter ran. A request whose serving endpoint has no estimate, such
// as a decode endpoint when the filter ran in the prefill profile, is not recorded.
func (p *Plugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrh.Response, target *fwkdl.EndpointMetadata) {
	if p.pluginState == nil || request == nil || request.RequestID == "" || response == nil {
		return
	}
	if response.StartOfStream {
		if state, err := fwkplugin.ReadPluginStateKey[*ttftEstimates](p.pluginState, request.RequestID, ttftEstimateStateKey); err == nil && target != nil {
			if estimated, ok := state.byEndpoint[target.ID.String()]; ok {
				recordTTFT(p.typedName, state.targetModel, estimated, time.Since(state.at))
			}
		}
		p.pluginState.Delete(request.RequestID)
		return
	}
	if response.EndOfStream {
		p.pluginState.Delete(request.RequestID)
	}
}

// inFlightLoad returns an endpoint's in-flight load. If the attribute is absent,
// it returns ok=false so the caller can treat the signal as missing.
func (p *Plugin) inFlightLoad(ep fwksched.Endpoint) (*attrconcurrency.InFlightLoad, bool) {
	if raw, ok := ep.Get(p.inFlightLoadDataKey); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return load, true
		}
	}
	return nil, false
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
