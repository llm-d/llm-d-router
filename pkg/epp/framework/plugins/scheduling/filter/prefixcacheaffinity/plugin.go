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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
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
}

func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}, nil
}

func (c *Config) validate() error {
	if c.AffinityThreshold > 1.0 {
		return fmt.Errorf("affinityThreshold must be <= 1.0, got %f", c.AffinityThreshold)
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

func (p *Plugin) Filter(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)

	if len(endpoints) <= 1 || p.config.AffinityThreshold <= 0 {
		return endpoints
	}

	// Exploration: skip the gate with configured probability.
	if rand.Float64() < p.config.ExplorationProbability {
		logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: exploration skip, keeping all",
			"affinityThreshold", p.config.AffinityThreshold, "total", len(endpoints))
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

	// No sticky endpoints found, keep all.
	if len(sticky) == 0 {
		logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: no sticky endpoints",
			"affinityThreshold", p.config.AffinityThreshold, "total", len(endpoints))
		return endpoints
	}

	// TTFT load gate: break stickiness if sticky endpoints are too slow.
	if p.config.MaxTTFTPenaltyMs > 0 && len(nonSticky) > 0 {
		bestStickyTTFT := p.bestTTFT(sticky)
		bestNonStickyTTFT := p.bestTTFT(nonSticky)
		if bestStickyTTFT-bestNonStickyTTFT > p.config.MaxTTFTPenaltyMs {
			logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: TTFT load gate broken",
				"bestStickyTTFT", bestStickyTTFT, "bestNonStickyTTFT", bestNonStickyTTFT,
				"penalty", bestStickyTTFT-bestNonStickyTTFT, "maxPenalty", p.config.MaxTTFTPenaltyMs)
			return endpoints
		}
	}

	logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: narrowed to sticky",
		"affinityThreshold", p.config.AffinityThreshold, "sticky", len(sticky), "total", len(endpoints))
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
	return fwkplugin.DataDependencies{Required: required}
}

func (p *Plugin) prefixCacheScore(ep fwksched.Endpoint) float64 {
	if raw, ok := ep.Get(p.prefixMatchDataKey.String()); ok {
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
func (p *Plugin) bestTTFT(endpoints []fwksched.Endpoint) float64 {
	best := math.MaxFloat64
	for _, ep := range endpoints {
		if ttft := p.endpointTTFT(ep); ttft < best {
			best = ttft
		}
	}
	return best
}

// endpointTTFT returns the predicted TTFT (ms) for an endpoint, either from the
// latency predictor or estimated from in-flight tokens and peak prefill
// throughput. Endpoints missing the required attribute contribute no signal:
// MaxFloat64 on the predictor path (never the fastest), 0 in-flight tokens on
// the throughput path (no observed load).
func (p *Plugin) endpointTTFT(ep fwksched.Endpoint) float64 {
	if p.config.usesLatencyPredictor() {
		if raw, ok := ep.Get(p.latencyPredictionInfoDataKey.String()); ok {
			info := raw.(*attrlatency.LatencyPredictionInfo)
			return info.TTFT()
		}
		return math.MaxFloat64
	}
	return float64(p.inFlightTokens(ep)) / p.config.PeakPrefillThroughput * 1000
}

// inFlightTokens returns an endpoint's in-flight token count, or 0 when the
// attribute is absent (no observed load).
func (p *Plugin) inFlightTokens(ep fwksched.Endpoint) int64 {
	if raw, ok := ep.Get(p.inFlightLoadDataKey.String()); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return load.Tokens
		}
	}
	return 0
}
