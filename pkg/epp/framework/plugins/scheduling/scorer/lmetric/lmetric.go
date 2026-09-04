package lmetric

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	LMetricScorerType = "lmetric-scorer"
)

type Config struct {
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`
	InFlightLoadProducerName    string `json:"inFlightLoadProducerName,omitempty"`
}

var _ fwksched.Scorer = &Scorer{}

type Scorer struct {
	typedName           fwkplugin.TypedName
	prefixMatchDataKey  fwkplugin.DataKey
	inFlightLoadDataKey fwkplugin.DataKey
}

func Factory(name string, params *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := Config{}
	if params != nil {
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal lmetric scorer config: %w", err)
		}
	}

	return &Scorer{
		typedName:           fwkplugin.TypedName{Type: LMetricScorerType, Name: name},
		prefixMatchDataKey:  attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(cfg.PrefixMatchInfoProducerName),
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
	}, nil
}

func (s *Scorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *Scorer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

func (s *Scorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{
		s.prefixMatchDataKey:  attrprefix.PrefixCacheMatchInfo{},
		s.inFlightLoadDataKey: attrconcurrency.InFlightLoad{},
	}}
}

func (s *Scorer) Score(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	costs := make(map[fwksched.Endpoint]float64, len(endpoints))
	minCost := math.Inf(1)
	maxCost := math.Inf(-1)
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		cost, ok := s.cost(endpoint)
		if !ok {
			continue
		}
		costs[endpoint] = cost
		minCost = math.Min(minCost, cost)
		maxCost = math.Max(maxCost, cost)
	}

	for _, endpoint := range endpoints {
		cost, ok := costs[endpoint]
		if !ok {
			scores[endpoint] = 0
			logger.V(logutil.DEBUG).Info("LMetricScorer scoring",
				"endpoint", endpoint.GetMetadata().ID.String(),
				"score", 0)
			continue
		}
		score := 1.0
		if maxCost > minCost {
			score = (maxCost - cost) / (maxCost - minCost)
		}
		scores[endpoint] = score
		logger.V(logutil.DEBUG).Info("LMetricScorer scoring",
			"endpoint", endpoint.GetMetadata().ID.String(),
			"cost", cost,
			"score", score)
	}

	return scores
}

func (s *Scorer) cost(endpoint fwksched.Endpoint) (float64, bool) {
	uncachedTokens, ok := s.uncachedPrefillTokens(endpoint)
	if !ok {
		return 0, false
	}
	decodeBatchSize := s.decodeBatchSize(endpoint)
	return uncachedTokens * decodeBatchSize, true
}

func (s *Scorer) uncachedPrefillTokens(endpoint fwksched.Endpoint) (float64, bool) {
	raw, ok := endpoint.Get(s.prefixMatchDataKey.String())
	if !ok {
		return 0, false
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil || info.TotalBlocks() <= 0 {
		return 0, false
	}
	matchBlocks := info.MatchBlocks()
	if matchBlocks < 0 {
		matchBlocks = 0
	}
	if matchBlocks > info.TotalBlocks() {
		matchBlocks = info.TotalBlocks()
	}
	blockSizeTokens := info.BlockSizeTokens()
	if blockSizeTokens <= 0 {
		blockSizeTokens = 1
	}
	return float64(info.TotalBlocks()-matchBlocks) * float64(blockSizeTokens), true
}

func (s *Scorer) decodeBatchSize(endpoint fwksched.Endpoint) float64 {
	requests := int64(0)
	if raw, ok := endpoint.Get(s.inFlightLoadDataKey.String()); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			requests = load.Requests
		}
	}
	if requests < 0 {
		requests = 0
	}
	return float64(requests + 1)
}
