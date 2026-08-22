package lmetric

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestLMetricScorerPrefersLowerDecodeBatchSizeForSamePrefixMiss(t *testing.T) {
	scorer := newTestScorer()
	lowBatch := makeEndpoint("low-batch", 80, 100, 16, 1, 1000, scorer)
	highBatch := makeEndpoint("high-batch", 80, 100, 16, 8, 1000, scorer)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{lowBatch, highBatch})

	assert.Equal(t, 1.0, scores[lowBatch])
	assert.Equal(t, 0.0, scores[highBatch])
}

func TestLMetricScorerBalancesPrefixMissAgainstDecodeBatchSize(t *testing.T) {
	scorer := newTestScorer()
	betterPrefix := makeEndpoint("better-prefix", 90, 100, 16, 4, 10000, scorer)
	lowerBatch := makeEndpoint("lower-batch", 40, 100, 16, 1, 0, scorer)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{betterPrefix, lowerBatch})

	assert.Equal(t, 1.0, scores[betterPrefix])
	assert.Equal(t, 0.0, scores[lowerBatch])
}

func TestLMetricScorerDecodeBatchSizeCanOutweighPrefixHit(t *testing.T) {
	scorer := newTestScorer()
	betterPrefixButBusy := makeEndpoint("better-prefix-busy", 90, 100, 16, 20, 0, scorer)
	worsePrefixButIdle := makeEndpoint("worse-prefix-idle", 40, 100, 16, 0, 0, scorer)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{betterPrefixButBusy, worsePrefixButIdle})

	assert.Equal(t, 0.0, scores[betterPrefixButBusy])
	assert.Equal(t, 1.0, scores[worsePrefixButIdle])
}

func TestLMetricScorerUsesDecodeBatchSizeNotInFlightTokens(t *testing.T) {
	scorer := newTestScorer()
	shortInFlightTokens := makeEndpoint("short-in-flight", 80, 100, 16, 3, 100, scorer)
	longInFlightTokens := makeEndpoint("long-in-flight", 80, 100, 16, 3, 10000, scorer)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{shortInFlightTokens, longInFlightTokens})

	assert.Equal(t, 1.0, scores[shortInFlightTokens])
	assert.Equal(t, 1.0, scores[longInFlightTokens])
}

func TestLMetricScorerIsInvariantToCommonISLScale(t *testing.T) {
	scorer := newTestScorer()
	shortA := makeEndpoint("short-a", 75, 100, 16, 1, 100, scorer)
	shortB := makeEndpoint("short-b", 50, 100, 16, 2, 200, scorer)
	longA := makeEndpoint("long-a", 750, 1000, 16, 1, 1000, scorer)
	longB := makeEndpoint("long-b", 500, 1000, 16, 2, 2000, scorer)

	shortScores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{shortA, shortB})
	longScores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{longA, longB})

	assert.Equal(t, shortScores[shortA], longScores[longA])
	assert.Equal(t, shortScores[shortB], longScores[longB])
}

func TestLMetricScorerInvalidPrefixDataScoresZero(t *testing.T) {
	scorer := newTestScorer()
	validPrefix := makeEndpoint("valid-prefix", 80, 100, 16, 1, 100, scorer)
	missingPrefix := makeEndpoint("missing-prefix", -1, 100, 16, 1, 100, scorer)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{validPrefix, missingPrefix})

	assert.Equal(t, 1.0, scores[validPrefix])
	assert.Equal(t, 0.0, scores[missingPrefix])
}

func TestFactoryValidConfig(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"prefixMatchInfoProducerName":"precise","inFlightLoadProducerName":"load"}`)), nil)

	assert.NoError(t, err)
	assert.Equal(t, LMetricScorerType, plugin.TypedName().Type)
	assert.Equal(t, "test", plugin.TypedName().Name)
}

func newTestScorer() *Scorer {
	return &Scorer{
		typedName:           fwkplugin.TypedName{Type: LMetricScorerType, Name: "test"},
		prefixMatchDataKey:  attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(""),
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(""),
	}
}

func makeEndpoint(name string, matchBlocks, totalBlocks int, blockSizeTokens int, inFlightRequests, inFlightTokens int64, scorer *Scorer) fwksched.Endpoint {
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: name}}, fwkdl.NewMetrics(), nil)
	if matchBlocks >= 0 {
		endpoint.Put(scorer.prefixMatchDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens))
	}
	endpoint.Put(scorer.inFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Requests: inFlightRequests, Tokens: inFlightTokens})
	return endpoint
}
