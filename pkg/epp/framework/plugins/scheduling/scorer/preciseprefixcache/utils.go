package preciseprefixcache

import (
	"context"
	"time"

	"github.com/jellydator/ttlcache/v3"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type endpointToKeyFunc func(endpoint scheduling.Endpoint) (string, bool)

// absoluteScoredPods returns score/totalBlocks per endpoint, clipped to [0, 1].
// totalBlocks <= 0 → all zeros.
func absoluteScoredPods(endpoints []scheduling.Endpoint, endpointToKey endpointToKeyFunc,
	scores map[string]float64, totalBlocks int) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64, len(endpoints))
	if totalBlocks <= 0 {
		for _, endpoint := range endpoints {
			scoredEndpoints[endpoint] = 0.0
		}
		return scoredEndpoints
	}

	denom := float64(totalBlocks)
	for _, endpoint := range endpoints {
		key, ok := endpointToKey(endpoint)
		if !ok {
			continue
		}
		raw, ok := scores[key]
		if !ok || raw <= 0 {
			scoredEndpoints[endpoint] = 0.0
			continue
		}
		ratio := raw / denom
		if ratio > 1.0 {
			ratio = 1.0
		}
		scoredEndpoints[endpoint] = ratio
	}
	return scoredEndpoints
}

func cleanCachePeriodically[K comparable, V any](ctx context.Context, cache *ttlcache.Cache[K, V], requestTimeout time.Duration) {
	ticker := time.NewTicker(requestTimeout)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cache.DeleteExpired()
		}
	}
}
