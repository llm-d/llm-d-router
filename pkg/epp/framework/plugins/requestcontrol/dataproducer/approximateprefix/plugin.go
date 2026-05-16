/*
Copyright 2026 The Kubernetes Authors.

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

package approximateprefix

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	ApproxPrefixCachePluginType = "approx-prefix-cache-producer"
)

var (
	_ requestcontrol.DataProducer = &dataProducer{}
	_ requestcontrol.PreRequest   = &dataProducer{}
)

// dataProducer is a plugin that produces data consumed by approx prefix cache aware scheduling.
type dataProducer struct {
	typedName   plugin.TypedName
	config      config
	indexerInst indexerInterface
	pluginState *plugin.PluginState
	wg          sync.WaitGroup // Used for waiting on async cache updates in tests.
}

// TypedName returns the type and name of the plugin.
func (p *dataProducer) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces returns the data produced by the plugin.
func (p *dataProducer) Produces() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// newDataProducer returns a new DataProducer plugin.
func newDataProducer(ctx context.Context, config config, handle plugin.Handle) (*dataProducer, error) {
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Prefix DataProducer initialized", "config", config)

	//nolint:staticcheck // BlockSize is deprecated, but we check it here to provide a migration path for users.
	if config.BlockSize > 0 && config.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: BlockSize (%d) is deprecated; please use BlockSizeTokens instead to define the cache block size in tokens", config.BlockSize)
	}

	if !config.AutoTune && config.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: BlockSizeTokens must be > 0 when AutoTune is disabled (current value: %d)", config.BlockSizeTokens)
	}
	if config.MaxPrefixTokensToMatch < 0 {
		return nil, fmt.Errorf("invalid configuration: MaxPrefixTokensToMatch must be >= 0 (current value: %d)", config.MaxPrefixTokensToMatch)
	}
	indexer := newIndexer(ctx, config.LRUCapacityPerServer)

	p := &dataProducer{
		typedName: plugin.TypedName{
			Type: ApproxPrefixCachePluginType,
			Name: ApproxPrefixCachePluginType,
		},
		config:      config,
		indexerInst: indexer,
		pluginState: plugin.NewPluginState(ctx),
	}

	if handle != nil {
		go p.CleanUpInactivePods(ctx, handle)
	}

	return p, nil
}

// CleanUpInactivePods starts a goroutine that periodically removes inactive pods from the indexer.
func (p *dataProducer) CleanUpInactivePods(ctx context.Context, handle plugin.Handle) {
	ticker := time.NewTicker(podActiveCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			podNames := handle.PodList()
			activePods := make(map[ServerID]struct{}, len(podNames))
			for _, nsn := range podNames {
				activePods[ServerID(nsn)] = struct{}{}
			}

			for _, pod := range p.indexerInst.Pods() {
				if _, ok := activePods[pod]; !ok {
					p.indexerInst.RemovePod(pod)
					log.FromContext(ctx).V(logutil.VERBOSE).Info("Removed pod not in active set", "pod", pod)
				}
			}
		}
	}
}

// indexer returns the shared indexer.
func (p *dataProducer) indexer() indexerInterface {
	return p.indexerInst
}

// PluginState returns the shared plugin state.
func (p *dataProducer) PluginState() *plugin.PluginState {
	return p.pluginState
}

// Produce is called by the director before scheduling requests.
func (p *dataProducer) Produce(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) error {
	blockSize := p.GetBlockSize(pods)
	maxBlocks := p.config.MaxPrefixBlocksToMatch
	if p.config.MaxPrefixTokensToMatch > 0 && blockSize > 0 {
		maxBlocks = p.config.MaxPrefixTokensToMatch / blockSize
	}
	hashes := hashPrompt(ctx, request, blockSize, maxBlocks)
	total := len(hashes)
	prefixCacheServers := p.matchLongestPrefix(ctx, hashes)

	for _, pod := range pods {
		matchLen := prefixCacheServers[ServerID(pod.GetMetadata().NamespacedName)]
		pod.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(matchLen, total, blockSize))
	}

	state := &SchedulingContextState{
		PrefixHashes:       hashes,
		PrefixCacheServers: prefixCacheServers,
	}

	// Store the state in shared plugin state for later use in PreRequest.
	// NOTE: We use the prefix plugin's type name as part of the key so that the scorer can read it.
	p.pluginState.Write(request.RequestID, plugin.StateKey(ApproxPrefixCachePluginType), state)

	return nil
}

// PreRequest records in the shared indexer the result of the scheduling selection.
// It updates the indexer with the prefix hashes for the selected endpoint(s).
func (p *dataProducer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) {
	// Delete the state to avoid memory leak.
	defer p.pluginState.Delete(request.RequestID)
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if len(primaryProfileResult.TargetEndpoints) == 0 {
		return
	}

	targetEndpoint := primaryProfileResult.TargetEndpoints[0]
	servers := []server{p.makeserver(targetEndpoint)}

	// Also record for prefill node if present in P/D disaggregated mode.
	if pr, exists := schedulingResult.ProfileResults[experimentalDefaultPrefillProfile]; exists && len(pr.TargetEndpoints) > 0 {
		servers = append(servers, p.makeserver(pr.TargetEndpoints[0]))
	}

	// Read state saved during Produce.
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin state", "requestID", request.RequestID)
		return
	}

	// Update indexer asynchronously to avoid blocking the request path.
	p.wg.Go(func() {
		for _, s := range servers {
			p.indexerInst.Add(state.PrefixHashes, s)
		}
	})

	// Record metrics.
	total := len(state.PrefixHashes)
	matchLen := state.PrefixCacheServers[ServerID(targetEndpoint.GetMetadata().NamespacedName)]
	blockSize := p.GetBlockSize(primaryProfileResult.TargetEndpoints)
	avgChars := averageCharactersPerToken
	metrics.RecordPrefixCacheMatch(matchLen*blockSize*avgChars, total*blockSize*avgChars)
}

func (p *dataProducer) makeserver(targetEndpoint fwksched.Endpoint) server {
	gpuBlocks := defaultLRUCapacityPerServer
	if p.config.AutoTune && targetEndpoint.GetMetrics().CacheNumBlocks > 0 {
		gpuBlocks = targetEndpoint.GetMetrics().CacheNumBlocks
	}
	return server{
		ServerID:       ServerID(targetEndpoint.GetMetadata().NamespacedName),
		NumOfGPUBlocks: gpuBlocks,
	}
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches, prefix length is defined in blocks.
func (p *dataProducer) matchLongestPrefix(ctx context.Context, hashes []blockHash) map[ServerID]int {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	res := make(map[ServerID]int)

	// Use a greedy strategy to search from the longest prefix.
	for _, hash := range hashes {
		cachedServers := p.indexerInst.Get(hash)
		if len(cachedServers) == 0 {
			break
		}
		loggerTrace.Info("Found cached servers", "cachedServers", cachedServers, "total # blocks", len(hashes))
		for server := range cachedServers {
			res[server]++
		}
	}
	return res
}

// GetBlockSize returns the block size in tokens, potentially auto-tuned from endpoint metrics.
func (p *dataProducer) GetBlockSize(endpoints []fwksched.Endpoint) int {
	if !p.config.AutoTune || len(endpoints) == 0 {
		return p.config.BlockSizeTokens
	}

	if endpoint := endpoints[0]; endpoint.GetMetrics() != nil {
		cacheBlockSize := endpoint.GetMetrics().CacheBlockSize
		if cacheBlockSize > 0 {
			return cacheBlockSize
		}
	}
	return p.config.BlockSizeTokens
}

// ApproxPrefixCacheFactory is the factory function for the prefix cache data producer plugin.
func ApproxPrefixCacheFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := defaultConfig
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal prefix cache parameters: %w", err)
		}
	}

	// pluginState will be initialized by newDataProducer as we pass nil here.
	p, err := newDataProducer(handle.Context(), parameters, handle)
	if err != nil {
		return nil, err
	}

	return p, nil
}
