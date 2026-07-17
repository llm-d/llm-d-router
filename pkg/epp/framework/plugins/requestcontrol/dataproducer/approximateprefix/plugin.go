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
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	approxprefixconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/constants"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

const (
	ApproxPrefixCachePluginType = approxprefixconstants.ApproxPrefixCachePluginType
)

// minBlockSizeTokens is the floor applied to the block size returned by
// GetBlockSize. The routing-side indexer keeps one LRU entry per (pod, block),
// so very small block sizes cause gigabyte-scale memory growth at scale. 64
// tokens is coarse enough to bound memory while still preserving useful
// prefix-match signal. Defined as a var so tests in this package can lower it
// to exercise prefix-match logic at small block sizes without leaking a knob
// into the public configuration surface. See
// https://github.com/llm-d/llm-d-router/issues/1158.
var minBlockSizeTokens = 64

var (
	_ requestcontrol.DataProducer = &dataProducer{}
	_ requestcontrol.PreRequest   = &dataProducer{}
	_ plugin.StateDumper          = &dataProducer{}
)

// dataProducer is a plugin that produces data consumed by approx prefix cache aware scheduling.
type dataProducer struct {
	typedName   plugin.TypedName
	config      config
	indexerInst indexerInterface
	pluginState *plugin.PluginState
	wg          sync.WaitGroup // Used for waiting on async cache updates in tests.
	dk          plugin.DataKey
}

// TypedName returns the type and name of the plugin.
func (p *dataProducer) TypedName() plugin.TypedName {
	return p.typedName
}

const maxDebugDumpPods = 100

// prefixIndexState is the sanitized snapshot returned by DumpState. It carries
// per-pod block counts only; the prompt-derived block hashes are never exposed.
// The dump is partial when TotalPods exceeds MaxPods.
type prefixIndexState struct {
	Pods      []podBlockCount `json:"pods"`
	TotalPods int             `json:"totalPods"`
	MaxPods   int             `json:"maxPods"`
}

type podBlockCount struct {
	Pod    string `json:"pod"`
	Blocks int    `json:"blocks"`
}

// DumpState reports how many prefix-cache blocks the indexer currently tracks
// per pod, ordered by block count and capped to maxDebugDumpPods so the debug
// payload stays bounded when a pool has many pods.
func (p *dataProducer) DumpState() (json.RawMessage, error) {
	return json.Marshal(p.snapshotState())
}

func (p *dataProducer) snapshotState() prefixIndexState {
	state := prefixIndexState{MaxPods: maxDebugDumpPods}
	if p.indexerInst == nil {
		return state
	}

	counts := p.indexerInst.PodBlockCounts()
	state.TotalPods = len(counts)
	state.Pods = make([]podBlockCount, 0, len(counts))
	for pod, blocks := range counts {
		state.Pods = append(state.Pods, podBlockCount{Pod: pod.String(), Blocks: blocks})
	}

	sort.SliceStable(state.Pods, func(a, b int) bool {
		if state.Pods[a].Blocks != state.Pods[b].Blocks {
			return state.Pods[a].Blocks > state.Pods[b].Blocks
		}
		return state.Pods[a].Pod < state.Pods[b].Pod
	})

	if len(state.Pods) > maxDebugDumpPods {
		state.Pods = state.Pods[:maxDebugDumpPods]
	}
	return state
}

// Produces returns the data produced by the plugin.
func (p *dataProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}}
}

// Consumes declares the TokenizedPrompt dependency so the data-layer DAG orders
// the token-producer before this producer runs and auto-creates one when none
// is configured.
func (p *dataProducer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: fwksched.TokenizedPrompt{}},
	}
}

// newDataProducer returns a new DataProducer plugin.
func newDataProducer(ctx context.Context, name string, config config, handle plugin.Handle) (*dataProducer, error) {
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Prefix DataProducer initialized", "config", config)

	// Note: 'blockSize' deprecation handling lives in ApproxPrefixCacheFactory so it
	// applies to JSON-decoded configs uniformly with the strict-parsing policy (#1068).
	// Direct callers of newDataProducer are expected to populate BlockSizeTokens.

	if !config.AutoTune && config.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: BlockSizeTokens must be > 0 when AutoTune is disabled (current value: %d)", config.BlockSizeTokens)
	}
	if config.MaxPrefixTokensToMatch < 0 {
		return nil, fmt.Errorf("invalid configuration: MaxPrefixTokensToMatch must be >= 0 (current value: %d)", config.MaxPrefixTokensToMatch)
	}
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if err := RegisterMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	// Surface the override to the operator so a too-small configured value is
	// not silently swallowed. The clamp itself happens at request time in
	// GetBlockSize and applies uniformly across endpoint metric, autotune
	// fallback, and manual configuration.
	if config.BlockSizeTokens > 0 && config.BlockSizeTokens < minBlockSizeTokens {
		log.FromContext(ctx).Info(
			"WARNING: configured blockSizeTokens is below the recommended minimum, overriding it.",
			"blockSizeTokens", config.BlockSizeTokens,
			"minimum", minBlockSizeTokens,
			"issue", "https://github.com/llm-d/llm-d-router/issues/1158",
		)
	}

	indexer := newIndexer(ctx, config.LRUCapacityPerServer, name, ApproxPrefixCachePluginType)

	p := &dataProducer{
		typedName: plugin.TypedName{
			Type: ApproxPrefixCachePluginType,
			Name: name,
		},
		config:      config,
		indexerInst: indexer,
		pluginState: plugin.NewPluginState(ctx),
		dk:          attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
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
	perPromptHashes := prefixhash.GetBlockHashes(ctx, request, blockSize, p.resolveMaxBlocks(blockSize))

	prefixCacheServers := make(map[ServerID]int)
	totalBlocks := 0
	for _, hashes := range perPromptHashes {
		for server, matchLen := range p.matchLongestPrefix(ctx, hashes) {
			prefixCacheServers[server] += matchLen
		}
		totalBlocks += len(hashes)
	}

	endpointHitRatios := make([]float64, 0, len(pods))
	for _, pod := range pods {
		matchLen := prefixCacheServers[ServerID(pod.GetMetadata().NamespacedName)]
		pod.Put(p.dk.String(), attrprefix.NewPrefixCacheMatchInfo(matchLen, totalBlocks, blockSize))
		if totalBlocks > 0 {
			endpointHitRatios = append(endpointHitRatios, float64(matchLen)/float64(totalBlocks))
		}
	}

	max, avg, std := calculateHitRatioStats(endpointHitRatios)
	recordPrefixCacheHitRatioStats(p.typedName.Name, p.typedName.Type, max, avg, std)

	state := &SchedulingContextState{
		PerPromptHashes:    perPromptHashes,
		PrefixCacheServers: prefixCacheServers,
	}

	p.pluginState.Write(request.RequestID, plugin.StateKey(p.typedName.Name), state)

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
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(p.typedName.Name))
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin state", "requestID", request.RequestID)
		return
	}

	// Update indexer asynchronously to avoid blocking the request path.
	p.wg.Go(func() {
		for _, s := range servers {
			for _, hashes := range state.PerPromptHashes {
				p.indexerInst.Add(hashes, s)
			}
		}
	})

	// Record metrics. Lengths are reported as a byte estimate (~averageCharactersPerToken bytes/token).
	total := 0
	for _, hashes := range state.PerPromptHashes {
		total += len(hashes)
	}
	matchLen := state.PrefixCacheServers[ServerID(targetEndpoint.GetMetadata().NamespacedName)]
	blockSize := p.GetBlockSize(primaryProfileResult.TargetEndpoints)
	const averageCharactersPerToken = 4
	recordPrefixCacheMatch(p.typedName.Name, p.typedName.Type, matchLen*blockSize*averageCharactersPerToken, total*blockSize*averageCharactersPerToken)
}

func (p *dataProducer) makeserver(targetEndpoint fwksched.Endpoint) server {
	gpuBlocks := defaultLRUCapacityPerServer
	if p.config.AutoTune && targetEndpoint.GetMetrics() != nil && targetEndpoint.GetMetrics().CacheNumBlocks > 0 {
		gpuBlocks = targetEndpoint.GetMetrics().CacheNumBlocks
	} else if p.config.LRUCapacityPerServer > 0 {
		gpuBlocks = p.config.LRUCapacityPerServer
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
//
// Values below minBlockSizeTokens are overridden with minBlockSizeTokens
// regardless of where they originate (endpoint metric, autotune fallback, or
// manual configuration). The routing-side indexer holds one LRU entry per
// (pod, block), so a small block size (e.g., vLLM's default of 16) would
// inflate indexer memory by ~64x. Routing intentionally measures matches at
// coarser granularity than the model server's true block size; a startup
// warning is logged in newDataProducer when a configured value triggers the
// override. See #1158.
func (p *dataProducer) GetBlockSize(endpoints []fwksched.Endpoint) int {
	blockSize := p.config.BlockSizeTokens
	if p.config.AutoTune && len(endpoints) > 0 {
		if endpoint := endpoints[0]; endpoint.GetMetrics() != nil {
			if metric := endpoint.GetMetrics().CacheBlockSize; metric > 0 {
				blockSize = metric
			}
		}
	}
	if blockSize < minBlockSizeTokens {
		return minBlockSizeTokens
	}
	return blockSize
}

// resolveMaxBlocks returns the per-request cap on hashed prefix blocks.
//
// MaxPrefixTokensToMatch wins when set, converting tokens to blocks at the
// effective block size. Zeroing both caps means unlimited: prompt length is
// already bounded by the model server's context window, so the whole prompt is
// at most one context window of tokens. A token cap smaller than the block size
// still resolves to 0 and hashes nothing; that is a misconfiguration, not a
// request for unlimited matching.
func (p *dataProducer) resolveMaxBlocks(blockSize int) int {
	if p.config.MaxPrefixTokensToMatch > 0 && blockSize > 0 {
		return p.config.MaxPrefixTokensToMatch / blockSize
	}
	if p.config.MaxPrefixBlocksToMatch == 0 {
		return unlimitedPrefixBlocks
	}
	return p.config.MaxPrefixBlocksToMatch
}

// ApproxPrefixCacheFactory is the factory function for the prefix cache data producer plugin.
func ApproxPrefixCacheFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := defaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal prefix cache parameters: %w", err)
		}
	}

	// Deprecated 'blockSize' is accepted with a warning and mapped to
	// 'blockSizeTokens'. Removed (truly unknown) fields are rejected by the
	// strict decoder above. See #1068.
	if parameters.BlockSize > 0 {
		log.FromContext(handle.Context()).V(logutil.DEFAULT).Info(
			"'blockSize' is deprecated; use 'blockSizeTokens' instead",
			"blockSize", parameters.BlockSize,
		)
		if parameters.BlockSizeTokens == defaultBlockSizeTokens {
			// BlockSizeTokens left at its default — map the deprecated value into it.
			parameters.BlockSizeTokens = parameters.BlockSize
		}
		parameters.BlockSize = 0
	}

	// pluginState will be initialized by newDataProducer as we pass nil here.
	p, err := newDataProducer(handle.Context(), name, parameters, handle)
	if err != nil {
		return nil, err
	}

	return p, nil
}
