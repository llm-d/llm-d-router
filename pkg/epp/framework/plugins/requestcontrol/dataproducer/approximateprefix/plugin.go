/*
Copyright 2026 The Kubernetes Authors.
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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/observability/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	approxprefixconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/constants"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
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
	_ requestcontrol.DataProducer       = &dataProducer{}
	_ requestcontrol.PreRequest         = &dataProducer{}
	_ plugin.StateDumper                = &dataProducer{}
	_ datalayer.EndpointExtractor       = (*dataProducer)(nil)
	_ datalayer.Registrant              = &dataProducer{}
	_ datalayer.CrossReplicaContributor = (*dataProducer)(nil)
)

// dataProducer is a plugin that produces data consumed by approx prefix cache aware scheduling.
type dataProducer struct {
	typedName             plugin.TypedName
	config                config
	handle                plugin.Handle
	indexerInst           indexerInterface
	pluginState           *plugin.PluginState
	wg                    sync.WaitGroup // Used for waiting on async cache updates in tests.
	dk                    plugin.DataKey
	syncHashDK            plugin.DataKey
	syncCrossReplicaState bool
	recent                *recentWindow
	// registeredEndpoints maps endpointID to the Endpoint pointer last seen on
	// AddOrUpdate so Extract can ignore stale deletes for a replaced object.
	registeredEndpoints sync.Map
	// peerMerged tracks peer prompt keys already folded into the indexer per
	// endpoint so a stable peer window does not re-Add (and LRU-touch) on every
	// request. Each merge replaces the set with the active peer-only window so
	// the map stays bounded and prompts that leave and return can be re-indexed.
	// Values are *mergedKeySet.
	peerMerged sync.Map
}

// mergedKeySet is the per-endpoint set stored in peerMerged. keys holds only
// prompt keys present in the latest peer sync window for that endpoint.
// digest is the xxhash of the last processed PrefixHashSet.Prompts so Produce
// can skip snapshot/key work when the sync attribute is unchanged.
type mergedKeySet struct {
	mu         sync.Mutex
	keys       map[string]struct{}
	digest     uint64
	haveDigest bool
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
	out := map[plugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}}
	if p.syncCrossReplicaState {
		out[p.syncHashDK] = attrprefix.PrefixHashSet{}
	}
	return out
}

// Consumes declares the TokenizedRequest dependency so the data-layer DAG orders
// the token-producer before this producer runs and auto-creates one when none
// is configured. PrefixHashSet is optional: when a CrossReplicaSyncer is wired,
// Produce reads the syncer-backed attribute to merge peer recent windows.
func (p *dataProducer) Consumes() plugin.DataDependencies {
	deps := plugin.DataDependencies{
		Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: fwksched.TokenizedRequest{}},
	}
	if p.syncCrossReplicaState {
		deps.Optional = map[plugin.DataKey]any{p.syncHashDK: attrprefix.PrefixHashSet{}}
	}
	return deps
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
	if config.MaxPrefixBlocksToMatch < 0 {
		return nil, fmt.Errorf("invalid configuration: MaxPrefixBlocksToMatch must be >= 0 (current value: %d)", config.MaxPrefixBlocksToMatch)
	}
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	prefixmetrics.Register()
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

	syncCrossReplicaState := true
	if config.SyncCrossReplicaState != nil {
		syncCrossReplicaState = *config.SyncCrossReplicaState
	}
	maxSynced := syncedHashBudget(config)

	p := &dataProducer{
		typedName: plugin.TypedName{
			Type: ApproxPrefixCachePluginType,
			Name: name,
		},
		config:                config,
		handle:                handle,
		indexerInst:           indexer,
		pluginState:           plugin.NewPluginState(ctx),
		dk:                    attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		syncHashDK:            attrprefix.PrefixHashSetDataKey.WithNonEmptyProducerName(name),
		syncCrossReplicaState: syncCrossReplicaState,
		recent:                newRecentWindow(maxSynced),
	}

	go p.CleanUpInactivePods(ctx, handle)

	return p, nil
}

// syncedHashBudget converts MaxSyncedTokens into a hash-count cap for the
// recent window. Unset MaxSyncedTokens uses defaultMaxSyncedTokens.
func syncedHashBudget(cfg config) int {
	blockSize := cfg.BlockSizeTokens
	if blockSize <= 0 {
		blockSize = defaultBlockSizeTokens
	}
	tokens := cfg.MaxSyncedTokens
	if tokens <= 0 {
		tokens = defaultMaxSyncedTokens
	}
	n := tokens / blockSize
	if n <= 0 {
		return 1
	}
	return n
}

// syncerActive is true when this producer should publish/merge cross-replica
// state: sync is enabled and a CrossReplicaSyncer is wired on the handle.
func (p *dataProducer) syncerActive() bool {
	return p.syncCrossReplicaState && p.handle != nil && p.handle.CrossReplicaSyncer() != nil
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
					id := pod.String()
					if p.recent != nil {
						p.recent.forget(id)
					}
					p.forgetPeerMerged(id)
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
	p.mergeRemoteHashes(pods)

	blockSize := p.GetBlockSize(pods)
	perPromptHashes, perPromptTokens := prefixhash.GetBlockHashesWithPromptTokens(ctx, request, blockSize, p.resolveMaxBlocks(blockSize))

	candidates := make([]ServerID, len(pods))
	for i, pod := range pods {
		candidates[i] = ServerID(pod.GetMetadata().ID)
	}
	prefixCacheServers := make(map[ServerID]int)
	predictedCachedTokens := make(map[ServerID]int)
	totalBlocks := 0
	for i, hashes := range perPromptHashes {
		for j, matchLen := range p.indexerInst.MatchLongestPrefix(hashes, candidates) {
			if matchLen == 0 {
				continue
			}
			prefixCacheServers[candidates[j]] += matchLen
			predictedCachedTokens[candidates[j]] += min(matchLen*blockSize, perPromptTokens[i])
		}
		totalBlocks += len(hashes)
	}
	if v := log.FromContext(ctx).V(logutil.TRACE); v.Enabled() {
		matched := make(map[string]int, len(prefixCacheServers))
		for server, blocks := range prefixCacheServers {
			matched[server.String()] = blocks
		}
		v.Info("Prefix cache match", "matchedBlocks", matched, "totalBlocks", totalBlocks)
	}

	bestAvailable := 0
	for _, pod := range pods {
		id := ServerID(pod.GetMetadata().ID)
		pod.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(prefixCacheServers[id], totalBlocks, blockSize))
		bestAvailable = max(bestAvailable, predictedCachedTokens[id])
	}

	state := &SchedulingContextState{
		PerPromptHashes:           perPromptHashes,
		PrefixCacheServers:        prefixCacheServers,
		PredictedCachedTokens:     predictedCachedTokens,
		BestAvailableCachedTokens: bestAvailable,
	}

	p.pluginState.Write(request.RequestID, plugin.StateKey(p.typedName.Name), state)

	return nil
}

// mergeRemoteHashes folds peer recent-window prompts into the local indexer.
// Runs only when a CrossReplicaSyncer is configured. The syncer-backed
// attribute is Aggregate(local, peers); local prompts and prompts already
// merged for this endpoint are skipped so Produce does not re-warm or
// LRU-touch the indexer on every request. Each new peer prompt is Added on its
// own so indexer LRU eviction keeps prompt-prefix semantics. Peer prompts are
// not recorded into the recent window (no echo). When the attribute digest
// matches the last processed value, the pod is skipped without snapshot or
// prompt-key work.
func (p *dataProducer) mergeRemoteHashes(pods []fwksched.Endpoint) {
	if !p.syncerActive() || p.indexerInst == nil {
		return
	}
	for _, pod := range pods {
		set, ok := peekPrefixHashSet(pod, p.syncHashDK)
		if !ok || set == nil {
			continue
		}
		endpointID := pod.GetMetadata().ID.String()
		digest := set.ContentDigest()
		if p.peerMergedDigestMatches(endpointID, digest) {
			continue
		}
		if len(set.Prompts) == 0 {
			p.reconcilePeerMerged(endpointID, map[string]struct{}{})
			p.rememberPeerMergedDigest(endpointID, digest)
			continue
		}
		local := p.recent.snapshot(endpointID)
		localSet := make(map[string]struct{}, len(local))
		for _, prompt := range local {
			localSet[promptKey(prompt)] = struct{}{}
		}
		active := make(map[string]struct{}, len(set.Prompts))
		for _, prompt := range set.Prompts {
			if len(prompt) == 0 {
				continue
			}
			key := promptKeyUint64(prompt)
			if _, ok := localSet[key]; ok {
				continue
			}
			active[key] = struct{}{}
		}
		if p.peerMergedCovers(endpointID, active) {
			p.rememberPeerMergedDigest(endpointID, digest)
			continue
		}
		novel := p.reconcilePeerMerged(endpointID, active)
		if len(novel) == 0 {
			p.rememberPeerMergedDigest(endpointID, digest)
			continue
		}
		server := p.makeserver(pod)
		for _, prompt := range set.Prompts {
			if len(prompt) == 0 {
				continue
			}
			key := promptKeyUint64(prompt)
			if _, ok := novel[key]; !ok {
				continue
			}
			p.indexerInst.Add(uint64sToBlockHashes(prompt), server)
		}
		p.rememberPeerMergedDigest(endpointID, digest)
	}
}

// peekPrefixHashSet returns the sync attribute without cloning when the
// endpoint supports Peek. The result must not be mutated.
func peekPrefixHashSet(pod fwksched.Endpoint, key plugin.DataKey) (*attrprefix.PrefixHashSet, bool) {
	type peeker interface {
		Peek(plugin.DataKey) (datalayer.Cloneable, bool)
	}
	var val datalayer.Cloneable
	var ok bool
	if p, okp := pod.(peeker); okp {
		val, ok = p.Peek(key)
	} else {
		val, ok = pod.Get(key)
	}
	if !ok {
		return nil, false
	}
	set, ok := val.(*attrprefix.PrefixHashSet)
	return set, ok
}

func (p *dataProducer) peerMergedDigestMatches(endpointID string, digest uint64) bool {
	v, ok := p.peerMerged.Load(endpointID)
	if !ok {
		return false
	}
	s := v.(*mergedKeySet)
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.haveDigest && s.digest == digest
}

func (p *dataProducer) rememberPeerMergedDigest(endpointID string, digest uint64) {
	v, _ := p.peerMerged.LoadOrStore(endpointID, &mergedKeySet{keys: make(map[string]struct{})})
	s := v.(*mergedKeySet)
	s.mu.Lock()
	s.digest = digest
	s.haveDigest = true
	s.mu.Unlock()
}

// peerMergedCovers reports whether endpointID's remembered set already matches
// active, so Produce can skip reconcile and Add work on a stable peer window.
func (p *dataProducer) peerMergedCovers(endpointID string, active map[string]struct{}) bool {
	v, ok := p.peerMerged.Load(endpointID)
	if !ok {
		return false
	}
	s := v.(*mergedKeySet)
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.keys) != len(active) {
		return false
	}
	for k := range active {
		if _, ok := s.keys[k]; !ok {
			return false
		}
	}
	return true
}

// reconcilePeerMerged replaces the remembered peer keys for endpointID with
// active and returns keys that were not previously remembered (need Add).
func (p *dataProducer) reconcilePeerMerged(endpointID string, active map[string]struct{}) map[string]struct{} {
	v, _ := p.peerMerged.LoadOrStore(endpointID, &mergedKeySet{keys: make(map[string]struct{})})
	s := v.(*mergedKeySet)
	s.mu.Lock()
	defer s.mu.Unlock()
	novel := make(map[string]struct{})
	next := make(map[string]struct{}, len(active))
	for k := range active {
		next[k] = struct{}{}
		if _, ok := s.keys[k]; !ok {
			novel[k] = struct{}{}
		}
	}
	s.keys = next
	return novel
}

func (p *dataProducer) forgetPeerMerged(endpointID string) {
	p.peerMerged.Delete(endpointID)
}

// PreRequest records in the shared indexer the result of the scheduling selection.
// It updates the indexer with the prefix hashes for the selected endpoint(s).
func (p *dataProducer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	// Delete the state to avoid memory leak.
	defer p.pluginState.Delete(request.RequestID)
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if len(primaryProfileResult.TargetEndpoints) == 0 {
		return nil
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
		return nil
	}

	// Update indexer asynchronously to avoid blocking the request path.
	p.wg.Go(func() {
		for _, s := range servers {
			for _, hashes := range state.PerPromptHashes {
				p.indexerInst.Add(hashes, s)
				if p.syncerActive() {
					p.recent.record(s.String(), hashes)
				}
			}
		}
	})

	// Record metrics. Lengths are reported as a byte estimate (fwkrh.BytesPerToken bytes per token).
	total := 0
	for _, hashes := range state.PerPromptHashes {
		total += len(hashes)
	}
	matchLen := state.PrefixCacheServers[ServerID(targetEndpoint.GetMetadata().ID)]
	blockSize := p.GetBlockSize(primaryProfileResult.TargetEndpoints)
	recordPrefixCacheMatch(p.typedName.Name, p.typedName.Type, matchLen*blockSize*fwkrh.BytesPerToken, total*blockSize*fwkrh.BytesPerToken)
	if request.Body != nil {
		predictionProfile, role := prefixmetrics.PredictionTarget(schedulingResult, experimentalDefaultPrefillProfile)
		selected := state.PredictedCachedTokens[ServerID(predictionProfile.TargetEndpoints[0].GetMetadata().ID)]
		// A profile that reports no scored candidates leaves only the chosen
		// endpoint to go on, so selected stands in for both maxima. That keeps
		// the histograms on the same requests, at the cost of reading as a
		// perfect routing decision.
		bestPredicted, bestAvailable := selected, selected
		if scored := predictionProfile.ScoredCandidates; len(scored) > 0 {
			bestPredicted = bestAmongScored(scored, state.PredictedCachedTokens, selected)
			bestAvailable = state.BestAvailableCachedTokens
		}
		modality, _ := mmobs.Summary(request)
		prefixmetrics.RecordPrediction(p.typedName.Name, p.typedName.Type, role, modality, prefixmetrics.Prediction{
			Selected:      selected,
			BestPredicted: bestPredicted,
			BestAvailable: bestAvailable,
			PromptTokens:  request.Body.TokenizedRequest.TokenCount(),
		})
	}
	return nil
}

// bestAmongScored returns the highest prediction the picker could have chosen.
// It walks the scored candidates rather than the prediction map, because the
// map covers every candidate endpoint, including those the scheduler's filters
// removed before the picker ran.
func bestAmongScored(scored []fwksched.ScoredEndpoint, predicted map[ServerID]int, selected int) int {
	best := selected
	for _, candidate := range scored {
		if md := candidate.GetMetadata(); md != nil {
			best = max(best, predicted[ServerID(md.ID)])
		}
	}
	return best
}

// RegisterDependencies attaches this producer to endpoint lifecycle events so
// the cross-replica publisher can discover it as a contributor.
func (p *dataProducer) RegisterDependencies(r datalayer.Registrar) error {
	return r.Register(datalayer.PendingRegistration{
		Owner:         p.TypedName(),
		SourceType:    sourcenotifications.EndpointNotificationSourceType,
		Extractor:     p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// CrossReplicaState publishes recent prefix hashes for peer EPPs to merge.
func (p *dataProducer) CrossReplicaState() datalayer.CrossReplicaSpec {
	return datalayer.CrossReplicaSpec{
		StateKey:     datalayer.StateKey("approx-prefix:" + p.typedName.Name),
		AttributeKey: p.syncHashDK,
		SyncDisabled: !p.syncCrossReplicaState,
		Read: func(endpointID string) datalayer.Cloneable {
			return &attrprefix.PrefixHashSet{Prompts: promptsToUint64(p.recent.snapshot(endpointID))}
		},
		Aggregate: func(values []any) any {
			// Each input is already capped by the publishing replica's
			// maxSyncedTokens. Do not re-trim the union: redis-syncer orders
			// peer fields by replica ID, so a global cap would drop whole
			// replicas rather than oldest prompts.
			seen := make(map[string]struct{})
			var out [][]blockHash
			for _, v := range values {
				set, ok := v.(*attrprefix.PrefixHashSet)
				if !ok || set == nil {
					continue
				}
				for _, prompt := range set.Prompts {
					if len(prompt) == 0 {
						continue
					}
					hashes := uint64sToBlockHashes(prompt)
					key := promptKey(hashes)
					if _, ok := seen[key]; ok {
						continue
					}
					seen[key] = struct{}{}
					out = append(out, hashes)
				}
			}
			return &attrprefix.PrefixHashSet{Prompts: promptsToUint64(out)}
		},
	}
}

// Extract installs a local placeholder for the sync attribute. The cross-replica
// publisher overwrites it with a syncer-backed DynamicAttribute on add/update.
func (p *dataProducer) Extract(ctx context.Context, event datalayer.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	endpointID := event.Endpoint.GetMetadata().ID.String()
	switch event.Type {
	case datalayer.EventDelete:
		// Same pointer-identity guard as inflightload: an out-of-order delete for
		// a replaced endpoint must not clear the live recent window.
		if registered, ok := p.registeredEndpoints.Load(endpointID); ok && registered != event.Endpoint {
			log.FromContext(ctx).V(logutil.DEFAULT).Info("Ignoring stale delete for replaced endpoint", "endpoint", endpointID)
			return nil
		}
		p.registeredEndpoints.Delete(endpointID)
		if p.recent != nil {
			p.recent.forget(endpointID)
		}
		p.forgetPeerMerged(endpointID)
		return nil
	case datalayer.EventAddOrUpdate:
		p.registeredEndpoints.Store(endpointID, event.Endpoint)
		// Placeholder for the sync attribute only when a syncer will publish.
		// Without a syncer, installing a local-only DynamicAttribute would make
		// every Produce snapshot and skip the local window for no benefit.
		if !p.syncerActive() {
			return nil
		}
		event.Endpoint.GetAttributes().Put(p.syncHashDK, &datalayer.DynamicAttribute{
			Get: func() datalayer.Cloneable {
				return p.CrossReplicaState().Read(endpointID)
			},
		})
	}
	return nil
}

func (p *dataProducer) makeserver(targetEndpoint fwksched.Endpoint) server {
	gpuBlocks := defaultLRUCapacityPerServer
	if p.config.AutoTune && targetEndpoint.GetMetrics() != nil && targetEndpoint.GetMetrics().CacheNumBlocks > 0 {
		gpuBlocks = targetEndpoint.GetMetrics().CacheNumBlocks
		// CacheNumBlocks counts engine blocks, but each LRU entry is one router
		// block, which minBlockSizeTokens can make larger. Size the LRU to the
		// engine's token capacity so it does not outlive the engine's cache.
		if engineBlockSize := targetEndpoint.GetMetrics().CacheBlockSize; engineBlockSize > 0 {
			gpuBlocks = max(1, gpuBlocks*engineBlockSize/p.GetBlockSize([]fwksched.Endpoint{targetEndpoint}))
		}
	} else if p.config.LRUCapacityPerServer > 0 {
		gpuBlocks = p.config.LRUCapacityPerServer
	}
	return server{
		ServerID:       ServerID(targetEndpoint.GetMetadata().ID),
		NumOfGPUBlocks: gpuBlocks,
	}
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
			m := endpoint.GetMetrics()
			if pmu := m.CachePrefixMatchUnit; pmu > 0 {
				blockSize = pmu
			} else if bs := m.CacheBlockSize; bs > 0 {
				blockSize = bs
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

// RequestAbandoned releases the per-request state when the request is abandoned
// before reaching PreRequest.
func (p *dataProducer) RequestAbandoned(_ context.Context, request *fwksched.InferenceRequest) {
	p.pluginState.Delete(request.RequestID)
}
