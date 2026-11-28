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
	"slices"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/common/observability/multimodal"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	approxprefixconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/approximateprefix/constants"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixmetrics"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
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
	_ fwkrh.DataProducer                = &dataProducer{}
	_ fwkrh.PreRequest                  = &dataProducer{}
	_ fwkrh.ResponseBodyProcessor       = &dataProducer{}
	_ plugin.StateDumper                = &dataProducer{}
	_ datalayer.EndpointExtractor       = (*dataProducer)(nil)
	_ datalayer.Registrant              = &dataProducer{}
	_ datalayer.CrossReplicaContributor = (*dataProducer)(nil)
	_ datalayer.CrossReplicaImporter    = (*dataProducer)(nil)
)

// indexUpdateStateKey identifies the per-request record of indexer entries
// PreRequest added. The record lives until the response arrives so a failed
// response can remove the entries.
const indexUpdateStateKey plugin.StateKey = "indexUpdate"

// errEndpointNotReady is returned by ImportCrossReplica for an endpoint whose
// add event has not reached Extract yet; the runtime retries the import.
var errEndpointNotReady = errors.New("approx-prefix: endpoint not registered yet")

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
	// lifecycleMu orders Extract's registration changes with
	// CleanUpInactivePods, so the cleanup never removes an endpoint that was
	// registered after it checked.
	lifecycleMu sync.Mutex
	// peerMerged tracks peer prompt keys already folded into the indexer per
	// endpoint so a stable peer window does not re-Add (and LRU-touch) on every
	// request. Each merge replaces the set with the active peer-only window so
	// the map stays bounded and prompts that leave and return can be re-indexed.
	// Values are *mergedKeySet.
	peerMerged sync.Map
	// estimateTokens is set during startup dependency validation when the
	// token-producer this plugin consumes is the estimate backend.
	estimateTokens atomic.Bool
}

// mergedKeySet is the per-endpoint set stored in peerMerged. keys holds the
// prompt keys of the latest peer window that have been Added to the indexer.
// digest is the xxhash of the last processed PrefixHashSet.Prompts so the
// request-path merge can skip snapshot/key work when the sync attribute is
// unchanged. importMu is held for a whole import of this endpoint.
type mergedKeySet struct {
	importMu   sync.Mutex
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
// is configured. PrefixHashSet is optional: when peer state is merged on the
// request path, Produce reads the syncer-backed attribute.
func (p *dataProducer) Consumes() plugin.DataDependencies {
	deps := plugin.DataDependencies{
		Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: plugin.ConsumedData{
			Type:    fwkrequest.TokenizedRequest{},
			Matches: p.recordTokenBackend,
		}},
	}
	if p.syncCrossReplicaState && !p.peerImportOffPath() {
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
	for _, tier := range config.CapacityTiers {
		if tier != capacityTierHBM && tier != capacityTierCPU {
			return nil, fmt.Errorf("invalid configuration: CapacityTiers entries must be %q or %q (current value: %q)", capacityTierHBM, capacityTierCPU, tier)
		}
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
// recent window. Unset MaxSyncedTokens uses defaultMaxSyncedTokens. The block
// size is raised to minBlockSizeTokens, as GetBlockSize does when prompts are
// hashed, so each hash in the budget covers the tokens it is hashed over.
func syncedHashBudget(cfg config) int {
	blockSize := cfg.BlockSizeTokens
	if blockSize <= 0 {
		blockSize = defaultBlockSizeTokens
	}
	blockSize = max(blockSize, minBlockSizeTokens)
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

// peerImportOffPath reports whether the runtime imports peer state through
// ImportCrossReplica, so Produce must not merge on the request path. It is
// evaluated per call because the syncer is set on the handle after plugins
// are constructed.
func (p *dataProducer) peerImportOffPath() bool {
	if !p.syncerActive() {
		return false
	}
	_, ok := p.handle.CrossReplicaSyncer().(datalayer.CrossReplicaPeerReader)
	return ok
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
			p.removeInactivePods(ctx, handle.PodList())
		}
	}
}

// removeInactivePods drops indexer, recent-window and merged-key state of pods
// that are neither in podNames nor registered. A registered endpoint is
// removed by its delete event; the pod list may lag behind a registration.
func (p *dataProducer) removeInactivePods(ctx context.Context, podNames []k8stypes.NamespacedName) {
	activePods := make(map[ServerID]struct{}, len(podNames))
	for _, nsn := range podNames {
		activePods[ServerID(nsn)] = struct{}{}
	}
	for _, pod := range p.indexerInst.Pods() {
		if _, ok := activePods[pod]; ok {
			continue
		}
		id := pod.String()
		p.lifecycleMu.Lock()
		if _, registered := p.registeredEndpoints.Load(id); !registered {
			p.indexerInst.RemovePod(pod)
			if p.recent != nil {
				p.recent.forget(id)
			}
			p.forgetPeerMerged(id)
			log.FromContext(ctx).V(logutil.VERBOSE).Info("Removed pod not in active set", "pod", pod)
		}
		p.lifecycleMu.Unlock()
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
	if !p.peerImportOffPath() {
		p.mergeRemoteHashes(pods)
	}

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
		BlockSize:                 blockSize,
	}

	p.pluginState.Write(request.RequestID, plugin.StateKey(p.typedName.Name), state)

	return nil
}

// mergeRemoteHashes folds peer recent-window prompts into the local indexer on
// the request path. It runs when a CrossReplicaSyncer is configured that
// cannot provide peer snapshots for background import. The syncer-backed
// attribute is Aggregate(local, peers). When the attribute digest matches the
// last processed value, the pod is skipped without snapshot or prompt-key
// work. A pod that another request is importing is skipped as well.
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
		keys := p.mergedKeys(endpointID)
		if !keys.importMu.TryLock() {
			continue
		}
		if err := p.importPeerSet(context.Background(), keys, endpointID, func() server { return p.makeserver(pod) }, set); err == nil {
			keys.rememberDigest(digest)
		}
		keys.importMu.Unlock()
	}
}

// ImportCrossReplica implements datalayer.CrossReplicaImporter. peer is the
// peer-only PrefixHashSet for endpointID; nil or empty means no peer
// publishes, which clears the recorded keys and leaves the indexer as is.
func (p *dataProducer) ImportCrossReplica(ctx context.Context, endpointID string, peer any) error {
	registered, ok := p.registeredEndpoints.Load(endpointID)
	if !ok {
		return errEndpointNotReady
	}
	endpoint, ok := registered.(endpointCapacity)
	if !ok {
		return fmt.Errorf("approx-prefix: endpoint %s has unexpected type %T", endpointID, registered)
	}
	set, ok := peer.(*attrprefix.PrefixHashSet)
	if !ok && peer != nil {
		return fmt.Errorf("approx-prefix: peer state for %s has unexpected type %T", endpointID, peer)
	}
	if set == nil {
		set = &attrprefix.PrefixHashSet{}
	}
	keys := p.mergedKeys(endpointID)
	keys.importMu.Lock()
	defer keys.importMu.Unlock()
	return p.importPeerSet(ctx, keys, endpointID, func() server { return p.makeserver(endpoint) }, set)
}

// importPeerSet Adds the prompts of set that are neither in the local recent
// window nor already recorded for endpointID, and records each key only after
// its Add. Keys that left the peer window are dropped so a prompt that leaves
// and returns is Added again. Merged prompts are not recorded into the recent
// window (no echo). Callers hold keys.importMu.
func (p *dataProducer) importPeerSet(ctx context.Context, keys *mergedKeySet, endpointID string, newServer func() server, set *attrprefix.PrefixHashSet) error {
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
	novel := keys.pruneAndDiff(active)
	if len(novel) == 0 {
		return nil
	}
	srv := newServer()
	for _, prompt := range set.Prompts {
		if len(prompt) == 0 {
			continue
		}
		key := promptKeyUint64(prompt)
		if _, ok := novel[key]; !ok {
			continue
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		p.indexerInst.Add(uint64sToBlockHashes(prompt), srv)
		if !keys.record(ctx, key) {
			return ctx.Err()
		}
	}
	return nil
}

func (p *dataProducer) mergedKeys(endpointID string) *mergedKeySet {
	v, _ := p.peerMerged.LoadOrStore(endpointID, &mergedKeySet{keys: make(map[string]struct{})})
	return v.(*mergedKeySet)
}

// pruneAndDiff drops recorded keys that are not in active and returns the
// active keys not recorded yet.
func (s *mergedKeySet) pruneAndDiff(active map[string]struct{}) map[string]struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k := range s.keys {
		if _, ok := active[k]; !ok {
			delete(s.keys, k)
		}
	}
	novel := make(map[string]struct{})
	for k := range active {
		if _, ok := s.keys[k]; !ok {
			novel[k] = struct{}{}
		}
	}
	return novel
}

// record marks key as Added unless ctx is done.
func (s *mergedKeySet) record(ctx context.Context, key string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ctx.Err() != nil {
		return false
	}
	s.keys[key] = struct{}{}
	return true
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

// rememberDigest records the digest of the window that was just imported.
// Callers hold s.importMu, so digests are stored in the order the imports ran
// and a set that an endpoint delete dropped meanwhile is not stored again.
func (s *mergedKeySet) rememberDigest(digest uint64) {
	s.mu.Lock()
	s.digest = digest
	s.haveDigest = true
	s.mu.Unlock()
}

func (p *dataProducer) forgetPeerMerged(endpointID string) {
	p.peerMerged.Delete(endpointID)
}

// PreRequest records in the shared indexer the result of the scheduling selection.
// It updates the indexer with the prefix hashes for the selected endpoint(s).
// The entries are optimistic: ResponseBody removes them when the response
// reports the request never ran, so a retry is not routed as cached.
// See https://github.com/llm-d/llm-d-router/issues/3919.
func (p *dataProducer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	// Release the scheduling state; the index-update record and continuation
	// state written below are kept until the response arrives.
	defer p.pluginState.DeleteKey(request.RequestID, plugin.StateKey(p.typedName.Name))
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if len(primaryProfileResult.TargetEndpoints) == 0 {
		p.pluginState.Delete(request.RequestID)
		return nil
	}

	targetEndpoint := primaryProfileResult.TargetEndpoints[0]
	pr := schedulingResult.PrefillProfile()
	disaggregated := pr != nil && len(pr.TargetEndpoints) > 0
	servers := make([]server, 0, 2)
	var decode *server
	if !disaggregated || !p.config.ExcludeDecodeWhenDisaggregated {
		d := p.makeserver(targetEndpoint)
		servers = append(servers, d)
		decode = &d
	}
	// Also record for prefill node if present in P/D disaggregated mode.
	if disaggregated {
		servers = append(servers, p.makeserver(pr.TargetEndpoints[0]))
	}

	// Read state saved during Produce.
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(p.typedName.Name))
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin state", "requestID", request.RequestID)
		p.pluginState.Delete(request.RequestID)
		return nil
	}
	promptIndexed := p.trackContinuation(ctx, request, decode, state.BlockSize)

	record := &indexUpdateRecord{
		servers:         servers,
		perPromptHashes: state.PerPromptHashes,
		inserted:        make(map[ServerID][][]blockHash, len(servers)),
		cell:            &updateCell{},
	}
	p.pluginState.Write(request.RequestID, indexUpdateStateKey, record)

	// Update indexer asynchronously to avoid blocking the request path. The
	// cell guards against a rollback that lands before this goroutine runs.
	p.wg.Go(func() {
		var decodeEntry []blockHash
		if promptIndexed != nil {
			defer func() { promptIndexed <- decodeEntry }()
		}
		record.cell.mu.Lock()
		defer record.cell.mu.Unlock()
		if record.cell.rolledBack {
			return
		}
		for i, s := range servers {
			for _, hashes := range state.PerPromptHashes {
				record.inserted[s.ServerID] = append(record.inserted[s.ServerID], p.indexerInst.Add(hashes, s))
				if p.syncerActive() {
					entry := p.recent.record(s.String(), hashes)
					if i == 0 && decode != nil {
						decodeEntry = entry
					}
				}
			}
		}
	})

	// Record metrics. Lengths are reported as a byte estimate (fwkrequest.BytesPerToken bytes per token).
	total := 0
	for _, hashes := range state.PerPromptHashes {
		total += len(hashes)
	}
	matchLen := state.PrefixCacheServers[ServerID(targetEndpoint.GetMetadata().ID)]
	blockSize := p.GetBlockSize(primaryProfileResult.TargetEndpoints)
	recordPrefixCacheMatch(p.typedName.Name, p.typedName.Type, matchLen*blockSize*fwkrequest.BytesPerToken, total*blockSize*fwkrequest.BytesPerToken)
	if request.Body != nil {
		predictionProfile, role := prefixmetrics.PredictionTarget(schedulingResult)
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

// ResponseBody releases the index-update record when the response stream ends,
// removing the entries PreRequest added when the response reports the request
// never ran on the model server. A prompt whose request failed must not look
// cached: neither endpoint computed it, and a retry routed on the stale
// entries would skip prefill. Otherwise it indexes the generated text of a
// request tracked for continuation.
func (p *dataProducer) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest, response *fwkrh.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	record, err := plugin.ReadPluginStateKey[*indexUpdateRecord](p.pluginState, request.RequestID, indexUpdateStateKey)
	if err != nil {
		// PreRequest never ran, or the staleness janitor already reaped the record.
		return
	}
	continuation, _ := plugin.ReadPluginStateKey[*continuationState](p.pluginState, request.RequestID, continuationStateKey)
	p.pluginState.Delete(request.RequestID)
	if !responseFailed(response) {
		p.indexContinuation(request, response, continuation)
		return
	}

	record.cell.mu.Lock()
	defer record.cell.mu.Unlock()
	if record.cell.rolledBack {
		return
	}
	record.cell.rolledBack = true
	for _, s := range record.servers {
		// The indexer keeps only what this request's Add newly inserted; a
		// shared block an earlier request contributed stays indexed.
		for _, hashes := range record.inserted[s.ServerID] {
			p.indexerInst.Remove(hashes, s.ServerID)
		}
		// The recent window publishes whole prompts, so the withdrawal is the
		// request's full prompts, not the inserted subset.
		if p.syncerActive() {
			for _, hashes := range record.perPromptHashes {
				p.recent.removePrompt(s.String(), hashes)
			}
		}
	}
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Removed prefix index entries for failed request",
		"requestID", request.RequestID, "terminationCause", response.TerminationCause)
}

// responseFailed reports whether the response says the request did not run to
// completion on the model server. A non-natural termination counts only when
// the stream ended before the model server produced a chunk: the director sets
// StartOfStream on the final call only when no body chunk preceded it, and
// once output exists the pod keeps the prefilled blocks, so the entries stay.
// A response delivered in full can still carry an error status, which
// TerminationCauseNatural does not distinguish, so the status header is read
// as well.
func responseFailed(response *fwkrh.Response) bool {
	if response.TerminationCause != "" && response.TerminationCause != fwkrh.TerminationCauseNatural {
		return response.StartOfStream
	}
	status := response.Headers[reqcommon.PseudoHeaderStatus]
	if status == "" {
		status = response.Headers[reqcommon.HeaderStatus]
	}
	code, err := strconv.Atoi(status)
	return err == nil && code >= 400
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
		p.lifecycleMu.Lock()
		defer p.lifecycleMu.Unlock()
		// Same pointer-identity guard as inflightload: an out-of-order delete for
		// a replaced endpoint must not clear the live recent window.
		if registered, ok := p.registeredEndpoints.Load(endpointID); ok && registered != event.Endpoint {
			log.FromContext(ctx).V(logutil.DEFAULT).Info("Ignoring stale delete for replaced endpoint", "endpoint", endpointID)
			return nil
		}
		p.registeredEndpoints.Delete(endpointID)
		// An endpoint that returns under the same name may have restarted with an
		// empty KV cache, so its entries must not survive until the next cleanup tick.
		if p.indexerInst != nil {
			p.indexerInst.RemovePod(ServerID(event.Endpoint.GetMetadata().ID))
		}
		if p.recent != nil {
			p.recent.forget(endpointID)
		}
		p.forgetPeerMerged(endpointID)
		return nil
	case datalayer.EventAddOrUpdate:
		p.lifecycleMu.Lock()
		p.registeredEndpoints.Store(endpointID, event.Endpoint)
		p.lifecycleMu.Unlock()
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

// coversTier reports whether the auto-tuned capacity covers tier. An empty
// CapacityTiers covers every tier.
func (p *dataProducer) coversTier(tier string) bool {
	return len(p.config.CapacityTiers) == 0 || slices.Contains(p.config.CapacityTiers, tier)
}

// endpointCapacity is the part of an endpoint makeserver reads. Both the
// scheduling endpoint and the data-layer endpoint delivered to Extract satisfy it.
type endpointCapacity interface {
	GetMetadata() *datalayer.EndpointMetadata
	GetMetrics() *datalayer.Metrics
}

// makeserver sizes the pod's prefix index. With AutoTune, the capacity is the
// largest tier in CapacityTiers that the endpoint reports, in router blocks:
// the HBM tier from CacheNumBlocks and the offload-extended tier from
// KvCacheMaxTokenCapacity. An offload tier holds a copy of what the HBM tier
// holds (SGLang HiCache, vLLM CPU offloading), so the tiers are compared
// rather than added. Without metrics the configured capacity applies.
func (p *dataProducer) makeserver(targetEndpoint endpointCapacity) server {
	capacity := defaultLRUCapacityPerServer
	fromMetrics := false
	offloadCapacityUnknown := false
	if m := targetEndpoint.GetMetrics(); p.config.AutoTune && m != nil {
		// Each LRU entry is one router block, which minBlockSizeTokens can make
		// larger than the engine's block, so capacities convert through tokens
		// and the index does not outlive the engine's cache.
		routerBlockSize := p.blockSizeFor(m)
		if p.coversTier(capacityTierHBM) && m.CacheNumBlocks > 0 {
			capacity = m.CacheNumBlocks
			if engineBlockSize := m.CacheBlockSize; engineBlockSize > 0 {
				capacity = max(1, capacity*engineBlockSize/routerBlockSize)
			}
			fromMetrics = true
		}
		if p.coversTier(capacityTierCPU) {
			if m.KvCacheMaxTokenCapacity > 0 {
				if blocks := max(1, m.KvCacheMaxTokenCapacity/routerBlockSize); blocks > capacity || !fromMetrics {
					capacity = blocks
				}
				fromMetrics = true
			} else if m.KvCacheOffloadDetected && fromMetrics {
				offloadCapacityUnknown = true
			}
		}
	}
	if !fromMetrics && p.config.LRUCapacityPerServer > 0 {
		capacity = p.config.LRUCapacityPerServer
	}
	return server{
		ServerID:               ServerID(targetEndpoint.GetMetadata().ID),
		NumOfGPUBlocks:         capacity,
		CapacityFromMetrics:    fromMetrics,
		OffloadCapacityUnknown: offloadCapacityUnknown,
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
	var m *datalayer.Metrics
	if len(endpoints) > 0 {
		m = endpoints[0].GetMetrics()
	}
	return p.blockSizeFor(m)
}

// blockSizeFor is GetBlockSize for one endpoint's metrics, which may be nil.
func (p *dataProducer) blockSizeFor(m *datalayer.Metrics) int {
	blockSize := p.config.BlockSizeTokens
	if p.config.AutoTune && m != nil {
		if pmu := m.CachePrefixMatchUnit; pmu > 0 {
			blockSize = pmu
		} else if bs := m.CacheBlockSize; bs > 0 {
			blockSize = bs
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
