/*
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

// Package multimodal provides a data producer for multimodal encoder-cache
// affinity. It extracts request media identifiers once, matches them against
// recent pod placements, and stores reusable match data on endpoints.
package multimodal

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

const (
	// ProducerType is the type name used to register the multimodal data producer.
	ProducerType = "mm-embeddings-cache-producer"

	podCleanupInterval = 2 * time.Minute

	// defaultCacheSizeInMB is 4 GiB (4096 MiB).
	defaultCacheSizeInMB = 4096

	// bytesPerImage is the assumed memory per tracked image.
	bytesPerImage = 2 * 1024 * 1024
)

var (
	// ProducedKey is the data key emitted by this producer.
	ProducedKey = attrmm.EncoderCacheMatchInfoKey

	_ requestcontrol.DataProducer = &Producer{}
	_ requestcontrol.PreRequest   = &Producer{}
	_ fwkdl.EndpointExtractor     = &Producer{}
	_ plugin.StateDumper          = &Producer{}
)

// Parameters configures the multimodal encoder-cache data producer.
type Parameters struct {
	// CacheSizeInMBPerServer is the per-endpoint LRU memory budget in mebibytes (MiB).
	CacheSizeInMBPerServer int `json:"cacheSizeInMBPerServer"`
}

// lruCapacityFromCacheSizeMB converts a MiB budget to a maximum LRU entry count.
func lruCapacityFromCacheSizeMB(mb int) int {
	if mb <= 0 {
		mb = defaultCacheSizeInMB
	}
	n := (int64(mb) * 1024 * 1024) / bytesPerImage
	if n < 1 {
		return 1
	}
	if n > int64(^uint(0)>>1) {
		return int(^uint(0) >> 1)
	}
	return int(n)
}

// Factory creates a multimodal encoder-cache data producer.
func Factory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", ProducerType, err)
		}
	}

	return New(handle.Context(), name, &parameters, handle.PodList)
}

// Producer tracks multimodal content hashes and the pods that likely hold their
// encoder-cache entries. Each pod has its own LRU cache of hashes, so eviction
// is scoped per endpoint rather than global.
type Producer struct {
	typedName   plugin.TypedName
	dk          plugin.DataKey
	caches      map[string]*lru.Cache[string, struct{}]
	cacheSize   int
	pluginState *plugin.PluginState
	podList     func() []k8stypes.NamespacedName
	mutex       sync.RWMutex
	wg          sync.WaitGroup
}

type requestState struct {
	items []attrmm.MatchItem
}

func (s *requestState) Clone() plugin.StateData {
	if s == nil {
		return nil
	}
	return &requestState{items: attrmm.CloneMatchItems(s.items)}
}

// New creates a Producer.
func New(ctx context.Context, name string, params *Parameters, podList func() []k8stypes.NamespacedName) (*Producer, error) {
	cacheSizeMB := 0
	if params != nil {
		cacheSizeMB = params.CacheSizeInMBPerServer
	}
	cacheSize := lruCapacityFromCacheSizeMB(cacheSizeMB)

	registerEncoderCacheMetrics()

	p := &Producer{
		typedName:   plugin.TypedName{Type: ProducerType, Name: name},
		dk:          attrmm.EncoderCacheMatchInfoKey.WithNonEmptyProducerName(name),
		caches:      make(map[string]*lru.Cache[string, struct{}]),
		cacheSize:   cacheSize,
		pluginState: plugin.NewPluginState(ctx),
		podList:     podList,
	}
	if podList != nil {
		go p.cleanupLoop(ctx)
	}
	return p, nil
}

// getOrCreatePodCache returns the LRU cache for the given pod, creating one if absent.
// Must be called with p.mutex held for write.
func (p *Producer) getOrCreatePodCache(pod string) *lru.Cache[string, struct{}] {
	if c, ok := p.caches[pod]; ok {
		return c
	}
	c, _ := lru.New[string, struct{}](p.cacheSize)
	p.caches[pod] = c
	return c
}

func (p *Producer) cleanupLoop(ctx context.Context) {
	ticker := time.NewTicker(podCleanupInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.removeStalePods()
		}
	}
}

// TypedName returns the plugin type/name.
func (p *Producer) TypedName() plugin.TypedName {
	return p.typedName
}

const maxDebugDumpPods = 100

// encoderCacheState is the snapshot returned by DumpState: the known pods from
// the datalayer and per-pod cached-item counts. The multimodal content hashes
// (the cache keys) are not exposed. Both lists are capped at MaxPods (PodList by
// name, Pods by item count) and a list is partial when its matching total
// exceeds MaxPods. PodList is sampled just before the per-pod view rather than
// atomically with it, so the snapshot is best-effort: the two need not be
// perfectly consistent, and a freshly tracked pod can appear in Pods but not yet
// in PodList.
type encoderCacheState struct {
	PodList        []string       `json:"podList"`
	TotalKnownPods int            `json:"totalKnownPods"`
	Pods           []podItemCount `json:"pods"`
	TotalPods      int            `json:"totalPods"`
	MaxPods        int            `json:"maxPods"`
}

type podItemCount struct {
	Pod   string `json:"pod"`
	Items int    `json:"items"`
}

// DumpState reports the known pods from the datalayer and how many encoder-cache
// items are tracked per pod, ordered by count (most first) and capped to
// maxDebugDumpPods so the payload stays bounded. Pod identities and counts are
// exposed; the content hashes (cache keys) are not.
func (p *Producer) DumpState() (json.RawMessage, error) {
	// podList() is the datalayer accessor; call it outside the cache lock, as
	// removeStalePods does.
	podList := []string{}
	var totalKnownPods int
	if p.podList != nil {
		known := p.podList()
		totalKnownPods = len(known)
		podList = make([]string, 0, len(known))
		for _, nn := range known {
			podList = append(podList, nn.String())
		}
		sort.Strings(podList)
		if len(podList) > maxDebugDumpPods {
			podList = podList[:maxDebugDumpPods]
		}
	}

	p.mutex.RLock()
	state := encoderCacheState{
		PodList:        podList,
		TotalKnownPods: totalKnownPods,
		MaxPods:        maxDebugDumpPods,
		TotalPods:      len(p.caches),
	}
	pods := make([]podItemCount, 0, len(p.caches))
	for pod, cache := range p.caches {
		pods = append(pods, podItemCount{Pod: pod, Items: cache.Len()})
	}
	p.mutex.RUnlock()

	sort.SliceStable(pods, func(a, b int) bool {
		if pods[a].Items != pods[b].Items {
			return pods[a].Items > pods[b].Items
		}
		return pods[a].Pod < pods[b].Pod
	})
	if len(pods) > maxDebugDumpPods {
		pods = pods[:maxDebugDumpPods]
	}
	state.Pods = pods
	return json.Marshal(state)
}

// Produces returns the data keys this plugin produces.
func (p *Producer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: attrmm.EncoderCacheMatchInfo{}}
}

// Consumes declares the TokenizedPrompt dependency so the data-layer DAG orders
// the token-producer before this producer runs and auto-creates one when none
// is configured; multimodal features come from the tokenizer output.
func (p *Producer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: scheduling.TokenizedPrompt{}},
	}
}

// PluginState returns request-scoped state shared between producer extension points.
func (p *Producer) PluginState() *plugin.PluginState {
	return p.pluginState
}

// Produce attaches multimodal encoder-cache match data to endpoints.
func (p *Producer) Produce(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	requestItems := ExtractMMItems(request)
	if len(requestItems) == 0 {
		logger.Info("No multimodal content found, skipping encoder-cache match data")
		return nil
	}

	p.recordItemLookups(requestItems)

	if request != nil && request.RequestID != "" {
		p.pluginState.Write(request.RequestID, plugin.StateKey(ProducerType), &requestState{items: requestItems})
	}
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			continue
		}
		matchedItems := p.matchedItemsForPod(metadata.NamespacedName.String(), requestItems)
		p.recordHitRatio(len(matchedItems), len(requestItems))
		endpoint.Put(p.dk.String(), attrmm.NewEncoderCacheMatchInfo(
			matchedItems,
			requestItems,
		))
	}

	return nil
}

// ExtractMMItems returns deterministic, unique multimodal encoder-cache items
// derived from the tokenized prompt's multimodal features.
func ExtractMMItems(request *scheduling.InferenceRequest) []attrmm.MatchItem {
	if request == nil || request.Body == nil || request.Body.TokenizedPrompt == nil {
		return nil
	}

	itemsByHash := map[string]attrmm.MatchItem{}
	for _, feature := range request.Body.TokenizedPrompt.MultiModalFeatures {
		if feature.Hash == "" {
			continue
		}
		addItem(itemsByHash, feature.Hash, string(feature.Modality))
	}
	return itemSlice(itemsByHash)
}

func addItem(itemsByHash map[string]attrmm.MatchItem, hash, modality string) {
	itemsByHash[hash] = attrmm.MatchItem{Hash: hash, Size: 1, Modality: modality}
}

func itemSlice(itemsByHash map[string]attrmm.MatchItem) []attrmm.MatchItem {
	if len(itemsByHash) == 0 {
		return nil
	}
	items := make([]attrmm.MatchItem, 0, len(itemsByHash))
	for _, item := range itemsByHash {
		items = append(items, item)
	}
	return items
}

// recordItemLookups increments the queries counter for each item and, for every
// endpoint whose LRU contains the hash, increments that endpoint's hits counter.
// Contains is used instead of Get to avoid altering recency during a read-only path.
func (p *Producer) recordItemLookups(items []attrmm.MatchItem) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	pluginType, pluginName := p.typedName.Type, p.typedName.Name
	for _, item := range items {
		encoderCacheQueriesTotal.WithLabelValues(pluginType, pluginName, item.Modality).Inc()
		for pod, podCache := range p.caches {
			if podCache.Contains(item.Hash) {
				encoderCacheHitsTotal.WithLabelValues(pluginType, pluginName, pod, item.Modality).Inc()
			}
		}
	}
}

// recordHitRatio observes the fraction of a request's multimodal items that
// matched a single endpoint's LRU. A zero total is not a meaningful ratio and
// is not observed.
func (p *Producer) recordHitRatio(matchedItems, totalItems int) {
	if totalItems == 0 {
		return
	}
	ratio := float64(matchedItems) / float64(totalItems)
	encoderCacheHitRatio.WithLabelValues(p.typedName.Type, p.typedName.Name).Observe(ratio)
}

func (p *Producer) matchedItemsForPod(pod string, requestItems []attrmm.MatchItem) []attrmm.MatchItem {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	podCache, ok := p.caches[pod]
	if !ok {
		return nil
	}
	matchedItemsByHash := map[string]attrmm.MatchItem{}
	for _, item := range requestItems {
		if podCache.Contains(item.Hash) {
			matchedItemsByHash[item.Hash] = item
		}
	}
	return itemSlice(matchedItemsByHash)
}

func (p *Producer) removeStalePods() {
	if p.podList == nil {
		return
	}
	podList := p.podList()
	if len(podList) == 0 {
		return
	}
	validPods := make(map[string]struct{}, len(podList))
	for _, pod := range podList {
		validPods[pod.String()] = struct{}{}
	}

	p.mutex.Lock()
	defer p.mutex.Unlock()
	for pod := range p.caches {
		if _, ok := validPods[pod]; !ok {
			delete(p.caches, pod)
		}
	}
}

// Extract removes deleted endpoints from the best-effort multimodal
// cache-affinity state when endpoint lifecycle events are wired through the data layer.
func (p *Producer) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	if event.Type != fwkdl.EventDelete || event.Endpoint == nil {
		return nil
	}
	metadata := event.Endpoint.GetMetadata()
	if metadata == nil || metadata.NamespacedName.Name == "" {
		return nil
	}
	p.removePod(metadata.NamespacedName.String())
	log.FromContext(ctx).V(logging.DEBUG).Info("Removed stale pod from multimodal encoder-cache state",
		"pod", metadata.NamespacedName.String())
	return nil
}

func (p *Producer) removePod(pod string) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	delete(p.caches, pod)
}
