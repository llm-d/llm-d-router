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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"sort"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

const (
	// ProducerType is the type name used to register the multimodal data producer.
	ProducerType = "multimodal-encoder-producer"

	// ProducedKey is the data key emitted by this producer.
	ProducedKey = attrmm.EncoderCacheMatchInfoKey

	defaultCacheSize = 10000
)

var (
	_ requestcontrol.DataProducer = &Producer{}
	_ requestcontrol.PreRequest   = &Producer{}
)

// Parameters configures the multimodal encoder-cache data producer.
type Parameters struct {
	// CacheSize defines the maximum number of mm_hash -> pod-set entries to track.
	CacheSize int `json:"cacheSize"`
}

// Factory creates a multimodal encoder-cache data producer.
func Factory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", ProducerType, err)
		}
	}

	p, err := New(handle.Context(), &parameters, handle.PodList)
	if err != nil {
		return nil, err
	}
	return p.WithName(name), nil
}

// Producer tracks multimodal content hashes and the pods that likely hold their
// encoder-cache entries.
type Producer struct {
	typedName   plugin.TypedName
	cache       *lru.Cache[string, map[string]struct{}]
	pluginState *plugin.PluginState
	podList     func() []k8stypes.NamespacedName
	mutex       sync.RWMutex
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
func New(ctx context.Context, params *Parameters, podList func() []k8stypes.NamespacedName) (*Producer, error) {
	cacheSize := defaultCacheSize
	if params != nil && params.CacheSize > 0 {
		cacheSize = params.CacheSize
	}

	cache, err := lru.New[string, map[string]struct{}](cacheSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create multimodal encoder-cache LRU with size %d: %w", cacheSize, err)
	}

	return &Producer{
		typedName:   plugin.TypedName{Type: ProducerType},
		cache:       cache,
		pluginState: plugin.NewPluginState(ctx),
		podList:     podList,
	}, nil
}

// TypedName returns the plugin type/name.
func (p *Producer) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the plugin instance name.
func (p *Producer) WithName(name string) *Producer {
	p.typedName.Name = name
	return p
}

// Produces returns the data keys this plugin produces.
func (p *Producer) Produces() map[string]any {
	return map[string]any{ProducedKey: attrmm.EncoderCacheMatchInfo{}}
}

// Consumes returns the data keys this plugin requires.
func (p *Producer) Consumes() map[string]any {
	return nil
}

// PluginState returns request-scoped state shared between producer extension points.
func (p *Producer) PluginState() *plugin.PluginState {
	return p.pluginState
}

// PrepareRequestData attaches multimodal encoder-cache match data to endpoints.
func (p *Producer) Produce(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	requestItems := ExtractMMItems(request)
	if len(requestItems) == 0 {
		logger.Info("No multimodal content found, skipping encoder-cache match data")
		return nil
	}

	if request != nil && request.RequestID != "" {
		p.pluginState.Write(request.RequestID, plugin.StateKey(ProducerType), &requestState{items: requestItems})
	}
	// TODO(#1144): Removal of stale pods should happen in background for better performance.
	p.removeStalePods()
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			continue
		}
		matchedItems := p.matchedItemsForPod(metadata.NamespacedName.String(), requestItems)
		endpoint.Put(attrmm.EncoderCacheMatchInfoKey, attrmm.NewEncoderCacheMatchInfo(
			matchedItems,
			requestItems,
		))
	}

	return nil
}

// PreRequest records the selected endpoint(s) for each hash in the current request.
func (p *Producer) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	if request == nil || request.RequestID == "" {
		return
	}
	defer p.pluginState.Delete(request.RequestID)

	state, err := plugin.ReadPluginStateKey[*requestState](p.pluginState, request.RequestID, plugin.StateKey(ProducerType))
	if err != nil || len(state.items) == 0 {
		logger.Info("No multimodal request state found, skipping encoder-cache update")
		return
	}

	targets := targetEndpoints(schedulingResult)
	if len(targets) == 0 {
		logger.Info("No target endpoints found, skipping encoder-cache update")
		return
	}

	p.mutex.Lock()
	defer p.mutex.Unlock()
	for _, item := range state.items {
		pods := map[string]struct{}{}
		if existing, ok := p.cache.Get(item.Hash); ok {
			pods = maps.Clone(existing)
		}
		for _, endpoint := range targets {
			if metadata := endpoint.GetMetadata(); metadata != nil {
				pods[metadata.NamespacedName.String()] = struct{}{}
			}
		}
		if len(pods) > 0 {
			p.cache.Add(item.Hash, pods)
		}
	}
}

// ExtractMMItems returns deterministic, unique multimodal encoder-cache items
// for a request. Parser-provided multimodal features are preferred; if
// unavailable, typed structured media blocks are hashed from stable identifiers.
func ExtractMMItems(request *scheduling.InferenceRequest) []attrmm.MatchItem {
	if request == nil || request.Body == nil {
		return nil
	}

	if request.Body.TokenizedPrompt != nil && len(request.Body.TokenizedPrompt.MultiModalFeatures) > 0 {
		return itemsFromTokenizedPrompt(request.Body.TokenizedPrompt.MultiModalFeatures)
	}

	if request.Body.ChatCompletions != nil {
		return itemsFromChat(request.Body.ChatCompletions)
	}

	return nil
}

func itemsFromTokenizedPrompt(features []fwkrh.MultiModalFeature) []attrmm.MatchItem {
	itemsByHash := map[string]attrmm.MatchItem{}
	for _, feature := range features {
		if feature.Hash == "" {
			continue
		}
		addItem(itemsByHash, feature.Hash, 1)
	}
	return sortedItems(itemsByHash)
}

func itemsFromChat(request *fwkrh.ChatCompletionsRequest) []attrmm.MatchItem {
	itemsByHash := map[string]attrmm.MatchItem{}
	for _, message := range request.Messages {
		for _, block := range message.Content.Structured {
			addBlockItem(itemsByHash, block)
		}
	}
	return sortedItems(itemsByHash)
}

func addBlockItem(itemsByHash map[string]attrmm.MatchItem, block fwkrh.ContentBlock) {
	switch {
	case block.ImageURL.URL != "":
		addItem(itemsByHash, contentHash("image_url", block.ImageURL.URL), 1)
	case block.VideoURL.URL != "":
		addItem(itemsByHash, contentHash("video_url", block.VideoURL.URL), 1)
	case block.InputAudio.Data != "":
		addItem(itemsByHash, contentHash("input_audio", block.InputAudio.Format+":"+block.InputAudio.Data), 1)
	}
}

func contentHash(kind, identifier string) string {
	sum := sha256.Sum256([]byte(kind + "\x00" + identifier))
	return hex.EncodeToString(sum[:])
}

func addItem(itemsByHash map[string]attrmm.MatchItem, hash string, size int) {
	itemsByHash[hash] = attrmm.MatchItem{Hash: hash, Size: size}
}

func sortedItems(itemsByHash map[string]attrmm.MatchItem) []attrmm.MatchItem {
	if len(itemsByHash) == 0 {
		return nil
	}
	hashes := make([]string, 0, len(itemsByHash))
	for hash := range itemsByHash {
		hashes = append(hashes, hash)
	}
	sort.Strings(hashes)
	items := make([]attrmm.MatchItem, 0, len(hashes))
	for _, hash := range hashes {
		items = append(items, itemsByHash[hash])
	}
	return items
}

func (p *Producer) matchedItemsForPod(pod string, requestItems []attrmm.MatchItem) []attrmm.MatchItem {
	matchedItemsByHash := map[string]attrmm.MatchItem{}
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	for _, item := range requestItems {
		pods, ok := p.cache.Get(item.Hash)
		if !ok {
			continue
		}
		if _, ok := pods[pod]; ok {
			matchedItemsByHash[item.Hash] = item
		}
	}
	return sortedItems(matchedItemsByHash)
}

func targetEndpoints(schedulingResult *scheduling.SchedulingResult) []scheduling.Endpoint {
	if schedulingResult == nil || schedulingResult.PrimaryProfileName == "" || schedulingResult.ProfileResults == nil {
		return nil
	}
	result := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if result == nil {
		return nil
	}
	return result.TargetEndpoints
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
	for _, hash := range p.cache.Keys() {
		pods, ok := p.cache.Get(hash)
		if !ok {
			continue
		}
		for pod := range pods {
			if _, ok := validPods[pod]; !ok {
				delete(pods, pod)
			}
		}
		if len(pods) == 0 {
			p.cache.Remove(hash)
			continue
		}
		p.cache.Add(hash, pods)
	}
}

func (p *Producer) cacheSnapshot() map[string]map[string]struct{} {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	snapshot := make(map[string]map[string]struct{}, p.cache.Len())
	for _, hash := range p.cache.Keys() {
		if pods, ok := p.cache.Get(hash); ok {
			snapshot[hash] = maps.Clone(pods)
		}
	}
	return snapshot
}

func (p *Producer) putCacheEntry(hash string, pods ...k8stypes.NamespacedName) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	podSet := map[string]struct{}{}
	if existing, ok := p.cache.Get(hash); ok {
		podSet = maps.Clone(existing)
	}
	for _, pod := range pods {
		podSet[pod.String()] = struct{}{}
	}
	p.cache.Add(hash, podSet)
}

func cloneMatchItems(items []attrmm.MatchItem) []attrmm.MatchItem {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]attrmm.MatchItem, len(items))
	copy(cloned, items)
	return cloned
}
