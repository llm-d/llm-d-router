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
	k8stypes "k8s.io/apimachinery/pkg/types"
)

const (
	// ProducerType is the type name used to register the multimodal data producer.
	ProducerType = "mm-embeddings-cache-producer"

	defaultCacheSize   = 10000
	podCleanupInterval = 2 * time.Minute
)

var (
	// ProducedKey is the data key emitted by this producer.
	ProducedKey = attrmm.EncoderCacheMatchInfoKey

	_ requestcontrol.DataProducer = &Producer{}
	_ requestcontrol.PreRequest   = &Producer{}
	_ fwkdl.EndpointExtractor     = &Producer{}
)

// Parameters configures the multimodal encoder-cache data producer.
type Parameters struct {
	// CacheSize defines the maximum number of mm_hash -> pod-set entries to track.
	CacheSize int `json:"cacheSize"`
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
// encoder-cache entries.
type Producer struct {
	typedName   plugin.TypedName
	dk          plugin.DataKey
	cache       *lru.Cache[string, map[string]struct{}]
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
	cacheSize := defaultCacheSize
	if params != nil && params.CacheSize > 0 {
		cacheSize = params.CacheSize
	}

	cache, err := lru.New[string, map[string]struct{}](cacheSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create multimodal encoder-cache LRU with size %d: %w", cacheSize, err)
	}

	p := &Producer{
		typedName:   plugin.TypedName{Type: ProducerType, Name: name},
		dk:          attrmm.EncoderCacheMatchInfoKey.WithNonEmptyProducerName(name),
		cache:       cache,
		pluginState: plugin.NewPluginState(ctx),
		podList:     podList,
	}
	if podList != nil {
		go p.cleanupLoop(ctx)
	}
	return p, nil
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

// Produces returns the data keys this plugin produces.
func (p *Producer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: attrmm.EncoderCacheMatchInfo{}}
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

	if request != nil && request.RequestID != "" {
		p.pluginState.Write(request.RequestID, plugin.StateKey(ProducerType), &requestState{items: requestItems})
	}
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			continue
		}
		matchedItems := p.matchedItemsForPod(metadata.NamespacedName.String(), requestItems)
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
		addItem(itemsByHash, feature.Hash)
	}
	return itemSlice(itemsByHash)
}

func addItem(itemsByHash map[string]attrmm.MatchItem, hash string) {
	itemsByHash[hash] = attrmm.MatchItem{Hash: hash, Size: 1}
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
	for _, hash := range p.cache.Keys() {
		pods, ok := p.cache.Get(hash)
		if !ok {
			continue
		}
		delete(pods, pod)
		if len(pods) == 0 {
			p.cache.Remove(hash)
			continue
		}
		p.cache.Add(hash, pods)
	}
}
