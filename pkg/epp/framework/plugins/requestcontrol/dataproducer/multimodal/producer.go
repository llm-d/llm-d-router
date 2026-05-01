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
	"strings"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/jellydator/ttlcache/v3"
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
	ProducerType = "multimodal-encoder-cache-data-producer"

	// ProducedKey is the data key emitted by this producer.
	ProducedKey = attrmm.EncoderCacheMatchInfoKey

	defaultCacheSize       = 10000
	defaultRequestStateTTL = 30 * time.Second
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
	typedName       plugin.TypedName
	cache           *lru.Cache[string, map[string]struct{}]
	requestStates   *ttlcache.Cache[string, map[string]int]
	requestStateTTL time.Duration
	podList         func() []k8stypes.NamespacedName
	mutex           sync.RWMutex
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

	requestStates := ttlcache.New[string, map[string]int](
		ttlcache.WithTTL[string, map[string]int](defaultRequestStateTTL),
		ttlcache.WithDisableTouchOnHit[string, map[string]int](),
	)
	go cleanRequestStates(ctx, requestStates, defaultRequestStateTTL)

	return &Producer{
		typedName:       plugin.TypedName{Type: ProducerType},
		cache:           cache,
		requestStates:   requestStates,
		requestStateTTL: defaultRequestStateTTL,
		podList:         podList,
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

// PrepareRequestData attaches multimodal encoder-cache match data to endpoints.
func (p *Producer) PrepareRequestData(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	hashToWeight := ExtractMMHashesWithWeights(request)
	if len(hashToWeight) == 0 {
		logger.Info("No multimodal content found, skipping encoder-cache match data")
		return nil
	}

	if request != nil && request.RequestId != "" {
		p.requestStates.Set(request.RequestId, maps.Clone(hashToWeight), p.requestStateTTL)
	}

	p.removeStalePods()
	total := totalWeight(hashToWeight)
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			continue
		}
		matchedHashes := p.matchedHashesForPod(metadata.NamespacedName.String(), hashToWeight)
		endpoint.Put(attrmm.EncoderCacheMatchInfoKey, attrmm.NewEncoderCacheMatchInfo(
			totalWeight(matchedHashes),
			total,
			matchedHashes,
		))
	}

	return nil
}

// PreRequest records the selected endpoint(s) for each hash in the current request.
func (p *Producer) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	if request == nil || request.RequestId == "" {
		return
	}

	stateItem := p.requestStates.Get(request.RequestId)
	p.requestStates.Delete(request.RequestId)
	if stateItem == nil || len(stateItem.Value()) == 0 {
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
	for hash := range stateItem.Value() {
		pods := map[string]struct{}{}
		if existing, ok := p.cache.Get(hash); ok {
			pods = maps.Clone(existing)
		}
		for _, endpoint := range targets {
			if metadata := endpoint.GetMetadata(); metadata != nil {
				pods[metadata.NamespacedName.String()] = struct{}{}
			}
		}
		if len(pods) > 0 {
			p.cache.Add(hash, pods)
		}
	}
}

// ExtractMMHashesWithWeights returns deterministic mm_hash -> encoder-work weight
// pairs for a request. Parser-provided multimodal features are preferred; if
// unavailable, structured media blocks are hashed from stable media identifiers.
func ExtractMMHashesWithWeights(request *scheduling.InferenceRequest) map[string]int {
	if request == nil || request.Body == nil {
		return nil
	}

	if request.Body.TokenizedPrompt != nil && len(request.Body.TokenizedPrompt.MultiModalFeatures) > 0 {
		return hashesFromTokenizedPrompt(request.Body.TokenizedPrompt.MultiModalFeatures)
	}

	if request.Body.ChatCompletions != nil {
		return hashesFromChat(request.Body.ChatCompletions)
	}

	if request.Body.Responses != nil {
		return hashesFromAny(request.Body.Responses.Input)
	}

	if request.Body.Conversations != nil {
		return hashesFromAny(request.Body.Conversations.Items)
	}

	if request.Body.Payload != nil && request.Body.Payload.IsParsed() {
		return hashesFromAny(request.Body.Payload)
	}

	return nil
}

func hashesFromTokenizedPrompt(features []fwkrh.MultiModalFeature) map[string]int {
	hashToWeight := map[string]int{}
	for _, feature := range features {
		if feature.Hash == "" {
			continue
		}
		weight := feature.Length
		if weight <= 0 {
			weight = 1
		}
		addMaxWeight(hashToWeight, feature.Hash, weight)
	}
	return emptyToNil(hashToWeight)
}

func hashesFromChat(request *fwkrh.ChatCompletionsRequest) map[string]int {
	hashToWeight := map[string]int{}
	for _, message := range request.Messages {
		for _, block := range message.Content.Structured {
			addBlockHash(hashToWeight, block)
		}
	}
	return emptyToNil(hashToWeight)
}

func hashesFromAny(value any) map[string]int {
	hashToWeight := map[string]int{}
	walkAny(value, hashToWeight)
	return emptyToNil(hashToWeight)
}

func addBlockHash(hashToWeight map[string]int, block fwkrh.ContentBlock) {
	switch {
	case block.ImageURL.Url != "":
		addMaxWeight(hashToWeight, contentHash("image_url", block.ImageURL.Url), 1)
	case block.VideoURL.Url != "":
		addMaxWeight(hashToWeight, contentHash("video_url", block.VideoURL.Url), 1)
	case block.InputAudio.Data != "":
		addMaxWeight(hashToWeight, contentHash("input_audio", block.InputAudio.Format+":"+block.InputAudio.Data), 1)
	}
}

func walkAny(value any, hashToWeight map[string]int) {
	switch typed := value.(type) {
	case fwkrh.PayloadMap:
		for k, v := range typed {
			walkNamedValue(k, v, hashToWeight)
		}
	case map[string]any:
		for k, v := range typed {
			walkNamedValue(k, v, hashToWeight)
		}
	case []any:
		for _, item := range typed {
			walkAny(item, hashToWeight)
		}
	case []fwkrh.ConversationItem:
		for _, item := range typed {
			walkAny(item.Content, hashToWeight)
		}
	case fwkrh.Content:
		for _, block := range typed.Structured {
			addBlockHash(hashToWeight, block)
		}
	case []fwkrh.ContentBlock:
		for _, block := range typed {
			addBlockHash(hashToWeight, block)
		}
	}
}

func walkNamedValue(name string, value any, hashToWeight map[string]int) {
	normalized := strings.ToLower(name)
	switch normalized {
	case "image_url", "video_url":
		if identifier := mediaIdentifier(value); identifier != "" {
			addMaxWeight(hashToWeight, contentHash(normalized, identifier), 1)
			return
		}
	case "input_audio":
		if identifier := mediaIdentifier(value); identifier != "" {
			addMaxWeight(hashToWeight, contentHash(normalized, identifier), 1)
			return
		}
	}
	walkAny(value, hashToWeight)
}

func mediaIdentifier(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case map[string]any:
		for _, key := range []string{"url", "data"} {
			if raw, ok := typed[key].(string); ok {
				return raw
			}
		}
	default:
		bytes, err := json.Marshal(typed)
		if err == nil && string(bytes) != "null" {
			return string(bytes)
		}
	}
	return ""
}

func contentHash(kind, identifier string) string {
	sum := sha256.Sum256([]byte(kind + "\x00" + identifier))
	return kind + ":" + hex.EncodeToString(sum[:])
}

func addMaxWeight(hashToWeight map[string]int, hash string, weight int) {
	if hash == "" {
		return
	}
	if weight <= 0 {
		weight = 1
	}
	if current, ok := hashToWeight[hash]; !ok || weight > current {
		hashToWeight[hash] = weight
	}
}

func emptyToNil(hashToWeight map[string]int) map[string]int {
	if len(hashToWeight) == 0 {
		return nil
	}
	return hashToWeight
}

func totalWeight(hashToWeight map[string]int) int {
	total := 0
	for _, weight := range hashToWeight {
		total += weight
	}
	return total
}

func (p *Producer) matchedHashesForPod(pod string, hashToWeight map[string]int) map[string]int {
	matched := map[string]int{}
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	hashes := make([]string, 0, len(hashToWeight))
	for hash := range hashToWeight {
		hashes = append(hashes, hash)
	}
	sort.Strings(hashes)
	for _, hash := range hashes {
		pods, ok := p.cache.Get(hash)
		if !ok {
			continue
		}
		if _, ok := pods[pod]; ok {
			matched[hash] = hashToWeight[hash]
		}
	}
	return matched
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

func cleanRequestStates(ctx context.Context, cache *ttlcache.Cache[string, map[string]int], interval time.Duration) {
	ticker := time.NewTicker(interval)
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
