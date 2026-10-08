/*
Copyright 2025 The Kubernetes Authors.
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
	"maps"
	"sync"
	"time"

	"github.com/hashicorp/golang-lru/v2/simplelru"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// indexer implements the indexerInterface interface. It keeps one LRU of block
// hashes per pod and answers prefix queries per candidate pod.
type indexer struct {
	mu             sync.RWMutex // guards pods; each podCache guards its own LRU
	pods           map[ServerID]*podCache
	defaultLRUSize int
	pluginName     string
	pluginType     string
}

type podCache struct {
	mu  sync.RWMutex
	lru *simplelru.LRU[blockHash, struct{}]
}

// newIndexer initializes an indexer with size limits and starts cache size reporting.
func newIndexer(ctx context.Context, defaultLRUSize int, pluginName, pluginType string) indexerInterface {
	i := &indexer{
		pods:           make(map[ServerID]*podCache),
		defaultLRUSize: defaultLRUSize,
		pluginName:     pluginName,
		pluginType:     pluginType,
	}

	go i.reportLRUSize(ctx, time.Second)
	return i
}

func (i *indexer) podCacheFor(pod server) *podCache {
	i.mu.RLock()
	pc := i.pods[pod.ServerID]
	i.mu.RUnlock()
	if pc != nil {
		return pc
	}

	i.mu.Lock()
	defer i.mu.Unlock()
	if pc = i.pods[pod.ServerID]; pc == nil {
		lruSize := pod.NumOfGPUBlocks
		if lruSize <= 0 {
			lruSize = i.defaultLRUSize
		}
		// We ignore the error since the only possible error is if size <= 0.
		l, _ := simplelru.NewLRU[blockHash, struct{}](lruSize, nil)
		pc = &podCache{lru: l}
		i.pods[pod.ServerID] = pc
	}
	return pc
}

// Add inserts hashes tail-first: matching is anchored at the first block, so
// the leading blocks are the most valuable entries and must stay cached
// longest; an oversized batch naturally keeps exactly its leading blocks.
// Because block hashes are chained, every Add that inserts block j of a prompt
// also inserts blocks 0..j-1 after it, so within a pod's LRU the blocks of any
// prompt that remain cached form a contiguous run from block 0, which is what
// the galloping search in MatchLongestPrefix requires.
//
// If RemovePod runs between looking up the pod's cache and inserting, the
// insert lands in the removed cache and is discarded with it, as if Add had
// completed before RemovePod.
func (i *indexer) Add(hashes []blockHash, pod server) {
	pc := i.podCacheFor(pod)
	pc.mu.Lock()
	defer pc.mu.Unlock()
	for idx := len(hashes) - 1; idx >= 0; idx-- {
		pc.lru.Add(hashes[idx], struct{}{})
	}
}

// MatchLongestPrefix finds each candidate's run by galloping search over its
// LRU.
func (i *indexer) MatchLongestPrefix(hashes []blockHash, candidates []ServerID) []int {
	res := make([]int, len(candidates))
	if len(hashes) == 0 {
		return res
	}
	// Release i.mu before waiting on pod locks, so a pending RemovePod or new
	// pod does not stall other queries behind an in-progress Add.
	caches := make([]*podCache, len(candidates))
	i.mu.RLock()
	for j, c := range candidates {
		caches[j] = i.pods[c]
	}
	i.mu.RUnlock()

	for j, pc := range caches {
		if pc == nil {
			continue
		}
		pc.mu.RLock()
		res[j] = runLength(pc.lru, hashes)
		pc.mu.RUnlock()
	}
	return res
}

// runLength returns the largest n such that the LRU holds hashes[n-1], relying
// on the contiguous run described on Add. It uses only Contains, which does not
// update recency, so callers need only the pod's read lock.
func runLength(l *simplelru.LRU[blockHash, struct{}], hashes []blockHash) int {
	if !l.Contains(hashes[0]) {
		return 0
	}
	// held is a length known to be held; missing is one known not to be.
	held, missing := 1, len(hashes)+1
	for step := 1; held+step < missing; step *= 2 {
		if !l.Contains(hashes[held+step-1]) {
			missing = held + step
			break
		}
		held += step
	}
	for missing-held > 1 {
		mid := held + (missing-held)/2
		if l.Contains(hashes[mid-1]) {
			held = mid
		} else {
			missing = mid
		}
	}
	return held
}

// reportLRUSize starts a goroutine that periodically reports the LRU cache size metric.
func (i *indexer) reportLRUSize(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			i.reportOnce(ctx)
		}
	}
}

func (i *indexer) reportOnce(ctx context.Context) {
	counts := i.PodBlockCounts()

	totalEntries := 0
	maxPodEntries := 0
	var maxPodName ServerID
	for pod, size := range counts {
		totalEntries += size
		if size > maxPodEntries {
			maxPodEntries = size
			maxPodName = pod
		}
	}

	numPods := len(counts)
	avg := 0.0
	if numPods > 0 {
		avg = float64(totalEntries) / float64(numPods)
	}

	recordPrefixCacheSize(i.pluginName, i.pluginType, int64(totalEntries))

	log.FromContext(ctx).V(logutil.TRACE).Info("Prefix cache state",
		"total entries", totalEntries,
		"# pods", numPods,
		"avg entries per pod", avg,
		"pod with max cache", maxPodName,
		"max pod size", maxPodEntries,
		"global max LRU cache capacity per pod", i.defaultLRUSize,
	)
}

// RemovePod removes a pod and its associated entries from the indexer.
func (i *indexer) RemovePod(pod ServerID) {
	i.mu.Lock()
	defer i.mu.Unlock()
	delete(i.pods, pod)
}

// Pods returns the list of all pods currently tracked in the indexer.
func (i *indexer) Pods() []ServerID {
	i.mu.RLock()
	defer i.mu.RUnlock()

	pods := make([]ServerID, 0, len(i.pods))
	for pod := range i.pods {
		pods = append(pods, pod)
	}
	return pods
}

// PodBlockCounts returns the number of cached blocks currently tracked per pod.
func (i *indexer) PodBlockCounts() map[ServerID]int {
	i.mu.RLock()
	caches := maps.Clone(i.pods)
	i.mu.RUnlock()

	counts := make(map[ServerID]int, len(caches))
	for pod, pc := range caches {
		pc.mu.RLock()
		counts[pod] = pc.lru.Len()
		pc.mu.RUnlock()
	}
	return counts
}
