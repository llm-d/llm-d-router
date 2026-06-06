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

package multimodal

import (
	k8stypes "k8s.io/apimachinery/pkg/types"
)

// cacheSnapshot returns a hash→pod-set view of the per-endpoint caches for assertions.
func (p *Producer) cacheSnapshot() map[string]map[string]struct{} {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	snapshot := map[string]map[string]struct{}{}
	for pod, podCache := range p.caches {
		for _, hash := range podCache.Keys() {
			if snapshot[hash] == nil {
				snapshot[hash] = map[string]struct{}{}
			}
			snapshot[hash][pod] = struct{}{}
		}
	}
	return snapshot
}

func (p *Producer) putCacheEntry(hash string, pods ...k8stypes.NamespacedName) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	for _, pod := range pods {
		p.getOrCreatePodCache(pod.String()).Add(hash, struct{}{})
	}
}
